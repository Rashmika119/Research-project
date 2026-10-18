"""Final-study contract checks: no optimizer steps or research training."""
import ast
import copy
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

import nbformat
import torch
import yaml
from torch_geometric.nn import RGATConv
from tests.helpers import fake_lm
from tests.report_fixtures import fixture_results
from training.final_study import CONFIGURATIONS, SEEDS, COUNTS, make_config, expected_budget, build_commands, main, preflight, validate_seeds
from training.final_reports import final_report, validate_runs, PLOTS
from training.research_pipeline import model_diagnostics, state_fingerprint, trainable_state, load_model_state, run_experiment
from training.reproducibility import seed_everything, assert_exact_state
from training.variants import build_scratch_model
from training.experiment_io import write_json, file_digest

ROOT = Path(__file__).resolve().parents[1]


def fixtures():
    """Fabricated test-only records; never saved as research measurements."""
    templates = fixture_results()
    results = []
    for i,c in enumerate(CONFIGURATIONS):
        for s in SEEDS:
            r = copy.deepcopy(templates[i])
            r.update(configuration=c, architecture=CONFIGURATIONS[c][0], settings=asdict(make_config(c,s)),
                     initial_structural_sha256=f'structural-{s}', initial_projection_sha256=None if c == 'baseline' else f'projection-{s}',
                     environment={'git_commit':'fixture','source_sha256':'fixture'}, numerical_policy={'deterministic_algorithms':True},
                     initial_validation_epoch=None if c == 'baseline' else 10, final_epoch=100, peak_gpu_reserved_bytes=5*2**30,
                     gpu_memory_start={'free_bytes':14*2**30,'total_bytes':15*2**30},
                     architecture_diagnostics={'trainable_parameters':(1000,2000)[i], 'unique_rgat_layers':2,
                                               'warmup_layer_executions':2,'integrated_layer_executions':2 if c == 'baseline' else 4},
                     learned_gates={} if c == 'baseline' else {'entity_text_gate':.01,'relation_text_gate':.01,'refinement_gate':.01})
            for h in r['history']:
                h.update(peak_gpu_reserved_bytes=5*2**30,learned_gates=r['learned_gates'].copy())
                if 'validation' in h:
                    h['validation']['MRR'] += i*.001+s*.0001
            r['validation']=r['history'][74]['validation'].copy()
            r['final_epoch_validation']=r['history'][-1]['validation'].copy()
            results.append(r)
    return results


class FinalStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    @patch('models.frozen_lm.load_pretrained_model',side_effect=fake_lm)
    def test_architectures_pairing_warmup_and_two_path_gradients(self,_):
        edges=torch.tensor([[0,1,2,3,1,2,3,0],[1,2,3,0,0,1,2,3]])
        types=torch.tensor([0,1,0,1,2,3,2,3])
        for seed in SEEDS:
            models=[]
            for c in CONFIGURATIONS:
                seed_everything(seed)
                with patch('torch.load',side_effect=AssertionError('No checkpoint transfer')):
                    m=build_scratch_model(make_config(c,seed).variant,4,2).eval()
                models.append(m)
            self.assertEqual(state_fingerprint(models[0]),state_fingerprint(models[1].structural))
            self.assertFalse(hasattr(models[0], 'bridge'))
            self.assertGreater(model_diagnostics(models[1])['trainable_parameters'],model_diagnostics(models[0])['trainable_parameters'])
            with self.assertRaises(ValueError):load_model_state(models[1],trainable_state(models[0]))
            for i,m in enumerate(models):
                layers=[l for l in m.modules() if isinstance(l,RGATConv)]
                self.assertEqual(len(layers),2)

                calls={id(l):0 for l in layers}
                def hook(l,*_): calls[id(l)]+=1
                handles=[l.register_forward_hook(hook) for l in layers]
                structural = m if i == 0 else m.structural
                h=structural.encode(edges,types)
                self.assertEqual(list(calls.values()),[1,1])
                h.square().mean().backward()
                if i == 0:
                    for handle in handles: handle.remove()
                    continue
                self.assertTrue(all(p.grad is None for p in m.bridge.parameters()))
                m.zero_grad(set_to_none=True)
                calls.update({k:0 for k in calls})
                ep=torch.randn(4,m.bridge.lm.hidden_size);rp=torch.randn(2,m.bridge.lm.hidden_size)
                with patch.object(m.bridge.lm,'encode_text',side_effect=AssertionError('Pooled cache bypasses LM')):
                    e,r=m(edges,types,torch.arange(4),torch.arange(2),None,None,edges,entity_pooled=ep,relation_pooled=rp)
                self.assertEqual(list(calls.values()),[2,2])
                for handle in handles:handle.remove()
                (e.square().mean()+r.square().mean()).backward()
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in m.bridge.lm_to_kg.parameters()))
                self.assertTrue(all(p.grad is None for p in m.bridge.lm.parameters()))
                optimizer=torch.optim.Adam(p for p in m.parameters() if p.requires_grad)
                params=optimizer.param_groups[0]['params']
                self.assertEqual(len(params),len({id(p) for p in params}))
                # Chain-rule audit: total shared gradient = structural path + refinement path.
                params=[m.structural.encoder.layers[j].weight for j in (0,1)]
                u=m.structural.encode(edges,types)+m.entity_text_gate.tanh()*m.bridge(pooled=ep)
                gate=m.refinement_gate.tanh()
                loss=(u+gate*m.structural.refine(u,edges,types)).square().mean()
                gu,*total=torch.autograd.grad(loss,[u,*params],retain_graph=True)
                first=torch.autograd.grad(u,params,grad_outputs=gu,retain_graph=True)
                ud=u.detach()
                second=torch.autograd.grad((ud+gate*m.structural.refine(ud,edges,types)).square().mean(),params)
                for g,a,b in zip(total,first,second):
                    self.assertGreater(a.abs().sum(),0);self.assertGreater(b.abs().sum(),0)
                    torch.testing.assert_close(g,a+b,rtol=1e-5,atol=1e-7)

    def test_protocol_two_commands_and_seed_zero_only(self):
        protocol=yaml.safe_load((ROOT/'experiments/configs/baseline_shared_c.yaml').read_text(encoding='utf-8'))
        self.assertEqual(list(CONFIGURATIONS),list(protocol['configurations']))
        self.assertEqual(tuple(protocol['seeds']),SEEDS)
        args=SimpleNamespace(seeds=SEEDS,raw_dir='raw',text_dir='text',cache_dir='cache',text_batch_size=256,eval_batch_size=512,skip_completed=True,resume=True)
        commands=build_commands(args,Path('fresh'),Path('manifest.json'))
        self.assertEqual(len(commands),2);self.assertEqual(len({str(d) for _,_,d,_ in commands}),2)
        for c,s,_,cmd in commands:
            cfg=make_config(c,s)
            self.assertEqual((cfg.max_entities,cfg.epochs,cfg.warmup),(0,100,CONFIGURATIONS[c][1]))
            self.assertEqual(expected_budget(c)['optimizer_steps'],900)
            for k in ('model','lr','train_batch_size','epochs','text_batch_size','eval_batch_size'):
                self.assertEqual(asdict(cfg)[k],protocol[k])
            with patch.object(sys,'argv',cmd[2:]),patch('torch.cuda.is_available',return_value=True),patch('training.final_study.run_experiment') as run:
                main();self.assertEqual(run.call_args.args[0].variant,cfg.variant)
        with self.assertRaises(ValueError):validate_seeds([0,1,2])
        with self.assertRaises(ValueError):make_config('shared-c-gated',1)
        with self.assertRaises(ValueError):validate_seeds([0,0])

    def test_report_statistics_selection_and_rejections(self):
        rows=fixtures()
        with tempfile.TemporaryDirectory() as d:
            summary=final_report(d,rows)
            self.assertEqual(summary['selected_by_validation'],'shared-c-gated')
            self.assertIsNone(summary['configurations']['shared-c-gated']['validation']['MRR']['std'])
            for p in PLOTS:
                for ext in ('png','pdf'):self.assertGreater((Path(d)/(p+'.'+ext)).stat().st_size,1000)
            self.assertEqual(len((Path(d)/'steps.csv').read_text().splitlines()),1801)
        mutations=[lambda r:r.pop(),lambda r:r[0].update(best_epoch=10),lambda r:r[0]['settings'].update(max_entities=2000),
                   lambda r:r[0].update(initial_projection_sha256='wrong'),lambda r:r[0]['history'][0]['step_losses'].pop(),
                   lambda r:r[0]['architecture_diagnostics'].update(unique_rgat_layers=4),
                   lambda r:r[0]['learned_gates'].update(refinement_gate=.01),lambda r:r[0]['history'][99].update(example_weighted_loss=999)]
        for mutate in mutations:
            changed=copy.deepcopy(rows);mutate(changed)
            with self.assertRaises(ValueError):validate_runs(changed)

    def test_parent_orchestration_report_only_and_oom_diagnostics(self):
        with tempfile.TemporaryDirectory() as d:
            records={(r['configuration'],r['settings']['seed']):r for r in fixtures()}
            def child(cmd,log):
                key=(cmd[cmd.index('--single')+1],int(cmd[cmd.index('--seed')+1]))
                write_json(Path(cmd[cmd.index('--output-dir')+1])/'results.json',records[key])
            manifest={'statistics':dict(zip(('entities','relations','train_triples','valid_triples','test_triples'),COUNTS)),
                      'max_entities':0,'selection_strategy':'full_original_splits','sha256':'fixture'}
            argv=['runner','--output-dir',d,'--resume','--skip-completed']
            with patch.object(sys,'argv',argv),patch('torch.cuda.is_available',return_value=True),patch('training.final_study.prepare_data',return_value=(None,None,None,manifest)),patch('training.final_study.run_logged',side_effect=child) as launch,patch('training.final_reports.final_report',return_value={'selected_by_validation':'fixture'}),redirect_stdout(io.StringIO()):
                main();self.assertEqual(launch.call_count,2)
                with patch.object(sys,'argv',argv+['--seeds','0','1','2','3','4']):
                    with self.assertRaisesRegex(ValueError,'only seed 0'):main()
                with patch.object(sys,'argv',argv+['--report-only']),patch('torch.cuda.is_available',side_effect=AssertionError('No CUDA check')),patch('torch.load',side_effect=AssertionError('No checkpoint loading')):
                    main()
                self.assertEqual(launch.call_count,2)
        with tempfile.TemporaryDirectory() as d:
            def fail(*args,**kw):
                kw['progress'].update(epoch=11,phase='main',operation='training',optimizer_steps=90)
                raise torch.cuda.OutOfMemoryError('fixture OOM')
            with patch('training.research_pipeline._run_experiment',side_effect=fail):
                with self.assertRaises(torch.cuda.OutOfMemoryError):run_experiment(make_config('shared-c-gated',0),d)
            failure=json.loads((Path(d)/'failure.json').read_text())
            self.assertEqual((failure['seed'],failure['epoch'],failure['phase'],failure['optimizer_steps']),(0,11,'main',90))

    def test_preflight_config_runtime_and_checkpoint_hash(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=make_config('shared-c-gated',0);name='shared-c-gated_seed0'
            manifest={'max_entities':0,'selection_strategy':'full_original_splits','sha256':'fixture','statistics':dict(zip(('entities','relations','train_triples','valid_triples','test_triples'),COUNTS))}
            env=dict(source_sha256='s',packages={},python='p',cuda='c',device='d')
            run=Path(d)/name;run.mkdir();(run/'best.pt').write_bytes(b'fixture')
            config=dict(config=asdict(cfg),manifest_sha256='fixture',run_id='id',**env)
            write_json(run/'config.json',config)
            write_json(run/'results.json',{'run_id':'id','status':'complete','checkpoint_sha256':{'best.pt':file_digest(run/'best.pt')}})
            with patch('training.experiment_io.environment',return_value=env),patch('torch.load',side_effect=AssertionError('No tensor load')),redirect_stdout(io.StringIO()):
                preflight(d,{name:cfg},manifest)
                (run/'best.pt').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'Changed checkpoint'):preflight(d,{name:cfg},manifest)
                config['source_sha256']='wrong';write_json(run/'config.json',config)
                with self.assertRaisesRegex(ValueError,'source/runtime'):preflight(d,{name:cfg},manifest)

    def test_notebook_syntax_protection_and_ordered_mock_execution(self):
        from notebooks.build_final_comparison_notebook import notebook,build
        nb=json.loads((ROOT/'notebooks/baseline_vs_shared_c_full.ipynb').read_text(encoding='utf-8'))
        self.assertEqual(nb,notebook());nbformat.validate(nbformat.from_dict(nb))
        for c in nb['cells']:
            if c['cell_type']=='code':ast.parse(c['source'])
        with self.assertRaises(FileExistsError):build()
        with tempfile.TemporaryDirectory() as d:
            calls=[]
            def output(cmd,**kw):
                if '-c' in cmd:return str(torch.__version__)
                return '' if 'status' in cmd else 'a'*40
            def prepare(raw,count,seed,path):
                self.assertEqual(count,0)
                manifest={'max_entities':0,'selection_strategy':'full_original_splits','sha256':'fixture','dataset_sha256':'fixture',
                          'statistics':dict(zip(('entities','relations','train_triples','valid_triples','test_triples'),COUNTS))}
                write_json(path,manifest)
                return SimpleNamespace(train=[(0,0,1)]),SimpleNamespace(edge_index=range(544230)),[(0,0,1)],manifest
            def research(cmd,log):
                calls.append(cmd);dest=Path(cmd[cmd.index('--output-dir')+1])
                for r in fixtures():write_json(dest/f"{r['configuration']}_seed{r['settings']['seed']}"/'results.json',r)
            scope={'__name__':'__notebook_test__'}
            with patch.dict(sys.modules,{'google.colab':SimpleNamespace(drive=SimpleNamespace(mount=lambda p:None))}),patch.dict(os.environ),patch('os.chdir'),patch('subprocess.run',return_value=subprocess.CompletedProcess([],0)),patch('subprocess.check_output',side_effect=output),patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.get_device_name',return_value='fixture'),patch('torch.cuda.get_device_properties',return_value=SimpleNamespace(total_memory=15*2**30)),patch('torch.cuda.mem_get_info',return_value=(14*2**30,15*2**30)),patch('training.reproducibility.seed_everything'),patch('training.experiment_io.environment',return_value={}),patch('training.experiment_io.prepare_data',side_effect=prepare),patch('models.pretrained.prepare_pretrained',return_value=SimpleNamespace(path='fixture',revision='e2da8e2f811d1448a5b465c236feacd80ffbac7b',files=[])),patch('training.process.run_logged',side_effect=research),patch('IPython.display.display'),redirect_stdout(io.StringIO()):
                for i,c in enumerate(nb['cells']):
                    if c['cell_type']!='code':continue
                    s=c['source']
                    s=s.replace("Path('/content/Research-project')",f'Path({str(ROOT)!r})')
                    for folder in ('scratch_pipeline_v1','baseline_shared_c_100'):
                        s=s.replace(f"Path('/content/drive/MyDrive/Research/{folder}')",f'Path({str(Path(d)/folder)!r})')
                    exec(compile(s,f'final_cell_{i}','exec'),scope)
                    if i<12:self.assertFalse(calls)
            self.assertEqual(len(calls),1);self.assertIn('run_final_comparison.py',calls[0])
            self.assertTrue((scope['EXPERIMENT_DIR']/'summary.json').exists())
            lock = scope['LOCK'].read_text(encoding='utf-8')
            self.assertEqual(len(lock.splitlines()), len(scope['packages']))
            self.assertNotIn('\\n', lock)

if __name__=='__main__':unittest.main()
