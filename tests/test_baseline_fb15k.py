"""Offline contracts and fabricated reports only; no optimizer updates/downloads."""
import ast
import copy
from contextlib import ExitStack, redirect_stdout
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

import nbformat
import torch
import yaml

from preprocessing.dataset import load_fb15k, validate_fb15k
from tests.test_final_study import fixtures
from training.baseline_fb15k import make_config, expected_budget, verify_manifest, verify_protocol, main, COUNTS
from training.baseline_fb15k_reports import baseline_report, validate_result, PLOTS
from training.experiment_io import prepare_data, write_json
from training.research_pipeline import run_experiment

ROOT = Path(__file__).resolve().parents[1]


def manifest_fixture():
    return {'dataset':'fb15k', 'max_entities':0, 'subset_seed':0,
            'selection_strategy':'full_original_splits', 'sha256':'fixture', 'dataset_sha256':'fixture',
            'statistics':dict(zip(('entities','relations','train_triples','valid_triples','test_triples'),COUNTS))}


def result_fixture():
    r = fixtures()[0]
    r.update(settings=asdict(make_config()), candidate_entities=14951,
             final_epoch=300, main_epochs=300, best_epoch=200,
             subset_statistics=manifest_fixture()['statistics'], **{k:v for k,v in expected_budget().items() if k!='last_batch_size'})
    history = []
    for e in range(1,301):
        row = {'epoch':e, 'phase':'main', 'loss':1.5-e*.001, 'example_weighted_loss':1.5-e*.001,
               'optimizer_steps':15*e, 'train_seconds':1., 'epoch_seconds':2., 'learned_gates':{},
               'peak_gpu_memory_bytes':2**30, 'peak_gpu_reserved_bytes':2**31,
               'step_losses':[{'optimizer_step':(e-1)*15+s+1,'loss':1.5-e*.001,
                               'positive_count':32768 if s<14 else 24390} for s in range(15)]}
        if e%10==0:
            row['validation'] = dict(r['validation'], MRR=.3 if e in (200,210) else .1+e*.0001)
            row['validation_model'] = 'baseline'
        history.append(row)
    r.update(history=history, validation=history[199]['validation'].copy(),
             final_epoch_validation=history[-1]['validation'].copy(), final_training_loss=history[-1]['loss'])
    return r


class FB15KTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_width100_ten_negatives_real_baseline_forward_backward(self):
        import random
        from training.variants import build_scratch_model
        from training.negative_sampling import sample_negatives
        from training.losses import bce_loss
        cfg=make_config()
        with patch('models.frozen_lm.load_pretrained_model',side_effect=AssertionError('No LM')):
            model=build_scratch_model('baseline',4,2,cfg.model)
        edges=torch.tensor([[0,1,2,3,1,2,3,0],[1,2,3,0,0,1,2,3]])
        types=torch.tensor([0,1,0,1,2,3,2,3])
        positives=torch.tensor([[0,0,1],[1,1,2]])
        known=set(map(tuple,positives.tolist()))
        negatives=sample_negatives(positives,4,known,cfg.num_negatives,random.Random(0))
        self.assertEqual(tuple(negatives.shape),(2,10,3))
        self.assertFalse(known & set(map(tuple,negatives.reshape(-1,3).tolist())))
        entities=model.encode(edges,types)
        self.assertEqual(tuple(entities.shape),(4,100))
        self.assertEqual(entities.chunk(2,-1)[0].shape[-1],50)
        loss=bce_loss(model.score_triples(entities,positives),
                      model.score_triples(entities,negatives.reshape(-1,3)).reshape(2,10))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.entity_emb.weight.grad)
        self.assertTrue(torch.isfinite(model.entity_emb.weight.grad).all())

    def test_protocol_and_dataset_identity(self):
        cfg = make_config()
        protocol = yaml.safe_load((ROOT/'experiments/configs/fb15k_baseline_300.yaml').read_text(encoding='utf-8'))
        verify_protocol(protocol,cfg)
        self.assertEqual((cfg.dataset,cfg.model['dim'],cfg.num_negatives,cfg.epochs,cfg.warmup,cfg.seed),
                         ('fb15k',100,10,300,0,0))
        self.assertEqual(divmod(COUNTS[2],cfg.train_batch_size),(14,24390))
        verify_manifest(manifest_fixture())
        bad=manifest_fixture();bad['dataset']='fb15k237'
        with self.assertRaises(ValueError):verify_manifest(bad)
        protocol['config']['seed']=1
        with self.assertRaises(ValueError):verify_protocol(protocol,cfg)
        cfg.variant='shared-c-gated'
        with self.assertRaises(ValueError):cfg.validate()

    def test_loader_staging_mapping_rejection_and_no_fallback(self):
        content={'train.txt':'a\tr\tb\nb\tr\tc\n','valid.txt':'a\tr\tc\n','test.txt':'c\tr\ta\n'}
        def download(url,path): Path(path).write_text(content[url.rsplit('/',1)[-1]],encoding='utf-8')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'fb15k'
            with patch('preprocessing.dataset.urllib.request.urlretrieve',side_effect=download),patch('preprocessing.dataset.FB15K_COUNTS',(3,1,2,1,1)):
                data=load_fb15k(root)
                self.assertEqual(data.entity2id,{'a':0,'b':1,'c':2})
                self.assertEqual(data.train,[(0,0,1),(1,0,2)])
                with patch('preprocessing.dataset.urllib.request.urlretrieve',side_effect=AssertionError('No redownload')):
                    self.assertEqual(load_fb15k(root),data)
            with self.assertRaises(ValueError):validate_fb15k(data)
            with patch('preprocessing.dataset.urllib.request.urlretrieve',side_effect=download):
                bad=Path(directory)/'wrong_counts'
                with self.assertRaises(ValueError):load_fb15k(bad)
                self.assertFalse((bad/'train.txt').exists())
            partial=Path(directory)/'partial';partial.mkdir();(partial/'train.txt').write_text('evidence')
            with self.assertRaises(ValueError):load_fb15k(partial)
            self.assertEqual((partial/'train.txt').read_text(),'evidence')
        with patch('training.experiment_io.load_fb15k',return_value=data) as fb,patch('training.experiment_io.validate_fb15k'),patch('training.experiment_io.load_dataset',side_effect=AssertionError('Wrong dataset')):
            _,graph,known,m=prepare_data('fixture',0,0,dataset='fb15k')
            self.assertEqual(m['dataset'],'fb15k');self.assertEqual(len(graph.edge_type),4)
            self.assertEqual(set(known),set(data.train));fb.assert_called_once()
        with patch('training.research_pipeline.prepare_data',side_effect=RuntimeError('stop before training')) as prepare:
            with self.assertRaisesRegex(RuntimeError,'stop before training'):run_experiment(make_config(),'unused')
            self.assertEqual(prepare.call_args.kwargs['dataset'],'fb15k')

    def test_reports_and_rejection_of_wrong_budgets(self):
        r=result_fixture()
        with tempfile.TemporaryDirectory() as directory:
            summary=baseline_report(directory,r)
            self.assertEqual(summary['best_epoch'],200)
            self.assertEqual(len((Path(directory)/'steps.csv').read_text().splitlines()),4501)
            for p in PLOTS:
                for ext in ('png','pdf'):self.assertGreater((Path(directory)/f'{p}.{ext}').stat().st_size,1000)
        mutations=[lambda r:r.update(best_epoch=210),lambda r:r['settings'].update(dataset='fb15k237'),
                   lambda r:r.update(optimizer_steps=4499),lambda r:r['history'][-1]['step_losses'].pop(),
                   lambda r:r['history'][9].pop('validation'),lambda r:r.update(lm_identity={'unexpected':True})]
        for mutation in mutations:
            bad=copy.deepcopy(r);mutation(bad)
            with self.assertRaises(ValueError):validate_result(bad)

    def test_cli_launch_and_report_only(self):
        with tempfile.TemporaryDirectory() as directory:
            args=['runner','--output-dir',directory,'--resume','--skip-completed']
            with patch.object(sys,'argv',args),patch('torch.cuda.is_available',return_value=True),patch('training.baseline_fb15k.prepare_data',return_value=(None,None,None,manifest_fixture())) as prepare,patch('training.baseline_fb15k.run_experiment',return_value=result_fixture()) as run,patch('training.baseline_fb15k_reports.baseline_report'),redirect_stdout(io.StringIO()):
                main()
                self.assertEqual(run.call_args.args[0],make_config())
                self.assertTrue(run.call_args.kwargs['resume'])
                self.assertEqual(prepare.call_args.kwargs['dataset'],'fb15k')
                write_json(Path(directory)/'baseline_seed0/results.json',result_fixture())
                with patch.object(sys,'argv',args+['--report-only']),patch('torch.cuda.is_available',side_effect=AssertionError('No GPU access')),patch('torch.load',side_effect=AssertionError('No tensor loading')):
                    main()
                self.assertEqual(run.call_count,1)

    def test_notebook_mock_execution_and_no_language_model(self):
        from notebooks.build_fb15k_baseline_notebook import notebook,build
        nb=json.loads((ROOT/'notebooks/fb15k_baseline_300.ipynb').read_text(encoding='utf-8'))
        self.assertEqual(nb,notebook());nbformat.validate(nbformat.from_dict(nb))
        for c in nb['cells']:
            if c['cell_type']=='code':ast.parse(c['source'])
        with self.assertRaises(FileExistsError):build()
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
            calls=[]
            def prepare(raw,count,seed,path,*,dataset):
                self.assertEqual((count,seed,dataset),(0,0,'fb15k'))
                write_json(path,manifest_fixture())
                return SimpleNamespace(train=[(0,0,1)]),SimpleNamespace(edge_type=range(966284)),[(0,0,1)],manifest_fixture()
            def launch(cmd,log):
                calls.append(cmd)
                write_json(Path(cmd[cmd.index('--output-dir')+1])/'baseline_seed0/results.json',result_fixture())
            def output(cmd,**kwargs):
                return str(torch.__version__) if '-c' in cmd else '' if 'status' in cmd else 'a'*40
            patches=[patch.dict(sys.modules,{'google.colab':SimpleNamespace(drive=SimpleNamespace(mount=lambda p:None))}),
                     patch.dict(os.environ),patch('os.chdir'),patch('subprocess.run',return_value=subprocess.CompletedProcess([],0)),
                     patch('subprocess.check_output',side_effect=output),patch('torch.cuda.is_available',return_value=True),
                     patch('torch.cuda.get_device_name',return_value='fixture'),patch('torch.cuda.get_device_properties',return_value=SimpleNamespace(total_memory=96*2**30)),
                     patch('torch.cuda.mem_get_info',return_value=(95*2**30,96*2**30)),patch('training.experiment_io.environment',return_value={}),
                     patch('training.experiment_io.prepare_data',side_effect=prepare),patch('training.process.run_logged',side_effect=launch),
                     patch('models.pretrained.prepare_pretrained',side_effect=AssertionError('Baseline may not prepare LM')),
                     patch('IPython.display.display'),redirect_stdout(io.StringIO())]
            for p in patches:stack.enter_context(p)
            scope={}
            for i,c in enumerate(nb['cells']):
                if c['cell_type']!='code':continue
                source=c['source'].replace("Path('/content/Research-project')",f'Path({str(ROOT)!r})')
                for folder in ('scratch_pipeline_v1','fb15k_baseline_300'):
                    source=source.replace(f"Path('/content/drive/MyDrive/Research/{folder}')",f'Path({str(Path(directory)/folder)!r})')
                exec(compile(source,f'fb15k_cell_{i}','exec'),scope)
                if i<10:self.assertFalse(calls)
            self.assertEqual(len(calls),1)
            self.assertIn('run_baseline_fb15k.py',calls[0])
            self.assertEqual(scope['REPO_REF'],'main')
            self.assertTrue((scope['EXPERIMENT_DIR']/'summary.json').exists())
