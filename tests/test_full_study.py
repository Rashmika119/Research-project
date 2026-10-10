"""Non-training verification only: contracts, reports, notebook, cache and wiring.

All scores below are fabricated report fixtures, never research measurements.
No forward/backward, optimizer step, GPU workload or model download is executed.
"""
import ast
import copy
from dataclasses import asdict
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

import nbformat
import torch
import yaml

from training.full_study import CONFIGURATIONS, SEEDS, COUNTS, make_config, expected_budget, build_commands, verify_manifest
from training.full_reports import full_report, validate_runs, PLOTS
from training.research_pipeline import ExperimentConfig, validation_due, can_select_initial

ROOT = Path(__file__).resolve().parents[1]
PROTECTED = {
    'notebooks/research_pipeline.ipynb': '24fc8df77eff6473b016c7ee84cc7e1384af88d33924ad8bc5a917501d2270c8',
    'notebooks/warmup_ablation_5000.ipynb': 'd5137bc4e16917fb8b46bf641cdbd532815c5bcfa5cc32f1fbfb031e979973f4',
}


def fixture_results():
    def metrics(mrr):
        return {'MRR': mrr, 'Hits@1': mrr/2, 'Hits@3': mrr, 'Hits@10': mrr*2,
                'Optimistic_MRR': mrr+.001, 'Tie_query_fraction': .01,
                'Mean_tied_candidates': 1.01, 'Max_tied_candidates': 2}
    results = []
    for index, c in enumerate(CONFIGURATIONS):
        for seed in SEEDS:
            cfg = make_config(c, seed)
            history = []
            for e in range(1, 301):
                h = {'epoch': e, 'phase': 'structural_warmup' if e <= cfg.warmup else 'main',
                     'loss': 1.5-e*.001, 'example_weighted_loss': 1.5-e*.001,
                     'optimizer_steps': e*9, 'train_seconds': 1., 'epoch_seconds': 2.,
                     'step_losses': [{'optimizer_step': (e-1)*9+s+1, 'loss': 1.5-e*.001,
                                      'positive_count': 32768 if s < 8 else 9971} for s in range(9)]}
                if e % 5 == 0:
                    h['validation'] = metrics(.8 if e <= cfg.warmup else
                        .05 + index*.01 + seed*.001 + (min(e,250)-max(0,e-250)*.1)*.0001)
                    h['validation_model'] = 'baseline' if c == 'baseline' else 'structural' if e <= cfg.warmup else 'integrated'
                history.append(h)
            results.append({'configuration': c, 'architecture': cfg.variant, 'settings': asdict(cfg),
                'status': 'complete', 'initialization': 'scratch', 'baseline_checkpoint_used': False,
                'dataset_scope': 'full', 'candidate_entities': 14541, 'ranking_policy': 'average_exact_ties',
                'manifest_sha256': 'fabricated-manifest', 'dataset_sha256': 'fabricated-data',
                'environment': {'source_sha256': 'fixture'}, 'numerical_policy': {'fixture': True},
                'lm_identity': None if c == 'baseline' else {'name': 'fixture-LM'},
                'text_cache_keys': None if c == 'baseline' else {'entities': 'fixture', 'relations': 'fixture'},
                'initial_structural_sha256': f'paired-structural-{seed}',
                'initial_trainable_sha256': f'{cfg.variant}-{seed}',
                'warmup_epochs': cfg.warmup, 'main_epochs': 300-cfg.warmup,
                **expected_budget(c), 'best_epoch': 250, 'validation': history[249]['validation'],
                'test': metrics(.2-index*.01), 'final_epoch_validation': history[-1]['validation'],
                'initial_validation': None if c == 'baseline' else metrics(.9),
                'duration_seconds': 700.+index*100+seed, 'peak_gpu_memory_bytes': (index+1)*2**30,
                'final_training_loss': history[-1]['loss'], 'history': history,
                'subset_statistics': dict(zip(('entities','relations','train_triples','valid_triples','test_triples'), COUNTS))})
    return results


class FullStudyTests(unittest.TestCase):
    def test_configs_budgets_schedule_and_pairing(self):
        self.assertEqual(divmod(272115,32768), (8,9971))
        for c in CONFIGURATIONS:
            for seed in SEEDS:
                cfg = make_config(c, seed)
                self.assertEqual([e for e in range(1,301) if validation_due(cfg,e,cfg.warmup)], list(range(5,301,5)))
                self.assertFalse(can_select_initial(cfg))
                self.assertEqual(sum(expected_budget(c)[k] for k in ('baseline_optimizer_steps','structural_optimizer_steps','integrated_optimizer_steps')),2700)
        self.assertTrue(can_select_initial(ExperimentConfig()))
        a,b = (asdict(make_config(c,0)) for c in list(CONFIGURATIONS)[1:])
        self.assertEqual({k for k in a if a[k] != b[k]}, {'warmup'})
        protocol = yaml.safe_load((ROOT/'experiments/configs/full_dataset_300.yaml').read_text())
        self.assertEqual(list(protocol['configurations']), list(CONFIGURATIONS))
        for key in ('epochs','train_batch_size','lr','weight_decay','num_negatives','model','text_batch_size','eval_batch_size','max_text_length','eval_every','selection_policy','record_step_losses'):
            self.assertEqual(asdict(make_config('baseline',0))[key],protocol[key])

    def test_manifest_contract(self):
        m = {'statistics': dict(zip(('entities','relations','train_triples','valid_triples','test_triples'),COUNTS)),
             'max_entities':0,'selection_strategy':'full_original_splits'}
        verify_manifest(m)
        m['statistics']['entities'] = 5000
        with self.assertRaises(ValueError): verify_manifest(m)

    def test_nine_independent_commands_and_child_configs(self):
        args = SimpleNamespace(raw_dir='raw',text_dir='text',cache_dir='cache',text_batch_size=256,
                               eval_batch_size=512,skip_completed=True,resume=True)
        commands = build_commands(args,Path('fresh'),Path('fresh/dataset_manifest.json'))
        self.assertEqual(len(commands),9)
        self.assertEqual(len({str(x[2]) for x in commands}),9)
        from training.full_study import main
        for c,s,d,command in commands:
            self.assertNotIn('--checkpoint',command)
            self.assertIn('--resume',command)
            with patch('sys.argv',command[2:]), patch('torch.cuda.is_available',return_value=True), \
                 patch('training.full_study.run_experiment') as run:
                main()
                cfg = run.call_args.args[0]
                self.assertEqual(cfg,make_config(c,s,raw_dir='raw',text_dir='text'))
                self.assertEqual(Path(run.call_args.kwargs['manifest_path']),Path('fresh/dataset_manifest.json'))

    def test_report_statistics_plots_and_selection(self):
        runs = fixture_results()
        with tempfile.TemporaryDirectory() as directory:
            summary = full_report(directory,runs)
            self.assertEqual(summary['selected_by_validation'],'residual-no-softprompt-no-warmup')
            self.assertAlmostEqual(summary['configurations']['baseline']['validation']['MRR']['std'],.001)
            self.assertEqual(len(summary['paired_comparisons']),33)
            for name in PLOTS:
                p=Path(directory)/name
                self.assertGreater(p.stat().st_size,1000)
                self.assertEqual(p.read_bytes()[:8],b'\x89PNG\r\n\x1a\n')
            self.assertEqual(len((Path(directory)/'steps.csv').read_text().splitlines()),24301)
            self.assertIn('not evidence of statistical significance', (Path(directory)/'research_analysis.md').read_text())
            behavior=json.loads((Path(directory)/'training_behavior.json').read_text())
            self.assertTrue(all(r['best_before_final'] for r in behavior))

    def test_parent_orchestration_and_changed_settings_guard(self):
        from training.full_study import main
        from training.experiment_io import write_json
        fixture={(r['configuration'],r['settings']['seed']):r for r in fixture_results()}
        with tempfile.TemporaryDirectory() as directory:
            def prepare(raw,count,seed,path):
                m={'statistics':dict(zip(('entities','relations','train_triples','valid_triples','test_triples'),COUNTS)),
                   'max_entities':0,'selection_strategy':'full_original_splits','sha256':'fixture'}
                write_json(path,m)
                return None,None,None,m
            def child(command,log):
                c=command[command.index('--single')+1]
                seed=int(command[command.index('--seed')+1])
                output=Path(command[command.index('--output-dir')+1])
                write_json(output/'results.json',fixture[c,seed])
            argv=['runner','--output-dir',directory,'--skip-completed','--resume']
            with (patch('torch.cuda.is_available',return_value=True),patch('sys.argv',argv),
                  patch('training.full_study.prepare_data',side_effect=prepare),
                  patch('training.full_study.run_logged',side_effect=child) as launch,
                  patch('training.full_reports.full_report',return_value={'selected_by_validation':'fixture'}) as report,
                  redirect_stdout(io.StringIO())):
                main()
                self.assertEqual(launch.call_count,9)
                self.assertEqual(len(report.call_args.args[1]),9)
                self.assertTrue((Path(directory)/'comparison_config.json').exists())
                with patch('sys.argv',argv+['--eval-batch-size','256']):
                    with self.assertRaisesRegex(ValueError,'settings changed'):main()
                self.assertEqual(launch.call_count,9)
                with patch('sys.argv',['runner','--output-dir',directory,'--report-only']), \
                     patch('torch.cuda.is_available',return_value=False):
                    main()
                self.assertEqual(launch.call_count,9)

    def test_reports_reject_mixed_or_incomplete_results(self):
        original=fixture_results()
        mutations = [lambda r:r.pop(), lambda r:r[0].update(manifest_sha256='other'),
            lambda r:r[3].update(initial_trainable_sha256='other'),
            lambda r:r[3].update(lm_identity={'revision':'other'}),
            lambda r:r[0].update(integrated_optimizer_steps=2700),
            lambda r:r[3].update(best_epoch=30),
            lambda r:r[0]['settings'].update(train_batch_size=16384),
            lambda r:r[0]['history'][0]['step_losses'].pop(),
            lambda r:r[0]['history'][0].update(validation=r[0]['validation'])]
        for mutation in mutations:
            runs=copy.deepcopy(original); mutation(runs)
            with self.assertRaises(ValueError): validate_runs(runs)

    def test_notebook_schema_syntax_protection_and_optional_flags(self):
        from notebooks.build_full_dataset_notebook import notebook, build
        actual=json.loads((ROOT/'notebooks/full_dataset_comparison_300.ipynb').read_text())
        self.assertEqual(actual,notebook())
        nbformat.validate(nbformat.from_dict(actual))
        sections=[c['source'] for c in actual['cells'] if c['cell_type']=='markdown' and c['source'].startswith('## ')]
        self.assertEqual(len(sections),8)
        for c in actual['cells']:
            if c['cell_type']=='code':
                ast.parse(c['source'])
                self.assertEqual(c['outputs'],[])
        config=actual['cells'][6]['source']
        for flag in ('RUN_OFFLINE_TRAINING_TESTS','RUN_REAL_LM_SMOKE_TRAINING','RUN_CUDA_RESUME_SMOKE_TRAINING','RUN_SYNTHETIC_TRAINING_TESTS'):
            self.assertIn(flag+' = False',config)
        # Actually execute the optional cell with all flags disabled; any training call fails.
        optional=actual['cells'][10]['source']
        ns={'os':__import__('os'),'RUN_OFFLINE_TRAINING_TESTS':False,'RUN_REAL_LM_SMOKE_TRAINING':False,
            'RUN_CUDA_RESUME_SMOKE_TRAINING':False,'RUN_SYNTHETIC_TRAINING_TESTS':False,
            'run_logged':lambda *a,**k:self.fail('Optional training ran automatically')}
        exec(optional,ns)
        with self.assertRaises(FileExistsError):build()
        from notebooks.build_warmup_ablation_notebook import build as old_build
        with self.assertRaises(FileExistsError):old_build()
        for path,sha in PROTECTED.items():
            self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),sha)

    def test_no_training_or_lm_for_baseline_preparation(self):
        from training.variants import build_scratch_model, ARCHITECTURES
        self.assertIn('no-refinement',ARCHITECTURES)
        self.assertNotIn('original',ARCHITECTURES)
        with patch('training.variants.KGTextRefinement', side_effect=AssertionError('Baseline used LM')):
            model=build_scratch_model('baseline',3,2)
            self.assertFalse(hasattr(model,'bridge'))
        from training import research_pipeline
        source=inspect.getsource(research_pipeline.run_experiment)
        self.assertIn("if can_select_initial(cfg):",source)
        self.assertIn("'baseline_optimizer_steps': optimizer_steps if cfg.variant == 'baseline' else 0",source)

    def test_cached_pooled_text_does_not_encode_or_move_lm(self):
        from training.text_cache import pooled_text
        lm = torch.nn.Linear(2,2)
        lm.hidden_size = 2
        lm.lm = SimpleNamespace(config=SimpleNamespace(_attn_implementation='fixture'))
        tokens = {'input_ids':torch.ones(3,4,dtype=torch.long),'attention_mask':torch.ones(3,4,dtype=torch.long)}
        with tempfile.TemporaryDirectory() as directory:
            torch.save({'key':'fixture','pooled':torch.zeros(3,2)},Path(directory)/'fixture.pt')
            with patch('training.text_cache.lm_identity',return_value={'fixture':True}), \
                 patch('training.text_cache.digest',return_value='fixture'), \
                 patch.object(lm,'to',side_effect=AssertionError('Cache hit moved LM')):
                pooled,key=pooled_text(lm,tokens,256,torch.device('cpu'),directory)
                self.assertEqual(key,'fixture')
                self.assertEqual(tuple(pooled.shape),(3,2))
                self.assertFalse(pooled.requires_grad)

    def test_full_notebook_ordered_execution_with_external_actions_mocked(self):
        from notebooks.build_full_dataset_notebook import notebook
        from training.experiment_io import write_json
        calls=[]
        with tempfile.TemporaryDirectory() as directory:
            def fake_run(command, **kwargs):
                calls.append(list(map(str,command)))
                return subprocess.CompletedProcess(command,0)
            def fake_output(command, **kwargs):
                if '-c' in command:
                    return str(torch.__version__)
                return '' if 'status' in command else 'fixture-commit'
            def fake_prepare(raw, count, seed, manifest_path):
                self.assertEqual(count,0)
                m={'statistics':dict(zip(('entities','relations','train_triples','valid_triples','test_triples'),COUNTS)),
                   'max_entities':0,'selection_strategy':'full_original_splits','sha256':'fixture','dataset_sha256':'fixture'}
                write_json(manifest_path,m)
                return SimpleNamespace(train=[(0,0,1)]),SimpleNamespace(edge_index=range(544230)),[(0,0,1)],m
            def fake_research(command, log_path, **kwargs):
                calls.append(list(map(str,command)))
                self.assertIn('run_full_dataset_comparison.py',command)
                dest=Path(command[command.index('--output-dir')+1])
                for r in fixture_results():
                    write_json(dest/f"{r['configuration']}_seed{r['settings']['seed']}"/'results.json',r)
            scope={'__name__':'__notebook_test__'}
            with (patch.dict(sys.modules,{'google.colab':SimpleNamespace(drive=SimpleNamespace(mount=lambda p:None))}),
                  patch.dict(os.environ),patch('os.chdir'),patch('subprocess.run',side_effect=fake_run),
                  patch('subprocess.check_output',side_effect=fake_output),
                  patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.get_device_name',return_value='T4 fixture'),
                  patch('torch.cuda.get_device_properties',return_value=SimpleNamespace(total_memory=16*2**30)),
                  patch('torch.cuda.mem_get_info',return_value=(15*2**30,16*2**30)),
                  patch('training.experiment_io.environment',return_value={'fixture':True}),
                  patch('training.experiment_io.prepare_data',side_effect=fake_prepare),
                  patch('models.pretrained.prepare_pretrained',return_value=SimpleNamespace(path='verified-fixture',revision='fixture',files=[])),
                  patch('training.process.run_logged',side_effect=fake_research),patch('IPython.display.display'),
                  redirect_stdout(io.StringIO())):
                for i,cell in enumerate(notebook()['cells']):
                    if cell['cell_type']!='code':continue
                    source=cell['source'].replace("Path('/content/Research-project')",f'Path({str(ROOT)!r})')
                    source=source.replace("Path('/content/drive/MyDrive/Research/scratch_pipeline_v1')",f'Path({str(Path(directory)/"shared")!r})')
                    source=source.replace("Path('/content/drive/MyDrive/Research/full_dataset_comparison_300')",f'Path({str(Path(directory)/"stage")!r})')
                    exec(compile(source,f'full_cell_{i}','exec'),scope)
                    if i<12:
                        self.assertFalse(any('run_full_dataset_comparison.py' in c for c in calls))
            self.assertEqual(sum('run_full_dataset_comparison.py' in c for c in calls),1)
            self.assertFalse(any('run_scratch_smoke_test.py' in c or 'run_resume_diagnostic.py' in c for c in calls))
            self.assertTrue((scope['EXPERIMENT_DIR']/'summary.json').is_file())


if __name__=='__main__':
    unittest.main()
