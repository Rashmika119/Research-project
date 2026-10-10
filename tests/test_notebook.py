"""New notebook wiring and twelve-run orchestration; old notebook is a protected artifact."""
from dataclasses import asdict
import hashlib
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

from preprocessing.dataset import load_synthetic_toy_graph
from training.experiment_io import write_json
from training.ablation_reports import ablation_report, METRICS
from training.research_pipeline import ExperimentConfig
from training.variants import CONFIGURATIONS, configuration_settings

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks' / 'warmup_ablation_5000.ipynb'
OLD_NOTEBOOK = ROOT / 'notebooks' / 'research_pipeline.ipynb'
OLD_SHA256 = '24fc8df77eff6473b016c7ee84cc7e1384af88d33924ad8bc5a917501d2270c8'


def fixture_result(identifier, seed, cfg=None):
    """Fictional mechanics-only fixture. Not a scientific result."""
    architecture, warm = configuration_settings(identifier, 1)
    cfg = cfg or ExperimentConfig(variant=architecture, seed=seed, max_entities=10, epochs=3, warmup=warm)
    index = list(CONFIGURATIONS).index(identifier)
    def metrics(mrr):
        return {'MRR': mrr, 'Hits@1': .05, 'Hits@3': .1, 'Hits@10': .3,
                'Optimistic_MRR': mrr, 'Tie_query_fraction': 0., 'Mean_tied_candidates': 1., 'Max_tied_candidates': 1}
    validation, test = metrics(.1 + index * .03 + seed * .01), metrics(.8 - index * .04)
    history = [{'epoch': epoch, 'phase': 'structural_warmup' if epoch <= cfg.warmup else 'main',
                'loss': 1.4 - epoch * .01, 'validation': validation, 'optimizer_steps': epoch,
                'epoch_seconds': .1} for epoch in range(1, cfg.epochs + 1)]
    return {'configuration': identifier, 'architecture': architecture, 'settings': asdict(cfg),
            'status': 'complete', 'initialization': 'scratch', 'baseline_checkpoint_used': False,
            'dataset_scope': 'subset', 'dataset_sha256': 'fixture', 'manifest_sha256': 'same_fixture',
            'candidate_entities': cfg.max_entities, 'ranking_policy': 'average_exact_ties',
            'initial_structural_sha256': f'seed{seed}',
            'initial_trainable_sha256': f'{architecture}_seed{seed}',
            'best_epoch': cfg.epochs, 'warmup_epochs': cfg.warmup, 'main_epochs': cfg.epochs - cfg.warmup,
            'optimizer_steps': cfg.epochs, 'steps_per_epoch': 1,
            'structural_optimizer_steps': cfg.warmup, 'integrated_optimizer_steps': cfg.epochs - cfg.warmup,
            'duration_seconds': 1. + index, 'peak_gpu_memory_bytes': None,
            'history': history, 'validation': validation, 'test': test}


class NotebookTests(unittest.TestCase):
    def test_original_notebook_untouched_and_generator_cannot_target_it(self):
        self.assertEqual(hashlib.sha256(OLD_NOTEBOOK.read_bytes()).hexdigest(), OLD_SHA256)
        from notebooks import build_colab_notebook as historical
        import inspect
        self.assertNotIn("with_name('research_pipeline.ipynb')", inspect.getsource(historical.build))

    def test_schema_syntax_sections_and_builder_match(self):
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        nbformat.validate(notebook)
        from notebooks.build_warmup_ablation_notebook import cells
        # The researcher edited and executed the pilot notebook after the study.
        # Validate its actual syntax, then validate the historical template separately.
        # Never demand regeneration to make a manual notebook match its old builder.
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                compile(cell.source, '<manual pilot cell>', 'exec')
        notebook = nbformat.from_dict({'cells': cells, 'metadata': {}, 'nbformat': 4, 'nbformat_minor': 5})
        sections = [c.source for c in notebook.cells if c.cell_type == 'markdown' and c.source.startswith('## ')]
        self.assertEqual(len(sections), 8)
        for number, section in enumerate(sections, 1):
            self.assertTrue(section.startswith(f'## {number}.'))
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == 'code':
                compile(cell.source, f'cell_{index}', 'exec')
                self.assertIsNone(cell.execution_count)
                self.assertFalse(cell.outputs)

    def test_twelve_run_orchestrator_uses_configuration_cli(self):
        from training.comparisons import main as comparison_main
        from training.research_cli import main as scratch_main
        from training.experiment_io import prepare_data
        synthetic = load_synthetic_toy_graph(12, 3, 35, 5, 5, seed=3)
        calls, manifests = [], []
        with tempfile.TemporaryDirectory() as directory:
            def prepare(*args, **kwargs):
                return prepare_data(*args, **kwargs, full_dataset=synthetic)
            def record(cfg, output_dir, **kwargs):
                from training.variants import configuration_id
                calls.append(cfg)
                manifests.append(json.loads(Path(kwargs['manifest_path']).read_text())['sha256'])
                result = fixture_result(configuration_id(cfg.variant, cfg.warmup), cfg.seed, cfg)
                write_json(Path(output_dir) / 'results.json', result)
                return result
            def child(command, *args, **kwargs):
                self.assertNotIn('--checkpoint', command)
                self.assertNotIn('--legacy-pretrained', command)
                with patch.object(sys, 'argv', command[1:]):
                    scratch_main()
                return subprocess.CompletedProcess(command, 0)
            argv = ['run_warmup_ablation.py', '--output-dir', directory, '--max-entities', '10',
                    '--epochs', '3', '--warmup-epochs', '1']
            with (patch.object(sys, 'argv', argv),
                  patch('training.comparisons.prepare_data', side_effect=prepare),
                  patch('training.comparisons.run_logged', side_effect=child),
                  patch('training.research_cli.run_experiment', side_effect=record)):
                comparison_main()
            self.assertEqual(len(calls), 12)
            self.assertEqual({(c.variant, c.warmup, c.seed) for c in calls},
                             {(a, int(enabled), s) for a, enabled in CONFIGURATIONS.values() for s in (0, 1, 2)})
            self.assertEqual(len(set(manifests)), 1)
            self.assertTrue(all(c.epochs == 3 and c.train_batch_size == 0 and c.eval_every == 1 for c in calls))
            self.assertTrue((Path(directory) / 'summary.json').is_file())

    def test_ordered_execution_all_new_cells(self):
        # Exercise the preserved template, not manually removed historical cells.
        from notebooks.build_warmup_ablation_notebook import cells
        notebook = nbformat.from_dict({'cells': cells, 'metadata': {}, 'nbformat': 4, 'nbformat_minor': 5})
        synthetic = load_synthetic_toy_graph(12, 3, 35, 5, 5, seed=3)
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            def fake_run(command, **kwargs):
                command = list(map(str, command))
                calls.append(command)
                if '--offline-only' in command:
                    self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
                    self.assertNotIn('RESEARCH_MODEL_BACKUP', kwargs['env'])
                if len(command) > 1 and command[1] == 'run_warmup_ablation.py':
                    dest = Path(command[command.index('--output-dir') + 1])
                    results = [fixture_result(c, s) for c in CONFIGURATIONS for s in (0, 1, 2)]
                    for r in results:
                        write_json(dest / f"{r['configuration']}_seed{r['settings']['seed']}" / 'results.json', r)
                    ablation_report(dest, results)
                return subprocess.CompletedProcess(command, 0)
            scope = {'__name__': '__notebook_test__'}
            with (patch.dict(sys.modules, {'google.colab': SimpleNamespace(drive=SimpleNamespace(mount=lambda path: None))}),
                  patch.dict(os.environ), patch('os.chdir'), patch('subprocess.run', side_effect=fake_run),
                  patch('training.process.run_logged', side_effect=lambda command, log_path, **kw: fake_run(command, **kw)),
                  patch('subprocess.check_output', return_value='fixture_sha'),
                  patch('torch.cuda.is_available', return_value=True),
                  patch('torch.cuda.get_device_name', return_value='Tesla T4 (fixture)'),
                  patch('run_pretrained_check.select_device', return_value='cuda'),
                  patch('training.experiment_io.load_dataset', return_value=synthetic),
                  patch('training.experiment_io.validate_fb15k237'), patch('IPython.display.display')):
                for index, cell in enumerate(notebook.cells):
                    if cell.cell_type != 'code':
                        continue
                    source = cell.source.replace("Path('/content/Research-project')", f'Path({str(ROOT)!r})')
                    source = source.replace("Path('/content/drive/MyDrive/Research/scratch_pipeline_v1')", f'Path({str(Path(directory)/"shared")!r})')
                    source = source.replace("Path('/content/drive/MyDrive/Research/warmup_ablation_5000')", f'Path({str(Path(directory)/"stage")!r})')
                    source = source.replace('MAX_ENTITIES = 5000', 'MAX_ENTITIES = 10')
                    exec(compile(source, f'cell_{index}', 'exec'), scope)
                scope['PRETRAINED_VERIFIED'] = False
                scope['run_logged'] = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('pretrained failure'))
                with self.assertRaisesRegex(RuntimeError, 'pretrained failure'):
                    scope['launch_study']()
            scripts = [c[1] for c in calls if len(c) > 1]
            self.assertEqual(scripts.count('run_warmup_ablation.py'), 1)
            self.assertLess(scripts.index('run_pretrained_check.py'), scripts.index('run_warmup_ablation.py'))
            self.assertEqual(scripts.count('run_resume_diagnostic.py'), 2)
