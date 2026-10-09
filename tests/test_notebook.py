"""Validate notebook schema and run all cells in order with external-work fixtures."""
from dataclasses import asdict
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
from training.reports import comparison_report
from training.research_pipeline import ExperimentConfig
from training.variants import VARIANTS


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks' / 'research_pipeline.ipynb'


def fixture_result(variant, seed, full=False):
    """Explicit fictional values for exercising notebook wiring, never benchmarks."""
    cfg = ExperimentConfig(variant=variant, seed=seed, max_entities=0 if full else 1000)
    return {'settings': asdict(cfg), 'status': 'complete', 'initialization': 'scratch',
            'dataset_scope': 'full' if full else 'subset', 'dataset_sha256': 'fixture',
            'manifest_sha256': 'full' if full else 'subset', 'ranking_policy': 'average_exact_ties',
            'initial_structural_sha256': f'fixture_seed{seed}', 'best_epoch': 6,
            'warmup_epochs': 0 if variant in ('baseline', 'original-no-warmup') else 5,
            'main_epochs': 30 if variant in ('baseline', 'original-no-warmup') else 25,
            'optimizer_steps': 30, 'duration_seconds': 1., 'peak_gpu_memory_bytes': None,
            'history': [{'epoch': 6, 'phase': 'main', 'loss': 1.0,
                         'validation': {'MRR': .1}}],
            'validation': {m: .1 for m in ('MRR', 'Hits@1', 'Hits@3', 'Hits@10')},
            'test': {m: .2 for m in ('MRR', 'Hits@1', 'Hits@3', 'Hits@10')}}


class NotebookTests(unittest.TestCase):
    def test_fifteen_run_orchestrator_uses_scratch_cli(self):
        from training.comparisons import main as comparison_main
        from training.research_cli import main as scratch_main
        from training.experiment_io import prepare_data
        synthetic = load_synthetic_toy_graph(12, 3, 35, 5, 5, seed=3)
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            def prepare(*args, **kwargs):
                return prepare_data(*args, **kwargs, full_dataset=synthetic)

            def record_run(cfg, output_dir, **kwargs):
                calls.append(cfg)
                result = fixture_result(cfg.variant, cfg.seed)
                result['settings'] = asdict(cfg)
                write_json(Path(output_dir) / 'results.json', result)
                return result

            def child(command, **kwargs):
                self.assertNotIn('--checkpoint', command)
                self.assertNotIn('--legacy-pretrained', command)
                with patch.object(sys, 'argv', command[1:]):
                    scratch_main()
                return subprocess.CompletedProcess(command, 0)

            argv = ['run_phase5_comparisons.py', '--output-dir', directory,
                    '--max-entities', '10', '--epochs', '3', '--warmup-epochs', '1']
            with (patch.object(sys, 'argv', argv),
                  patch('training.comparisons.prepare_data', side_effect=prepare),
                  patch('training.comparisons.subprocess.run', side_effect=child),
                  patch('training.research_cli.run_experiment', side_effect=record_run)):
                comparison_main()
            self.assertEqual({(c.variant, c.seed) for c in calls},
                             {(v, s) for v in VARIANTS for s in (0, 1, 2)})
            self.assertEqual(len(calls), 15)
            self.assertTrue(all(c.epochs == 3 and c.warmup == 1 and c.subset_seed == 0 for c in calls))
            self.assertTrue((Path(directory) / 'summary.json').is_file())

    def test_schema_syntax_and_sections(self):
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        nbformat.validate(notebook)
        sections = [c.source for c in notebook.cells if c.cell_type == 'markdown' and c.source.startswith('## ')]
        self.assertEqual(len(sections), 16)
        for number, section in enumerate(sections, 1):
            self.assertTrue(section.startswith(f'## {number}.'))
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == 'code':
                compile(cell.source, f'notebook_cell_{index}', 'exec')
                self.assertIsNone(cell.execution_count)
                self.assertFalse(cell.outputs)

    def test_ordered_execution_including_optional_final_stage(self):
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        synthetic = load_synthetic_toy_graph(12, 3, 35, 5, 5, seed=3)
        launched = []
        with tempfile.TemporaryDirectory() as directory:
            def fake_run(command, **kwargs):
                command = list(map(str, command))
                if len(command) > 1 and command[1].endswith('.py'):
                    script = command[1]
                    if script in ('run_research_baseline.py', 'run_phase5_training.py', 'run_phase5_comparisons.py'):
                        launched.append(command)
                        self.assertNotIn('--checkpoint', command)
                        dest = Path(command[command.index('--output-dir') + 1])
                        if script == 'run_phase5_comparisons.py':
                            results = [fixture_result(v, seed) for v in VARIANTS for seed in (0, 1, 2)]
                            for r in results:
                                write_json(dest / f"{r['settings']['variant']}_seed{r['settings']['seed']}" / 'results.json', r)
                            comparison_report(dest, results)
                        else:
                            variant = 'baseline' if script == 'run_research_baseline.py' else command[command.index('--variant') + 1]
                            if script == 'run_phase5_training.py':
                                self.assertEqual(command[command.index('--max-entities') + 1], '0')
                            write_json(dest / 'results.json', fixture_result(variant, 0, full=True))
                return subprocess.CompletedProcess(command, 0)

            fake_colab = SimpleNamespace(drive=SimpleNamespace(mount=lambda path: None))
            scope = {'__name__': '__notebook_test__', 'display': lambda *a, **k: None}
            with (patch.dict(sys.modules, {'google.colab': fake_colab}),
                  patch.dict(os.environ), patch('os.chdir'),
                  patch('subprocess.run', side_effect=fake_run),
                  patch('subprocess.check_output', return_value='fixture_commit'),
                  patch('preprocessing.dataset.load_dataset', return_value=synthetic),
                  patch('training.experiment_io.load_dataset', return_value=synthetic),
                  patch('training.experiment_io.validate_fb15k237'),
                  patch('IPython.display.display')):
                # prepare_data's production validation is patched only here;
                # it still executes the real sampler and graph-integrity checks.
                for index, cell in enumerate(notebook.cells):
                    if cell.cell_type != 'code':
                        continue
                    source = cell.source.replace("Path('/content/Research-project')", f'Path({str(ROOT)!r})')
                    source = source.replace("Path('/content/drive/MyDrive/Research/scratch_pipeline_v1')",
                                            f'Path({directory!r})')
                    source = source.replace('RUN_FINAL_FULL_DATASET = False', 'RUN_FINAL_FULL_DATASET = True')
                    exec(compile(source, f'notebook_cell_{index}', 'exec'), scope)
            self.assertEqual(len(launched), 3)
            self.assertTrue((Path(directory) / 'run1' / 'exports' / 'final_comparison.csv').is_file())


if __name__ == '__main__':
    unittest.main()
