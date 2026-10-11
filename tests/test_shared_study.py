"""Shared-layer contracts, derivatives and mocked orchestration; no training."""
import ast
import copy
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

import nbformat
import torch
from torch_geometric.nn import RGATConv

from tests.test_research_pipeline import fake_lm, make_tokens
from training.variants import build_scratch_model
from training.shared_study import CONFIGURATIONS, make_config, expected_budget, build_commands, main
from training.shared_reports import shared_report, validate_runs, PLOTS
from training.research_pipeline import trainable_state, load_model_state
from training.experiment_io import write_json, save_checkpoint
from training.reproducibility import (assert_exact_state, capture_rng, numerical_policy,
                                      restore_training_state, seed_everything)

ROOT = Path(__file__).resolve().parents[1]


def fixtures():
    """Fabricated report records, not measurements or experimental evidence."""
    from tests.test_full_study import fixture_results
    template = fixture_results()[1]
    result = []
    for i, (c, (_, gated)) in enumerate(CONFIGURATIONS.items()):
        r = copy.deepcopy(template)
        cfg = make_config(c, 0)
        history = []
        for e in range(1, 31):
            h = {'epoch': e, 'phase': 'structural_warmup' if e <= 5 else 'main',
                 'loss': 1-e*.01, 'optimizer_steps': e, 'train_seconds': 1.,
                 'step_losses': [{'optimizer_step': e, 'positive_count': 1234, 'loss': 1-e*.01}]}
            if e % 5 == 0:
                h['validation'] = {**template['validation'], 'MRR': .1+i*.01+(20-abs(e-20))*.001}
                h['validation_model'] = 'structural' if e <= 5 else 'integrated'
            history.append(h)
        r.update(configuration=c, architecture=c, settings=asdict(cfg), candidate_entities=1000,
                 dataset_scope='subset', warmup_epochs=5, main_epochs=25,
                 subset_statistics={'entities': 1000, 'relations': 237, 'train_triples': 1234,
                                    'valid_triples': 40, 'test_triples': 50},
                 **expected_budget(c, 1234), history=history, best_epoch=20,
                 validation=history[19]['validation'], final_epoch_validation=history[-1]['validation'],
                 initial_trainable_sha256=f'paired-gated-{gated}', initial_structural_sha256='paired-all',
                 learned_gates={'entity_text_gate': .01, 'relation_text_gate': .01,
                                **({'refinement_gate': .01} if gated else {})})
        result.append(r)
    return result


class SharedStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @patch('models.frozen_lm.load_pretrained_model', side_effect=fake_lm)
    def test_layer_reuse_warmup_gradients_and_fusion_equations(self, _):
        edges = torch.tensor([[0, 1, 2, 3, 1, 2, 3, 0], [1, 2, 3, 0, 0, 1, 2, 3]])
        types = torch.tensor([0, 1, 0, 1, 2, 3, 2, 3])
        reference = None
        for c, (arrangement, gated) in CONFIGURATIONS.items():
            seed_everything(3)
            with patch('torch.load', side_effect=AssertionError('Scratch cannot load a checkpoint')):
                model = build_scratch_model(c, 4, 2).eval()
            self.assertEqual(sum(isinstance(m, RGATConv) for m in model.modules()), 2)
            common = {k: v for k, v in trainable_state(model).items() if k != 'refinement_gate'}
            if reference is None:
                reference = common
            assert_exact_state(reference, common)
            self.assertFalse(model.soft_prompt)
            self.assertEqual(hasattr(model, 'refinement_gate'), gated)
            self.assertEqual(len({id(p) for p in model.parameters()}), len(list(model.parameters())))
            counts = [0, 0]
            def hook(index):
                def count(*_): counts[index] += 1
                return count
            handles = [l.register_forward_hook(hook(i)) for i, l in enumerate(model.structural.encoder.layers)]
            h = model.structural.encode(edges, types)
            self.assertEqual(counts, [1, int(arrangement == 'c')])
            h.square().mean().backward()
            self.assertIsNotNone(model.structural.encoder.layers[0].weight.grad)
            self.assertEqual(model.structural.encoder.layers[1].weight.grad is not None, arrangement == 'c')
            self.assertTrue(all(p.grad is None for p in model.bridge.parameters()))
            model.zero_grad(set_to_none=True)
            counts[:] = [0, 0]
            et, rt = make_tokens(4), make_tokens(2)
            ep, rp = model.bridge.lm.encode_text(**et), model.bridge.lm.encode_text(**rt)
            with patch.object(model.bridge.lm, 'encode_text', side_effect=AssertionError('Cached text must bypass LM')):
                e, r = model(edges, types, torch.arange(4), torch.arange(2), et, rt, edges,
                             text_batch_size=2, entity_pooled=ep, relation_pooled=rp)
            self.assertEqual(counts, {'a': [2, 1], 'b': [1, 1], 'c': [2, 2]}[arrangement])
            for handle in handles: handle.remove()
            fused = h + model.entity_text_gate.tanh() * model.bridge(pooled=ep)
            v = fused
            if arrangement in ('a', 'c'):
                v = model.structural.encoder.dropout(torch.relu(model.structural.encoder.layers[0](v, edges, types)))
            v = model.structural.encoder.layers[1](v, edges, types)
            torch.testing.assert_close(e, fused+(model.refinement_gate.tanh()*v if gated else v))
            torch.testing.assert_close(r, model.structural.scorer.relation_emb.weight + model.relation_text_gate.tanh()*model.bridge(pooled=rp))
            (e.square().mean()+r.square().mean()).backward()
            for module in (*model.structural.encoder.layers, model.bridge.lm_to_kg):
                grads = [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
                self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))
                self.assertTrue(any(g.abs().sum() > 0 for g in grads))
            self.assertTrue(all(p.grad is None for p in model.bridge.lm.parameters()))

    @patch('models.frozen_lm.load_pretrained_model', side_effect=fake_lm)
    def test_shared_checkpoint_round_trip_and_restore_without_updates(self, _):
        for c in CONFIGURATIONS:
            seed_everything(0)
            model = build_scratch_model(c, 4, 2)
            optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad))
            for p in optimizer.param_groups[0]['params']:
                optimizer.state[p] = {'step': torch.tensor(5.), 'exp_avg': torch.ones_like(p)*.1, 'exp_avg_sq': torch.ones_like(p)*.2}
            negative, shuffle = random.Random(0), torch.Generator().manual_seed(0)
            saved = {'model_state': trainable_state(model), 'optimizer_state': optimizer.state_dict(),
                     'rng': capture_rng(negative, shuffle), 'module_modes': {n: m.training for n, m in model.named_modules()},
                     'checkpoint_boundary': 'epoch_end', 'epoch': 5, 'optimizer_steps': 5,
                     'history': [{'epoch': e, 'optimizer_steps': e} for e in range(1, 6)],
                     'scheduler_state': None, 'grad_scaler_state': None, 'numerical_policy': numerical_policy()}
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'last.pt'
                save_checkpoint(path, saved)
                clone = build_scratch_model(c, 4, 2)
                adam = torch.optim.Adam((p for p in clone.parameters() if p.requires_grad))
                audit = restore_training_state(clone, adam, torch.load(path, weights_only=True), negative, shuffle)
                self.assertTrue(audit['optimizer_exact'] and audit['rng_exact'])
                assert_exact_state(trainable_state(clone), saved['model_state'])
                opposite = c.replace('-ungated', '-gated') if c.endswith('-ungated') else c.replace('-gated', '-ungated')
                with self.assertRaises(ValueError):
                    load_model_state(build_scratch_model(opposite, 4, 2), saved['model_state'])

    def test_protocol_yaml_commands_and_old_registry_unchanged(self):
        import yaml
        from training.full_study import CONFIGURATIONS as full, make_config as full_config
        self.assertEqual(len(full), 3)
        self.assertEqual(full_config('baseline', 0).epochs, 100)
        protocol = yaml.safe_load((ROOT/'experiments/configs/shared_rgat_1000.yaml').read_text())
        self.assertEqual(list(CONFIGURATIONS), list(protocol['configurations']))
        for c in CONFIGURATIONS:
            cfg = make_config(c, 0)
            self.assertEqual((cfg.max_entities, cfg.epochs, cfg.warmup, cfg.seed), (1000, 30, 5, 0))
            for key in ('model', 'lr', 'train_batch_size', 'text_batch_size', 'eval_batch_size', 'eval_every'):
                self.assertEqual(asdict(cfg)[key], protocol[key])
        self.assertEqual(expected_budget('shared-a-gated', 32769)['optimizer_steps'], 60)
        args = SimpleNamespace(seed=0, raw_dir='raw', text_dir='text', cache_dir='cache',
                               text_batch_size=256, eval_batch_size=512, resume=True, skip_completed=True)
        commands = build_commands(args, Path('fresh'), Path('manifest.json'))
        self.assertEqual(len(commands), 6)
        self.assertEqual(len({str(d) for _, _, d, _ in commands}), 6)
        for c, _, _, cmd in commands:
            with patch.object(sys, 'argv', cmd[2:]), patch('torch.cuda.is_available', return_value=True), patch('training.shared_study.run_experiment') as run:
                main()
                self.assertEqual(run.call_args.args[0].variant, c)

    def test_reports_selection_gate_effects_and_rejection(self):
        runs = fixtures()
        with tempfile.TemporaryDirectory() as directory:
            summary = shared_report(directory, runs)
            self.assertEqual(summary['selected_by_validation'], 'shared-c-ungated')
            self.assertEqual(len(summary['gate_effects']), 3)
            for plot in PLOTS:
                self.assertGreater((Path(directory)/plot).stat().st_size, 1000)
            self.assertEqual(len((Path(directory)/'steps.csv').read_text().splitlines()), 181)
        mutations = [lambda r: r.pop(), lambda r: r[0]['settings'].update(seed=1),
                     lambda r: r[0].update(best_epoch=5), lambda r: r[0].update(initial_structural_sha256='wrong'),
                     lambda r: r[0]['history'][0]['step_losses'].clear(), lambda r: r[0]['learned_gates'].pop('refinement_gate')]
        for mutate in mutations:
            changed = copy.deepcopy(runs); mutate(changed)
            with self.assertRaises(ValueError): validate_runs(changed)

    def test_parent_runner_and_report_only_do_not_launch_extra_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            def prepare(*args):
                return None, None, None, {'statistics': {'entities': 1000, 'relations': 237,
                    'train_triples': 1234, 'valid_triples': 40, 'test_triples': 50}, 'max_entities': 1000,
                    'subset_seed': 0, 'selection_strategy': 'train_bfs_induced_v1', 'sha256': 'fixture'}
            runs = {r['configuration']: r for r in fixtures()}
            def child(command, log):
                c = command[command.index('--single')+1]
                write_json(Path(command[command.index('--output-dir')+1])/'results.json', runs[c])
            argv = ['runner', '--output-dir', directory, '--resume', '--skip-completed']
            with patch.object(sys, 'argv', argv), patch('torch.cuda.is_available', return_value=True), patch('training.shared_study.prepare_data', side_effect=prepare), patch('training.shared_study.run_logged', side_effect=child) as launch, patch('training.shared_reports.shared_report', return_value={'selected_by_validation': 'fixture'}), redirect_stdout(io.StringIO()):
                main()
                self.assertEqual(launch.call_count, 6)
                with patch.object(sys, 'argv', argv+['--seed', '1']):
                    with self.assertRaisesRegex(ValueError, 'settings changed'): main()
                with patch.object(sys, 'argv', argv+['--report-only']), patch('torch.cuda.is_available', return_value=False): main()
                self.assertEqual(launch.call_count, 6)

    def test_notebook_schema_syntax_and_generation_protection(self):
        from notebooks.build_shared_rgat_notebook import notebook, build
        actual = json.loads((ROOT/'notebooks/shared_rgat_1000.ipynb').read_text())
        self.assertEqual(actual, notebook())
        nbformat.validate(nbformat.from_dict(actual))
        for cell in actual['cells']:
            if cell['cell_type'] == 'code': ast.parse(cell['source'])
        self.assertIn('RUN_STATIC_CHECKS = False', actual['cells'][6]['source'])
        self.assertIn('run_shared_rgat_comparison.py', actual['cells'][12]['source'])
        with self.assertRaises(FileExistsError): build()


    def test_full_notebook_ordered_execution_with_external_actions_mocked(self):
        from notebooks.build_shared_rgat_notebook import notebook
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
                self.assertEqual(count,1000)
                m={'statistics':dict(zip(('entities','relations','train_triples','valid_triples','test_triples'),(1000,237,1234,40,50))),
                   'max_entities':1000,'subset_seed':0,'selection_strategy':'train_bfs_induced_v1','sha256':'fixture','dataset_sha256':'fixture'}
                write_json(manifest_path,m)
                return SimpleNamespace(train=[(0,0,1)]*1234),SimpleNamespace(edge_index=range(2468)),[(0,0,1)],m
            def fake_research(command, log_path, **kwargs):
                calls.append(list(map(str,command)))
                self.assertIn('run_shared_rgat_comparison.py',command)
                dest=Path(command[command.index('--output-dir')+1])
                for r in fixtures():
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
                    source=source.replace("Path('/content/drive/MyDrive/Research/shared_rgat_1000')",f'Path({str(Path(directory)/"stage")!r})')
                    exec(compile(source,f'full_cell_{i}','exec'),scope)
                    if i<12:
                        self.assertFalse(any('run_shared_rgat_comparison.py' in c for c in calls))
            self.assertEqual(sum('run_shared_rgat_comparison.py' in c for c in calls),1)
            self.assertFalse(any('run_scratch_smoke_test.py' in c or 'run_resume_diagnostic.py' in c for c in calls))
            self.assertTrue((scope['EXPERIMENT_DIR']/'summary.json').is_file())

if __name__ == '__main__':
    unittest.main()
