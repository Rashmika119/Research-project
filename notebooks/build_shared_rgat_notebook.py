"""Build a separate shared-RGAT notebook; never overwrite existing notebooks."""
import copy
import json
from pathlib import Path
from notebooks.build_full_dataset_notebook import notebook as full_notebook


def notebook():
    nb = copy.deepcopy(full_notebook())
    def source(index, text):
        nb['cells'][index]['source'] = text.strip() + '\n'
    source(0, """# Six shared-RGAT experiments: 1,000 entities, 30 epochs, 5 warmup
A: L1 -> fusion -> same L1 -> L2. B: L1 -> fusion -> L2.
C: L1 -> L2 -> fusion -> same L1 -> same L2.
Each runs with a learned refinement gate and without it. Ungated is U + V;
the residual addition and both text gates remain. Seed 0, scratch initialization.
L1 is RGATConv/ReLU/dropout at every use; L2 has no added activation.
Run Sections 1-4 for setup; Section 6 starts the six actual experiments.
No preliminary training runs automatically. GPU execution is not locally verified.
""")
    nb['cells'][2]['source'] = nb['cells'][2]['source'].replace('all three runs', 'all six runs')
    setup = nb['cells'][4]['source']
    setup = setup.replace('full_dataset_comparison_100', 'shared_rgat_1000')
    setup = setup.replace('full_fb15k237_100_seed0_v1_run1', 'shared_rgat_1000_30_5_seed0_v1_run1')
    setup = setup.replace('run_full_dataset_comparison.py', 'run_shared_rgat_comparison.py')
    setup = setup.replace('Select the commit containing this full study.', 'Push/select the commit containing the six-run shared RGAT study.')
    source(4, setup)
    source(6, """import json
import yaml
import pandas as pd
from dataclasses import asdict
from datetime import datetime
from IPython.display import display, Image, Markdown
from training.shared_study import CONFIGURATIONS, make_config, expected_budget, verify_manifest
from training.experiment_io import write_json, environment
from training.process import run_logged

SEED = 0
MAX_ENTITIES = 1000
TOTAL_EPOCHS = 30
WARMUP_EPOCHS = 5
TEXT_BATCH_SIZE = 256
EVAL_BATCH_SIZE = 512
SKIP_COMPLETED = True
RESUME_INTERRUPTED = True
RUN_STATIC_CHECKS = False  # Optional, CPU-only, no optimizer steps.
assert (SEED, MAX_ENTITIES, TOTAL_EPOCHS, WARMUP_EPOCHS) == (0, 1000, 30, 5)
EXPERIMENT_DIR = RUN_ROOT / 'runs'
MANIFEST = EXPERIMENT_DIR / 'dataset_manifest.json'
RAW_DIR = SHARED_ROOT / 'data' / 'fb15k237'
TEXT_DIR = SHARED_ROOT / 'data' / 'text'
POOLED_CACHE = SHARED_ROOT / 'pooled_text_cache'
LOG_DIR = RUN_ROOT / 'logs' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
LOG_DIR.mkdir(parents=True, exist_ok=True)
protocol = yaml.safe_load(Path('experiments/configs/shared_rgat_1000.yaml').read_text())
assert list(CONFIGURATIONS) == list(protocol['configurations'])
assert (protocol['epochs'], protocol['warmup'], protocol['max_entities'], protocol['seeds']) == (30, 5, 1000, [0])
configs = [make_config(c, SEED, raw_dir=RAW_DIR, text_dir=TEXT_DIR,
                      text_batch_size=TEXT_BATCH_SIZE, eval_batch_size=EVAL_BATCH_SIZE) for c in CONFIGURATIONS]
display(pd.DataFrame([{'configuration': c.variant, 'warmup_epochs': c.warmup,
                      'integrated_epochs': c.epochs-c.warmup, 'train_batch_size': c.train_batch_size,
                      'eval_every': c.eval_every} for c in configs]))
write_json(LOG_DIR / 'environment.json', environment())
with (LOG_DIR / 'pip_freeze.txt').open('w') as f:
    subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=f, check=True)
""")
    source(7, """## 4. Non-training subset and model-file verification
Select 1,000 entities using training facts only, then retain all induced facts
from each original split. All six runs use this exact manifest. Candidate set:
1,000 entities, not the full dataset. Cached text is keyed by subset/token identity.
This cell downloads/verifies inputs but performs no forward/backward or training.
""")
    source(8, """from training.experiment_io import prepare_data
from models.pretrained import prepare_pretrained

data, graph, known_train, manifest = prepare_data(str(RAW_DIR), 1000, 0, MANIFEST)
verify_manifest(manifest)
assert len(graph.edge_index) == 2 * len(data.train)
assert set(data.train) == set(known_train)
print('Subset statistics:', manifest['statistics'])
print('Dataset SHA256:', manifest['dataset_sha256'])
print('Manifest SHA256:', manifest['sha256'])
display(pd.DataFrame([{'configuration': c, **expected_budget(c, len(data.train))} for c in CONFIGURATIONS]))
write_json(RUN_ROOT / 'dataset_statistics.json', manifest['statistics'])
prepared = prepare_pretrained()
write_json(LOG_DIR / 'pretrained_files.json', {'revision': prepared.revision, 'files': prepared.files})
print('Verified pretrained files:', prepared.path)
print('Output directory:', EXPERIMENT_DIR)
del data, graph, known_train, manifest
""")
    source(9, '## 5. Optional non-training developer checks')
    source(10, """if RUN_STATIC_CHECKS:
    run_logged([sys.executable, 'run_static_checks.py'], LOG_DIR / 'static_checks.log')
else:
    print('Optional checks disabled. No preliminary training is required.')
""")
    source(11, """## 6. Execute all six experiments
Each child starts from scratch and runs 5 structural + 25 integrated epochs.
Every epoch prints loss, phase, elapsed time and peak GPU memory. Validation
prints every five epochs. Epoch 5 structural validation and initial integrated
validation are diagnostic only; eligible selection epochs are 10,15,20,25,30.
The selected checkpoint is evaluated on test. No baseline weights are reused.
""")
    source(12, """command = [sys.executable, '-u', 'run_shared_rgat_comparison.py',
           '--seed', str(SEED), '--raw-dir', str(RAW_DIR), '--text-dir', str(TEXT_DIR),
           '--cache-dir', str(POOLED_CACHE), '--text-batch-size', str(TEXT_BATCH_SIZE),
           '--eval-batch-size', str(EVAL_BATCH_SIZE), '--output-dir', str(EXPERIMENT_DIR)]
if SKIP_COMPLETED:
    command.append('--skip-completed')
if RESUME_INTERRUPTED:
    command.append('--resume')
run_logged(command, LOG_DIR / 'shared_study.log')
""")
    source(13, """## 7. Checkpoints and disconnection recovery
Rerun setup and Section 6 with exactly the same RUN_TAG, source, packages,
settings and GPU. Completed runs are verified/skipped; partial runs restore
their own atomic epoch-end last.pt, including Adam, RNG streams and selected state.
An interrupted, uncommitted epoch replays. Never reuse full-study or another
configuration's checkpoints. Changed identities require a fresh output directory.
""")
    source(14, """status = []
for c in CONFIGURATIONS:
    folder = EXPERIMENT_DIR / f'{c}_seed{SEED}'
    path = folder / 'results.json'
    r = json.loads(path.read_text()) if path.exists() else None
    status.append({'configuration': c, 'status': 'complete' if r else ('partial' if (folder/'last.pt').exists() else 'not started'),
                   'best_epoch': r['best_epoch'] if r else None, 'checkpoint': str(folder/'last.pt')})
display(pd.DataFrame(status))
""")
    source(15, """## 8. Final comparison, gate effects and graphs
Selection uses validation MRR, never test. Report A/B/C gated-minus-ungated
scores and runtime/memory. A, B and C apply graph layers 3, 2 and 4 times per
integrated forward respectively; C also warms up two layers rather than one.
These are single-seed descriptive comparisons, with no across-seed error bars.
Subset scores are not directly comparable with full-dataset scores.
""")
    source(16, """from training.shared_reports import shared_report, PLOTS
paths = [EXPERIMENT_DIR / f'{c}_seed{SEED}' / 'results.json' for c in CONFIGURATIONS]
if all(p.is_file() for p in paths):
    summary = shared_report(EXPERIMENT_DIR, [json.loads(p.read_text()) for p in paths], seed=SEED)
    display(pd.read_csv(EXPERIMENT_DIR / 'comparison.csv').sort_values('validation_MRR', ascending=False))
    display(pd.read_csv(EXPERIMENT_DIR / 'gate_effects.csv'))
    display(Markdown((EXPERIMENT_DIR / 'research_analysis.md').read_text()))
    for filename in PLOTS:
        display(Image(filename=str(EXPERIMENT_DIR / filename)))
else:
    print('Complete all six runs before making a comparison.')
print('Persistent results:', EXPERIMENT_DIR)
print('Logs:', LOG_DIR)
""")
    nb['metadata']['colab']['name'] = 'shared_rgat_1000.ipynb'
    for i, cell in enumerate(nb['cells']):
        cell['id'] = f'shared-rgat-{i:03d}'
    return nb


def build():
    target = Path(__file__).with_name('shared_rgat_1000.ipynb')
    if target.exists():
        raise FileExistsError('Preserve existing notebook cells/outputs; create a reviewed preview instead.')
    target.write_text(json.dumps(notebook(), indent=2)+'\n', encoding='utf-8')
    print(target)


if __name__ == '__main__':
    build()
