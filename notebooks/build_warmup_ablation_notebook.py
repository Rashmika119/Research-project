"""Build only warmup_ablation_5000.ipynb; never read/write the user's old notebook."""
import json
from pathlib import Path
from textwrap import dedent

cells = []


def md(source):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': dedent(source).strip() + '\n'})


def code(source):
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                  'outputs': [], 'source': dedent(source).strip() + '\n'})


md('''
# Warmup ablation: four configurations, 5,000 entities, three seeds
This is a new research stage. Your earlier `research_pipeline.ipynb` and its results
remain historical artifacts. Run this notebook from top to bottom on a GPU runtime.
The twelve runs are substantial; they have not been executed during code development.

All KG-specific parameters start from scratch. Only frozen pretrained RoBERTa and
independent frozen text caches are reused. No baseline or earlier-run weights are
transferred. Warmup is 5 structural + 25 integrated updates; no-warmup is 30 integrated
updates. Validation chooses checkpoints and the final configuration; test never selects.
''')
md('## 1. Environment, repository and GPU setup')
code('''
import os
import sys
import subprocess
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['PYTHONUNBUFFERED'] = '1'
REPO_URL = 'https://github.com/Rashmika119/Research-project.git'
REPO_REF = 'feature/phase_03'  # Push this stage first; a fixed commit SHA is preferable.
REPO_DIR = Path('/content/Research-project')
if not REPO_DIR.exists():
    subprocess.run(['git', 'clone', REPO_URL, str(REPO_DIR)], check=True)
    subprocess.run(['git', 'checkout', REPO_REF], cwd=REPO_DIR, check=True)
os.chdir(REPO_DIR)
assert Path('run_warmup_ablation.py').is_file(), 'Checkout the commit containing the new warmup study.'
print('Repository SHA:', subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
import torch
print('Python:', sys.version)
print('PyTorch:', torch.__version__, '; CUDA:', torch.version.cuda)
assert torch.cuda.is_available(), 'Select Runtime > Change runtime type > T4 GPU.'
GPU_NAME = torch.cuda.get_device_name(0)
print('GPU:', GPU_NAME, '; Tesla T4 selected:', 'T4' in GPU_NAME)
if 'T4' not in GPU_NAME:
    print('Using another GPU; keep hardware consistent across all twelve runs.')
''')
md('## 2. Google Drive and verified local model cache')
code('''
from google.colab import drive
drive.mount('/content/drive')
SHARED_ROOT = Path('/content/drive/MyDrive/Research/scratch_pipeline_v1')
STAGE_ROOT = Path('/content/drive/MyDrive/Research/warmup_ablation_5000')
STAGE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ['HF_HOME'] = '/content/hf_cache'
os.environ['HF_HUB_CACHE'] = '/content/hf_cache/hub'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/content/hf_cache/hub'
os.environ.pop('TRANSFORMERS_CACHE', None)
os.environ['RESEARCH_MODEL_BACKUP'] = str(SHARED_ROOT / 'verified_roberta_backup')
print('Model loads from:', os.environ['HF_HUB_CACHE'])
print('Verified persistent backup:', os.environ['RESEARCH_MODEL_BACKUP'])
print('New experiment root:', STAGE_ROOT)
''')
md('## 3. Four configurations and fixed training budget')
code('''
import json
import yaml
import pandas as pd
from datetime import datetime
from IPython.display import display, Image
from training.variants import CONFIGURATIONS, configuration_settings
from training.process import run_logged

MAX_ENTITIES = 5000
SEEDS = [0, 1, 2]
TOTAL_EPOCHS = 30
WARMUP_EPOCHS = 5
SUBSET_SEED = 0
RUN_ALL_CONFIGURATIONS = True
SKIP_COMPLETED = True
RESUME_INTERRUPTED = True
RUN_OFFLINE_TESTS = True
RUN_REAL_LM_SMOKE = True
RUN_CUDA_RESUME_CHECK = True
RUN_REAL_LM_RESUME_CHECK = True
TEXT_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 64
MAX_TEXT_LENGTH = 64
RUN_TAG = 'warmup5000_induced_v1_run1'  # Reuse only to resume; change for changed settings/code.

protocol = yaml.safe_load(Path('experiments/configs/research_protocol.yaml').read_text(encoding='utf-8-sig'))
assert list(CONFIGURATIONS) == protocol['study']['configurations']
assert 0 < WARMUP_EPOCHS < TOTAL_EPOCHS and MAX_ENTITIES >= 2
assert SEEDS and len(SEEDS) == len(set(SEEDS))
RUN_ROOT = STAGE_ROOT / RUN_TAG
EXPERIMENT_DIR = RUN_ROOT / 'runs'
MANIFEST = EXPERIMENT_DIR / 'subset_manifest.json'
RAW_DIR = SHARED_ROOT / 'data' / 'fb15k237'
TEXT_DIR = SHARED_ROOT / 'data' / 'text'
POOLED_CACHE = SHARED_ROOT / 'pooled_text_cache'
LOG_DIR = RUN_ROOT / 'logs' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
RUN_ROOT.mkdir(parents=True, exist_ok=True)
CPU_ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1')
CPU_ENV.pop('RESEARCH_MODEL_BACKUP', None)  # Corruption fixtures must never touch Drive backups.
PRETRAINED_VERIFIED = False

def ensure_pretrained():
    global PRETRAINED_VERIFIED
    if not PRETRAINED_VERIFIED:
        run_logged([sys.executable, 'run_pretrained_check.py', '--device', 'cuda',
                    '--report', str(LOG_DIR / 'pretrained_report.json')], LOG_DIR / 'pretrained.log')
        PRETRAINED_VERIFIED = True

rows = []
for identifier in CONFIGURATIONS:
    architecture, warm = configuration_settings(identifier, WARMUP_EPOCHS)
    rows.append({'configuration': identifier, 'architecture': architecture,
                 'structural_updates': warm, 'integrated_updates': TOTAL_EPOCHS - warm,
                 'total_updates': TOTAL_EPOCHS, 'seeds': str(SEEDS)})
display(pd.DataFrame(rows))
print('Runs:', len(CONFIGURATIONS) * len(SEEDS), '; LR=0.001; KG width=32; Adam; four negatives; full-subset updates')
print('No-Refinement remains available through the single-run CLI; it is excluded here.')
''')
md('''
## 4. Dataset preparation and shared manifest
The historical traversal retained only some training edges. The new version selects
entities by the same train-only seeded BFS, then keeps **every original training fact**
with both endpoints selected. Held-out facts are filtered into their own original splits.
Entity count is exact; no adaptive reduction occurs. All candidate rankings use this
same entity set. The saved manifest includes mappings, facts, strategy and fingerprints.
''')
code('''
from training.experiment_io import prepare_data, write_json
subset, graph, known_train, manifest = prepare_data(str(RAW_DIR), MAX_ENTITIES, SUBSET_SEED, MANIFEST)
assert subset.num_entities == MAX_ENTITIES
assert set(subset.train) == set(known_train), 'Induced training subset must retain every covered training fact.'
stats = manifest['statistics']
display(pd.DataFrame([{k: v for k, v in stats.items() if k not in ('isolated_entity_ids', 'component_sizes', 'definition')}]))
print('Disconnected/isolated IDs:', stats['isolated_entity_ids'])
print('Weak component sizes:', stats['component_sizes'])
print('Dataset SHA256:', manifest['dataset_sha256'])
print('Manifest SHA256:', manifest['sha256'])
print('Sampling:', manifest['selection_strategy'])
write_json(RUN_ROOT / 'subset_statistics.json', stats)
''')
md('''
## 5. Independent reliability tests and pretrained validation
CPU regressions use temporary fixtures, with the persistent model backup removed
from their environment. GPU checks use actual pretrained weights. Explicit CUDA
requests fail if no GPU exists. Run these cells independently when diagnosing a
failure. Logs preserve complete tracebacks and exit codes. Preflight remains
mandatory for the experiment launch even if optional tests are disabled.
''')
code('''
if RUN_OFFLINE_TESTS:
    run_logged([sys.executable, 'run_scratch_smoke_test.py', '--offline-only', '--device', 'cpu'],
               LOG_DIR / 'offline.log', env=CPU_ENV)
    for script in ('run_complex_scorer_check.py', 'run_tie_ranking_check.py'):
        run_logged([sys.executable, script], LOG_DIR / (Path(script).stem + '.log'), env=CPU_ENV)
''')
code('''
ensure_pretrained()
if RUN_REAL_LM_SMOKE:
    run_logged([sys.executable, 'run_scratch_smoke_test.py', '--real-lm-only', '--device', 'cuda'],
               LOG_DIR / 'real_roberta.log')
''')
code('''
DIAGNOSTIC_DIR = RUN_ROOT / 'diagnostics' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
if RUN_CUDA_RESUME_CHECK:
    from run_pretrained_check import select_device
    select_device('cuda')
    for repeat in range(3):
        run_logged([sys.executable, '-m', 'unittest',
                    'tests.test_research_pipeline.PipelineTests.test_resume_matches_uninterrupted', '-v'],
                   LOG_DIR / f'cuda_resume_targeted_{repeat}.log')
    run_logged([sys.executable, 'run_resume_diagnostic.py', '--device', 'cuda', '--repeats', '3',
                '--output-dir', str(DIAGNOSTIC_DIR / 'tiny_lm')], LOG_DIR / 'cuda_resume.log')
''')
code('''
if RUN_REAL_LM_RESUME_CHECK:
    run_logged([sys.executable, 'run_resume_diagnostic.py', '--device', 'cuda', '--real-lm',
                '--variant', 'residual', '--repeats', '1', '--output-dir', str(DIAGNOSTIC_DIR / 'real_roberta')],
               LOG_DIR / 'real_roberta_resume.log')
''')
md('''
## 6. Run all twelve experiments sequentially
Each child process prints its architecture, warmup setting and seed, followed by
every epoch's phase, loss, validation MRR, validation Hits@10, peak tracked GPU
memory and elapsed time. An epoch may take a long time before its completion line.
Each run starts fresh; warmup preserves only that run's structural weights/Adam
state at its transition. Separate processes release GPU allocations between runs.
''')
code('''
def launch_study():
    ensure_pretrained()
    command = [sys.executable, 'run_warmup_ablation.py', '--max-entities', str(MAX_ENTITIES),
               '--epochs', str(TOTAL_EPOCHS), '--warmup-epochs', str(WARMUP_EPOCHS),
               '--seeds', *map(str, SEEDS), '--subset-seed', str(SUBSET_SEED),
               '--raw-dir', str(RAW_DIR), '--text-dir', str(TEXT_DIR), '--cache-dir', str(POOLED_CACHE),
               '--text-batch-size', str(TEXT_BATCH_SIZE), '--eval-batch-size', str(EVAL_BATCH_SIZE),
               '--max-text-length', str(MAX_TEXT_LENGTH), '--output-dir', str(EXPERIMENT_DIR)]
    if SKIP_COMPLETED:
        command.append('--skip-completed')
    if RESUME_INTERRUPTED:
        command.append('--resume')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_logged(command, LOG_DIR / f'study_{stamp}.log')

if RUN_ALL_CONFIGURATIONS:
    launch_study()
else:
    print('Twelve-run experiment disabled. Setup and tests only.')
''')
md('''
## 7. Resume and checkpoint inspection
After a disconnect, reconnect Drive, use the same repository/dependency versions
and RUN_TAG/settings, and rerun setup plus Section 6 with SKIP_COMPLETED and
RESUME_INTERRUPTED enabled. Compatible completed runs are skipped; partial runs
resume only from their own committed `last.pt`. Architecture, warmup, seed, data,
code, LM and numerical policy mismatches are rejected. Never supply an old baseline
or another configuration's checkpoint. Earlier 1,000-entity runs are not compatible.

Every run saves config/manifest, best/last checkpoints, history JSON/CSV, optimizer
and complete RNG state, selected validation/test metrics, time/memory and resume
audits. Best selection can occur at the initial integrated evaluation immediately
after warmup (epoch 5), or before any update without warmup (epoch 0).
''')
code('''
artifacts = []
for identifier in CONFIGURATIONS:
    for seed in SEEDS:
        folder = EXPERIMENT_DIR / f'{identifier}_seed{seed}'
        result_path = folder / 'results.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else None
        artifacts.append({'configuration': identifier, 'seed': seed,
                          'status': 'complete' if result else ('partial' if (folder / 'last.pt').exists() else 'not started'),
                          'best_epoch': result['best_epoch'] if result else None,
                          'optimizer_steps': result['optimizer_steps'] if result else None,
                          'checkpoint': str(folder / 'last.pt')})
display(pd.DataFrame(artifacts))
''')
md('''
## 8. Final analysis, paired ablations and graphs
Primary selection is mean validation MRR. Tables report sample SD across seeds,
per-seed differences and descriptive percentage changes; three seeds do not support
claims of statistical significance. Test values describe the validation-selected
checkpoints. Previous 1,000-entity scores are a different candidate/data scope.

Dashed curve segments are structural warmup; solid segments are integrated training.
Warmup validation also uses the structural model, and does not select checkpoints.
Shading marks the warmup window for enabled configurations only. No mean combines
different phases of a configuration. Total time includes setup/evaluation/checkpoint
I/O; epoch times help interpret soft-prompt cost despite independent-text cache hits.
''')
code('''
from training.ablation_reports import PLOTS
summary_path = EXPERIMENT_DIR / 'summary.json'
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
    display(pd.read_csv(EXPERIMENT_DIR / 'comparison.csv').sort_values('validation_MRR_mean', ascending=False))
    display(pd.read_csv(EXPERIMENT_DIR / 'runs.csv'))
    display(pd.DataFrame(summary['paired_comparisons']))
    display(pd.read_csv(EXPERIMENT_DIR / 'paired_seed_differences.csv'))
    print('Selected by validation:', summary['selected_by_validation'])
    print('Selected configuration test summary:', summary['selected_test_results'])
    behavior = json.loads((EXPERIMENT_DIR / 'training_behavior.json').read_text())
    display(pd.DataFrame(behavior))
    for row in behavior:
        if row['severe_tie_flags']:
            print('TIE WARNING:', row['configuration'], 'seed', row['seed'], row['severe_tie_flags'])
    display(pd.read_csv(EXPERIMENT_DIR / 'ties.csv'))
    for filename in PLOTS:
        display(Image(filename=str(EXPERIMENT_DIR / filename)))
else:
    print('No complete comparison yet. Finish all configured runs before interpreting a winner.')

with (RUN_ROOT / 'environment.txt').open('w') as stream:
    subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=stream, check=True)
print('Reports, curves and all per-run artifacts:', EXPERIMENT_DIR)
print('Diagnostic logs:', LOG_DIR)
''')


def build():
    notebook = {'cells': cells, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11'},
        'colab': {'name': 'warmup_ablation_5000.ipynb'}, 'accelerator': 'GPU'},
        'nbformat': 4, 'nbformat_minor': 5}
    for index, cell in enumerate(cells):
        cell['id'] = f'warmup-ablation-{index:03d}'
    target = Path(__file__).with_name('warmup_ablation_5000.ipynb')
    if target.exists():
        raise FileExistsError('Historical notebook contains manual edits; do not overwrite it')
    target.write_text(json.dumps(notebook, indent=2) + '\n', encoding='utf-8')
    print(target)


if __name__ == '__main__':
    build()
