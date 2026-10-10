"""Regenerate the checked-in Colab notebook from readable, reviewable cells."""
import json
from pathlib import Path
from textwrap import dedent


cells = []


def markdown(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': dedent(text).strip() + '\n'})


def code(text):
    cells.append({'cell_type': 'code', 'execution_count': None, 'metadata': {},
                  'outputs': [], 'source': dedent(text).strip() + '\n'})


markdown('''
# FB15k-237: independent baseline and five scratch architectures

Run cells in order. Select a GPU runtime before starting. The default research run
trains a 100-epoch full-data graph baseline and 15 pilot runs of 30 epochs each.
These are substantial experiments; smoke tests alone do not establish performance.

All task-specific parameters start from scratch. Only frozen RoBERTa uses pretrained
weights. User-selected budget: **5 structural warmup + 25 integrated epochs**;
`original-no-warmup` uses **30 integrated epochs**, with identical architecture.
Warmup uses the fixed learning rate, not a learning-rate scheduler. Pilot runs have
one optimizer step per epoch. Validation selects checkpoints and architectures.

Use a repository ref containing these changes. For a private repository, arrange
GitHub access or upload the repository first; do not paste access tokens into cells.
Completed compatible runs are verified and skipped. Interrupted runs resume from
their own `last.pt`; baseline/pilot weights never initialize other experiments.
''')
markdown('## 1. Environment setup and GPU verification')
code('''
import os
import sys
import subprocess
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTHONUNBUFFERED', '1')
import torch
print('Python:', sys.version)
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
else:
    print('CPU fallback active. Choose Runtime > Change runtime type > GPU for research training.')
''')
markdown('## 2. Clone repository and install dependencies')
code('''
REPO_URL = 'https://github.com/Rashmika119/Research-project.git'
REPO_REF = 'feature/phase_03'  # Push these changes first; optionally use a fixed commit.
REPO_DIR = Path('/content/Research-project')
if not REPO_DIR.exists():
    subprocess.run(['git', 'clone', REPO_URL, str(REPO_DIR)], check=True)
    subprocess.run(['git', 'checkout', REPO_REF], cwd=REPO_DIR, check=True)
os.chdir(REPO_DIR)
assert Path('run_research_baseline.py').exists(), 'Checkout a ref containing the new scratch pipeline.'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
print('Repository commit:', subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
''')
markdown('## 3. Mount Google Drive')
code('''
from google.colab import drive
drive.mount('/content/drive')
RESEARCH_ROOT = Path('/content/drive/MyDrive/Research/scratch_pipeline_v1')
RESEARCH_ROOT.mkdir(parents=True, exist_ok=True)
# Configure before importing Transformers/Hub. Never memory-map weights on Drive.
LOCAL_HF_HOME = Path('/content/hf_cache')
os.environ['HF_HOME'] = str(LOCAL_HF_HOME)
os.environ['HF_HUB_CACHE'] = str(LOCAL_HF_HOME / 'hub')
os.environ['HUGGINGFACE_HUB_CACHE'] = str(LOCAL_HF_HOME / 'hub')
os.environ.pop('TRANSFORMERS_CACHE', None)
# Only verified files are copied to/from this persistent backup.
os.environ['RESEARCH_MODEL_BACKUP'] = str(RESEARCH_ROOT / 'verified_roberta_backup')
print('Local model cache:', os.environ['HF_HUB_CACHE'])
print('Verified Drive backup:', os.environ['RESEARCH_MODEL_BACKUP'])
''')
markdown('## 4. Configure paths, seeds, and training settings')
code('''
import json
import yaml
import pandas as pd
from IPython.display import display

RUN_BASELINE = True
RUN_PILOT = True
RUN_FINAL_FULL_DATASET = False
RUN_OFFLINE_TESTS = True
RUN_REAL_LM_SMOKE = True
RUN_CUDA_RESUME_CHECK = True
RUN_REAL_LM_RESUME_CHECK = True
SKIP_COMPLETED = True
RESUME_INTERRUPTED = True
RUN_TAG = 'run2_verified'  # New code/RNG schema requires a new run, not an old checkpoint.

protocol = yaml.safe_load(Path('experiments/configs/research_protocol.yaml').read_text())
SEEDS = protocol['pilot']['seeds']
SUBSET_SEED = protocol['pilot']['subset_seed']
PILOT_ENTITIES = protocol['pilot']['max_entities']
PILOT_EPOCHS = protocol['pilot']['epochs']
WARMUP_EPOCHS = protocol['pilot']['warmup_epochs']
TEXT_BATCH_SIZE = protocol['pilot']['text_batch_size']
FINAL_EPOCHS = protocol['final']['epochs']
FINAL_SEED = protocol['final']['seed']

RAW_DIR = RESEARCH_ROOT / 'data' / 'fb15k237'
TEXT_DIR = RESEARCH_ROOT / 'data' / 'text'
POOLED_CACHE = RESEARCH_ROOT / 'pooled_text_cache'
RUN_ROOT = RESEARCH_ROOT / RUN_TAG
BASELINE_DIR = RUN_ROOT / 'baseline_full_seed0'
PILOT_DIR = RUN_ROOT / 'pilot_1000'
baseline_path = BASELINE_DIR / 'best.pt'  # Evaluation artifact only; never passed to variants.
RUN_ROOT.mkdir(parents=True, exist_ok=True)
from training.process import run_logged
from datetime import datetime
LOG_DIR = RUN_ROOT / 'logs' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
CPU_ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='')
PRETRAINED_VERIFIED = False

def ensure_pretrained():
    global PRETRAINED_VERIFIED
    if not PRETRAINED_VERIFIED:
        run_logged([sys.executable, 'run_pretrained_check.py', '--device', 'cuda',
                    '--report', str(LOG_DIR / 'pretrained_report.json')], LOG_DIR / 'pretrained.log')
        PRETRAINED_VERIFIED = True

def launch(script, *arguments):
    # Also protects running an experiment cell directly without running test cells.
    ensure_pretrained()
    command = [sys.executable, script, *map(str, arguments)]
    if SKIP_COMPLETED:
        command.append('--skip-completed')
    if RESUME_INTERRUPTED:
        command.append('--resume')
    print('Starting:', script, flush=True)
    run_logged(command, LOG_DIR / (Path(script).stem + '.log'))

print('Pilot:', len(protocol['pilot']['variants']) * len(SEEDS), 'independent runs')
print('Budget:', WARMUP_EPOCHS, '+', PILOT_EPOCHS - WARMUP_EPOCHS,
      '; no-warmup:', PILOT_EPOCHS, 'integrated epochs')
''')
markdown('## 5. Download and validate FB15k-237')
code('''
from preprocessing.dataset import load_dataset
from preprocessing.graph_builder import build_train_graph, assert_no_leakage
from training.experiment_io import validate_fb15k237, prepare_data

dataset = load_dataset(RAW_DIR, download_if_missing=True, use_synthetic_fallback=False)
validate_fb15k237(dataset)
graph = build_train_graph(dataset.train, dataset.num_entities, dataset.num_relations)
assert_no_leakage(graph, dataset.train, dataset.valid, dataset.test)
print('Full dataset:', dataset.num_entities, 'entities;', dataset.num_relations, 'relations;',
      len(dataset.train), len(dataset.valid), len(dataset.test), 'train/valid/test')
# Each child verifies the shared manifest against these same original files.
pilot_data, _, _, pilot_manifest = prepare_data(str(RAW_DIR), PILOT_ENTITIES, SUBSET_SEED)
print('Pilot:', pilot_data.num_entities, 'entities;', len(pilot_data.train),
      len(pilot_data.valid), len(pilot_data.test), 'train/valid/test')
print('Pilot manifest SHA256:', pilot_manifest['sha256'])
''')
markdown('''
## 6. Verify pretrained weights and run independent test groups
Preflight is mandatory before any experiment, even if optional smoke tests are off.
Offline regressions run in a CPU child process. The next cells independently test
actual pretrained RoBERTa and strict CUDA checkpoint resume. Explicit GPU requests
fail if CUDA is unavailable. Full logs and JSON diagnostics persist on Drive.
The resume diagnostic compares a second uninterrupted control as well as a resumed
run, including losses, weights, Adam state, RNG streams, progress and metrics.
''')
code('''
if RUN_BASELINE or RUN_PILOT or RUN_FINAL_FULL_DATASET or RUN_REAL_LM_SMOKE or RUN_REAL_LM_RESUME_CHECK:
    ensure_pretrained()
''')
code('''
# CPU regressions: can be rerun independently of the GPU cells.
for script in ('run_tie_ranking_check.py', 'run_complex_scorer_check.py'):
    run_logged([sys.executable, script], LOG_DIR / (Path(script).stem + '.log'), env=CPU_ENV)
if RUN_OFFLINE_TESTS:
    run_logged([sys.executable, 'run_scratch_smoke_test.py', '--offline-only', '--device', 'cpu'],
               LOG_DIR / 'offline_tests.log', env=CPU_ENV)
''')
code('''
# Real pretrained GPU forward/backward gate for all five architectures.
if RUN_REAL_LM_SMOKE:
    run_logged([sys.executable, 'run_scratch_smoke_test.py', '--real-lm-only', '--device', 'cuda'],
               LOG_DIR / 'real_roberta.log')
''')
code('''
# Dedicated CUDA resume verification; never covered up by the CPU test setting.
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
# A separate real-RoBERTa GPU training interruption/resume experiment.
if RUN_REAL_LM_RESUME_CHECK:
    run_logged([sys.executable, 'run_resume_diagnostic.py', '--device', 'cuda', '--real-lm',
                '--variant', 'residual', '--repeats', '1',
                '--output-dir', str(DIAGNOSTIC_DIR / 'real_roberta')], LOG_DIR / 'real_roberta_resume.log')
''')
markdown('''
## 7. Train or load the independent graph-only baseline
Full FB15k-237, RGAT + ComplEx, width 32, 100 epochs, four negatives, fixed LR 0.001.
Validation runs every five epochs. A compatible completed run is loaded by verified
skip; a bare historical `.pt` file lacks this experiment's history/provenance and is
not silently treated as a completed run. Checkpoints go directly to Drive each epoch.
''')
code('''
if RUN_BASELINE:
    launch('run_research_baseline.py', '--output-dir', BASELINE_DIR, '--raw-dir', RAW_DIR,
           '--epochs', protocol['baseline']['epochs'], '--lr', protocol['baseline']['lr'],
           '--num-negatives', protocol['baseline']['num_negatives'],
           '--train-batch-size', protocol['baseline']['train_batch_size'],
           '--eval-every', protocol['baseline']['eval_every'])
''')
markdown('## 8. Save and evaluate baseline results')
code('''
baseline_results_file = BASELINE_DIR / 'results.json'
baseline_result = json.loads(baseline_results_file.read_text()) if baseline_results_file.exists() else None
if baseline_result:
    assert baseline_result['dataset_scope'] == 'full'
    assert baseline_result['ranking_policy'] == 'average_exact_ties'
    display(pd.DataFrame([{'split': split, **baseline_result[split]}
                          for split in ('validation', 'test')]))
    print('Best epoch:', baseline_result['best_epoch'], '; checkpoint:', baseline_path)
else:
    print('No baseline result yet. Scratch pilot models can still run independently.')
''')
markdown('''
## 9. Configure the five variants from scratch
All variants use fresh entity/relation tables, first RGAT and other trainable modules.
`original-no-warmup` has exactly the same architecture as `original`.
`residual-no-softprompt` retains second RGAT and refinement residual, uses masked
mean text pooling, and caches only frozen 768-wide vectors, never projected vectors.
The baseline checkpoint is deliberately absent from the command below.
''')
code('''
display(pd.DataFrame([
    {'variant': 'original', 'soft_prompt': True, 'residual': False, 'second_RGAT': True, 'warmup': WARMUP_EPOCHS},
    {'variant': 'residual', 'soft_prompt': True, 'residual': True, 'second_RGAT': True, 'warmup': WARMUP_EPOCHS},
    {'variant': 'no-refinement', 'soft_prompt': True, 'residual': True, 'second_RGAT': False, 'warmup': WARMUP_EPOCHS},
    {'variant': 'original-no-warmup', 'soft_prompt': True, 'residual': False, 'second_RGAT': True, 'warmup': 0},
    {'variant': 'residual-no-softprompt', 'soft_prompt': False, 'residual': True, 'second_RGAT': True, 'warmup': WARMUP_EPOCHS},
]))
''')
markdown('''
## 10. Execute all three seeds per variant
Fifteen sequential child processes release GPU memory between runs. The same subset
manifest and candidate set are verified for every run. Use the same command to resume
after a disconnect. Changing settings requires a new RUN_TAG.
''')
code('''
if RUN_PILOT:
    launch('run_phase5_comparisons.py', '--output-dir', PILOT_DIR,
           '--max-entities', PILOT_ENTITIES, '--epochs', PILOT_EPOCHS,
           '--warmup-epochs', WARMUP_EPOCHS, '--subset-seed', SUBSET_SEED,
           '--seeds', *SEEDS, '--raw-dir', RAW_DIR, '--text-dir', TEXT_DIR,
           '--cache-dir', POOLED_CACHE, '--text-batch-size', TEXT_BATCH_SIZE)
''')
markdown('## 11. Individual checkpoints and histories')
code('''
run_files = sorted(PILOT_DIR.glob('*_seed*/results.json'))
artifact_rows = []
for file in run_files:
    r = json.loads(file.read_text())
    artifact_rows.append({'variant': r['settings']['variant'], 'seed': r['settings']['seed'],
        'best_epoch': r['best_epoch'], 'warmup_epochs': r['warmup_epochs'], 'main_epochs': r['main_epochs'],
        'optimizer_steps': r['optimizer_steps'], 'duration_seconds': r['duration_seconds'],
        'peak_GPU_GiB': (r['peak_gpu_memory_bytes'] / 2**30) if r['peak_gpu_memory_bytes'] is not None else None,
        'checkpoint': str(file.parent / 'best.pt')})
display(pd.DataFrame(artifact_rows))
print('Every run saves config.json, manifest.json, best.pt, last.pt, history.json/CSV, metrics.csv and results.json.')
''')
markdown('## 12. Compare validation results')
code('''
summary_file = PILOT_DIR / 'summary.json'
summary = json.loads(summary_file.read_text()) if summary_file.exists() else None
if summary:
    comparison = pd.read_csv(PILOT_DIR / 'comparison.csv')
    display(comparison.sort_values('validation_MRR_mean', ascending=False))
else:
    print('The comparison report appears only after all configured pilot runs finish.')
''')
markdown('## 13. Select the best variant using mean validation MRR only')
code('''
selected_variant = summary['selected_by_validation'] if summary else None
print('Selected architecture:', selected_variant)
if summary:
    assert summary['selection_metric'] == 'mean validation MRR'
    print('Seeds:', summary['seeds'])
    print('Pilot candidates:', PILOT_ENTITIES, '; full-data candidates:', dataset.num_entities)
    print('Do not compare pilot MRR directly with full-dataset baseline MRR.')
''')
markdown('## 14. Visualize training loss and validation MRR')
code('''
from IPython.display import Image, display
if summary:
    display(Image(filename=str(PILOT_DIR / 'training_curves.png')))
    display(Image(filename=str(PILOT_DIR / 'validation_comparison.png')))
print('Warmup loss is included in loss curves; validation curves use the integrated model only.')
''')
markdown('''
## 15. Optional full-dataset training of the selected architecture
Disabled by default. This creates a fresh model with full-data entity/relation tables.
No pilot or baseline checkpoint is passed. The flag does not guarantee that a full
two-RGAT + RoBERTa training graph fits a T4. Start with text batch size 1 if needed;
text batching cannot remove full-graph RGAT memory costs. A larger GPU may be needed.
There is no automatic dimension reduction, sampling, precision change, or objective change.
The default final budget is 100 total epochs, including five warmup epochs if applicable.
''')
code('''
FINAL_DIR = RUN_ROOT / 'final_full' / f'{selected_variant}_seed{FINAL_SEED}' if selected_variant else None
if RUN_FINAL_FULL_DATASET:
    assert selected_variant is not None, 'Finish all pilot runs and validation selection first.'
    launch('run_phase5_training.py', '--variant', selected_variant, '--max-entities', 0,
           '--epochs', FINAL_EPOCHS, '--warmup-epochs', WARMUP_EPOCHS, '--seed', FINAL_SEED,
           '--raw-dir', RAW_DIR, '--text-dir', TEXT_DIR, '--cache-dir', POOLED_CACHE,
           '--text-batch-size', TEXT_BATCH_SIZE, '--eval-every', protocol['final']['eval_every'],
           '--output-dir', FINAL_DIR)
''')
markdown('''
## 16. Final evaluation and result export
Only compare full-dataset scores with full-dataset scores. Both models select checkpoints
using validation, then evaluate the untouched original test split. The export records
optimizer-step budgets because baseline minibatches and full-model full-batch training
have different update counts. Pilot architecture selection uses one fixed subset and
three seeds; the optional final run has one seed and is not a full multi-seed benchmark.
''')
code('''
from training.reports import final_comparison

final_result_file = FINAL_DIR / 'results.json' if FINAL_DIR else None
if final_result_file and final_result_file.exists():
    assert baseline_result is not None, 'Train/load the independent full baseline for final comparison.'
    final_result = json.loads(final_result_file.read_text())
    rows = final_comparison(baseline_result, final_result, RUN_ROOT / 'exports')
    display(pd.DataFrame(rows))
else:
    print('Final full-data experiment is disabled or incomplete; no full-model benchmark claimed.')

with (RUN_ROOT / 'environment.txt').open('w') as environment_file:
    subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=environment_file, check=True)
print('Persistent artifacts:', RUN_ROOT)
print('Pilot CSV/JSON and PNG files:', PILOT_DIR)
''')


def build():
    notebook = {'cells': cells, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11'},
        'colab': {'name': 'research_pipeline.ipynb'}, 'accelerator': 'GPU'},
        'nbformat': 4, 'nbformat_minor': 5}
    for index, cell in enumerate(notebook['cells']):
        cell['id'] = f'research-{index:03d}'
    target = Path(__file__).with_name('research_pipeline.ipynb')
    target.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(target)


if __name__ == '__main__':
    build()
