"""Generate only the new full-dataset notebook. Refuse accidental regeneration."""
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
# Full FB15k-237: three models, 300 epochs, three seeds
Nine independent scratch runs: RGAT + ComplEx baseline; Residual No-Softprompt
with 30 warmup epochs; identical No-Softprompt with no warmup. Each run performs
2,700 optimizer updates. Existing notebooks and results are historical artifacts.

Run Sections 1–4 for non-training setup and integrity checks. Optional training
tests in Section 5 are disabled. Real training begins only when you execute the
Section 6 cell. This notebook has not been validated on a full-data T4 training run.
No automatic batch/entity/precision changes occur after an out-of-memory error.
''')
md('## 1. CUDA environment and local cache settings')
code('''
import os
import sys
import subprocess
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['HF_HOME'] = '/content/hf_cache'
os.environ['HF_HUB_CACHE'] = '/content/hf_cache/hub'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/content/hf_cache/hub'
os.environ.pop('TRANSFORMERS_CACHE', None)
import torch
print('Python:', sys.version, '; PyTorch:', torch.__version__, '; CUDA:', torch.version.cuda)
assert torch.cuda.is_available(), 'Select Runtime > Change runtime type > T4 GPU.'
print('GPU:', torch.cuda.get_device_name(0))
print('Total GPU GiB:', torch.cuda.get_device_properties(0).total_memory / 2**30)
print('Free / total bytes:', torch.cuda.mem_get_info())
print('Keep GPU, packages and numerical policy consistent across all nine runs.')
''')
md('## 2. Drive, repository and dependencies')
code('''
from google.colab import drive
drive.mount('/content/drive')
SHARED_ROOT = Path('/content/drive/MyDrive/Research/scratch_pipeline_v1')
STAGE_ROOT = Path('/content/drive/MyDrive/Research/full_dataset_comparison_300')
RUN_TAG = 'full_fb15k237_300_v1_run1'  # Use the SAME tag to resume; a NEW tag for changed settings.
RUN_ROOT = STAGE_ROOT / RUN_TAG
RUN_ROOT.mkdir(parents=True, exist_ok=True)
os.environ['RESEARCH_MODEL_BACKUP'] = str(SHARED_ROOT / 'verified_roberta_backup')
REPO_URL = 'https://github.com/Rashmika119/Research-project.git'
REPO_REF = 'feature/phase_03'  # Push this stage first. Replace with its exact commit SHA for resume.
REPO_DIR = Path('/content/Research-project')
if not REPO_DIR.exists():
    subprocess.run(['git', 'clone', REPO_URL, str(REPO_DIR)], check=True)
dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO_DIR, text=True).strip()
assert not dirty, 'Repository has local edits. Preserve them before selecting the research commit.'
subprocess.run(['git', 'fetch', 'origin', REPO_REF], cwd=REPO_DIR, check=True)
subprocess.run(['git', 'checkout', '--detach', 'FETCH_HEAD'], cwd=REPO_DIR, check=True)
os.chdir(REPO_DIR)
assert Path('run_full_dataset_comparison.py').is_file(), 'Select the commit containing this full study.'
COMMIT = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
print('Exact repository SHA:', COMMIT)
LOCK = RUN_ROOT / 'dependencies.txt'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(LOCK if LOCK.exists() else Path('requirements.txt'))], check=True)
# The lock covers packages used by run identity. CUDA/Python are provided by Colab:
# a changed runtime must be made compatible, never silently accepted for resume.
import importlib.metadata as metadata
packages = ('torch', 'torch-geometric', 'transformers', 'huggingface-hub', 'safetensors',
            'numpy', 'matplotlib', 'pandas', 'pyyaml', 'nbformat', 'ipython')
versions = {name: metadata.version(name) for name in packages}
print('Package versions:', versions)
fresh_torch = subprocess.check_output([sys.executable, '-c', 'import torch; print(torch.__version__)'], text=True).strip()
assert str(torch.__version__) == fresh_torch, 'PyTorch changed during installation; restart runtime and rerun setup.'
if not LOCK.exists():
    LOCK.write_text(''.join(f'{name}=={version}\\n' for name, version in versions.items()))
print('Persistent dependency lock:', LOCK)
print('Verified weights load locally from:', os.environ['HF_HUB_CACHE'])
''')
md('## 3. Approved experiment configuration')
code('''
import json
import yaml
import pandas as pd
from dataclasses import asdict
from datetime import datetime
from IPython.display import display, Image, Markdown
from training.full_study import CONFIGURATIONS, SEEDS as APPROVED_SEEDS, make_config, expected_budget, verify_manifest
from training.experiment_io import write_json, environment
from training.process import run_logged

MAX_ENTITIES = 0
SEEDS = [0, 1, 2]
TOTAL_EPOCHS = 300
WARMUP_EPOCHS = 30
TRAIN_BATCH_SIZE = 32768
TEXT_BATCH_SIZE = 256
EVAL_BATCH_SIZE = 512
MAX_TEXT_LENGTH = 64
EVAL_EVERY = 5
SKIP_COMPLETED = True
RESUME_INTERRUPTED = True
RUN_OFFLINE_TRAINING_TESTS = False
RUN_REAL_LM_SMOKE_TRAINING = False
RUN_CUDA_RESUME_SMOKE_TRAINING = False
RUN_SYNTHETIC_TRAINING_TESTS = False
assert (MAX_ENTITIES, SEEDS, TOTAL_EPOCHS, WARMUP_EPOCHS, TRAIN_BATCH_SIZE, MAX_TEXT_LENGTH, EVAL_EVERY) == (0, list(APPROVED_SEEDS), 300, 30, 32768, 64, 5)
assert TEXT_BATCH_SIZE > 0 and EVAL_BATCH_SIZE > 0
EXPERIMENT_DIR = RUN_ROOT / 'runs'
MANIFEST = EXPERIMENT_DIR / 'dataset_manifest.json'
RAW_DIR = SHARED_ROOT / 'data' / 'fb15k237'
TEXT_DIR = SHARED_ROOT / 'data' / 'text'
POOLED_CACHE = SHARED_ROOT / 'pooled_text_cache'
LOG_DIR = RUN_ROOT / 'logs' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
LOG_DIR.mkdir(parents=True, exist_ok=True)
protocol = yaml.safe_load(Path('experiments/configs/full_dataset_300.yaml').read_text())
assert list(CONFIGURATIONS) == list(protocol['configurations'])
rows = []
for identifier in CONFIGURATIONS:
    cfg = make_config(identifier, 0, raw_dir=RAW_DIR, text_dir=TEXT_DIR,
                      text_batch_size=TEXT_BATCH_SIZE, eval_batch_size=EVAL_BATCH_SIZE)
    rows.append({'configuration': identifier, 'warmup_epochs': cfg.warmup,
                 'main_epochs': 300-cfg.warmup, **expected_budget(identifier)})
display(pd.DataFrame(rows))
print('9 runs; 24,300 committed updates; full graph recomputed for every optimizer step.')
write_json(LOG_DIR / 'environment.json', environment())
with (LOG_DIR / 'pip_freeze.txt').open('w') as f:
    subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=f, check=True)
''')
md('''
## 4. Non-training dataset and model-file verification
No forward/backward or optimizer steps run here. Model verification checks pinned
files and verified backup copies; it does not instantiate RoBERTa. A baseline-only
CLI run has no LM requirement. Dataset loading rejects synthetic fallback.
''')
code('''
from training.experiment_io import prepare_data
from models.pretrained import prepare_pretrained
data, graph, known_train, manifest = prepare_data(str(RAW_DIR), 0, 0, MANIFEST)
verify_manifest(manifest)
assert len(graph.edge_index) == 544230
assert set(data.train) == set(known_train)
print('Full dataset statistics:', manifest['statistics'])
print('Dataset SHA256:', manifest['dataset_sha256'])
print('Manifest SHA256:', manifest['sha256'])
print('All 14,541 entities remain candidates, including training-isolated entities.')
write_json(RUN_ROOT / 'dataset_statistics.json', manifest['statistics'])
prepared = prepare_pretrained()
write_json(LOG_DIR / 'pretrained_files.json', {'revision': prepared.revision, 'files': prepared.files})
print('Verified pretrained files:', prepared.path)
print('Output directory:', EXPERIMENT_DIR)
del data, graph, known_train, manifest
''')
md('''
## 5. Optional developer training tests — disabled
These commands run only if you explicitly change their flags. They are not required
before Section 6. Final test-set evaluation after real training is always enabled.
''')
code('''
CPU_ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1')
CPU_ENV.pop('RESEARCH_MODEL_BACKUP', None)
if RUN_OFFLINE_TRAINING_TESTS:
    run_logged([sys.executable, 'run_scratch_smoke_test.py', '--offline-only', '--device', 'cpu'],
               LOG_DIR / 'optional_offline.log', env=CPU_ENV)
if RUN_REAL_LM_SMOKE_TRAINING:
    run_logged([sys.executable, 'run_scratch_smoke_test.py', '--real-lm-only', '--device', 'cuda'],
               LOG_DIR / 'optional_real_lm.log')
if RUN_CUDA_RESUME_SMOKE_TRAINING:
    run_logged([sys.executable, 'run_resume_diagnostic.py', '--device', 'cuda', '--repeats', '3',
                '--output-dir', str(LOG_DIR / 'optional_resume')], LOG_DIR / 'optional_resume.log')
if RUN_SYNTHETIC_TRAINING_TESTS:
    run_logged([sys.executable, '-m', 'unittest', 'tests.test_research_pipeline', '-v'],
               LOG_DIR / 'optional_synthetic.log', env=CPU_ENV)
print('Optional test flags:', RUN_OFFLINE_TRAINING_TESTS, RUN_REAL_LM_SMOKE_TRAINING,
      RUN_CUDA_RESUME_SMOKE_TRAINING, RUN_SYNTHETIC_TRAINING_TESTS)
''')
md('''
## 6. Execute the nine real research runs
Execute the following cell when ready to train. It performs no preliminary training.
Every completed epoch prints loss, phase, elapsed time and peak GPU memory; epochs
5,10,...,300 also print validation MRR/Hits@10. Full traces are streamed and saved.

Structural validation cannot select integrated weights. The initial integrated
evaluation at epoch 0 or the epoch-30 boundary is diagnostic only. Baseline and
no-warmup selection start at epoch 5; warmup integrated selection starts at 35.
All runs finish 300 epochs and evaluate their validation-selected model on test.
''')
code('''
command = [sys.executable, '-u', 'run_full_dataset_comparison.py',
           '--raw-dir', str(RAW_DIR), '--text-dir', str(TEXT_DIR),
           '--cache-dir', str(POOLED_CACHE), '--text-batch-size', str(TEXT_BATCH_SIZE),
           '--eval-batch-size', str(EVAL_BATCH_SIZE), '--output-dir', str(EXPERIMENT_DIR)]
if SKIP_COMPLETED:
    command.append('--skip-completed')
if RESUME_INTERRUPTED:
    command.append('--resume')
run_logged(command, LOG_DIR / 'full_study.log')
''')
md('''
## 7. Checkpoints and disconnection recovery
Reconnect with the same RUN_TAG, exact source commit, package versions, GPU and
settings. Rerun setup and the training cell. Completed runs are verified/skipped;
partial runs restore their own atomic epoch-end last.pt. The unfinished epoch
is replayed (up to nine updates); committed budget remains 2,700. No mid-epoch
resume or cross-hardware exact-reproducibility claim is made.

Each run saves config.json, manifest.json, best.pt, last.pt, history.json/CSV,
steps.csv, metrics.csv, results.json and a resume audit when applicable. Last
checkpoint includes Adam, all RNG streams, modes, progress and selected weights.
Changed identities fail explicitly. Never pass an earlier baseline as initialization.
''')
code('''
status = []
for identifier in CONFIGURATIONS:
    for seed in SEEDS:
        folder = EXPERIMENT_DIR / f'{identifier}_seed{seed}'
        result_path = folder / 'results.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else None
        status.append({'configuration': identifier, 'seed': seed,
                       'status': 'complete' if result else ('partial' if (folder / 'last.pt').exists() else 'not started'),
                       'best_epoch': result['best_epoch'] if result else None,
                       'optimizer_steps': result['optimizer_steps'] if result else None,
                       'checkpoint': str(folder / 'last.pt')})
display(pd.DataFrame(status))
''')
md('''
## 8. Final comparison and research analysis
Select by mean validation MRR, then report test metrics separately. Tables include
sample SD, paired seed effects, Hits@1/3/10, ties, memory, time and best/final epochs.
Dashed warmup, solid integrated and dotted baseline curves represent distinct phases.
Loss versus updates includes all 2,700 step losses; epoch loss is the mean of nine
update losses. The 9,971-positive last batch has a separate example-weighted diagnostic.

Paired signs show whether improvements agree across seeds. Ranking instability
compares MRR changes over matched epochs 35–300. A falling loss and stalled/worsening
validation is an overfitting warning, not proof. Observed-best convergence thresholds
are descriptive, not equal absolute performance targets. No significance claims.
''')
code('''
from training.full_reports import PLOTS, full_report
result_paths = [EXPERIMENT_DIR / f'{c}_seed{s}' / 'results.json' for c in CONFIGURATIONS for s in SEEDS]
if all(p.is_file() for p in result_paths):
    # Report regeneration only: no training, LM loading or test-set reevaluation.
    summary = full_report(EXPERIMENT_DIR, [json.loads(p.read_text()) for p in result_paths])
    display(pd.read_csv(EXPERIMENT_DIR / 'comparison.csv').sort_values('validation_MRR_mean', ascending=False))
    display(pd.read_csv(EXPERIMENT_DIR / 'runs.csv'))
    display(pd.DataFrame(summary['paired_comparisons']))
    display(pd.read_csv(EXPERIMENT_DIR / 'paired_seed_differences.csv'))
    display(pd.DataFrame(json.loads((EXPERIMENT_DIR / 'training_behavior.json').read_text())))
    display(pd.read_csv(EXPERIMENT_DIR / 'ties.csv'))
    print('Selected by validation:', summary['selected_by_validation'])
    print('Selected configuration test results:', summary['selected_test_results'])
    print(summary['interpretation'])
    display(Markdown((EXPERIMENT_DIR / 'research_analysis.md').read_text()))
    for filename in PLOTS:
        display(Image(filename=str(EXPERIMENT_DIR / filename)))
else:
    print('Complete all nine runs before making a configuration comparison.')
print('Persistent results:', EXPERIMENT_DIR)
print('Logs:', LOG_DIR)
''')


def notebook():
    result = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}, 'colab': {'name': 'full_dataset_comparison_300.ipynb'}, 'accelerator': 'GPU'},
        'nbformat': 4, 'nbformat_minor': 5}
    for i, cell in enumerate(cells):
        cell['id'] = f'full-study-{i:03d}'
    return result


def build():
    target = Path(__file__).with_name('full_dataset_comparison_300.ipynb')
    if target.exists():
        raise FileExistsError('Notebook already exists; preserve manual edits. Generate a reviewed preview using notebook().')
    target.write_text(json.dumps(notebook(), indent=2) + '\n', encoding='utf-8')
    print(target)


if __name__ == '__main__':
    build()
