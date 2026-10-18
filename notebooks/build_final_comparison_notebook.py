"""Generate the final full-dataset notebook without overwriting research evidence."""
import json
from pathlib import Path
RUNTIME_SETUP = r"""import os
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
print('Keep GPU, packages and numerical policy consistent across both runs.')

torch.use_deterministic_algorithms(True, warn_only=False)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
"""

REPOSITORY_SETUP = r"""from google.colab import drive
drive.mount('/content/drive')
SHARED_ROOT = Path('/content/drive/MyDrive/Research/scratch_pipeline_v1')
STAGE_ROOT = Path('/content/drive/MyDrive/Research/baseline_shared_c_100')
RUN_TAG = 'baseline_shared_c_100_seed0_v1_run1'  # Use the SAME tag to resume; a NEW tag for changed settings.
RUN_ROOT = STAGE_ROOT / RUN_TAG
RUN_ROOT.mkdir(parents=True, exist_ok=True)
os.environ['RESEARCH_MODEL_BACKUP'] = str(SHARED_ROOT / 'verified_roberta_backup')
REPO_URL = 'https://github.com/Rashmika119/Research-project.git'
REPO_REF = 'main'  # Fetch the latest pushed main; record its resolved commit below.
REPO_DIR = Path('/content/Research-project')
if not REPO_DIR.exists():
    subprocess.run(['git', 'clone', REPO_URL, str(REPO_DIR)], check=True)
dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO_DIR, text=True).strip()
assert not dirty, 'Repository has local edits. Preserve them before selecting the research commit.'
subprocess.run(['git', 'fetch', 'origin', REPO_REF], cwd=REPO_DIR, check=True)
subprocess.run(['git', 'checkout', '--detach', 'FETCH_HEAD'], cwd=REPO_DIR, check=True)
os.chdir(REPO_DIR)
assert Path('run_final_comparison.py').is_file(), 'Select the commit containing this full study.'
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
    LOCK.write_text(''.join(f'{name}=={version}\n' for name, version in versions.items()))
print('Persistent dependency lock:', LOCK)
print('Verified weights load locally from:', os.environ['HF_HUB_CACHE'])
"""


def notebook():
    nb = {'nbformat': 4, 'nbformat_minor': 5,
          'metadata': {'colab': {}, 'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
          'cells': [{'cell_type': 'code' if i >= 2 and i % 2 == 0 else 'markdown',
                     'metadata': {}, 'source': '',
                     **({'execution_count': None, 'outputs': []} if i >= 2 and i % 2 == 0 else {})}
                    for i in range(17)]}
    def source(i, text): nb['cells'][i]['source'] = text.strip()+'\n'
    source(0, '''# Full-dataset comparison: KG-only baseline versus Shared C gated
Two models, seed 0: two independent scratch runs.
Full FB15k-237, 100 epochs and 900 optimizer updates per run.
Baseline: 100 structural epochs, no RoBERTa. Shared C gated: 10 warmup + 90 integrated
epochs, frozen independent RoBERTa, live projection, text/refinement gates and ComplEx.
Sections 1-5 do not train. Section 6 alone starts research training.
Full-data T4 capacity is unverified; OOM never silently changes the protocol.
''')
    source(1, '## 1. Runtime and CUDA environment')
    source(2, RUNTIME_SETUP)
    source(3, '## 2. Drive, repository and dependencies')
    source(4, REPOSITORY_SETUP)
    source(5,'## 3. Final experiment configuration')
    source(6,'''import json
import yaml
import pandas as pd
from dataclasses import asdict
from datetime import datetime
from IPython.display import display, Image, Markdown
from training.final_study import CONFIGURATIONS, SEEDS as APPROVED_SEEDS, PROTOCOL, make_config, expected_budget, verify_manifest, preflight
from training.experiment_io import write_json, environment
from training.reproducibility import seed_everything, numerical_policy
from training.process import run_logged

SEEDS = [0]  # Exactly two architecture runs, seed 0 only.
SKIP_COMPLETED = True
RESUME_INTERRUPTED = True
RUN_STATIC_CHECKS = False  # Non-training checks only; CPU/offline.
EXPERIMENT_DIR = RUN_ROOT / 'runs'
MANIFEST = EXPERIMENT_DIR / 'dataset_manifest.json'
RAW_DIR = SHARED_ROOT / 'data' / 'fb15k237'
TEXT_DIR = SHARED_ROOT / 'data' / 'text'
POOLED_CACHE = SHARED_ROOT / 'pooled_text_cache'
LOG_DIR = RUN_ROOT / 'logs' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
LOG_DIR.mkdir(parents=True, exist_ok=True)
protocol = yaml.safe_load(Path('experiments/configs/baseline_shared_c.yaml').read_text(encoding='utf-8'))
assert protocol['protocol'] == PROTOCOL, 'Notebook/repository protocol mismatch; select the final-study commit.'
assert list(CONFIGURATIONS) == list(protocol['configurations'])
assert SEEDS == list(APPROVED_SEEDS) == protocol['seeds']
configs = {f'{c}_seed{s}': make_config(c,s,raw_dir=RAW_DIR,text_dir=TEXT_DIR) for c in CONFIGURATIONS for s in SEEDS}
for cfg in configs.values():
    assert cfg.warmup == protocol['configurations'][cfg.variant]['warmup']
    for key in ('max_entities','epochs','train_batch_size','lr','weight_decay','num_negatives','grad_clip','model','text_batch_size','eval_batch_size','max_text_length','eval_every','selection_policy','record_step_losses','deterministic','lm_name'):
        assert asdict(cfg)[key] == protocol[key], f'Runner/YAML mismatch: {key}'
seed_everything(0, True)
print(json.dumps(protocol, indent=2))
print('Numerical policy:', numerical_policy())
display(pd.DataFrame([{'configuration':c,'seed':s,**expected_budget(c)} for c in CONFIGURATIONS for s in SEEDS]))
print('2 independent runs; 1,800 committed optimizer updates; no preliminary training.')
write_json(LOG_DIR/'environment.json',environment())
with (LOG_DIR/'pip_freeze.txt').open('w') as f:
    subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)
''')
    source(7,'''## 4. Dataset, pretrained files and existing run identities
Verify the full original mappings/splits, train-only graph with inverses, and pinned
RoBERTa files. Read saved configuration/checkpoint hashes if resuming. No model
forward/backward, optimizer steps or training occur. The child also verifies full
text and checkpoint identity before resume or skip.
''')
    source(8,'''from training.experiment_io import prepare_data
from models.pretrained import prepare_pretrained

data, graph, known_train, manifest = prepare_data(str(RAW_DIR),0,0,MANIFEST)
verify_manifest(manifest)
assert len(graph.edge_index) == 544230
assert set(data.train) == set(known_train)
print('Full dataset:',manifest['statistics'])
print('Dataset SHA:',manifest['dataset_sha256'],'; manifest SHA:',manifest['sha256'])
write_json(RUN_ROOT/'dataset_statistics.json',manifest['statistics'])
preflight(EXPERIMENT_DIR,configs,manifest)
prepared = prepare_pretrained()
assert prepared.revision == protocol['lm_revision']
write_json(LOG_DIR/'pretrained_files.json',{'revision':prepared.revision,'files':prepared.files})
print('Verified weights:',prepared.path)
print('Persistent results:',EXPERIMENT_DIR)
del data,graph,known_train,manifest
''')
    source(9,'## 5. Optional developer verification (disabled by default)')
    source(10,'''if RUN_STATIC_CHECKS:
    run_logged([sys.executable,'run_static_checks.py'],LOG_DIR/'static_checks.log')
else:
    print('Optional non-training checks disabled; no smoke or preliminary training runs.')
''')
    source(11,'''## 6. Actual final research training
Two child processes run sequentially: baseline, then Shared C gated, seed 0.
Baseline trains structurally for all 100 epochs; selection starts at epoch 5.
Shared C uses warmup epochs 1-10, integrated epochs 11-100, selection from epoch 15.
Its warmup and integrated-boundary evaluations are diagnostic only.
Both select by scheduled validation MRR; earliest equal best wins.
Each epoch prints phase/loss/updates/validation/time/allocated and reserved peaks.
To resume, rerun setup and this cell with the same commit, packages, GPU and tag.
Completed runs are verified/skipped; interrupted runs restore their own last.pt.
An uncommitted epoch replays. Preserve old outputs if identities change.
''')
    source(12,'''command = [sys.executable,'-u','run_final_comparison.py','--output-dir',str(EXPERIMENT_DIR),
           '--raw-dir',str(RAW_DIR),'--text-dir',str(TEXT_DIR),'--cache-dir',str(POOLED_CACHE),
           '--seeds',*map(str,SEEDS)]
if SKIP_COMPLETED: command.append('--skip-completed')
if RESUME_INTERRUPTED: command.append('--resume')
run_logged(command,LOG_DIR/'final_study.log')
''')
    source(13,'## 7. Results and analysis')
    source(14,'''from training.final_reports import PLOTS
if (EXPERIMENT_DIR/'summary.json').is_file():
    display(pd.read_csv(EXPERIMENT_DIR/'comparison.csv'))
    display(pd.read_csv(EXPERIMENT_DIR/'per_seed_validation.csv'))
    display(pd.read_csv(EXPERIMENT_DIR/'per_seed_test.csv'))
    display(pd.read_csv(EXPERIMENT_DIR/'paired_summary.csv'))
    display(Markdown((EXPERIMENT_DIR/'research_analysis.md').read_text(encoding='utf-8')))
    for name in PLOTS:
        display(Image(filename=str(EXPERIMENT_DIR/(name+'.png'))))
else:
    print('Complete both runs before making the final comparison.')
print('Results, PNG/PDF figures and checkpoints:',EXPERIMENT_DIR)
''')
    source(15,'''## 8. Report regeneration only
Run independently after setting EXPERIMENT_DIR and SEEDS (no GPU required):
`python run_final_comparison.py --report-only --output-dir /path/to/runs --seeds 0`
This reads saved result histories only: no model evaluation or checkpoint loading.
No figures are produced from partial studies or synthetic research measurements.
''')
    source(16,'''paths = [EXPERIMENT_DIR/f'{c}_seed{s}'/'results.json' for c in CONFIGURATIONS for s in SEEDS]
if all(p.is_file() for p in paths):
    from training.final_reports import final_report
    summary = final_report(EXPERIMENT_DIR,[json.loads(p.read_text(encoding='utf-8')) for p in paths],seeds=SEEDS)
    print('Selected by mean validation MRR:',summary['selected_by_validation'])
else:
    print('Missing completed runs:',[str(p) for p in paths if not p.is_file()])
''')
    nb['metadata']['colab']['name']='baseline_vs_shared_c_full.ipynb'
    for i,c in enumerate(nb['cells']): c['id']=f'final-study-{i:03d}'
    return nb


def build():
    path=Path(__file__).with_name('baseline_vs_shared_c_full.ipynb')
    if path.exists(): raise FileExistsError('Preserve existing notebook cells and outputs; write a reviewed preview instead.')
    path.write_text(json.dumps(notebook(),indent=2)+'\n',encoding='utf-8')
    print(path)

if __name__=='__main__': build()
