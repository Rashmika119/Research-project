"""Generate the final full-dataset notebook without overwriting research evidence."""
import copy
import json
from pathlib import Path
from notebooks.build_full_dataset_notebook import notebook as previous_notebook


def notebook():
    nb = copy.deepcopy(previous_notebook())
    def source(i, text): nb['cells'][i]['source'] = text.strip()+'\n'
    source(0, '''# Final full-dataset comparison: Residual versus Shared C
Three existing architectures, seed 0: three independent scratch runs.
Full FB15k-237, 100 total epochs = 10 warmup + 90 integrated, 900 updates/run.
All use frozen independent RoBERTa, trainable projection, text gates and ComplEx.
Shared C ungated retains the skip addition and both text gates.
Sections 1-5 do not train. Section 6 alone starts research training.
Full-data T4 capacity is unverified; OOM never silently changes the protocol.
''')
    source(1, '## 1. Runtime and CUDA environment')
    source(2, nb['cells'][2]['source'].replace('all three runs','all three runs') + '''
torch.use_deterministic_algorithms(True, warn_only=False)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
''')
    setup=nb['cells'][4]['source'].replace('full_dataset_comparison_100','final_three_architectures_100').replace('full_fb15k237_100_seed0_v1_run1','full_fb15k237_residual_vs_shared_c_100_10_seed0_v1_run1').replace('run_full_dataset_comparison.py','run_final_comparison.py')
    setup=setup.replace("REPO_REF = 'feature/phase_03'  # Push this stage first. Replace with its exact commit SHA for resume.", "REPO_REF = 'SET_EXACT_PUSHED_COMMIT_SHA'\nimport re\nassert re.fullmatch(r'[0-9a-fA-F]{40}', REPO_REF), 'Push this final study and enter its exact 40-character Git commit SHA.'")
    source(4,setup)
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

SEEDS = [0]  # Exactly three architecture runs, seed 0 only.
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
protocol = yaml.safe_load(Path('experiments/configs/final_residual_vs_shared_c.yaml').read_text(encoding='utf-8'))
assert protocol['protocol'] == PROTOCOL, 'Notebook/repository protocol mismatch; select the final-study commit.'
assert list(CONFIGURATIONS) == list(protocol['configurations'])
assert SEEDS == list(APPROVED_SEEDS) == protocol['seeds']
configs = {f'{c}_seed{s}': make_config(c,s,raw_dir=RAW_DIR,text_dir=TEXT_DIR) for c in CONFIGURATIONS for s in SEEDS}
for cfg in configs.values():
    for key in ('max_entities','epochs','warmup','train_batch_size','lr','weight_decay','num_negatives','grad_clip','model','text_batch_size','eval_batch_size','max_text_length','eval_every','selection_policy','record_step_losses','deterministic','lm_name'):
        assert asdict(cfg)[key] == protocol[key], f'Runner/YAML mismatch: {key}'
seed_everything(0, True)
print(json.dumps(protocol, indent=2))
print('Numerical policy:', numerical_policy())
display(pd.DataFrame([{'configuration':c,'seed':s,**expected_budget(c)} for c in CONFIGURATIONS for s in SEEDS]))
print('3 independent runs; 2,700 committed optimizer updates; no preliminary training.')
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
Three child processes run sequentially: A (independent gated), B (shared C gated),
C (shared C ungated), each with seed 0. Epochs 1-10 are structural warmup;
11-100 are integrated. Validation at 5/10 and the integrated boundary is diagnostic.
Selection begins at 15 and uses validation MRR only; earliest equal best wins.
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
    print('Complete all three runs before making the final comparison.')
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
    nb['metadata']['colab']['name']='final_residual_vs_shared_c_full.ipynb'
    for i,c in enumerate(nb['cells']): c['id']=f'final-study-{i:03d}'
    return nb


def build():
    path=Path(__file__).with_name('final_residual_vs_shared_c_full.ipynb')
    if path.exists(): raise FileExistsError('Preserve existing notebook cells and outputs; write a reviewed preview instead.')
    path.write_text(json.dumps(notebook(),indent=2)+'\n',encoding='utf-8')
    print(path)

if __name__=='__main__': build()
