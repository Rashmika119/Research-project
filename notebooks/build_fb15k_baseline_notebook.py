"""Build the baseline-only FB15K Colab notebook; preserve existing notebooks."""
import json
from pathlib import Path

from notebooks.build_final_comparison_notebook import RUNTIME_SETUP, REPOSITORY_SETUP


def notebook():
    cells = []
    def markdown(source):
        cells.append({'cell_type':'markdown', 'metadata':{}, 'source':source.strip()+'\n'})
    def code(source):
        cells.append({'cell_type':'code', 'metadata':{}, 'source':source.strip()+'\n',
                      'outputs':[], 'execution_count':None})
    markdown('''# FB15K baseline: YAML-configured RGAT + ComplEx
This notebook fetches main and reads `baseline_config.yaml` from the repository root.
Edit and push that YAML to change epochs, dimensions, batching, validation interval,
backend setting and run tag. The default study is 300 epochs, width64, seed0.
Only section5 launches training. No RoBERTa/text is loaded.
A changed budget starts a new scratch run; it does not extend the completed 100-epoch run.
''')
    markdown('## 1. GPU runtime')
    code(RUNTIME_SETUP.replace('T4 GPU','GPU').replace('both runs','resumed sessions'))
    markdown('## 2. Fetch main, mount Drive and install dependencies')
    setup = REPOSITORY_SETUP.replace('baseline_shared_c_100', 'fb15k_baseline_300')
    setup = setup.replace('run_final_comparison.py', 'run_baseline_fb15k.py')
    setup = '\n'.join(line for line in setup.splitlines()
                      if not line.startswith(('RUN_TAG =', 'RUN_ROOT =', 'RUN_ROOT.mkdir'))
                      and 'RESEARCH_MODEL_BACKUP' not in line
                      and 'Verified weights load locally' not in line)
    setup = setup.replace("LOCK = RUN_ROOT / 'dependencies.txt'", '''# Read the freshly fetched YAML before choosing the persistent run folder.
subprocess.run([sys.executable, '-m', 'pip', 'install', 'pyyaml'], check=True)
import yaml
CONFIG_PATH = REPO_DIR / 'baseline_config.yaml'
protocol = yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8'))
RUN_TAG = protocol['run']['tag']
assert RUN_TAG not in ('.', '..') and '/' not in RUN_TAG and chr(92) not in RUN_TAG
RUN_ROOT = STAGE_ROOT / RUN_TAG
RUN_ROOT.mkdir(parents=True, exist_ok=True)
LOCK = RUN_ROOT / 'dependencies.txt'
''')
    code(setup)
    markdown('## 3. Approved configuration and full dataset verification')
    code('''import json
import yaml
import pandas as pd
from datetime import datetime
from IPython.display import display, Image, Markdown
from training.baseline_fb15k import make_config, expected_budget, verify_protocol, verify_manifest
from training.experiment_io import prepare_data, write_json, environment
from training.process import run_logged

EXPERIMENT_DIR = RUN_ROOT / 'runs'
RAW_DIR = SHARED_ROOT / 'data' / 'fb15k'
MANIFEST = EXPERIMENT_DIR / 'dataset_manifest.json'
LOG_DIR = RUN_ROOT / 'logs' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
LOG_DIR.mkdir(parents=True, exist_ok=True)
SKIP_COMPLETED = True
RESUME_INTERRUPTED = True
RUN_STATIC_CHECKS = False
protocol = yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8'))
cfg = make_config(RAW_DIR, CONFIG_PATH)
verify_protocol(protocol, cfg)
print('Fetched configuration:', CONFIG_PATH, protocol)
write_json(LOG_DIR/'requested_configuration.json', protocol)
display(pd.DataFrame([{'dataset':'FB15K', 'model':'RGAT + ComplEx', 'seed':0,
                      'width':cfg.model['dim'], 'negatives':cfg.num_negatives,
                      'optimizer':cfg.optimizer, 'lr':cfg.lr, 'loss':cfg.loss, 'l2':cfg.l2_coefficient,
                      'epochs':cfg.epochs, 'warmup':cfg.warmup,
                      'effective_batch':cfg.train_batch_size, 'micro_batch':cfg.micro_batch_size,
                      'accumulation_steps':cfg.accumulation_steps, **expected_budget(cfg)}]))
data, graph, known_train, manifest = prepare_data(str(RAW_DIR),0,0,MANIFEST,dataset='fb15k')
verify_manifest(manifest)
assert len(graph.edge_type) == 966284
assert set(data.train) == set(known_train)
print('Full FB15K:', manifest['statistics'])
print('Dataset SHA256:', manifest['dataset_sha256'])
print('Manifest SHA256:', manifest['sha256'])
write_json(RUN_ROOT/'dataset_statistics.json', manifest['statistics'])
write_json(LOG_DIR/'environment.json', environment())
print('One baseline run; no pretrained model or text cache needed.')
print('Persistent outputs:', EXPERIMENT_DIR)
del data, graph, known_train, manifest
''')
    markdown('## 4. Optional offline CPU checks (no training)')
    code('''if RUN_STATIC_CHECKS:
    run_logged([sys.executable,'run_static_checks.py'],LOG_DIR/'static_checks.log')
else:
    print('Optional checks disabled. No preliminary training is launched.')
''')
    markdown('''## 5. Train the existing baseline
Runs one child process using `--config baseline_config.yaml`. Logs every epoch.
Validation and total epochs come from YAML; final-epoch validation also runs.
Scoring micro-batches use YAML sizes and one RGAT encoding per optimizer update.
Partial final batches are weighted by actual example counts.
Clipping and the configured optimizer run once per effective batch. Full-graph activations still remain resident.
Each completed epoch commits last.pt; best validation saves best.pt.
Resume uses the same tag, source, settings and runtime. If main changes, set
REPO_REF to the recorded original SHA when resuming. Never mix old study outputs.
No automatic batch-size, precision or epoch-budget changes on OOM.
Your compute-unit balance is not a runtime guarantee: use observed epoch timings.
''')
    code('''command = [sys.executable,'-u','run_baseline_fb15k.py',
           '--output-dir',str(EXPERIMENT_DIR),'--raw-dir',str(RAW_DIR),
           '--config',str(CONFIG_PATH)]
if SKIP_COMPLETED: command.append('--skip-completed')
if RESUME_INTERRUPTED: command.append('--resume')
run_logged(command,LOG_DIR/'baseline_training.log')
''')
    markdown('## 6. Results and progress (also usable after an interruption)')
    code('''from training.baseline_fb15k_reports import PLOTS
history_path = EXPERIMENT_DIR/'baseline_seed0/history.json'
if history_path.exists():
    history = json.loads(history_path.read_text(encoding='utf-8'))
    display(pd.DataFrame([{k:v for k,v in h.items() if k not in ('step_losses','validation','learned_gates')}
                          for h in history]).tail(10))
    if history:
        saved_settings = json.loads((EXPERIMENT_DIR/'baseline_seed0/config.json').read_text(encoding='utf-8'))['config']
        from training.research_pipeline import ExperimentConfig
        saved_budget = expected_budget(ExperimentConfig(**saved_settings))
        print('Committed progress:', history[-1]['epoch'], '/', saved_settings['epochs'], 'epochs;',
              history[-1]['optimizer_steps'], '/', saved_budget['optimizer_steps'], 'updates')
if (EXPERIMENT_DIR/'summary.json').exists():
    display(pd.read_csv(EXPERIMENT_DIR/'metrics.csv'))
    display(Markdown((EXPERIMENT_DIR/'research_analysis.md').read_text(encoding='utf-8')))
    for name in PLOTS:
        display(Image(filename=str(EXPERIMENT_DIR/(name+'.png'))))
else:
    print('No completed report yet; partial histories are not final test results.')
''')
    markdown('''## 7. Regenerate completed reports without training
Can run independently after setting EXPERIMENT_DIR. No GPU or checkpoint loading:
`python run_baseline_fb15k.py --report-only --output-dir /path/to/runs`
''')
    code('''result_path = EXPERIMENT_DIR/'baseline_seed0/results.json'
if result_path.exists():
    from training.baseline_fb15k_reports import baseline_report
    baseline_report(EXPERIMENT_DIR,json.loads(result_path.read_text(encoding='utf-8')))
    print('Reports regenerated:', EXPERIMENT_DIR)
else:
    print('Complete training before generating final reports.')
''')
    for i,c in enumerate(cells):
        c['id'] = f'fb15k-baseline-{i:03d}'
    return {'cells':cells, 'nbformat':4, 'nbformat_minor':5,
            'metadata':{'colab':{'name':'fb15k_baseline_300.ipynb'},
                        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}}}


def build():
    path = Path(__file__).with_name('fb15k_baseline_300.ipynb')
    if path.exists():
        raise FileExistsError('Preserve notebook outputs; generate a reviewed preview instead')
    path.write_text(json.dumps(notebook(),indent=2)+'\n',encoding='utf-8')
    print(path)


if __name__ == '__main__':
    build()
