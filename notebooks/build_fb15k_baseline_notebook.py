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
    markdown('''# FB15K baseline: RGAT + ComplEx
Existing two-layer baseline, seed0, width100 (50 complex dimensions), 10 negatives,
300 epochs, no warmup. Effective batch32768 = micro4096 × 8: 15 updates/epoch, 4500 total.
Validation every10 epochs; select by validation MRR and test only the best checkpoint.
This is not pure ComplEx paper reproduction. No RoBERTa/text is loaded.
Only section5 launches training. Full GPU memory/runtime capacity must be measured.
''')
    markdown('## 1. GPU runtime')
    code(RUNTIME_SETUP.replace('T4 GPU','GPU').replace('both runs','resumed sessions'))
    markdown('## 2. Fetch main, mount Drive and install dependencies')
    setup = REPOSITORY_SETUP.replace('baseline_shared_c_100', 'fb15k_baseline_300')
    setup = setup.replace('fb15k_baseline_300_seed0_v1_run1', 'fb15k_baseline_300_seed0_accum8_v2_run1')
    setup = setup.replace('run_final_comparison.py', 'run_baseline_fb15k.py')
    setup = '\n'.join(line for line in setup.splitlines() if 'RESEARCH_MODEL_BACKUP' not in line
                      and 'Verified weights load locally' not in line)
    code(setup)
    markdown('## 3. Approved configuration and full dataset verification')
    code('''import json
import yaml
import pandas as pd
from datetime import datetime
from IPython.display import display, Image, Markdown
from training.baseline_fb15k import PROTOCOL, make_config, expected_budget, verify_protocol, verify_manifest
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
protocol = yaml.safe_load(Path('experiments/configs/fb15k_baseline_300.yaml').read_text(encoding='utf-8'))
verify_protocol(protocol, make_config())
cfg = make_config(RAW_DIR)
display(pd.DataFrame([{'dataset':'FB15K', 'model':'RGAT + ComplEx', 'seed':0,
                      'width':cfg.model['dim'], 'negatives':cfg.num_negatives,
                      'epochs':cfg.epochs, 'warmup':cfg.warmup,
                      'effective_batch':cfg.train_batch_size, 'micro_batch':cfg.micro_batch_size,
                      'accumulation_steps':cfg.accumulation_steps, **expected_budget()}]))
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
Runs one child process. Logs every epoch; validation at10,20,...,300.
Effective batch32768 uses8 scoring micro-batches of4096 and one RGAT encoding.
The final24390 examples use6 micro-batches (5×4096 +3910), weighted by their sizes.
Clipping and Adam run once per effective batch. Full-graph activations still remain resident.
Each completed epoch commits last.pt; best validation saves best.pt.
Resume uses the same tag, source, settings and runtime. If main changes, set
REPO_REF to the recorded original SHA when resuming. Never mix old study outputs.
No automatic batch-size, precision or epoch-budget changes on OOM.
Your compute-unit balance is not a runtime guarantee: use observed epoch timings.
''')
    code('''command = [sys.executable,'-u','run_baseline_fb15k.py',
           '--output-dir',str(EXPERIMENT_DIR),'--raw-dir',str(RAW_DIR)]
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
        print('Committed progress:', history[-1]['epoch'], '/ 300 epochs;', history[-1]['optimizer_steps'], '/ 4500 updates')
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
