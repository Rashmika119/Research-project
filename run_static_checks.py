"""Explicit non-training checks; no optimizer steps, downloads or CUDA workloads."""
import os
import subprocess
import sys


if __name__ == '__main__':
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1')
    env.pop('RESEARCH_MODEL_BACKUP', None)
    modules = ('tests.test_final_study', 'tests.test_notebook', 'tests.test_selected_models',
               'tests.test_pretrained', 'tests.test_process', 'tests.test_reproducibility',
               'tests.test_dataset', 'tests.test_scoring', 'tests.test_baseline_fb15k',
               'tests.test_accumulation')
    raise SystemExit(subprocess.run([sys.executable, '-m', 'unittest', *modules, '-v'], env=env).returncode)
