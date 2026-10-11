"""Explicit non-training checks; no optimizer steps, downloads or CUDA workloads."""
import os
import subprocess
import sys


if __name__ == '__main__':
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1')
    env.pop('RESEARCH_MODEL_BACKUP', None)
    modules = ('tests.test_full_study', 'tests.test_notebook', 'tests.test_selected_models',
               'tests.test_pretrained', 'tests.test_process', 'tests.test_reproducibility',
               'tests.test_dataset.SubsetTests.test_induced_complete_exact_count_and_heldout_independence',
               'tests.test_dataset.SubsetTests.test_disconnected_training_components_fill_exactly')
    raise SystemExit(subprocess.run([sys.executable, '-m', 'unittest', *modules, '-v'], env=env).returncode)
