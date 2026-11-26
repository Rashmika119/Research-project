"""Only the two supported pipeline notebooks are published."""
from pathlib import Path
import unittest
import nbformat

class NotebookTests(unittest.TestCase):
    def test_only_supported_notebooks(self):
        root = Path(__file__).resolve().parents[1]
        paths = list((root/'notebooks').glob('*.ipynb'))
        self.assertEqual({p.name for p in paths}, {'fb15k_baseline_300.ipynb', 'baseline_vs_shared_c_full.ipynb'})
        for path in paths:
            nbformat.validate(nbformat.read(path, as_version=4))
