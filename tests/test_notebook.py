"""Protected historical notebooks remain byte-for-byte research evidence."""
import hashlib
from pathlib import Path
import unittest
import nbformat
from tests.test_full_study import PROTECTED

class NotebookTests(unittest.TestCase):
    def test_historical_notebooks_unchanged(self):
        root = Path(__file__).resolve().parents[1]
        for path, checksum in PROTECTED.items():
            with self.subTest(path=path):
                self.assertEqual(hashlib.sha256((root/path).read_bytes()).hexdigest(), checksum)
                nbformat.validate(nbformat.read(root/path, as_version=4))
