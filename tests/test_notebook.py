"""Immutable historical evidence remains readable after cleanup."""
import hashlib
import json
from pathlib import Path
import unittest
import nbformat

class NotebookTests(unittest.TestCase):
    def test_archived_evidence_unchanged(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root/'docs/archive/pre_two_model/manifest.json').read_text(encoding='utf-8'))
        for row in manifest['files']:
            with self.subTest(path=row['archived']):
                path = root/row['archived']
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),row['sha256'])
                if path.suffix == '.ipynb': nbformat.validate(nbformat.read(path,as_version=4))
