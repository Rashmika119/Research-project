"""Offline corruption tests use tiny safetensors, never pretend to be real RoBERTa."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from models import pretrained as p


class PretrainedTests(unittest.TestCase):
    def setUp(self):
        # Direct unittest/pytest execution must not inherit a Colab Drive backup.
        isolated = patch.dict(os.environ, {'RESEARCH_MODEL_BACKUP': '', 'HF_HUB_OFFLINE': '1'})
        isolated.start()
        self.addCleanup(isolated.stop)

    def test_missing_encoder_weights_are_fatal(self):
        model = SimpleNamespace(config=SimpleNamespace())
        prepared = p.PreparedModel(Path('verified-fixture'), p.REVISION, [])
        with (patch.object(p, 'prepare_pretrained', return_value=prepared),
              patch.object(p.AutoModel, 'from_pretrained', return_value=(model, {
                  'missing_keys': ['encoder.layer.0.attention.self.query.weight']})) as load):
            with self.assertRaisesRegex(RuntimeError, 'encoder did not load completely'):
                p.load_pretrained_model()
            self.assertTrue(load.call_args.kwargs['use_safetensors'])
            self.assertTrue(load.call_args.kwargs['local_files_only'])
            load.return_value = model, {'missing_keys': ['pooler.dense.weight', 'pooler.dense.bias']}
            self.assertIs(p.load_pretrained_model(), model)
            self.assertEqual(model.config._commit_hash, p.REVISION)

    def test_invalid_headers_and_payload_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.safetensors'
            for content, reason in (
                (b'version https://git-lfs.github.com/spec/v1\n', 'LFS pointer'),
                (b'<html>download denied</html>', 'HTML/XML'),
                (b'abc', 'header length'),
                ((200).to_bytes(8, 'little') + b'{}', 'truncated'),
                (b'12345678payload', 'header length')):
                with self.subTest(reason=reason):
                    path.write_bytes(content)
                    with self.assertRaisesRegex(p.InvalidModelFile, reason):
                        p.inspect_file(path)
            save_file({'weight': torch.ones(4)}, path)
            checksum = p.inspect_file(path)['sha256']
            content = bytearray(path.read_bytes())
            content[-1] ^= 1  # Valid header/offsets, corrupt tensor payload.
            path.write_bytes(content)
            with self.assertRaisesRegex(p.InvalidModelFile, 'SHA256 mismatch'):
                p.inspect_file(path, checksum)
            path.write_bytes(content[:-1])
            with self.assertRaises(p.InvalidModelFile):
                p.inspect_file(path)

    def test_targeted_recovery_backup_and_cache_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remote, hub = root / 'remote', root / 'hub'
            remote.mkdir()
            hub.mkdir()
            for name in p.FILES:
                (remote / name).write_text('{}' if name.endswith('.json') else 'merges', encoding='utf-8')
            save_file({'weight': torch.ones(4)}, remote / 'model.safetensors')
            checksum = hashlib.sha256((remote / 'model.safetensors').read_bytes()).hexdigest()
            corrupt = hub / 'model.safetensors'
            corrupt.write_text('version https://git-lfs.github.com/spec/v1\n')
            unrelated = hub / 'unrelated.pt'
            unrelated.write_bytes(b'preserve user data')
            calls = []
            def download(repo, name, **kwargs):
                calls.append((name, kwargs.get('force_download', False)))
                return remote / name if kwargs.get('force_download') or name != 'model.safetensors' else corrupt
            with (patch.object(p, 'WEIGHTS_SHA256', checksum),
                  patch.object(p, 'METADATA_SHA256', {}),
                  patch.object(p, 'hf_hub_download', side_effect=download)):
                prepared = p.prepare_pretrained(cache_dir=hub, backup_dir=root / 'backup')
                self.assertEqual([name for name, forced in calls if forced], ['model.safetensors'])
                self.assertTrue(all(report['valid'] for report in prepared.files))
                self.assertEqual(unrelated.read_bytes(), b'preserve user data')
                count = len(calls)
                p.prepare_pretrained(cache_dir=hub, backup_dir=root / 'backup')
                self.assertEqual(len(calls), count)
                # A fresh local cache can be populated offline from verified backup.
                with patch.object(p, 'hf_hub_download', side_effect=AssertionError('Unexpected download')):
                    p.prepare_pretrained(cache_dir=root / 'fresh', backup_dir=root / 'backup')
                # A corrupt backup is repaired only when a good local copy exists.
                (root / 'backup' / p.REVISION / 'model.safetensors').write_bytes(b'broken')
                p.prepare_pretrained(cache_dir=hub, backup_dir=root / 'backup')
                self.assertTrue(p.inspect_file(root / 'backup' / p.REVISION / 'model.safetensors', checksum)['valid'])

    def test_invalid_redownload_fails_without_random_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text('{}')
            (root / 'model.safetensors').write_bytes(b'broken')
            with (patch.object(p, 'METADATA_SHA256', {}),
                  patch.object(p, 'hf_hub_download', side_effect=lambda repo, name, **kw: root / name) as download):
                with self.assertRaises(p.InvalidModelFile):
                    p.prepare_pretrained(cache_dir=root / 'cache')
                self.assertTrue(download.call_args.kwargs['force_download'])
            with patch.object(p, 'hf_hub_download', side_effect=AssertionError('Local directory must stay read-only')):
                with self.assertRaises(p.InvalidModelFile):
                    p.prepare_pretrained(str(root))
            self.assertEqual((root / 'model.safetensors').read_bytes(), b'broken')
