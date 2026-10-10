"""Validate and stage the frozen RoBERTa checkpoint before Transformers opens it.

Only the named corrupt Hub file is force-downloaded. User-supplied directories
are read-only. No cache tree is ever deleted and no random-weight fallback exists.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = 'roberta-base'
REVISION = 'e2da8e2f811d1448a5b465c236feacd80ffbac7b'
# Published LFS SHA256 at this revision, also verified against our actual file.
WEIGHTS_SHA256 = '5bde1d28afb363d0103324efeb5afc8b2b397fe5e04beabb9b1ef355255ade81'
METADATA_SHA256 = {
    'config.json': 'ef0185e2aae6e06c5f105a285006952c340e20c7dbf43c86ec82601b13fc45e9',
    'tokenizer.json': '847bbeab6174d66a88898f729d52fa8d355fafe1bea101cf960dd404581df70e',
    'tokenizer_config.json': '994f46754c5bf4014f1aa92d34b1374319c3a6b3f702105cd5b742beaecd18ce',
    'vocab.json': '9e7f63c2d15d666b52e21d250d2e513b87c9b713cfa6987a82ed89e5e6e50655',
    'merges.txt': '1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5',
}
FILES = ('config.json', 'model.safetensors', 'tokenizer.json',
         'tokenizer_config.json', 'vocab.json', 'merges.txt')


class InvalidModelFile(ValueError):
    pass


def cache_root():
    # Explicit arguments avoid huggingface_hub's import-time environment snapshot.
    return Path(os.environ.get('HF_HUB_CACHE') or os.environ.get('HUGGINGFACE_HUB_CACHE')
                or str(Path(os.environ.get('HF_HOME', Path.home() / '.cache' / 'huggingface')) / 'hub'))


def inspect_file(path, expected_sha256=None):
    path = Path(path)
    report = {'path': str(path.absolute()), 'resolved_path': str(path.resolve()),
              'size_bytes': path.stat().st_size}
    with path.open('rb') as stream:
        first = stream.read(160)
        report['first_32_bytes_hex'] = first[:32].hex()
        if path.name.endswith('.safetensors'):
            report['header_length'] = int.from_bytes(first[:8], 'little') if len(first) >= 8 else None
        stream.seek(0)
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    report['sha256'] = checksum
    try:
        if first.startswith(b'version https://git-lfs.github.com/spec'):
            raise ValueError('Git LFS pointer text, not model weights')
        if first.lstrip().lower().startswith((b'<!doctype html', b'<html', b'<?xml', b'<error')):
            raise ValueError('HTML/XML response, not model weights')
        if path.name.endswith('.safetensors'):
            header = report['header_length']
            if header is None or header < 2 or header > 100_000_000:
                raise ValueError(f'invalid safetensors header length: {header}')
            if header + 8 > report['size_bytes']:
                raise ValueError('truncated safetensors header')
            # Checks JSON, offsets, tensor layouts and full payload extent.
            with safe_open(str(path), framework='pt', device='cpu') as handle:
                report['tensor_count'] = len(handle.keys())
                if not report['tensor_count']:
                    raise ValueError('empty tensor checkpoint')
        elif path.suffix == '.json':
            json.loads(path.read_text(encoding='utf-8'))
        elif not first:
            raise ValueError('empty file')
        if expected_sha256 and checksum != expected_sha256:
            raise ValueError(f'SHA256 mismatch; expected {expected_sha256}')
    except Exception as error:
        report.update(valid=False, reason=str(error))
        raise InvalidModelFile(json.dumps(report, indent=2)) from error
    return {**report, 'valid': True}


def atomic_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=destination.name + '.', suffix='.tmp', dir=destination.parent)
    os.close(descriptor)
    try:
        shutil.copyfile(source, name)
        os.replace(name, destination)
    finally:
        Path(name).unlink(missing_ok=True)


@dataclass
class PreparedModel:
    path: Path
    revision: str | None
    files: list


def prepare_pretrained(model_name=MODEL_NAME, revision=None, *, cache_dir=None, backup_dir=None):
    local = Path(model_name)
    if local.is_dir():
        # Never modify or repair an explicitly supplied model directory.
        reports = [inspect_file(local / name) for name in FILES]
        return PreparedModel(local, None, reports)
    if model_name not in (MODEL_NAME, 'FacebookAI/roberta-base'):
        raise ValueError('Verified Hub loader supports roberta-base; supply a validated local directory for a custom model')
    if revision not in (None, REVISION):
        raise ValueError(f'This research protocol pins RoBERTa to {REVISION}; requested {revision}')
    root = Path(cache_dir) if cache_dir is not None else cache_root()
    stage = root / 'verified_snapshots' / 'roberta-base' / REVISION
    backup = backup_dir or os.environ.get('RESEARCH_MODEL_BACKUP')
    backup = Path(backup) / REVISION if backup else None
    reports = []
    for name in FILES:
        expected = WEIGHTS_SHA256 if name == 'model.safetensors' else METADATA_SHA256.get(name)
        target = stage / name
        report = None
        if target.exists():
            try:
                report = inspect_file(target, expected)
            except InvalidModelFile as error:
                print('Invalid staged model file:\n' + str(error), flush=True)
        if report is None and backup and (backup / name).exists():
            try:
                inspect_file(backup / name, expected)
                atomic_copy(backup / name, target)
                report = inspect_file(target, expected)
            except InvalidModelFile as error:
                print('Invalid persistent model backup (will replace this file only):\n' + str(error), flush=True)
        if report is None:
            downloaded = Path(hf_hub_download(model_name, name, revision=REVISION, cache_dir=str(root)))
            try:
                inspect_file(downloaded, expected)
            except InvalidModelFile as error:
                print('Confirmed corrupt Hub cache entry:\n' + str(error), flush=True)
                print(f'Redownloading only {model_name}@{REVISION}/{name}', flush=True)
                downloaded = Path(hf_hub_download(model_name, name, revision=REVISION,
                                                  cache_dir=str(root), force_download=True))
                inspect_file(downloaded, expected)  # A second failure is fatal.
            atomic_copy(downloaded, target)
            report = inspect_file(target, expected)
        reports.append(report)
        if backup:
            try:
                if not (backup / name).exists() or inspect_file(backup / name, expected)['sha256'] != report['sha256']:
                    atomic_copy(target, backup / name)
            except InvalidModelFile:
                atomic_copy(target, backup / name)
    return PreparedModel(stage, REVISION, reports)


def load_pretrained_model(model_name=MODEL_NAME, revision=None):
    prepared = prepare_pretrained(model_name, revision)
    model, info = AutoModel.from_pretrained(str(prepared.path), local_files_only=True,
                                           use_safetensors=True, output_loading_info=True)
    # The MLM checkpoint has no pooler; that unused AutoModel output is harmless.
    missing = set(info.get('missing_keys', [])) - {'pooler.dense.weight', 'pooler.dense.bias'}
    if missing or info.get('mismatched_keys') or info.get('error_msgs'):
        raise RuntimeError(f'Pretrained encoder did not load completely: {info}')
    model.config._commit_hash = prepared.revision
    return model


def load_pretrained_tokenizer(model_name=MODEL_NAME, revision=None):
    prepared = prepare_pretrained(model_name, revision)
    return AutoTokenizer.from_pretrained(str(prepared.path), local_files_only=True)
