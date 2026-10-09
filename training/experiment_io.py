"""Dataset manifests, atomic artifacts, and reproducibility identities."""
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

import torch

from preprocessing.dataset import load_dataset
from preprocessing.graph_builder import build_train_graph, assert_no_leakage
from preprocessing.toy_subset import make_toy_subset


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def save_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def environment():
    packages = {}
    for name in ('torch', 'torch-geometric', 'transformers', 'numpy', 'matplotlib'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = 'not installed'
    root = Path(__file__).resolve().parents[1]
    sources = {str(p.relative_to(root)): file_digest(p)
               for folder in ('models', 'training', 'evaluation', 'preprocessing')
               for p in sorted((root / folder).glob('*.py'))}
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {'python': platform.python_version(), 'packages': packages,
            'source_sha256': digest(sources), 'git_commit': commit,
            'cuda': torch.version.cuda,
            'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}


def dataset_payload(data):
    return {key: getattr(data, key) for key in ('entity2id', 'relation2id', 'train', 'valid', 'test')}


def validate_fb15k237(data):
    actual = (data.num_entities, data.num_relations, len(data.train), len(data.valid), len(data.test))
    expected = (14541, 237, 272115, 17535, 20466)
    if actual != expected:
        raise ValueError(f'Expected real FB15k-237 {expected}, received {actual}; no synthetic fallback')


def prepare_data(raw_dir, max_entities, subset_seed, manifest_path=None, full_dataset=None):
    """Injected datasets are for offline tests; CLI always validates real FB15k-237."""
    full = full_dataset
    if full is None:
        full = load_dataset(raw_dir, download_if_missing=True, use_synthetic_fallback=False)
        validate_fb15k237(full)
    data = full if max_entities == 0 else make_toy_subset(full, max_entities, subset_seed)
    if not data.train or not data.valid or not data.test:
        raise ValueError('Empty split. Increase the subset size; never manufacture held-out facts.')
    original_to_local = {full.entity2id[name]: local for name, local in data.entity2id.items()}
    known_train = [(original_to_local[h], r, original_to_local[t]) for h, r, t in full.train
                   if h in original_to_local and t in original_to_local]
    payload = dataset_payload(data)
    manifest = {'schema': 1, 'dataset_sha256': digest(dataset_payload(full)),
                'max_entities': max_entities, 'subset_seed': subset_seed,
                'data': payload, 'known_train': known_train}
    manifest['sha256'] = digest(manifest)
    if manifest_path:
        path = Path(manifest_path)
        if path.exists():
            previous = json.loads(path.read_text(encoding='utf-8'))
            if digest(previous) != digest(manifest):
                raise ValueError('Dataset/subset differs from saved shared manifest')
        else:
            write_json(path, manifest)
    graph = build_train_graph(data.train, data.num_entities, data.num_relations)
    assert_no_leakage(graph, data.train, data.valid, data.test)
    if set(data.valid) & set(data.test):
        raise ValueError('Validation/test overlap')
    # Check the exact edges, not only their count.
    expected = [(h, t, r) for h, r, t in data.train]
    expected += [(t, h, r + data.num_relations) for h, r, t in data.train]
    actual = [(h, t, r) for (h, t), r in zip(graph.edge_index, graph.edge_type)]
    if sorted(actual) != sorted(expected):
        raise AssertionError('Training message graph differs from training triples')
    return data, graph, known_train, manifest
