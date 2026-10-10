"""Early pretrained-weight gate; inspect suspect files without modifying them."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def select_device(requested):
    import torch
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was explicitly requested but is unavailable. Select a GPU runtime; no CPU fallback was used.')
    return torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if requested == 'auto' else requested)


def verify(device='auto'):
    import torch
    from models.pretrained import (MODEL_NAME, REVISION, cache_root, prepare_pretrained,
                                   load_pretrained_tokenizer, load_pretrained_model)
    report = {'python': sys.version, 'packages': {name: importlib.metadata.version(name)
              for name in ('torch', 'transformers', 'huggingface-hub', 'safetensors', 'torch-geometric')},
              'model': MODEL_NAME, 'revision': REVISION, 'cache': str(cache_root()),
              'cache_environment': {name: os.environ.get(name) for name in
               ('HF_HOME', 'HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE', 'TRANSFORMERS_CACHE', 'RESEARCH_MODEL_BACKUP')}}
    print(json.dumps(report, indent=2), flush=True)
    runtime_device = select_device(device)
    prepared = prepare_pretrained()
    report['files'] = prepared.files
    print(json.dumps({'verified_files': prepared.files}, indent=2), flush=True)
    tokenizer = load_pretrained_tokenizer()
    model = load_pretrained_model().eval().to(runtime_device)
    inputs = tokenizer('A verified pretrained RoBERTa encoder.', return_tensors='pt').to(runtime_device)
    with torch.no_grad():
        hidden = model(**inputs).last_hidden_state
    if not torch.isfinite(hidden).all():
        raise RuntimeError('Pretrained forward produced nonfinite outputs')
    report.update(status='passed', device=str(runtime_device), output_shape=list(hidden.shape))
    print('Pretrained RoBERTa loading and forward PASSED on', runtime_device, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--inspect-file', type=Path, help='Read-only diagnosis of the original suspect weight file')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    try:
        if args.inspect_file:
            from models.pretrained import inspect_file, WEIGHTS_SHA256
            result = inspect_file(args.inspect_file, WEIGHTS_SHA256)
            print(json.dumps(result, indent=2))
        else:
            result = verify(args.device)
    except Exception as error:
        result = {'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)}
        print(json.dumps(result, indent=2), file=sys.stderr, flush=True)
        print('Check the reported file/path and Hub connectivity. Recovery replaces only a confirmed corrupt model file; '
              'no random weights or alternate model are used.', file=sys.stderr)
        raise
    finally:
        if args.report and 'result' in locals():
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
