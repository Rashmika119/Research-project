"""Content-addressed cache of frozen, independently pooled text representations."""
from pathlib import Path
import importlib.metadata

import torch

from training.experiment_io import digest, save_checkpoint


def lm_identity(lm):
    model = lm.lm
    revision = getattr(model.config, '_commit_hash', None)
    if revision is None:
        # Local/custom models lack a Hub commit: identify the actual weights.
        import hashlib
        h = hashlib.sha256()
        for name, tensor in sorted(model.state_dict().items()):
            h.update(name.encode())
            h.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        revision = h.hexdigest()
    return {'name': lm.model_name, 'revision_or_weights': revision,
            'hidden_size': lm.hidden_size}


def pooled_text(lm, tokens, batch_size, device, cache_dir=None):
    if batch_size < 1:
        raise ValueError('Text batch size must be positive')
    metadata = {'schema': 2, 'lm': lm_identity(lm),
                'pooling': 'attention_mask_mean_including_special_tokens',
                'torch': str(torch.__version__),
                'transformers': importlib.metadata.version('transformers'),
                'device_type': torch.device(device).type,
                'device_name': torch.cuda.get_device_name(device) if torch.device(device).type == 'cuda' else 'cpu',
                'cuda': torch.version.cuda, 'batch_size': batch_size,
                'dtype': str(next(lm.parameters()).dtype),
                'attention_implementation': getattr(lm.lm.config, '_attn_implementation', None),
                'input_ids': tokens['input_ids'].tolist(),
                'attention_mask': tokens['attention_mask'].tolist()}
    # Token content and ordering implicitly cover descriptions, tokenizer and truncation.
    key = digest(metadata)
    path = Path(cache_dir) / (key + '.pt') if cache_dir else None
    shape = (len(tokens['input_ids']), lm.hidden_size)
    if path and path.exists():
        cached = torch.load(path, map_location='cpu', weights_only=True)
        result = cached['pooled']
        if cached['key'] != key or tuple(result.shape) != shape or not torch.isfinite(result).all():
            raise ValueError('Invalid pooled-text cache: ' + str(path))
        print('Frozen pooled-text cache hit:', path, flush=True)
        return result.detach(), key
    parts = []
    original_device = next(lm.parameters()).device
    print('Computing frozen pooled text:', shape, '; batch size:', batch_size, flush=True)
    lm.to(device)
    try:
        for start in range(0, shape[0], batch_size):
            part = {k: v[start:start + batch_size].to(device) for k, v in tokens.items()}
            parts.append(lm.encode_text(part['input_ids'], part['attention_mask']).detach().cpu())
    finally:
        lm.to(original_device)
    result = torch.cat(parts)
    if tuple(result.shape) != shape or not torch.isfinite(result).all():
        raise ValueError('Invalid frozen text representations')
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(path, {'key': key, 'identity': metadata['lm'], 'pooled': result})
    return result, key
