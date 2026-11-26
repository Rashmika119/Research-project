"""Read baseline launch settings without importing PyTorch."""
from pathlib import Path
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'baseline_config.yaml'


def load_protocol(path=DEFAULT_CONFIG):
    protocol = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(protocol, dict) or set(protocol) != {'protocol', 'description', 'config', 'runtime', 'run'}:
        raise ValueError('Expected protocol, description, config, runtime and run in baseline YAML')
    if not isinstance(protocol['protocol'], str) or not protocol['protocol']:
        raise ValueError('Missing protocol name')
    if set(protocol['runtime']) != {'disable_native_jit'} or type(protocol['runtime']['disable_native_jit']) is not bool:
        raise ValueError('runtime.disable_native_jit must be a boolean')
    tag = protocol['run'].get('tag')
    if set(protocol['run']) != {'tag'} or not isinstance(tag, str) or not tag or tag in ('.', '..') or any(c in tag for c in '/\\'):
        raise ValueError('run.tag must be a single directory name')
    return protocol
