"""Launch both YAML-configured FB15K models with backend policy set before torch."""
import argparse
import os
from pathlib import Path
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', default=str(Path(__file__).with_name('comparison_config.yaml')))
    args, _ = parser.parse_known_args()
    protocol = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    os.environ['TORCH_DISABLE_NATIVE_JIT'] = '1' if protocol['runtime']['disable_native_jit'] else '0'
    from training.final_study import main
    main()
