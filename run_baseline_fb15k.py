"""Launch the YAML-configured baseline; configure native kernels before torch imports."""
import argparse
import os
from training.baseline_config import DEFAULT_CONFIG, load_protocol

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    args, _ = parser.parse_known_args()
    import sys
    if '--report-only' not in sys.argv and '--help' not in sys.argv:
        protocol = load_protocol(args.config)
        os.environ['TORCH_DISABLE_NATIVE_JIT'] = '1' if protocol['runtime']['disable_native_jit'] else '0'
    from training.baseline_fb15k import main
    main()
