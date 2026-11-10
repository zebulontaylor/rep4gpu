"""Generate one-use native q256 correlations without field conversion."""

import argparse
import json
from pathlib import Path

from rep4gpu.preprocessing_mascot_direct import run_native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-root', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--batch-size', type=int, default=1000)
    args = parser.parse_args()
    report = run_native(args.manifest, args.output, native_root=args.native_root,
                        timeout=args.timeout, batch_size=args.batch_size)
    fields = ('passed', 'ready', 'return_codes', 'compile_ms',
              'native_preprocessing_ms', 'preprocessing_ms')
    print(json.dumps({key: report.get(key) for key in fields}, indent=2))
    return 0 if report.get('passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
