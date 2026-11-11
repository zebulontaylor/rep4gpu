"""Check Trident-style online matrix products at the toy FFN's actual sizes.

Run ``python -m benchmarks.trident_linear --output results/trident_linear.json``.
Each layer is tested independently with public, floor-rounded fixture inputs.
The fixture supplies trusted correlations; preprocessing traffic is unknown.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.inference import fixture
from rep4gpu.trident_linear_reference import run_matmul


def run():
    torch.set_num_threads(1)
    _, tensors, _ = fixture(1, 2048, 8192, 2048, 14)
    x, w1, b1, w2, _ = [value.numpy() for value in tensors]
    raw1 = x @ w1
    hidden = np.maximum((raw1 + (b1 << 14)) >> 14, 0)
    cases = []
    for index, (name, left, right) in enumerate((('up', x, w1), ('down', hidden, w2))):
        expected = (left @ right).astype(np.uint32)
        result = run_matmul(left, right, seed=20261110 + index)
        output = result.pop('output')
        if not np.array_equal(output, expected):
            raise AssertionError(f'{name} matrix product differs from integer fixture')
        result.update(layer=name, input_shape=list(left.shape), weight_shape=list(right.shape),
                      exact_integer_match=True, output_sha256=hashlib.sha256(output.tobytes()).hexdigest())
        cases.append(result)
    sources = [Path(__file__), Path('rep4gpu/trident_linear_reference.py'),
               Path('rep4gpu/transport.py'), Path('benchmarks/inference.py')]
    return dict(
        methodology='Four role-local objects on IdealNetwork; measured serialized application payloads. '
                    'Independent layer tests, not a complete private FFN. No distributed latency measurement.',
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        results=cases, summed_linear_online_payload_bytes=sum(c['online_payload_bytes'] for c in cases),
        preprocessing_payload_bytes=None, full_private_ffn_validated=False,
        sub500kb_validated=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/trident_linear.json'))
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
