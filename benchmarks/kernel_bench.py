"""Time exact ring kernels and floating GEMM on the same CUDA device.

Run from the repository root: ``python -m benchmarks.kernel_bench --quick``.
Floating GEMMs are throughput baselines, not valid secret-share computations.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import triton

from benchmarks.timing import measure
from rep4gpu.kernels import ring_matmul


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sizes = [128, 512] if args.quick else [128, 512, 1024, 2048, 4096]
    rows = []
    rng = np.random.default_rng(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    for m in (1, 16, 128):
        for size in sizes:
            a = torch.from_numpy(rng.bit_generator.random_raw(m*size).view(np.int64).reshape(m,size)).cuda()
            b = torch.from_numpy(rng.bit_generator.random_raw(size*size).view(np.int64).reshape(size,size)).cuda()
            af, bf = a.float() / 2**63, b.float() / 2**63
            # Normalize floating baselines while retaining full-ring integer kernel inputs.
            ah, bh = af.half(), bf.half()
            for bits in (32, 64):
                for backend in ("decode", "tensor"):
                    fn = lambda: ring_matmul(a,b,bits,backend=backend)
                    fn()
                    timing = measure(fn)
                    row = dict(m=m,k=size,n=size,bits=bits,backend=backend,**timing)
                    rows.append(row)
                    print(json.dumps(row), flush=True)
            for dtype, aa, bb in (("fp32",af,bf),("fp16",ah,bh)):
                timing = measure(lambda: aa@bb)
                row = dict(m=m,k=size,n=size,backend=dtype,**timing)
                rows.append(row)
                print(json.dumps(row),flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(dict(device=torch.cuda.get_device_name(),results=rows),indent=2)+"\n")


if __name__ == "__main__":
    main()
