"""Compare shared-memory SIMT, tensor-limb, and clear CUDA matrix products."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.timing import measure
from rep4gpu import simt_kernels
from rep4gpu import kernels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 512, 1024])
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 16, 128])
    parser.add_argument("--bits", nargs="+", type=int, default=[64])
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = []
    rng = np.random.default_rng(1234)
    for m in args.rows:
        for k in args.sizes:
            xs = [torch.from_numpy(rng.bit_generator.random_raw(m*k).view(np.int64).reshape(m,k)).cuda() for _ in range(3)]
            ys = [torch.from_numpy(rng.bit_generator.random_raw(k*k).view(np.int64).reshape(k,k)).cuda() for _ in range(3)]
            ah, bh = (xs[0].float()/2**63).half(), (ys[0].float()/2**63).half()
            af, bf = ah.float(), bh.float()
            cases = [("fp16", lambda: ah@bh), ("fp32", lambda: af@bf)]
            for bits in args.bits:
                cases.extend([
                    (f"simt_single{bits}", lambda bits=bits: simt_kernels.ring_matmul(xs[0],ys[0],bits)),
                    (f"simt_local{bits}", lambda bits=bits: simt_kernels.local_products(*xs,*ys,bits)),
                    (f"tensor_single{bits}", lambda bits=bits: kernels.ring_matmul(xs[0],ys[0],bits)),
                    (f"tensor_local{bits}", lambda bits=bits: kernels.local_products(*xs,*ys,bits)),
                ])
            for name, fn in cases:
                row = dict(m=m,k=k,n=k,backend=name,**measure(fn,rep=args.rep))
                print(json.dumps(row),flush=True)
                results.append(row)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(dict(device=torch.cuda.get_device_name(),results=results),indent=2)+"\n")


if __name__ == "__main__":
    main()
