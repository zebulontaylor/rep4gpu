"""Benchmark fused local share products, optionally packing static weights.

Run from the repository root with ``python -m benchmarks.local_kernel_bench``.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import triton

from benchmarks.timing import measure
from rep4gpu.kernels import local_products, prepare_weight, ring_matmul


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sizes = (128,512) if args.quick else (128,512,1024,2048,4096)
    rng = np.random.default_rng(1337)
    results=[]
    for m in (1,16,128):
        for k in sizes:
            xs=[torch.from_numpy(rng.bit_generator.random_raw(m*k).view(np.int64).reshape(m,k)).cuda() for _ in range(3)]
            ys=[torch.from_numpy(rng.bit_generator.random_raw(k*k).view(np.int64).reshape(k,k)).cuda() for _ in range(3)]
            af,bf=xs[0].float()/2**63,ys[0].float()/2**63
            ah,bh=af.half(),bf.half()
            for bits in (32,64):
                packed=[prepare_weight(y,bits) for y in ys]
                cases=[("fused",lambda:local_products(*xs,*ys,bits)),
                       ("fused_prepared",lambda:local_products(*xs,*packed,bits)),
                       ("single_prepared",lambda:ring_matmul(xs[0],packed[0],bits))]
                for name,fn in cases:
                    fn()
                    timing=measure(fn)
                    row=dict(m=m,k=k,n=k,bits=bits,backend=name,**timing)
                    print(json.dumps(row),flush=True)
                    results.append(row)
            for name,a,b in [("fp32",af,bf),("fp16",ah,bh)]:
                timing=measure(lambda:a@b)
                row=dict(m=m,k=k,n=k,backend=name,**timing)
                print(json.dumps(row),flush=True)
                results.append(row)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(dict(device=torch.cuda.get_device_name(),results=results),indent=2)+"\n")


if __name__=="__main__":
    main()
