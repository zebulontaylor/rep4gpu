"""Check and time direct-register exact warp MMA kernels."""
import argparse
import json
from pathlib import Path

import torch

from benchmarks.timing import measure
from rep4gpu.kernels import local_products
from rep4gpu.warp_mma_kernels import local_products_mma, prepare_mma_weight


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m",type=int,default=128)
    parser.add_argument("--size",type=int,default=512)
    parser.add_argument("--bits",type=int,default=64)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    m,k,bits=args.m,args.size,args.bits
    xs=[torch.randint(-2**63,2**63-1,(m,k),device="cuda",dtype=torch.int64) for _ in range(3)]
    ys=[torch.randint(-2**63,2**63-1,(k,k),device="cuda",dtype=torch.int64) for _ in range(3)]
    packed=[prepare_mma_weight(y,bits) for y in ys]
    expected=local_products(*xs,*ys,bits)
    results=[]
    for warps in (4,8):
        fn=lambda:local_products_mma(*xs,*packed,bits,warps=warps)
        out=fn()
        for a,b in zip(out,expected):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        row=dict(m=m,k=k,n=k,bits=bits,warps=warps,**measure(fn,rep=40))
        print(json.dumps(row),flush=True)
        results.append(row)
    af,bf=xs[0].float()/2**63,ys[0].float()/2**63
    ah,bh=af.half(),bf.half()
    for backend,a,b in (("fp32",af,bf),("fp16",ah,bh)):
        row=dict(backend=backend,**measure(lambda:a@b,rep=40))
        print(json.dumps(row),flush=True)
        results.append(row)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(results,indent=2)+"\n")


if __name__=="__main__":
    main()
