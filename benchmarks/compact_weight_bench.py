"""Compare 32-bit Rep4 decode with raw, byte-plane, and compact word weights."""
import argparse
import json
from pathlib import Path

import torch

from benchmarks.timing import measure
from rep4gpu.kernels import local_products, prepare_weight


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes",type=int,nargs="+",default=[2048,4096])
    parser.add_argument("--output",type=Path)
    parser.add_argument("--ffn-width",type=int)
    args=parser.parse_args()
    rows=[]
    shapes=[(k,k) for k in args.sizes]
    if args.ffn_width:
        width=args.ffn_width
        shapes=[(width,4*width),(4*width,width)]
    for k,n in shapes:
        xs=[torch.randint(-2**63,2**63-1,(1,k),device="cuda",dtype=torch.int64) for _ in range(3)]
        ys=[torch.randint(-2**63,2**63-1,(k,n),device="cuda",dtype=torch.int64) for _ in range(3)]
        packed=[prepare_weight(y,32) for y in ys]
        compact=[prepare_weight(y,32,storage="words32") for y in ys]
        expected=local_products(*xs,*ys,32)
        for storage,weights in (("int64",ys),("limbs",packed),("words32",compact)):
            fn=lambda:local_products(*xs,*weights,32)
            for actual,want in zip(fn(),expected):
                torch.testing.assert_close(actual,want,rtol=0,atol=0)
            row=dict(m=1,k=k,n=n,bits=32,storage=storage,**measure(fn,rep=40))
            print(json.dumps(row),flush=True)
            rows.append(row)
        a,b=xs[0].float()/2**63,ys[0].float()/2**63
        for backend,af,bf in (("fp32",a,b),("fp16",a.half(),b.half())):
            row=dict(m=1,k=k,n=n,backend=backend,**measure(lambda:af@bf,rep=40))
            print(json.dumps(row),flush=True)
            rows.append(row)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(rows,indent=2)+"\n")


if __name__=="__main__":
    main()
