"""Compare resident packed exact kernels, including activation packing time."""
import argparse
import json
from pathlib import Path

import torch

from benchmarks.timing import measure
from rep4gpu.kernels import local_products, prepare_weight
from rep4gpu.packed_kernels import local_products_packed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m",type=int,default=128)
    parser.add_argument("--size",type=int,default=1024)
    parser.add_argument("--bits",type=int,default=64)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--large-tiles",action="store_true")
    args=parser.parse_args()
    m,k,bits=args.m,args.size,args.bits
    xs=[torch.randint(-2**63,2**63-1,(m,k),device="cuda",dtype=torch.int64) for _ in range(3)]
    ys=[torch.randint(-2**63,2**63-1,(k,k),device="cuda",dtype=torch.int64) for _ in range(3)]
    packed=[prepare_weight(y,bits) for y in ys]
    results=[]
    ref=local_products(*xs,*ys,bits)
    configs = [(tile,4,acc) for acc in (False,True)
               for tile in ((16,32,32),(16,32,64),(32,32,32),(32,64,32),(32,64,64),(64,64,32))]
    if args.large_tiles:
        configs = [(tile,warps,False) for tile in ((64,64,32),(64,128,32),(128,32,32),(128,64,32),(128,128,32))
                   for warps in (8,16)]
    for tile,warps,accumulate in configs:
            fn=lambda:local_products_packed(*xs,*packed,bits,tile=tile,warps=warps,accumulate=accumulate)
            try:
                out=fn()
                for a,b in zip(out,ref):
                    torch.testing.assert_close(a,b,rtol=0,atol=0)
                row=dict(m=m,k=k,n=k,bits=bits,tile=tile,warps=warps,accumulate=accumulate,**measure(fn,rep=30))
            except Exception as exc:
                row=dict(tile=tile,accumulate=accumulate,error=str(exc))
            print(json.dumps(row),flush=True)
            results.append(row)
    a,b=xs[0].float()/2**63,ys[0].float()/2**63
    for name,af,bf in (("fp32",a,b),("fp16",a.half(),b.half())):
        row=dict(backend=name,**measure(lambda:af@bf,rep=30))
        print(json.dumps(row),flush=True)
        results.append(row)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(results,indent=2)+"\n")


if __name__=="__main__":
    main()
