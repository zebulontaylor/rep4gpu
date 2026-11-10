"""Compare validated five-GEMM and six-GEMM ring32 local products."""
import argparse
import json
from pathlib import Path

import torch

from benchmarks.timing import measure
from rep4gpu.five_product import local_products_five,prepare_five_weights
from rep4gpu.kernels import local_products,prepare_weight


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width",type=int,default=2048)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    rows=[]
    for k,n in ((args.width,4*args.width),(4*args.width,args.width)):
        torch.manual_seed(k+n+0x50000)
        xs=[torch.randint(-2**63,2**63-1,(128,k),device="cuda",dtype=torch.int64) for _ in range(3)]
        ys=[prepare_weight(torch.randint(-2**63,2**63-1,(k,n),device="cuda",dtype=torch.int64),32,storage="words32") for _ in range(3)]
        torch.cuda.synchronize()
        allocated=torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        five=prepare_five_weights(*ys)
        torch.cuda.synchronize()
        peak_extra=torch.cuda.max_memory_allocated()-allocated
        expected=local_products(*xs,*ys,32)
        for a,b in zip(local_products_five(xs,five),expected):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        memory=dict(three_weight_bytes=12*k*n,five_weight_bytes=20*k*n,
                    additional_weight_bytes_when_both_retained=12*k*n,peak_prepare_extra_bytes=peak_extra)
        for backend,fn in (("six_gemm",lambda:local_products(*xs,*ys,32)),
                           ("five_gemm",lambda:local_products_five(xs,five))):
            row=dict(m=128,k=k,n=n,backend=backend,**memory,**measure(fn,rep=40))
            print(json.dumps(row),flush=True)
            rows.append(row)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(rows,indent=2)+"\n")


if __name__=="__main__":
    main()
