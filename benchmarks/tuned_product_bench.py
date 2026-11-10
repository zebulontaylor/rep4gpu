"""Compare exact ring32 local products on both FFN-width-2048 matrix shapes.

Run from the repository root: ``python -m benchmarks.tuned_product_bench``.
Integer inputs span the full ring; all candidate outputs must equal the baseline.
"""
import argparse
import json
from pathlib import Path

import torch

from benchmarks.timing import measure
from rep4gpu.five_product import local_products_five,prepare_five_weights
from rep4gpu.kernels import CompactWeight,local_products,prepare_weight
from rep4gpu.tuned_kernels import local_products_tuned,ring_matmul_tuned,local_products_five_tuned


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width",type=int,default=2048)
    parser.add_argument("--output",type=Path,default=Path("artifacts/tuned_product_ffn2048.json"))
    args=parser.parse_args()
    rows=[]
    for k,n in [(args.width,4*args.width),(4*args.width,args.width)]:
        torch.manual_seed(k+n+0x50000)
        xs=[torch.randint(-2**63,2**63-1,(128,k),device='cuda',dtype=torch.int64) for _ in range(3)]
        ys=[prepare_weight(torch.randint(-2**63,2**63-1,(k,n),device='cuda',dtype=torch.int64),32,storage='words32') for _ in range(3)]
        weights=prepare_five_weights(*ys)
        reference=local_products(*xs,*ys,32)
        for backend,fn in [('six_tuned',lambda:local_products_tuned(*xs,*ys,32)),
                           ('five_default',lambda:local_products_five(xs,weights)),
                           ('five_tuned',lambda:local_products_five_tuned(xs,weights))]:
            for actual,want in zip(fn(),reference):
                torch.testing.assert_close(actual,want,rtol=0,atol=0)
            row=dict(k=k,n=n,backend=backend,**measure(fn,rep=40))
            print(json.dumps(row),flush=True)
            rows.append(row)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':
    main()
