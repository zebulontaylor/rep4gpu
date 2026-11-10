"""Compare exact prime31 CUDA products with the previous three-ring-product path."""
import argparse
import json
from pathlib import Path

import torch

from benchmarks.timing import measure
from rep4gpu.field_kernels import PRIME,field_matmul
from rep4gpu.kernels import ring_matmul


def legacy_gpu(a,b):
    a,b=a.remainder(PRIME),b.remainder(PRIME)
    al,ah=a&65535,a>>16
    bl,bh=b&65535,b>>16
    low=ring_matmul(al,bl)
    high=ring_matmul(ah,bh)
    cross=ring_matmul(al+ah,bl+bh)-low-high
    result=(low.remainder(PRIME)+(cross.remainder(PRIME)<<16)).remainder(PRIME)
    return (result+(high.remainder(PRIME)<<32)).remainder(PRIME)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick",action="store_true")
    parser.add_argument("--output",type=Path,default=Path("artifacts/field_kernel_bench.json"))
    args=parser.parse_args()
    shapes=[(1,512,512),(128,256,256),(128,1024,1024)]
    if not args.quick:
        shapes.append((128,2048,8192))
    rows=[]
    for m,k,n in shapes:
        torch.manual_seed(m+k+n)
        a=torch.randint(0,PRIME,(m,k),device="cuda",dtype=torch.int64)
        b=torch.randint(0,PRIME,(k,n),device="cuda",dtype=torch.int64)
        torch.testing.assert_close(field_matmul(a,b),legacy_gpu(a,b),rtol=0,atol=0)
        for backend,fn in (("existing",lambda:legacy_gpu(a,b)),("prime31_tensor" if m>4 else "prime31_decode",lambda:field_matmul(a,b))):
            row=dict(m=m,k=k,n=n,backend=backend,**measure(fn,rep=20))
            print(json.dumps(row),flush=True)
            rows.append(row)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(rows,indent=2)+"\n")


if __name__=="__main__":
    main()
