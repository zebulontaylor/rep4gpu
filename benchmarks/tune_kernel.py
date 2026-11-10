"""Sweep exact tensor kernel tiles; timings exclude weight packing."""
import argparse
import json

import torch
import triton

from rep4gpu.kernels import _tensor_mm, prepare_weight


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m",type=int,default=128)
    parser.add_argument("--size",type=int,default=2048)
    parser.add_argument("--bits",type=int,default=64)
    args=parser.parse_args()
    m,k,bits=args.m,args.size,args.bits
    a=torch.randint(-2**63,2**63-1,(m,k),device="cuda",dtype=torch.int64)
    b=torch.randint(-2**63,2**63-1,(k,k),device="cuda",dtype=torch.int64)
    c=torch.empty((m,k),device="cuda",dtype=torch.int64)
    p=prepare_weight(b,bits)
    results=[]
    for prepared in (False,True):
        for bm in (16,32,64):
            if m<bm and bm!=16:
                continue
            for bn in (32,64,128):
                for bk in (32,64,128):
                    for warps in (4,8):
                        fn=lambda:_tensor_mm[(triton.cdiv(m,bm),triton.cdiv(k,bn))](
                            a,p.limbs if prepared else b,c,m,k,k,bits,bm,bn,bk,prepared,num_warps=warps)
                        try:
                            fn()
                            ms=triton.testing.do_bench(fn,warmup=10,rep=30)
                        except Exception as exc:
                            print(json.dumps(dict(prepared=prepared,bm=bm,bn=bn,bk=bk,warps=warps,error=str(exc)[:100])),flush=True)
                            continue
                        row=dict(prepared=prepared,bm=bm,bn=bn,bk=bk,warps=warps,ms=ms)
                        results.append(row)
                        print(json.dumps(row),flush=True)
    print("BEST",json.dumps(sorted(results,key=lambda x:x['ms'])[:10]),flush=True)


if __name__=="__main__":
    main()
