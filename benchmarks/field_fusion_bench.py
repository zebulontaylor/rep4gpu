"""Benchmark prime31 compact storage, authenticated products and MAC reductions."""
import argparse
import json
from pathlib import Path

import torch

from benchmarks.timing import measure
from rep4gpu.field_fast import field_matmul as fast
from rep4gpu.field_fusion import authenticated_local,field_mac_check
from rep4gpu.field_kernels import PRIME,field_matmul as original


def stacked(data,multiply,party=0):
    d,e,av,am,bv,bm,cv,cm,alpha=data
    h=alpha.numel()
    m,n=cv.shape
    public=multiply(d,e)
    right=multiply(torch.cat((av,*am.unbind(0)),0),e).reshape(h+1,m,n)
    left=multiply(d,torch.cat((bv,*bm.unbind(0)),1))
    value=(cv.to(torch.int64)+left[:,:n].to(torch.int64)+right[0].to(torch.int64)+(public.to(torch.int64) if party==0 else 0))%PRIME
    macs=(cm.to(torch.int64)+left[:,n:].reshape(m,h,n).permute(1,0,2).to(torch.int64)+right[1:].to(torch.int64)+alpha.to(torch.int64)[:,None,None]*public.to(torch.int64))%PRIME
    return value,macs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("artifacts/field_fusion_bench.json"))
    args=parser.parse_args()
    rows=[]
    for m,k,n in [(128,256,1024),(128,1024,4096)]:
        a=torch.randint(0,PRIME,(m,k),device="cuda",dtype=torch.int32)
        b=torch.randint(0,PRIME,(k,n),device="cuda",dtype=torch.int32)
        a64,b64=a.to(torch.int64),b.to(torch.int64)
        reference=original(a64,b64)
        for backend,fn in [("original64",lambda:original(a64,b64)),
                           ("fast32",lambda:fast(a,b,packed=True))]:
            torch.testing.assert_close(fn().to(torch.int64),reference,rtol=0,atol=0)
            row=dict(kind="matrix",m=m,k=k,n=n,backend=backend,**measure(fn,rep=30))
            print(json.dumps(row),flush=True);rows.append(row)
    for m,k,n in [(1,256,256),(128,256,1024),(128,1024,4096)]:
        h=10
        shapes=[(m,k),(k,n),(m,k),(h,m,k),(k,n),(h,k,n),(m,n),(h,m,n),(h,)]
        data=[torch.randint(0,PRIME,s,device="cuda",dtype=torch.int32) for s in shapes]
        data64=[x.to(torch.int64) for x in data]
        reference=stacked(data64,original)
        for backend,fn in [("original_stacked64",lambda:stacked(data64,original)),
                           ("fast_stacked32",lambda:stacked(data,fast)),
                           ("fused32",lambda:authenticated_local(*data,0))]:
            for x,y in zip(fn(),reference):torch.testing.assert_close(x.to(torch.int64),y,rtol=0,atol=0)
            row=dict(kind="authenticated",h=h,m=m,k=k,n=n,backend=backend,**measure(fn,rep=30))
            print(json.dumps(row),flush=True);rows.append(row)
        del data,data64,reference
    for n in [262144,1048576]:
        h=10
        macs=torch.randint(0,PRIME,(h,n),device="cuda",dtype=torch.int32)
        opened=torch.randint(0,PRIME,(n,),device="cuda",dtype=torch.int32)
        alpha=torch.randint(0,PRIME,(h,),device="cuda",dtype=torch.int32)
        chi=torch.randint(0,PRIME,(h,n),device="cuda",dtype=torch.int32)
        row=dict(kind="mac_check",h=h,n=n,**measure(lambda:field_mac_check(macs,opened,alpha,chi),rep=30))
        print(json.dumps(row),flush=True);rows.append(row)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(rows,indent=2)+"\n")


if __name__=="__main__":
    main()
