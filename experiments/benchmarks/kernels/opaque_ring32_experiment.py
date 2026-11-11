"""Compare opaque ring32 reconstruction with the maintained exact local kernel."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import triton
from rep4gpu.gpu.matmul.kernels import prepare_weight, _local_products
from rep4gpu.gpu.matmul.tuned_kernels import local_products_tuned
from rep4gpu.gpu.matmul.ring32_opaque import local_products_opaque


def legacy_local(xs,ys):
    """The previous SM89 uint64-reconstruction route and its fixed tile."""
    m,k=xs[0].shape
    n=ys[0].shape[1]
    out=torch.empty((3,m,n),device=xs[0].device,dtype=torch.int64)
    _local_products[(triton.cdiv(m,32),triton.cdiv(n,128),3)](
        *xs,*(y.words for y in ys),out,m,n,k,32,32,128,64,False,
        num_warps=4,num_stages=1)
    return out.unbind(0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    rng=np.random.default_rng(3280909)
    rows=[]
    torch.manual_seed(3280909)
    xc=[torch.randint(-2**63,2**63-1,(128,71),dtype=torch.int64) for _ in range(3)]
    yc=[torch.randint(-2**63,2**63-1,(71,39),dtype=torch.int64) for _ in range(3)]
    expected=[((xc[0]+xc[1])@yc[0]+xc[0]@yc[1])&0xffffffff,
              ((xc[1]+xc[2])@yc[1]+xc[1]@yc[2])&0xffffffff,
              (xc[0]@yc[2]+xc[2]@yc[0])&0xffffffff]
    xs=[x.cuda() for x in xc]
    ys=[prepare_weight(y.cuda(),32,storage='words32') for y in yc]
    assert all(torch.equal(a.cpu(),b) for a,b in zip(local_products_opaque(*xs,*ys),expected))
    del xc,yc,xs,ys,expected
    for m,k,n in [(128,71,39),(33,65,129),(128,2048,8192),(128,8192,2048)]:
        xs=[torch.from_numpy(rng.integers(0,2**32,size=(m,k),dtype=np.int64)).cuda() for _ in range(3)]
        raw=[torch.from_numpy(rng.integers(0,2**32,size=(k,n),dtype=np.int64)).cuda() for _ in range(3)]
        ys=[prepare_weight(y,32,storage='words32') for y in raw]
        expected=local_products_tuned(*xs,*ys,32)
        if k<100:
            a,b,c=[x.cpu() for x in xs];d,e,f=[y.cpu() for y in raw]
            cpu=((a+b)@d+a@e,(b+c)@e+b@f,a@f+c@d)
            assert all(torch.equal(x.cpu(),v&0xffffffff) for x,v in zip(expected,cpu))
        baseline=lambda:legacy_local(xs,ys)
        configs=([((32,128,32),4,False),((32,128,32),4,True),((32,128,64),4,True)] if args.paired else
                 [((32,128,64),4,False),((32,64,64),4,False),((64,64,64),4,False),((32,128,32),4,False),((32,128,64),8,False)])
        for tile,warps,separate in configs:
            fn=lambda:local_products_opaque(*xs,*ys,tile=tile,warps=warps,separate=separate)
            actual=fn()
            assert all(torch.equal(x,v) for x,v in zip(actual,expected)),(m,k,n,tile)
            if k<100: continue
            samples=[]
            for iteration in range(7 if args.paired else 3):
                functions=[('baseline',baseline),('candidate',fn)]
                if iteration%2:functions.reverse()
                samples.append({name:triton.testing.do_bench_cudagraph(function,rep=50) for name,function in functions})
            row=dict(shape=[m,k,n],tile=tile,warps=warps,separate=separate,all_outputs_exact=True,samples_ms=samples)
            rows.append(row);print(json.dumps(row),flush=True)
        del xs,raw,ys,expected,actual
    Path('experiments/results/kernels/opaque_ring32_paired.json' if args.paired else 'experiments/results/kernels/opaque_ring32_experiment.json').write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()
