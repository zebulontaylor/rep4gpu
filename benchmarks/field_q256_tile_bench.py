"""Bounded exact cross-limb tensor-core tile comparison."""
import itertools,json,statistics
import torch,triton
from rep4gpu.field_q256_tensor import _cross_packed,_pack_matrix,prepare_q256_weight


def main():
    torch.manual_seed(1256189)
    m,k,n=1,1024,1024
    a=torch.randint(65536,(m,k,16),device='cuda',dtype=torch.int32)
    b=torch.randint(65536,(k,n,16),device='cuda',dtype=torch.int32)
    a=_pack_matrix(a).reshape(m*37,k)
    b=prepare_q256_weight(b).digits
    expected=torch.empty((m*37,n*37),device='cuda',dtype=torch.int32)
    _cross_packed[(triton.cdiv(m*37,32),triton.cdiv(n*37,64))](a,b,expected,m*37,n*37,k,32,64,64,num_warps=4,num_stages=1)
    out=torch.empty_like(expected)
    results=[]
    for tm,tn,tk in itertools.product((32,64),(64,128),(64,128)):
        def fn():
            _cross_packed[(triton.cdiv(m*37,tm),triton.cdiv(n*37,tn))](a,b,out,m*37,n*37,k,tm,tn,tk,num_warps=4,num_stages=1)
        fn()
        torch.testing.assert_close(out,expected,rtol=0,atol=0)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):fn()
        samples=[]
        for _ in range(11):
            x,y=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            x.record()
            for _ in range(10):graph.replay()
            y.record();y.synchronize()
            samples.append(x.elapsed_time(y)/10)
        result=dict(tile=[tm,tn,tk],median_ms=statistics.median(samples),exact_cross_parity=True)
        results.append(result);print(json.dumps(result),flush=True)
    json.dump(results,open('artifacts/field_q256_tile_bench.json','w'),indent=2)


if __name__=='__main__':main()
