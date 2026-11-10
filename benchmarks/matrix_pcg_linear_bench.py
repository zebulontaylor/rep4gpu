"""Local linear-expansion component timing with synthetic resident field shares.

This script generates no cryptographic preprocessing. It times all6K signed
raw-term additions and the local product/final expansion on each path. Public
H setup, sparse seed validation, and the A=H@E materialization needed by both
protocol schedules are common setup outside these component measurements.
"""
import argparse
import json
import statistics
import time

import torch
import triton
from rep4gpu.field_fast import field_matmul
from rep4gpu.field_kernels import PRIME
from rep4gpu.matrix_pcg_linear import PublicLinearMap,RegularSparseColumns,LinearExpansionAccumulator,_accumulate


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--sizes',type=int,nargs='+',default=[128,256,512])
    parser.add_argument('--samples',type=int,default=3)
    parser.add_argument('--output',default='artifacts/matrix_pcg_linear_bench.json')
    args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(319412)
    results=[]
    for m in args.sizes:
        a,k,n,l,t=2*m,m,m,4*m,8
        h=torch.randint(PRIME,(a,l),device='cuda',dtype=torch.int32)
        plan=PublicLinearMap(h);identity=plan.identity
        positions=torch.stack([torch.randint(j*(l//t),(j+1)*(l//t),(k,),device='cuda',dtype=torch.int64) for j in range(t)],dim=1)
        values=torch.randint(1,PRIME,(k,t),device='cuda',dtype=torch.int32)
        sparse=RegularSparseColumns(plan,positions,values,map_identity=identity)
        b=torch.randint(PRIME,(k,n),device='cuda',dtype=torch.int32)
        local_a=plan.expand(sparse.materialize(),map_identity=identity)
        term_batch=torch.randint(PRIME,(8,l,n),device='cuda',dtype=torch.int32)
        terms=list(term_batch.unbind(0))
        count=6*k
        def fused():
            state=LinearExpansionAccumulator(plan,n,map_identity=identity)
            state.add_local(sparse,b,map_identity=identity)
            for index in range(count):
                state.add(terms[index%len(terms)],sign=1 if index%2==0 else -1,map_identity=identity)
            return state.finish(map_identity=identity)
        def fused_batch8():
            state=LinearExpansionAccumulator(plan,n,map_identity=identity)
            state.add_local(sparse,b,map_identity=identity)
            for _ in range(count//8):
                state.add_batch(term_batch,signs=(1,-1)*4,map_identity=identity)
            return state.finish(map_identity=identity)
        def unfused():
            result=field_matmul(local_a,b,packed=True)
            for index in range(count):
                expanded=plan.expand(terms[index%len(terms)],map_identity=identity)
                _accumulate[(triton.cdiv(result.numel(),256),)](expanded,result,result.numel(),1 if index%2==0 else -1,256,num_warps=4)
            return result
        expected=unfused();actual=fused();torch.cuda.synchronize()
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        torch.testing.assert_close(fused_batch8(),expected,rtol=0,atol=0)
        samples={name:dict(event_ms=[],wall_ms=[]) for name in ('unfused','fused','fused_batch8')}
        functions=dict(unfused=unfused,fused=fused,fused_batch8=fused_batch8)
        for repetition in range(args.samples):
            order=('unfused','fused','fused_batch8') if repetition%2==0 else ('fused_batch8','fused','unfused')
            for name in order:
                torch.cuda.synchronize()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin=time.perf_counter();start.record();result=functions[name]();end.record();end.synchronize()
                samples[name]['wall_ms'].append((time.perf_counter()-begin)*1000)
                samples[name]['event_ms'].append(start.elapsed_time(end))
                torch.testing.assert_close(result,expected,rtol=0,atol=0)
        medians={name:{kind:statistics.median(times) for kind,times in row.items()} for name,row in samples.items()}
        row=dict(m=m,a=a,l=l,k=k,n=n,t=t,raw_terms=count,parties=4,
                 fixture='8 preallocated full-field random buffers reused as synthetic streamed terms; not a cryptographic producer',
                 scope='one party local algebra, including sparse E@B or local A@B, all6K signed accumulations, and every required H expansion',
                 common_setup_excluded='public H binding, sparse seed validation, and A=H@E required for authentication on both paths',
                 median_ms=medians,samples_ms=samples,exact_fused_unfused_parity=True,
                 wall_speedup=medians['unfused']['wall_ms']/medians['fused']['wall_ms'],
                 batched_wall_speedup=medians['unfused']['wall_ms']/medians['fused_batch8']['wall_ms'],
                 arithmetic='p=2^31-1; exact full-field int32 shares and int64 reduced products/sums',device=torch.cuda.get_device_name())
        results.append(row);print(json.dumps(row),flush=True)
        with open(args.output,'w') as output:json.dump(results,output,indent=2)


if __name__=='__main__':main()
