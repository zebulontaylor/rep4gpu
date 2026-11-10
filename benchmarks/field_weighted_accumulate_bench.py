"""Measure the streamed MAC projection epilogue on fixed resident buffers."""
import argparse
import json
import random
import statistics
import time

import torch
from rep4gpu.field_kernels import PRIME
from rep4gpu.field_weighted import field_weighted_check
from rep4gpu.field_weighted_accumulate import field_weighted_accumulate


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output')
    parser.add_argument('--repetitions',type=int,default=30)
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(3771)
    results=[]
    for h,lengths in ((7,(512,2048,512)),(7,(1024,4096,1024)),(10,(1024,4096,1024))):
        pairs=[(torch.randint(PRIME,(h,n),device='cuda',dtype=torch.int32),torch.randint(PRIME,(h,n),device='cuda',dtype=torch.int32)) for n in lengths]
        rejected=[torch.zeros(1,device='cuda',dtype=torch.int32) for _ in pairs]
        total=torch.zeros(h,device='cuda',dtype=torch.int32)
        failure=torch.zeros(1,device='cuda',dtype=torch.int32)
        def original():
            result=total
            for (residual,challenge),status in zip(pairs,rejected):
                part=field_weighted_check(residual,challenge,packed=True)
                result=(result.long()+part).remainder(PRIME).int()
                failure.bitwise_or_(status)
            return result
        def fused():
            for (residual,challenge),status in zip(pairs,rejected):
                field_weighted_accumulate(residual,challenge,total,failure,status)
            return total
        want=original()
        actual=fused()
        torch.testing.assert_close(actual,want,rtol=0,atol=0)
        functions={'original':original,'fused':fused}
        graphs={}
        for name,fn in functions.items():
            for _ in range(3):fn()
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):fn()
            graphs[name]=graph
        timings={name:{'event_ms':[],'host_ms':[]} for name in functions}
        rng=random.Random(3771)
        order=list(functions)
        for _ in range(args.repetitions):
            rng.shuffle(order)
            for name in order:
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(10):graphs[name].replay()
                end.record();end.synchronize()
                timings[name]['event_ms'].append(start.elapsed_time(end)/10)
                torch.cuda.synchronize()
                begin=time.perf_counter()
                functions[name]()
                torch.cuda.synchronize()
                timings[name]['host_ms'].append((time.perf_counter()-begin)*1000)
        medians={name:{kind:statistics.median(values) for kind,values in row.items()} for name,row in timings.items()}
        result=dict(h=h,lengths=lengths,device=torch.cuda.get_device_name(),median_ms=medians,samples=timings,
                    contract='fixed independent coefficient rows for kernel timing only; production uses fresh protocol-derived rows')
        results.append(result)
        print(json.dumps(result),flush=True)
    if args.output:
        with open(args.output,'w') as output:json.dump(results,output,indent=2)


if __name__=='__main__':main()
