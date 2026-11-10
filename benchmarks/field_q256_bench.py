"""Primitive-only q256 SIMT timings beside the existing prime31 field kernel."""
import argparse
import json
import statistics
import time

import torch
from rep4gpu.field_q256 import field_matmul_q256
from rep4gpu.field_fast import field_matmul


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output')
    args=parser.parse_args()
    torch.set_num_threads(1)
    results=[]
    for m,k,n in ((1,16,16),(1,64,64),(16,64,64),(1,128,128)):
        a=torch.randint(65536,(m,k,16),device='cuda',dtype=torch.int32)
        b=torch.randint(65536,(k,n,16),device='cuda',dtype=torch.int32)
        # Rejection keeps the benchmark inputs uniform over the complete field.
        for source in (a,b):
            invalid=(source[...,1:]==65535).all(dim=-1)&(source[...,0]>=65347)
            while bool(invalid.any()):
                source[invalid]=torch.randint(65536,(int(invalid.sum()),16),device=source.device,dtype=torch.int32)
                invalid=(source[...,1:]==65535).all(dim=-1)&(source[...,0]>=65347)
        p=torch.randint(2147483647,(m,k),device='cuda',dtype=torch.int32)
        q=torch.randint(2147483647,(k,n),device='cuda',dtype=torch.int32)
        functions={'q256_one_field':lambda:field_matmul_q256(a,b),'p31_one_field':lambda:field_matmul(p,q,packed=True)}
        medians={}
        for name,fn in functions.items():
            for _ in range(3):fn()
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):fn()
            events=[];walls=[]
            for _ in range(15):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record();end.synchronize()
                events.append(start.elapsed_time(end))
                begin=time.perf_counter();fn();torch.cuda.synchronize()
                walls.append((time.perf_counter()-begin)*1000)
            medians[name]=dict(graph_event_ms=statistics.median(events),eager_wall_ms=statistics.median(walls))
        row=dict(m=m,k=k,n=n,median_ms=medians,device=torch.cuda.get_device_name(),scope='one unauthenticated field matrix product; excludes native preprocessing and all online protocol/check work')
        print(json.dumps(row),flush=True)
        results.append(row)
    if args.output:
        with open(args.output,'w') as output:json.dump(results,output,indent=2)


if __name__=='__main__':main()
