"""Exact field primitives on resident uniformly random full-width inputs."""
import argparse
import json
import statistics
import time

import torch
from rep4gpu.field_q256 import field_matmul_q256
from rep4gpu.field_q256_tensor import field_matmul_q256_tensor,prepare_q256_weight
from rep4gpu.field_fast import field_matmul
from rep4gpu.field_q256_bytes import field_matmul_q256_bytes,prepare_q256_bytes


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output')
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(256189)
    results=[]
    for m,k,n in ((1,64,64),(1,128,128),(1,1024,1024),(1,4096,1024)):
        a=torch.randint(65536,(m,k,16),device='cuda',dtype=torch.int32)
        b=torch.randint(65536,(k,n,16),device='cuda',dtype=torch.int32)
        for source in (a,b):
            invalid=(source[...,1:]==65535).all(dim=-1)&(source[...,0]>=65347)
            while bool(invalid.any()):
                source[invalid]=torch.randint(65536,(int(invalid.sum()),16),device=source.device,dtype=torch.int32)
                invalid=(source[...,1:]==65535).all(dim=-1)&(source[...,0]>=65347)
        a16,b16=a.to(torch.uint16),b.to(torch.uint16)
        p=torch.randint(2147483647,(m,k),device='cuda',dtype=torch.int32)
        q=torch.randint(2147483647,(k,n),device='cuda',dtype=torch.int32)
        prepared=prepare_q256_weight(b16)
        byte_prepared=prepare_q256_bytes(b16)
        functions={'q256_bytes_prepared':lambda:field_matmul_q256_bytes(a16,byte_prepared),'q256_bytes_raw':lambda:field_matmul_q256_bytes(a16,b16),'q256_prepared':lambda:field_matmul_q256_tensor(a16,prepared),'q256_simt':lambda:field_matmul_q256(a,b),
                   'q256_tensor':lambda:field_matmul_q256_tensor(a,b),
                   'q256_tensor_uint16':lambda:field_matmul_q256_tensor(a16,b16),
                   'p31_one_field':lambda:field_matmul(p,q,packed=True)}
        medians={};outputs={}
        for name,fn in functions.items():
            for _ in range(2):outputs[name]=fn()
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):fn()
            events=[];walls=[]
            for _ in range(11):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record();graph.replay();end.record();end.synchronize()
                events.append(start.elapsed_time(end))
                begin=time.perf_counter();fn();torch.cuda.synchronize()
                walls.append((time.perf_counter()-begin)*1000)
            medians[name]=dict(graph_event_ms=statistics.median(events),eager_wall_ms=statistics.median(walls))
        for name in ('q256_tensor','q256_tensor_uint16','q256_prepared','q256_bytes_prepared','q256_bytes_raw'):
            torch.testing.assert_close(outputs[name],outputs['q256_simt'],rtol=0,atol=0)
        row=dict(m=m,k=k,n=n,median_ms=medians,exact_simt_tensor_parity=True,
                 cross_product_temporary_bytes=dict(radix128=m*n*1369*4,centered_bytes=m*n*1024*4),
                 prepared_weight_bytes=dict(radix128=37*k*n,centered_bytes=32*k*n+32*n*4),device=torch.cuda.get_device_name(),
                 scope='one field product only; excludes preprocessing, authentication, checks and communication')
        print(json.dumps(row),flush=True)
        results.append(row)
    if args.output:
        with open(args.output,'w') as output:json.dump(results,output,indent=2)


if __name__=='__main__':main()
