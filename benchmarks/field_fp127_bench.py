"""Resident F127 products with setup separated from prepared multiplication."""
import argparse
import json
import statistics
import time

import numpy as np
import torch
from rep4gpu.field_fp127 import PRIME,field_matmul_fp127,prepare_fp127_left,prepare_fp127_weight


def random_field(shape):
    words=torch.randint(65536,(*shape,8),device='cuda',dtype=torch.int32)
    words[...,-1].bitwise_and_(32767)
    invalid=(words[...,:7]==65535).all(dim=-1)&(words[...,-1]==32767)
    while bool(invalid.any()):
        replacement=torch.randint(65536,(int(invalid.sum()),8),device='cuda',dtype=torch.int32)
        replacement[:,-1].bitwise_and_(32767)
        words[invalid]=replacement
        invalid=(words[...,:7]==65535).all(dim=-1)&(words[...,-1]==32767)
    return words.to(torch.uint16)


def integers(words):
    values=words.astype(object)
    return sum(values[...,i]<<(16*i) for i in range(8))


def sample_oracle(a,b,output):
    rows=sorted(set((0,a.shape[0]//2,a.shape[0]-1)))
    cols=sorted(set((0,b.shape[1]//2,b.shape[1]-1)))
    aa=integers(a.cpu().numpy()[rows])
    bb=integers(b.cpu().numpy()[:,cols])
    want=(aa@bb)%PRIME
    got=integers(output.cpu().numpy()[rows][:,cols])
    np.testing.assert_array_equal(got,want)


def measure(fn,samples,graph=True):
    for _ in range(2):fn()
    torch.cuda.synchronize()
    replay=None
    if graph:
        replay=torch.cuda.CUDAGraph()
        with torch.cuda.graph(replay):fn()
    events=[];walls=[]
    for _ in range(samples):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        if replay is not None:
            for _ in range(5):replay.replay()
        else:fn()
        end.record();end.synchronize()
        events.append(start.elapsed_time(end)/(5 if replay is not None else 1))
        begin=time.perf_counter();fn();torch.cuda.synchronize()
        walls.append((time.perf_counter()-begin)*1000)
    return dict(median_event_ms=statistics.median(events),median_eager_wall_ms=statistics.median(walls),
                event_kind='resident CUDA graph replay' if graph else 'eager setup',samples_event_ms=events,samples_wall_ms=walls)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='artifacts/field_fp127_bench.json')
    parser.add_argument('--samples',type=int,default=11)
    args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(127512)
    results=[]
    for m,k,n in ((128,512,128),(256,512,256),(128,1024,256),(256,1024,128),(1,1024,1024)):
        a,b=random_field((m,k)),random_field((k,n))
        left,right=prepare_fp127_left(a),prepare_fp127_weight(b)
        functions={'raw':lambda:field_matmul_fp127(a,b),
                   'prepared_left':lambda:field_matmul_fp127(left,b),
                   'prepared_right':lambda:field_matmul_fp127(a,right),
                   'prepared_both':lambda:field_matmul_fp127(left,right)}
        expected=functions['raw']()
        sample_oracle(a,b,expected)
        for fn in functions.values():torch.testing.assert_close(fn(),expected,rtol=0,atol=0)
        setup={'left':measure(lambda:prepare_fp127_left(a),args.samples,False),
               'right':measure(lambda:prepare_fp127_weight(b),args.samples,False)}
        timings={name:measure(fn,args.samples) for name,fn in functions.items()}
        row=dict(m=m,k=k,n=n,field='2**127-1',input_distribution='uniform over the complete canonical field by127-bit rejection',
                 input_storage='uint16 radix65536 limbs',timings=timings,setup=setup,
                 prepared_left_bytes=left.storage_bytes,prepared_right_bytes=right.storage_bytes,
                 cross_temporary_bytes=256*m*n*4,exact_sample_bigint_match=True,exact_all_prepared_modes_match=True,
                 scope='one exact field product; no preprocessing, authentication, transport, or native conversion included',
                 device=torch.cuda.get_device_name())
        results.append(row);print(json.dumps(row),flush=True)
        with open(args.output,'w') as output:json.dump(results,output,indent=2)


if __name__=='__main__':main()
