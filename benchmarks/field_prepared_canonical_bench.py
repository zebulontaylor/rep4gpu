"""Compare resident prepared decode kernels with paired CUDA-event timings."""
import argparse
import json
import random

import torch
from rep4gpu.field_kernels import PRIME
from rep4gpu.field_prepared import prepared_right_online
from rep4gpu.field_prepared_canonical import prepared_right_online_canonical


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output')
    parser.add_argument('--repetitions',type=int,default=20)
    args=parser.parse_args()
    torch.manual_seed(31107)
    results=[]
    for k,n in ((256,1024),(1024,4096),(4096,1024)):
        shapes=((1,k),(k,n),(10,k,n),(1,n),(10,1,n))
        inputs=[torch.randint(PRIME,s,device='cuda',dtype=torch.int32) for s in shapes]
        functions={'general':prepared_right_online,'canonical':prepared_right_online_canonical}
        graphs={}
        outputs={}
        for name,fn in functions.items():
            for _ in range(3):
                outputs[name]=fn(*inputs)
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs[name]=fn(*inputs)
            graphs[name]=graph
        for a,b in zip(outputs['general'],outputs['canonical']):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        measurements={name:[] for name in functions}
        order=list(functions)
        rng=random.Random(31107)
        for _ in range(args.repetitions):
            rng.shuffle(order)
            for name in order:
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(10):
                    graphs[name].replay()
                end.record()
                end.synchronize()
                measurements[name].append(begin.elapsed_time(end)/10)
        medians={name:float(torch.tensor(values).median()) for name,values in measurements.items()}
        result=dict(m=1,h=10,k=k,n=n,device=torch.cuda.get_device_name(),
                    timing='paired resident CUDA graph replay; CUDA events',
                    median_ms=medians,speedup=medians['general']/medians['canonical'],samples_ms=measurements)
        results.append(result)
        print(json.dumps(result),flush=True)
    if args.output:
        with open(args.output,'w') as output:
            json.dump(results,output,indent=2)


if __name__=='__main__':
    main()
