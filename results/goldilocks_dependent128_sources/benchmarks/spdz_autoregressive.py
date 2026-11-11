"""Dependent-step references and clear baselines for the trained micro stream."""
import copy
import statistics
import time

import numpy as np
import torch
from benchmarks.spdz_timing import wall_measurement
from scripts.run_spdz_prime256 import integer_reference


def integer_stream(parameters,seed,steps):
    """Compute each next input from the preceding integer logits' argmax."""
    if type(seed) is not int or seed not in range(8) or type(steps) is not int or steps<1:
        raise ValueError('seed must be a token in [0,8) and steps must be positive')
    token=seed;tokens=[token];logits=[]
    for _ in range(steps):
        x=[[int(i==token) for i in range(8)]]
        row=integer_reference([x,*parameters])[0]
        token=max(range(8),key=lambda i:row[i])
        logits.append(row);tokens.append(token)
    return dict(tokens=tokens,logits=logits)


def clear_baselines(model,seed,steps,*,repeats=10,gpu_device=None):
    """Time dependent streams; GPU resident steps never read tokens on the CPU.

    Host-delivery variants copy all eight logits after every step. Resident
    graph variants keep argmax and the next one-hot input on the GPU, with one
    synchronization at sequence end. Per-step traces are validated separately.
    """
    cpu=copy.deepcopy(model).cpu().float().eval()
    def cpu_stream():
        token=seed;tokens=[token];rows=[]
        with torch.inference_mode():
            for _ in range(steps):
                x=torch.nn.functional.one_hot(torch.tensor([token]),8).float()
                output=cpu(x);token=int(output.argmax(-1).item())
                rows.append(output[0].tolist());tokens.append(token)
        return dict(tokens=tokens,logits=rows)
    reference=cpu_stream()
    a=cpu[0].weight.detach().numpy().T;b=cpu[0].bias.detach().numpy()
    c=cpu[2].weight.detach().numpy().T;d=cpu[2].bias.detach().numpy()
    def numpy_stream():
        token=seed;tokens=[token];rows=[]
        for _ in range(steps):
            x=np.zeros((1,8),dtype=np.float32);x[0,token]=1
            row=(np.square(x@a+b)@c+d)[0]
            token=int(row.argmax());rows.append(row.tolist());tokens.append(token)
        return dict(tokens=tokens,logits=rows)
    timings={};measurements={};accuracy={}
    for name,compute in [('fp32_cpu',cpu_stream),('numpy_cpu',numpy_stream)]:
        actual=compute();measurement=wall_measurement(compute,'cpu',repeats,3)
        timings[name+'_sequence_ms']=measurement['median_ms'];measurements[name]=measurement
        accuracy[name]=dict(tokens=actual['tokens'],matches_fp32_tokens=actual['tokens']==reference['tokens'],
            max_logit_error=float(np.max(np.abs(np.array(actual['logits'])-np.array(reference['logits'])))))
    if gpu_device is not None:
        device=torch.device(gpu_device)
        for name,dtype in [('fp32',torch.float32),('fp16',torch.float16)]:
            local=copy.deepcopy(cpu).to(device=device,dtype=dtype)
            initial=torch.nn.functional.one_hot(torch.tensor([seed],device=device),8).to(dtype)
            state=initial.clone()
            def step():
                output=local(state)
                prediction=output.argmax(-1)
                state.zero_();state.scatter_(1,prediction[:,None],1)
                return output
            with torch.inference_mode():
                stream=torch.cuda.Stream(device=device);stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    for _ in range(20):step()
                torch.cuda.current_stream(device).wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):output=step()
                def execute(captured,deliver):
                    state.copy_(initial);rows=[];tokens=[seed]
                    for _ in range(steps):
                        if captured:graph.replay();current=output
                        else:current=step()
                        if deliver:
                            row=current.detach().float().cpu()[0]
                            rows.append(row.tolist());tokens.append(int(row.argmax().item()))
                    return dict(tokens=tokens,logits=rows) if deliver else None
                for captured in (False,True):
                    prefix=name+('_step_graph' if captured else '_eager')
                    actual=execute(captured,True)
                    accuracy[prefix]=dict(tokens=actual['tokens'],matches_fp32_tokens=actual['tokens']==reference['tokens'],
                        max_logit_error=float(np.max(np.abs(np.array(actual['logits'])-np.array(reference['logits'])))))
                    for deliver in (False,True):
                        key=prefix+('_host_logits' if deliver else '_resident')
                        measurement=wall_measurement(lambda:execute(captured,deliver),device,repeats,3)
                        timings[key+'_sequence_ms']=measurement['median_ms'];measurements[key]=measurement
                    if captured:
                        samples=[]
                        for _ in range(5):
                            state.copy_(initial);torch.cuda.synchronize(device)
                            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            begin.record()
                            for _ in range(steps):graph.replay()
                            end.record();end.synchronize();samples.append(begin.elapsed_time(end))
                        key=prefix+'_resident_events'
                        timings[key+'_sequence_ms']=statistics.median(samples)
                        measurements[key]=dict(group_samples_ms=samples,measurement='CUDA events around CPU-issued dependent step graph replays; includes submission starvation; no interstep device-to-host transfer')
    return dict(timings=timings,measurements=measurements,accuracy=accuracy,fp32_reference=reference,
                workload='One dependent fixed-length autoregressive stream; actual MLP, argmax and next one-hot input at every step',
                resident_gpu_contract='GPU argmax/state update; no interstep host delivery. Primary GPU comparator.',
                host_delivery_contract='All eight logits copied to CPU after every step; separate streaming comparator.')
