"""Warmed wall-clock and grouped CUDA-event measurements for clear baselines."""
import statistics
import time

import torch

from benchmarks.inference import synchronize


def wall_measurement(compute,device,repeats=20,warmups=20):
    minimum_seconds=.1 if torch.device(device).type=='cuda' else 0.
    started=time.perf_counter();count=0
    while count<warmups or time.perf_counter()-started<minimum_seconds:
        compute();count+=1
    synchronize(device)
    warmup_ms=(time.perf_counter()-started)*1000
    samples=[]
    for _ in range(repeats):
        start=time.perf_counter()
        compute();synchronize(device)
        samples.append((time.perf_counter()-start)*1000)
    return dict(median_ms=statistics.median(samples),minimum_ms=min(samples),maximum_ms=max(samples),
                samples_ms=samples,warmups=count,minimum_requested_warmups=warmups,
                minimum_gpu_warmup_ms=minimum_seconds*1000,actual_warmup_wall_ms=warmup_ms)


def graph_measurement(graph,device,groups=5,replays=200,warmups=20):
    started=time.perf_counter();count=0
    while count<warmups or time.perf_counter()-started<.1:
        graph.replay();count+=1
    synchronize(device)
    warmup_ms=(time.perf_counter()-started)*1000
    samples=[]
    for _ in range(groups):
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(replays):graph.replay()
        end.record();end.synchronize()
        samples.append(begin.elapsed_time(end)/replays)
    median=statistics.median(samples)
    return dict(median_ms=median,minimum_ms=min(samples),maximum_ms=max(samples),
                group_samples_ms=samples,replays_per_group=replays,groups=groups,warmups=count,minimum_requested_warmups=warmups,
                minimum_gpu_warmup_ms=100,actual_warmup_wall_ms=warmup_ms,
                relative_range=(max(samples)-min(samples))/median,
                groups_stable=(max(samples)-min(samples))/median<=.10,
                measurement='CUDA events around CPU-issued graph replays; includes any GPU starvation between submissions')


def device_batch_graph_measurement(compute,device,groups=5,replays=200,expected_output=None):
    """Time one captured batch of clear computations without host submission gaps."""
    stream=torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(20):compute()
    torch.cuda.current_stream(device).wait_stream(stream)
    outer=torch.cuda.CUDAGraph()
    with torch.cuda.graph(outer,stream=stream):
        for _ in range(replays):output=compute()
    for _ in range(20):outer.replay()
    synchronize(device)
    samples=[]
    for _ in range(groups):
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        begin.record();outer.replay();end.record();end.synchronize()
        samples.append(begin.elapsed_time(end)/replays)
    median=statistics.median(samples)
    return dict(median_ms=median,minimum_ms=min(samples),maximum_ms=max(samples),
                group_samples_ms=samples,replays_per_group=replays,groups=groups,
                relative_range=(max(samples)-min(samples))/median,
                groups_stable=(max(samples)-min(samples))/median<=.10,
                output_matches_single_graph=None if expected_output is None else torch.equal(output,expected_output),
                measurement='CUDA events around one graph containing200 clear computations; divided by200; excludes host gaps between computations')
