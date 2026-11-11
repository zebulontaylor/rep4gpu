"""Consistent cold-cache, resident CUDA-graph, and Python wall-clock timings."""
import time

import torch
import triton


def measure(fn, rep=50):
    fn()
    torch.cuda.synchronize()
    cold_ms = triton.testing.do_bench(fn, warmup=20, rep=rep)
    graph_ms = triton.testing.do_bench_cudagraph(fn, rep=rep)
    iterations = max(1, min(1000, int(rep / max(graph_ms, 0.001))))
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - start) * 1000 / iterations
    return dict(cold_ms=cold_ms, graph_ms=graph_ms, wall_ms=wall_ms)
