"""Measure narrow ring16 pointer storage without changing runtime contracts.

Run from the repository root::

    python -m experiments.benchmarks.kernels.packed_ring16_bench \
        --output experiments/results/kernels/packed_ring16.json

The direct kernel calls intentionally bypass int64-only runtime wrappers.
Repeated PRG nonces are deterministic benchmark fixtures, never live masks.
"""
import argparse
from itertools import chain
import json
from pathlib import Path
import statistics

import torch
import triton

from rep4gpu.gpu.crypto.gpu_prg_batch import generate_rows
from rep4gpu.gpu.crypto.gpu_prg_graph import _rows
from rep4gpu.gpu.matmul.kernels import _local_products, prepare_weight


def measure(fn, calls, repeats):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(calls):
                fn()
    stream.synchronize()
    samples = []
    for _ in range(repeats):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / calls)
    return dict(ms=statistics.median(samples), samples_ms=samples)


def matmul_cases(args):
    m, k = args.batch, args.input_width
    for n in args.hidden_widths:
        xs = tuple(torch.randint(0, 65536, (m, k), device='cuda', dtype=torch.int64)
                   for _ in range(3))
        ys = tuple(torch.randint(0, 65536, (k, n), device='cuda', dtype=torch.int64)
                   for _ in range(3))
        for t in (*xs, *ys):
            t.flatten()[:min(4, t.numel())] = torch.tensor(
                [0, 32767, 32768, 65535][:min(4, t.numel())], device='cuda')
        weights = tuple(prepare_weight(y, 16).limbs for y in ys)
        # CPU int64 sums cannot overflow at these checked matrix dimensions.
        a, b, c = (x[:min(m, 8)].cpu() for x in xs)
        d, e, f = (y.cpu() for y in ys)
        reference = torch.stack(((a+b)@d+a@e, (b+c)@e+b@f, a@f+c@d)) & 65535
        wide = None
        for dtype in (torch.int64, torch.uint16):
            inputs = tuple(x.to(dtype) for x in xs)
            output = torch.empty((3, m, n), dtype=dtype, device='cuda')

            def launch():
                _local_products[(triton.cdiv(m, 16), triton.cdiv(n, 32), 3)](
                    *inputs, *weights, output, m, n, k, 16, 16, 32, 64, True,
                    num_warps=4)

            timing = measure(launch, args.calls, args.repeats)
            actual = output.to(torch.int64)
            torch.testing.assert_close(actual[:, :reference.shape[1]].cpu(), reference,
                                       rtol=0, atol=0)
            if wide is None:
                wide = actual.clone()
            torch.testing.assert_close(actual, wide, rtol=0, atol=0)
            yield dict(kind='local_matmul', bits=16, batch=m, input_width=k,
                       hidden_width=n, storage=str(dtype), prepared_weights='int8_limbs',
                       input_bytes=3*m*k*output.element_size(),
                       output_bytes=output.numel()*output.element_size(),
                       prepared_weight_bytes=3*k*n*2,
                       exact_full_output_equality=True, exact_cpu_reference_rows=reference.shape[1],
                       **timing)


def prg_cases(args):
    count = args.batch*args.input_width
    key = torch.arange(8, device='cuda', dtype=torch.int64).to(torch.uint32)
    nonces = torch.tensor([17, 18], device='cuda', dtype=torch.uint64)
    for bits in (16, 32):
        for rows in (1, 2):
            reference = generate_rows((key,)*rows, (17, 18)[:rows], count, bits)
            for packed in (False, True):
                dtype = {16: torch.uint16, 32: torch.uint32}[bits] if packed else torch.int64
                output = torch.empty((rows, count), dtype=dtype, device='cuda')
                block = 256 if packed and bits == 32 and count >= 1024*1024 else 128

                def launch():
                    _rows[(triton.cdiv(triton.cdiv(count, 512//bits), block), rows)](
                        *((key,)*6), nonces, output, count, bits,
                        (True,)*rows+(False,)*(6-rows), block, num_warps=4)

                timing = measure(launch, args.calls, args.repeats)
                torch.testing.assert_close(output.to(torch.int64), reference, rtol=0, atol=0)
                stream_bytes = rows*count*(bits//8)
                output_bytes = output.numel()*output.element_size()
                yield dict(kind='prg', bits=bits, rows=rows, count_per_row=count,
                           storage=str(dtype), packed=packed, block=block,
                           stream_bytes=stream_bytes, output_bytes=output_bytes,
                           stream_GB_per_second=stream_bytes/timing['ms']/1e6,
                           store_GB_per_second=output_bytes/timing['ms']/1e6,
                           exact_stream_equality=True, **timing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, default=8192)
    parser.add_argument('--input-width', type=int, default=512)
    parser.add_argument('--hidden-widths', type=int, nargs='+', default=[32, 328])
    parser.add_argument('--calls', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1616)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.batch, args.input_width, args.calls, args.repeats, *args.hidden_widths) < 1:
        parser.error('dimensions and timing counts must be positive')
    if args.input_width*3*65535**2 >= 2**63:
        parser.error('input width exceeds the exact CPU reference accumulator bound')
    torch.manual_seed(args.seed)
    rows = []
    for row in chain(matmul_cases(args), prg_cases(args)):
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = dict(gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  triton_version=triton.__version__, cuda_version=torch.version.cuda,
                  arguments={**vars(args), 'output': str(args.output)},
                  timing_method='Three warmups, one CUDA graph of repeated calls, median '
                  'CUDA-event replay time divided by calls. Serialized on one GPU; '
                  'resident repeated buffers; allocation, preparation, and validation excluded.',
                  scope='Local kernels only; no protocol, network, or end-to-end timing. '
                  'Byte counts describe tensors, not measured DRAM transactions. '
                  'PRG reuses fixed nonces solely for this benchmark.', cases=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
