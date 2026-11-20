"""Compare exact ring32 CUDA kernels for the private 512-to-128 encoder."""
import argparse
from functools import partial
import hashlib
import inspect
import json
from pathlib import Path
import re
import statistics

import numpy as np
import torch
import triton

from rep4gpu.gpu.matmul.kernels import prepare_weight
from rep4gpu.gpu.matmul.tuned_kernels import local_products_tuned


def compiler_metadata(run):
    """Record the PTX and launch resources from kernels loaded on the first call."""
    kernels = []

    def on_load(module, function, name, files, kernel_hash):
        paths = {Path(path).suffix: Path(path) for path in files.values()}
        metadata = json.loads(paths['.json'].read_text())
        info = dict(name=name, triton_hash=kernel_hash, shared_bytes=metadata['shared'],
                    num_stages=metadata['num_stages'], num_warps=metadata['num_warps'])
        if '.ptx' in paths:
            ptx = paths['.ptx'].read_text()
            info.update(ptx_async_copy_instructions=len(re.findall(r'\bcp\.async\.(?:ca|cg)\.', ptx)),
                        ptx_mma_instructions=len(re.findall(r'\bmma\.sync\.', ptx)))
        kernels.append(info)

    hook = triton.knobs.runtime.kernel_load_end_hook
    hook.add(on_load)
    try:
        result = run()
    finally:
        hook.remove(on_load)
    return result, kernels


def cpu_reference(xs, ys, rows):
    # uint32 matmul and addition wrap at every operation in the reference ring.
    a, b, c = (x[rows].numpy().astype(np.uint32) for x in xs)
    d, e, f = (y.numpy().astype(np.uint32) for y in ys)
    values = ((a+b)@d+a@e, (b+c)@e+b@f, a@f+c@d)
    return tuple(torch.from_numpy(v.astype(np.int64)) for v in values)


def capture(run, calls):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            output = run()
    # Retain the final output allocation for the lifetime of the replay graph.
    return graph, output


def benchmark_batch(batch, stages, calls, repeats, seed, prepared=False):
    from rep4gpu.gpu.matmul.pipelined_encoder import local_products_pipelined

    gen = torch.Generator().manual_seed(seed + batch)

    def shares(shape):
        values = tuple(torch.randint(-(2**63), 2**63-1, shape, generator=gen,
                                     dtype=torch.int64) for _ in range(3))
        for i, value in enumerate(values):
            value[0, :6] = torch.tensor([0, -1, 2**31, 2**32-1, -2**63, 2**63-1]).roll(i)
        return values

    xs, ys = shares((batch, 512)), shares((512, 128))
    rows = sorted({0, batch-1, *(i * (batch-1) // 8 for i in range(1, 8))})
    reference = cpu_reference(xs, ys, rows)
    inputs = tuple(x.cuda() for x in xs)
    weights = tuple(prepare_weight(y.cuda(), 32, storage='words32') for y in ys)
    variants = {'current': partial(local_products_tuned, *inputs, *weights, bits=32)}
    compact_bytes = sum(w.words.nbytes for w in weights)
    if prepared:
        from rep4gpu.gpu.matmul.pipelined_encoder import (
            local_products_pipelined_prepared, prepare_pipelined_encoder,
        )
        packed = prepare_pipelined_encoder(*weights)
        run_pipeline = partial(local_products_pipelined_prepared, *inputs, packed)
        variant_prefix = 'prepared_stages'
        pipeline_weight_bytes = packed.limbs.nbytes
    else:
        run_pipeline = partial(local_products_pipelined, *inputs, *weights)
        variant_prefix = 'stages'
        pipeline_weight_bytes = compact_bytes
    variants.update({f'{variant_prefix}_{s}': partial(run_pipeline, stages=s) for s in stages})
    case = dict(batch=batch, input_width=512, output_width=128, exact_cpu_rows=rows, variants={})
    graphs, baseline = {}, None
    for name, run in variants.items():
        try:
            outputs, kernels = compiler_metadata(run)
        except triton.runtime.errors.OutOfResources as exc:
            if name == 'current':
                raise
            case['variants'][name] = dict(skipped=str(exc))
            continue
        assert len(outputs) == 3
        for i, result in enumerate(outputs):
            torch.testing.assert_close(result[rows].cpu(), reference[i], rtol=0, atol=0)
            if baseline is not None:
                torch.testing.assert_close(result, baseline[i], rtol=0, atol=0)
        if baseline is None:
            baseline = outputs
        graphs[name] = capture(run, calls)
        case['variants'][name] = dict(exact=True, kernels_loaded=kernels, samples_ms=[],
                                      persistent_weight_bytes=(compact_bytes if name == 'current'
                                                               else pipeline_weight_bytes))
    torch.cuda.synchronize()
    for graph, _ in graphs.values():
        graph.replay()
    torch.cuda.synchronize()

    for i in range(repeats):
        order = list(graphs)
        # Rotate starting position so each variant gets early and late slots.
        for name in order[i % len(order):] + order[:i % len(order)]:
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            graphs[name][0].replay()
            end.record()
            end.synchronize()
            case['variants'][name]['samples_ms'].append(start.elapsed_time(end) / calls)
    for name in graphs:
        result = case['variants'][name]
        result['median_ms'] = statistics.median(result['samples_ms'])
        result['speedup_vs_current'] = case['variants']['current']['median_ms'] / result['median_ms']
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batches', type=int, nargs='+', default=[4096, 8192, 129])
    parser.add_argument('--stages', type=int, choices=range(1, 6), nargs='+', default=[1, 2, 3, 4])
    parser.add_argument('--prepared', action='store_true',
                        help='compare the pipeline using prepacked weights with current compact dispatch')
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--calls', type=int, default=4)
    parser.add_argument('--seed', type=int, default=8128)
    args = parser.parse_args()
    if 5 in args.stages and not args.prepared:
        parser.error('five stages require --prepared')
    if min(*args.batches, *args.stages, args.repeats, args.calls) < 1:
        parser.error('batches, stages, calls and repeats must be positive')

    from rep4gpu.gpu.matmul.pipelined_encoder import local_products_pipelined
    source = Path(inspect.getfile(local_products_pipelined))
    report = dict(gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                  torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  triton_version=triton.__version__, kernel_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  seed=args.seed, calls=args.calls, repeats=args.repeats, prepared=args.prepared,
                  timing='CUDA event time per local product on warm resident data. One graph per '
                         'variant, rotated replay order. Preparation and communication excluded.',
                  cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for batch in args.batches:
        case = benchmark_batch(batch, args.stages, args.calls, args.repeats, args.seed, args.prepared)
        report['cases'].append(case)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(case), flush=True)


if __name__ == '__main__':
    main()
