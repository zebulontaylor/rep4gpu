"""Benchmark resident secret-weight ReLU FFN inference with GPU ring Rep4.

Evaluates a two-layer feed-forward block (width -> 4*width ReLU -> width)
using ``PhaseGraphParty`` (CUDA ChaCha20 masks, BLAKE3-256 digests, captured
CUDA phase graphs, optional Strassen ring32 matmuls, sparse/narrow truncation
corrections, and bounded-sign ReLU).

Reports:
  - Live fresh-mask 4-party execution on one GPU (includes inter-role contention)
  - Isolated-party zero-network GPU replay estimate (timing fixture only; excludes
    network latency and bandwidth)
  - PyTorch FP16 and FP32 CUDA-graph baselines on the same dimensions

Run from the repository root:
    python -m examples.ring_ffn_benchmark --quick
    python -m examples.ring_ffn_benchmark --batch 1 128 --width 2048 --storage strassen32
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import statistics
import time

import numpy as np
import torch

from rep4gpu import (
    DeviceIdealEndpoint,
    FixedPoint,
    IdealNetwork,
    PhaseGraphParty,
    make_seeds,
)
from rep4gpu.transports import DeviceRecorder, DeviceReplay


def _max_abs(t: torch.Tensor) -> int:
    return max(abs(int(t.min())), abs(int(t.max())))


def _checked_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if _max_abs(a) * _max_abs(b) * a.shape[-1] > (1 << 63) - 1:
        raise ValueError('int64 reference accumulation overflow')
    return a @ b


def make_ffn_fixture(batch: int, width: int, fractional_bits: int, bits: int, rounding: str):
    """Build a public FFN fixture and validate its signed ring and rounding bounds."""
    hidden = 4 * width
    g = torch.Generator().manual_seed(2026)
    scale = 1 << fractional_bits
    x_f = torch.randn(batch, width, generator=g) / (width ** 0.5)
    w1_f = torch.randn(width, hidden, generator=g) / (width ** 0.5)
    b1_f = torch.randn(hidden, generator=g) * 0.05
    w2_f = torch.randn(hidden, width, generator=g) / (hidden ** 0.5)
    b2_f = torch.randn(width, generator=g) * 0.05

    encode = lambda t: torch.round(t.double() * scale).long()
    integers = tuple(encode(t) for t in (x_f, w1_f, b1_f, w2_f, b2_f))
    x, w1, b1, w2, b2 = integers

    raw1 = _checked_matmul(x, w1) + b1 * scale
    low_h = (raw1 >> fractional_bits).clamp_min(0)
    if rounding == 'probabilistic':
        high_h = ((raw1 + (scale - 1)) >> fractional_bits).clamp_min(0)
        w2_pos, w2_neg = w2.clamp_min(0), w2.clamp_max(0)
        raw2_lo = _checked_matmul(low_h, w2_pos) + _checked_matmul(high_h, w2_neg)
        raw2_hi = _checked_matmul(high_h, w2_pos) + _checked_matmul(low_h, w2_neg)
        lower = (raw2_lo >> fractional_bits) + b2
        upper = ((raw2_hi + (scale - 1)) >> fractional_bits) + b2
    else:
        raw2_lo = raw2_hi = _checked_matmul(low_h, w2)
        lower = upper = (raw2_lo >> fractional_bits) + b2

    max_acc = max(_max_abs(raw1), _max_abs(raw2_lo), _max_abs(raw2_hi))
    if rounding == 'probabilistic' and max_acc >= (1 << (bits - 2)):
        raise ValueError(f'accumulator {max_acc} violates probabilistic limit {1 << (bits - 2)}')
    ring_limit = 1 << (bits - 1)
    for tensor in (*integers, raw1, raw2_lo, raw2_hi, lower, upper):
        if int(tensor.min()) < -ring_limit or int(tensor.max()) >= ring_limit:
            raise ValueError(f'fixture exceeds signed ring{bits}')
    return integers, (x_f, w1_f, b1_f, w2_f, b2_f), lower, upper, max_acc


def pytorch_graph_ms(floats, dtype: torch.dtype, repeats: int) -> float:
    x, w1, b1, w2, b2 = (t.to(device='cuda', dtype=dtype) for t in floats)
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        out = torch.empty_like(x)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for _ in range(3):
                out.copy_(torch.relu(x @ w1 + b1) @ w2 + b2)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out.copy_(torch.relu(x @ w1 + b1) @ w2 + b2)
        stream.synchronize()

        times = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            graph.replay()
            end.record(stream)
            stream.synchronize()
            times.append(start.elapsed_time(end))
        return float(statistics.median(times))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32


def benchmark_batch(batch: int, width: int, *, bits: int, fractional_bits: int,
                    storage: str, rounding: str, repeats: int,
                    bounded_sign: bool, sparse_products: bool,
                    narrow_products: bool, check_at_send: bool) -> dict:
    effective_storage = 'words32' if (storage == 'strassen32' and (batch % 2 != 0 or width % 2 != 0)) else storage
    if bits != 32 and effective_storage in ('words32', 'strassen32'):
        effective_storage = 'limbs'
    integers, floats, lower, upper, max_acc = make_ffn_fixture(
        batch, width, fractional_bits, bits, rounding,
    )
    device = torch.device('cuda')
    cuda_ints = tuple(t.to(device) for t in integers)

    def make_roles(transports, seeds):
        return [
            PhaseGraphParty(
                p, transports[p], {e: s for e, s in seeds.items() if e != p},
                device=device, bits=bits, binary_bits=bits,
                optimized=True, gpu_open=True, rounding=rounding,
                bounded_sign=bounded_sign, sparse_products=sparse_products,
                narrow_products=narrow_products,
            )
            for p in range(4)
        ]

    # 1. Live 4-party execution with fresh masks + transcript recording on warmup.
    network = IdealNetwork(timeout=120)
    seeds = make_seeds()
    recorders = [DeviceRecorder(DeviceIdealEndpoint(network.endpoint(p), device=device)) for p in range(4)]
    roles = make_roles(recorders, seeds)

    def share_all(p: int):
        try:
            r = roles[p]
            shared = [r.input(owner, val if p == owner else None, val.shape)
                      for owner, val in zip((0, 1, 1, 1, 1), cuda_ints)]
            x, w1, b1, w2, b2 = shared
            w1 = r.prepare_weight_share(w1, storage=effective_storage)
            w2 = r.prepare_weight_share(w2, storage=effective_storage)
            return x, w1, b1, w2, b2
        except BaseException as err:
            network.abort(err)
            raise

    with ThreadPoolExecutor(4) as pool:
        shared_state = list(pool.map(share_all, range(4)))

    plans = []
    for p in range(4):
        r = roles[p]
        x, w1, b1, w2, b2 = shared_state[p]
        fp = FixedPoint(r, fractional_bits)

        def evaluate(inp, _fp=fp, _r=r, _w1=w1, _b1=b1, _w2=w2, _b2=b2):
            h = _fp.matmul_relu(inp, _w1, _b1)
            return _r.open(_fp.add(_fp.matmul(h, _w2), _b2))

        plans.append(r.prepare_inference(evaluate, x, check_at_send=check_at_send))

    # Record one warmup transcript for isolated-party zero-network replay timing.
    for rec in recorders:
        rec.transcript.clear()
    saved_counters = [dict(r.counters) for r in roles]

    def run_once(p: int):
        try:
            return plans[p].run(shared_state[p][0]).cpu()
        except BaseException as err:
            network.abort(err)
            raise

    with ThreadPoolExecutor(4) as pool:
        warmup_out = list(pool.map(run_once, range(4)))
    if any(not torch.equal(warmup_out[0], o) for o in warmup_out[1:]):
        raise AssertionError('parties disagreed during warmup')
    if not bool(((warmup_out[0] >= lower) & (warmup_out[0] <= upper)).all()):
        raise AssertionError('warmup output violated certified rounding bounds')

    transcripts = [list(rec.transcript) for rec in recorders]
    total_payload_bytes = sum(
        sum(len(buf) for buf in frame.outgoing.values())
        for t in transcripts for frame in t
    )
    rounds = len(transcripts[0])

    # Disable recording for live timing repeats.
    for p in range(4):
        roles[p].transport = recorders[p].underlying

    live_ms = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with ThreadPoolExecutor(4) as pool:
            outs = list(pool.map(run_once, range(4)))
        torch.cuda.synchronize()
        live_ms.append((time.perf_counter() - t0) * 1000)
        if any(not torch.equal(outs[0], o) for o in outs[1:]):
            raise AssertionError('parties disagreed during live repeat')
        if not bool(((outs[0] >= lower) & (outs[0] <= upper)).all()):
            raise AssertionError('live output violated certified rounding bounds')

    # 2. Isolated-party zero-network GPU replay estimate (timing fixture only).
    role_phase_medians = []
    for p in range(4):
        replay = DeviceReplay(transcripts[p], device=device, staging='device', check_outgoing=True)
        roles[p].transport = replay
        # Verify one exact replay before timing.
        roles[p].counters = dict(saved_counters[p])
        replay.round = 0
        out = plans[p].run(shared_state[p][0]).cpu()
        if not torch.equal(out, warmup_out[0]):
            raise AssertionError(f'isolated replay mismatch for party {p}')
        replay.assert_consumed()
        replay.check_outgoing = False

        samples = []
        for _ in range(repeats):
            roles[p].counters = dict(saved_counters[p])
            replay.round = 0
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            plans[p].run(shared_state[p][0])
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1000)
        role_phase_medians.append(float(statistics.median(samples)))

    fp16_ms = pytorch_graph_ms(floats, torch.float16, repeats)
    fp32_ms = pytorch_graph_ms(floats, torch.float32, repeats)
    isolated_ms = max(role_phase_medians)

    return {
        'batch': batch,
        'width': width,
        'hidden': 4 * width,
        'bits': bits,
        'fractional_bits': fractional_bits,
        'storage': effective_storage,
        'rounding': rounding,
        'max_accumulator': max_acc,
        'rounds': rounds,
        'global_payload_mb': round(total_payload_bytes / (1024 * 1024), 3),
        'live_4role_ms': round(float(statistics.median(live_ms)), 3),
        'isolated_party_ms': round(isolated_ms, 3),
        'fp16_graph_ms': round(fp16_ms, 4),
        'fp32_graph_ms': round(fp32_ms, 4),
        'isolated_vs_fp16': round(isolated_ms / fp16_ms, 2),
        'isolated_vs_fp32': round(isolated_ms / fp32_ms, 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true', help='Run small fast shapes (batch=1,16 width=256)')
    parser.add_argument('--batch', type=int, nargs='+', default=[1, 128])
    parser.add_argument('--width', type=int, default=2048)
    parser.add_argument('--bits', type=int, choices=(32, 64), default=32)
    parser.add_argument('--fractional-bits', type=int, default=14)
    parser.add_argument('--storage', choices=('limbs', 'words32', 'strassen32'), default='words32')
    parser.add_argument('--rounding', choices=('floor', 'probabilistic'), default='probabilistic')
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for ring_ffn_benchmark')

    batches = [1, 16] if args.quick else args.batch
    width = 256 if args.quick else args.width
    repeats = 3 if args.quick else args.repeats
    fb = 12 if args.quick else args.fractional_bits

    results = [
        benchmark_batch(
            b, width,
            bits=args.bits,
            fractional_bits=fb,
            storage=args.storage,
            rounding=args.rounding,
            repeats=repeats,
            bounded_sign=(args.rounding == 'probabilistic'),
            sparse_products=True,
            narrow_products=(args.rounding == 'probabilistic' and fb <= 16),
            check_at_send=True,
        )
        for b in batches
    ]
    print(json.dumps({'gpu': torch.cuda.get_device_name(0), 'results': results}, indent=2))


if __name__ == '__main__':
    main()
