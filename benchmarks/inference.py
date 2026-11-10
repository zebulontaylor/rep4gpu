"""Measure checked Rep4 toy feed-forward inference and ideal-network replay.

Run from the repository root: python -m benchmarks.inference --device cpu --quick
The four-party run uses fresh masks. Replay is a fixed public test fixture used
only for timing; it must never serve requests or reuse masks with new inputs.
"""
import argparse
import hashlib
import importlib.metadata
import platform
import sys
from threadpoolctl import threadpool_info, threadpool_limits
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from rep4gpu.activations import FixedPoint
from rep4gpu.runtime import Party, make_seeds
from rep4gpu.transport import IdealNetwork, RecordingTransport, ReplayTransport


class TruncationPolicy:
    truncation = 'compact'

    def trunc(self, x, shift):
        if self.truncation == 'probabilistic':
            from rep4gpu.probabilistic import trunc_pr
            return trunc_pr(self, x, shift)
        if self.truncation == 'compact':
            from rep4gpu.exact_truncation import trunc
            return trunc(self, x, shift)
        return super().trunc(x, shift)

    def trunc_relu(self, x, shift):
        if self.truncation == 'probabilistic':
            truncated = self.trunc(x, shift)
            return self.mul(truncated, self.sub(self.public(1, like=x), self.sign_bit(truncated)))
        if self.truncation == 'compact':
            from rep4gpu.exact_truncation import trunc_relu
            return trunc_relu(self, x, shift)
        return super().trunc_relu(x, shift)


def maximum_abs(value):
    """Compute magnitude without signed-int64 abs overflow."""
    return max(abs(int(value.min())), abs(int(value.max())))


def checked_matmul(left, right):
    """Reject a conservative overflow bound before using int64 accumulation."""
    bound = maximum_abs(left) * maximum_abs(right) * left.shape[-1]
    if bound > torch.iinfo(torch.int64).max:
        raise ValueError('reference matrix-product bound exceeds signed int64')
    return left @ right


def checked_add(left, right):
    if maximum_abs(left) + maximum_abs(right) > torch.iinfo(torch.int64).max:
        raise ValueError('reference addition bound exceeds signed int64')
    return left + right


def provenance():
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve(), *sorted((root / 'rep4gpu').glob('*.py'))]
    packages = {}
    for name in ('torch', 'numpy', 'triton', 'threadpoolctl', 'cryptography'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(python=sys.version, platform=platform.platform(), packages=packages,
                cuda_runtime=torch.version.cuda,
                source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in paths})


def fixture(batch, width, hidden, out, fractional_bits=12):
    if not isinstance(fractional_bits, int) or not 0 <= fractional_bits < 63:
        raise ValueError("fractional_bits must be an integer in [0, 62]")
    rng = np.random.default_rng(20261109)
    arrays = [rng.normal(0, .5, (batch, width)),
              rng.normal(0, 1 / np.sqrt(width), (width, hidden)),
              rng.normal(0, .05, (hidden,)),
              rng.normal(0, 1 / np.sqrt(hidden), (hidden, out)),
              rng.normal(0, .05, (out,))]
    floats = [torch.from_numpy(a.astype(np.float32)) for a in arrays]
    encoded = [np.rint(a * (1 << fractional_bits)) for a in arrays]
    if any(not np.all(np.isfinite(a) & (a >= -(2.**63)) & (a < 2.**63)) for a in encoded):
        raise ValueError('fixture encoding must fit signed int64')
    integers = [torch.from_numpy(a.astype(np.int64)) for a in encoded]
    x, w1, b1, w2, b2 = integers
    h = checked_add(checked_matmul(x, w1) >> fractional_bits, b1).clamp_min(0)
    expected = checked_add(checked_matmul(h, w2) >> fractional_bits, b2)
    return floats, integers, expected


def plain(values):
    x, w1, b1, w2, b2 = values
    return torch.relu(x @ w1 + b1) @ w2 + b2


def infer(party, values, fractional_bits):
    x, w1, b1, w2, b2 = values
    fp = FixedPoint(party, fractional_bits)
    hidden = fp.matmul_relu(x, w1, b1)
    output = fp.add(fp.matmul(hidden, w2), b2)
    return party.open(output)


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def timed(fn, device, repeats=10):
    for _ in range(3):
        fn()
    synchronize(device)
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        synchronize(device)
        values.append((time.perf_counter() - start) * 1000)
    return statistics.median(values)


def baselines(values, device, repeats):
    results, errors = {}, {}
    reference = plain(values)
    for name, dtype in [('fp32', torch.float32), ('fp16', torch.float16)]:
        if name == 'fp16' and torch.device(device).type == 'cpu':
            continue
        inputs = [v.to(device=device, dtype=dtype) for v in values]
        actual = plain(inputs).cpu().float()
        errors[name] = dict(max_logit_error=float((actual-reference).abs().max()),
                            max_logprob_error=float((actual.log_softmax(-1)-reference.log_softmax(-1)).abs().max()))
        results[name + '_eager_ms'] = timed(lambda: plain(inputs), device, max(20, repeats))
        if torch.device(device).type == 'cuda':
            graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    plain(inputs)
            torch.cuda.current_stream().wait_stream(stream)
            with torch.cuda.graph(graph):
                output = plain(inputs)
            # Amortize Python replay overhead, then use GPU events for device time.
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(100):
                graph.replay()
            end.record()
            end.synchronize()
            results[name + '_graph_ms'] = begin.elapsed_time(end) / 100
            results[name + '_graph_wall_ms'] = timed(graph.replay, device, max(20, repeats))
    arrays = [v.numpy() for v in values]
    results['numpy_cpu_ms'] = timed(lambda: np.maximum(arrays[0] @ arrays[1] + arrays[2], 0) @ arrays[3] + arrays[4], 'cpu', max(20, repeats))
    return results, errors


def run_case(batch, width, hidden, out, device, repeats=3, fractional_bits=12, prepared=False,
             bits=64, truncation='compact', fused=False, prg='shake', binary_bits=64,
             weight_storage='limbs'):
    if repeats < 1:
        raise ValueError('repeats must be positive')
    floats, integers, expected = fixture(batch, width, hidden, out, fractional_bits)
    network = IdealNetwork()
    seeds = make_seeds()
    base = Party
    if fused:
        from rep4gpu.fused_runtime import FusedParty
        base = FusedParty
    party_type = type('InferenceParty', (TruncationPolicy, base), {'truncation': truncation})
    parties = [party_type(p, network.endpoint(p), {i: s for i, s in seeds.items() if i != p}, device,
                          bits=bits, prg=prg, binary_bits=binary_bits)
               for p in range(4)]
    # Conservative prebounds make the int64 reference arithmetic exact before inputs
    # are shared. It does not implement a private range check for live clients.
    x, w1, b1, w2, b2 = integers
    if maximum_abs(b1) * (1 << fractional_bits) > torch.iinfo(torch.int64).max:
        raise ValueError('scaled bias exceeds signed int64')
    raw1 = checked_add(checked_matmul(x, w1), b1 << fractional_bits)
    low_h = (raw1 >> fractional_bits).clamp_min(0)
    high_h = (checked_add(raw1, torch.tensor((1 << fractional_bits) - 1)) >> fractional_bits).clamp_min(0)
    raw2lo = checked_add(checked_matmul(low_h, w2.clamp_min(0)), checked_matmul(high_h, w2.clamp_max(0)))
    raw2hi = checked_add(checked_matmul(high_h, w2.clamp_min(0)), checked_matmul(low_h, w2.clamp_max(0)))
    lower = checked_add(raw2lo >> fractional_bits, b2)
    upper = checked_add(checked_add(raw2hi, torch.tensor((1 << fractional_bits) - 1)) >> fractional_bits, b2)
    maximum_accumulator = max(maximum_abs(v) for v in [raw1, raw2lo, raw2hi])
    bound = 1 << (bits - (2 if truncation == 'probabilistic' else 1))
    if maximum_accumulator >= bound:
        raise ValueError(f'fixture accumulator {maximum_accumulator} violates {bits}-bit {truncation} bound {bound}')
    pool = ThreadPoolExecutor(4)
    def run_parties(function):
        def guarded(party):
            try:
                return function(party)
            except BaseException as error:
                network.abort(error)
                raise
        return list(pool.map(guarded, parties))
    def distribute(p):
        values = [p.input(owner, a if p.party == owner else None, a.shape)
                  for owner, a in zip([0, 1, 1, 1, 1], integers)]
        if prepared and torch.device(device).type == 'cuda':
                values[1] = p.prepare_weight_share(values[1], storage=weight_storage)
                values[3] = p.prepare_weight_share(values[3], storage=weight_storage)
        return values
    start = time.perf_counter()
    try:
        shares = run_parties(distribute)
        synchronize(device)
        setup_ms = (time.perf_counter() - start) * 1000
        def check(outputs):
            if any(not torch.equal(outputs[0], output) for output in outputs[1:]):
                raise AssertionError('parties disagree on opened inference output')
            for output in outputs:
                actual = output.cpu()
                if truncation == 'probabilistic':
                    if not bool(((actual >= lower) & (actual <= upper)).all()):
                        raise AssertionError('secret inference exceeds exact propagated rounding bounds')
                elif not torch.equal(actual, expected):
                    raise AssertionError('secret inference differs from exact integer reference')
        # Warm caches/JIT, then record a fresh inference transcript.
        check(run_parties(lambda p: infer(p, shares[p.party], fractional_bits)))
        wall_ms, recorded, starts = [], None, None
        for repeat in range(repeats):
            starts = [dict(p.counters) for p in parties]
            recorded = [RecordingTransport(p.transport) for p in parties]
            for p, transport in zip(parties, recorded):
                p.transport = transport
            start = time.perf_counter()
            outputs = run_parties(lambda p: infer(p, shares[p.party], fractional_bits))
            synchronize(device)
            wall_ms.append((time.perf_counter() - start) * 1000)
            check(outputs)
            for p, transport in zip(parties, recorded):
                p.transport = transport.underlying
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    # Run each party alone on the GPU using the validated peer messages. Phase
    # maxima model four independent identical GPUs with zero network duration.
    replay_party_ms, replay_phases, replay_phase_samples = [], [], []
    for p in parties:
        transcript = recorded[p.party].transcript
        times, phases = [], []
        for repeat in range(repeats + 1):
            replay = ReplayTransport(transcript, check_outgoing=repeat == 0)
            local = party_type(p.party, replay, p.seeds, device, bits=bits, prg=prg, binary_bits=binary_bits)
            local.counters = dict(starts[p.party])
            replay.start(synchronize=lambda: synchronize(device))
            output = infer(local, shares[p.party], fractional_bits)
            replay.finish()
            replay.assert_consumed()
            check([output])
            if repeat:
                phases.append(replay.phase_times)
                times.append(sum(replay.phase_times) * 1000)
        replay_phase_samples.append(phases)
        replay_party_ms.append(statistics.median(times))
        replay_phases.append(np.median(np.asarray(phases), axis=0))
    ideal_ms = float(np.max(np.stack(replay_phases), axis=0).sum() * 1000)
    reference_float = plain(floats)
    decoded = outputs[0].cpu().float() / (1 << fractional_bits)
    baseline, baseline_errors = baselines(floats, device, repeats)
    max_logprob_error = float((decoded.log_softmax(-1) - reference_float.log_softmax(-1)).abs().max())
    return dict(batch=batch, width=width, hidden=hidden, out=out, device=str(device),
                fractional_bits=fractional_bits, ring_bits=bits, binary_bits=binary_bits,
                prg=prg, truncation=truncation, fused_runtime=fused,
                prepared_weights=prepared, weight_storage=weight_storage,
                maximum_accumulator=maximum_accumulator,
                rounding_bound_validated=True, exact_integer_match=truncation != 'probabilistic',
                max_logit_error=float((decoded - reference_float).abs().max()),
                max_logprob_error=max_logprob_error, setup_all_parties_ms=setup_ms,
                local_four_party_ms=statistics.median(wall_ms), local_samples_ms=wall_ms,
                replay_party_ms=replay_party_ms, ideal_network_phase_max_ms=ideal_ms,
                rounds=len(recorded[0].transcript),
                payload_bytes=sum(sum(len(v) for v in outgoing.values()) for r in recorded for outgoing, _ in r.transcript),
                replay_phase_samples_seconds=replay_phase_samples,
                samples=repeats, protocol_warmups=1, replay_warmups_per_party=1,
                baseline_warmups=3, baseline_samples=max(20, repeats),
                baseline_graph_device_replays=100 if torch.device(device).type == 'cuda' else 0,
                ideal_timing_kind='estimate: sum of per-phase maxima of isolated-party median times',
                baseline_accuracy=baseline_errors,
                baselines=baseline, ideal_slowdowns={k: ideal_ms / ms for k, ms in baseline.items()},
                methodology=f'Fresh-mask checked four-party colocated execution; fixed-fixture per-party replay with real {prg} PRG, BLAKE2b, host staging, and checked output. Ideal timing is an estimate summing per-phase maxima of medians; no network latency/bandwidth. Sharing/setup reported separately.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--batch', type=int, nargs='+', default=[1, 16, 128])
    parser.add_argument('--width', type=int, nargs='+', default=[128, 512, 1024])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--prepared', action='store_true')
    parser.add_argument('--weight-storage', choices=['limbs', 'words32'], default='limbs')
    parser.add_argument('--bits', type=int, choices=[32, 64], default=64)
    parser.add_argument('--binary-bits', type=int, choices=[32, 64], default=64)
    parser.add_argument('--fractional-bits', type=int, default=12)
    parser.add_argument('--truncation', choices=['reference', 'compact', 'probabilistic'], default='compact')
    parser.add_argument('--fused', action='store_true')
    parser.add_argument('--prg', choices=['shake', 'aes'], default='shake')
    parser.add_argument('--output', type=Path, default=Path('results/inference.json'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    cases = [(1, 8, 8, 4)] if args.quick else [(b, w, 4*w, w) for b in args.batch for w in args.width]
    report = dict(torch=torch.__version__, numpy=np.__version__, threads=torch.get_num_threads(),
                  gpu=torch.cuda.get_device_name() if args.device.startswith('cuda') else None,
                  blas_threadpools=threadpool_info(), provenance=provenance(), results=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for b, w, h, n in cases:
        row = run_case(b, w, h, n, args.device, args.repeats,
                       fractional_bits=args.fractional_bits, prepared=args.prepared,
                       bits=args.bits, truncation=args.truncation, fused=args.fused, prg=args.prg,
                       binary_bits=args.binary_bits, weight_storage=args.weight_storage)
        report['results'].append(row)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: row[k] for k in ('batch', 'width', 'ring_bits', 'truncation',
                         'prg', 'fused_runtime', 'local_four_party_ms', 'ideal_network_phase_max_ms',
                         'payload_bytes', 'max_logprob_error', 'ideal_slowdowns')}), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1, user_api='blas'):
        main()
