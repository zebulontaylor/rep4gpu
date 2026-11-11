"""Measure secret matrix multiplication independently of activation and opening.

Run from the repository root: python -m experiments.benchmarks.kernels.primitives --device cpu --m 1 --k 8
Replay uses a fixed fixture only for timing. Never reuse its masks for requests.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from threadpoolctl import threadpool_info, threadpool_limits

from experiments.benchmarks.inference.inference import checked_matmul, maximum_abs, provenance, synchronize, timed
from rep4gpu.arithmetic.exact_truncation import trunc, trunc_relu
from rep4gpu.protocol.runtime import Party, make_seeds
from rep4gpu.protocol.transport import IdealNetwork, RecordingTransport, ReplayTransport


def fixture(m, k, n, bits, fractional_bits=12):
    if min(m, k, n) < 1:
        raise ValueError('matrix dimensions must be positive')
    rng = np.random.default_rng(20261109)
    floats = [torch.from_numpy(a.astype(np.float32)) for a in
              (rng.normal(0, .5, (m, k)), rng.normal(0, 1 / np.sqrt(k), (k, n)))]
    integers = [torch.round(a.double() * (1 << fractional_bits)).long() for a in floats]
    product = checked_matmul(*integers)
    range_metadata = dict(maximum_accumulator_before_ring_wrap=maximum_abs(product),
                          minimum_accumulator_before_ring_wrap=int(product.min()),
                          maximum_signed_accumulator_before_ring_wrap=int(product.max()),
                          accumulator_fits_signed_ring=bool(((product >= -(1 << (bits-1))) &
                                                             (product < (1 << (bits-1)))).all()))
    if bits == 32:
        product = ((product + (1 << 31)) & 0xffffffff) - (1 << 31)
    return floats, integers, product, range_metadata


def operation(party, shares, name, shift):
    if name == 'trunc':
        return trunc(party, shares[0], shift)
    product = party.matmul(*shares)
    return trunc_relu(party, product, shift) if name == 'matmul_relu' else product


def plain(values, name):
    if name == 'trunc':
        return torch.floor(values[0] * 4096) / 4096
    product = values[0] @ values[1]
    return torch.relu(product) if name == 'matmul_relu' else product


def baselines(values, name, device, repeats):
    timings, errors = {}, {}
    reference = plain(values, name)
    for label, dtype in [('fp32', torch.float32), ('fp16', torch.float16)]:
        if label == 'fp16' and torch.device(device).type == 'cpu':
            continue
        inputs = [v.to(device=device, dtype=dtype) for v in values]
        compute = lambda: plain(inputs, name)
        errors[label] = float((compute().cpu().float() - reference).abs().max())
        timings[label + '_eager_ms'] = timed(compute, device, max(20, repeats))
        if torch.device(device).type == 'cuda':
            graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    compute()
            torch.cuda.current_stream().wait_stream(stream)
            with torch.cuda.graph(graph):
                output = compute()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(100):
                graph.replay()
            end.record()
            end.synchronize()
            timings[label + '_graph_ms'] = begin.elapsed_time(end) / 100
            timings[label + '_graph_wall_ms'] = timed(graph.replay, device, max(20, repeats))
    arrays = [v.numpy() for v in values]
    def numpy_compute():
        if name == 'trunc':
            return np.floor(arrays[0] * 4096) / 4096
        result = arrays[0] @ arrays[1]
        return np.maximum(result, 0) if name == 'matmul_relu' else result
    timings['numpy_cpu_ms'] = timed(numpy_compute, 'cpu', max(20, repeats))
    return timings, errors


def run_case(m, k, n, device='cpu', repeats=5, bits=64, binary_bits=64,
             fused=False, prg='shake', name='matmul', prepared=False, weight_storage='limbs'):
    if repeats < 1:
        raise ValueError('repeats must be positive')
    floats, integers, product, range_metadata = fixture(m, k, n, bits)
    shift = 12
    if name == 'trunc':
        integers = [product]
        floats = [(floats[0] @ floats[1])]
        expected = product >> shift
        # A raw Q24 product becomes Q12 after truncation.
        decoded_scale = 1 << shift
    elif name == 'matmul_relu':
        expected = (product >> shift).clamp_min(0)
        decoded_scale = 1 << shift
    else:
        expected, decoded_scale = product, 1 << (2 * shift)
    base = Party
    if fused:
        from rep4gpu.protocol.fused_runtime import FusedParty
        base = FusedParty
    network, seeds = IdealNetwork(), make_seeds()
    parties = [base(p, network.endpoint(p), {i:s for i,s in seeds.items() if i != p},
                    device=device, bits=bits, binary_bits=binary_bits, prg=prg) for p in range(4)]
    with ThreadPoolExecutor(4) as pool:
        def run_parties(function):
            def guarded(party):
                try:
                    return function(party)
                except BaseException as error:
                    network.abort(error)
                    raise
            return list(pool.map(guarded, parties))
        def distribute(p):
            values = [p.input(owner, value if p.party == owner else None, value.shape)
                      for owner, value in enumerate(integers)]
            if prepared and name != 'trunc' and torch.device(device).type == 'cuda':
                values[1] = p.prepare_weight_share(values[1], storage=weight_storage)
            return values
        start = time.perf_counter()
        shares = run_parties(distribute)
        synchronize(device)
        setup_ms = (time.perf_counter()-start)*1000
        opening_samples = []
        def validate(results):
            for excluded in range(4):
                replicas = [results[p.party].parts[(excluded-p.party-1) % 4]
                            for p in parties if p.party != excluded]
                if any(not torch.equal(replicas[0], value) for value in replicas[1:]):
                    raise AssertionError('replicated output components disagree')
            start = time.perf_counter()
            opened = run_parties(lambda p: p.open(results[p.party]))
            synchronize(device)
            opening_samples.append((time.perf_counter()-start)*1000)
            if any(not torch.equal(value.cpu(), expected) for value in opened):
                raise AssertionError('secret primitive differs from integer ring reference')
            return opened[0]
        validate(run_parties(lambda p: operation(p, shares[p.party], name, shift)))
        local_samples = []
        for _ in range(repeats):
            starts = [dict(p.counters) for p in parties]
            records = [RecordingTransport(p.transport) for p in parties]
            for p, transport in zip(parties, records):
                p.transport = transport
            start = time.perf_counter()
            results = run_parties(lambda p: operation(p, shares[p.party], name, shift))
            synchronize(device)
            local_samples.append((time.perf_counter()-start)*1000)
            for p, transport in zip(parties, records):
                p.transport = transport.underlying
            opened = validate(results)
    phase_samples, party_samples = [], []
    for p in parties:
        phases, totals = [], []
        for repeat in range(repeats+1):
            replay = ReplayTransport(records[p.party].transcript, check_outgoing=repeat == 0)
            local = base(p.party, replay, p.seeds, device=device, bits=bits,
                         binary_bits=binary_bits, prg=prg)
            local.counters = dict(starts[p.party])
            replay.start(synchronize=lambda: synchronize(device))
            result = operation(local, shares[p.party], name, shift)
            replay.finish()
            replay.assert_consumed()
            if any(not torch.equal(a,b) for a,b in zip(result.parts, results[p.party].parts)):
                raise AssertionError('replay shares differ from checked fresh execution')
            if repeat:
                phases.append(replay.phase_times)
                totals.append(sum(replay.phase_times)*1000)
        phase_samples.append(phases)
        party_samples.append(totals)
    estimate = float(np.max(np.median(np.asarray(phase_samples), axis=1), axis=0).sum()*1000)
    timings, errors = baselines(floats, name, device, repeats)
    return dict(**range_metadata, operation=name, m=m,k=k,n=n, device=str(device), ring_bits=bits,
                binary_bits=binary_bits, prg=prg, fused_runtime=fused, fractional_bits=shift,
                prepared_weights=prepared, weight_storage=weight_storage,
                integer_reference_contract='signed modular ring product; truncation and ReLU follow ring wrapping',
                exact_integer_match=True, max_float_error=float((opened.cpu().float()/decoded_scale-plain(floats,name)).abs().max()),
                setup_all_parties_ms=setup_ms, local_four_party_ms=statistics.median(local_samples),
                local_samples_ms=local_samples, verification_opening_samples_ms=opening_samples,
                replay_party_samples_ms=party_samples, replay_phase_samples_seconds=phase_samples,
                ideal_network_phase_max_ms=estimate,
                ideal_timing_kind='estimate: sum of per-phase maxima of isolated-party median times',
                rounds=len(records[0].transcript),
                payload_bytes=sum(sum(map(len,sent.values())) for r in records for sent,_ in r.transcript),
                baselines=timings, baseline_max_float_errors=errors,
                ideal_slowdowns={label:estimate/ms for label,ms in timings.items()},
                samples=repeats, protocol_warmups=1, replay_warmups_per_party=1,
                baseline_samples=max(20,repeats), baseline_warmups=3,
                timing_boundary='operation entry through final protocol consistency check and GPU synchronization; no output opening',
                methodology='Fresh-mask checked four-party execution; isolated fixed-fixture replay retains PRG, hashes and host staging; warmup also checks outgoing transcript bytes. Sharing and output opening are excluded from operation latency and reported separately. No network latency or bandwidth cost. Float baselines perform the corresponding clear operation; the truncation baseline rounds decoded values down to Q12.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--m', type=int,nargs='+',default=[1,16,128])
    parser.add_argument('--k', type=int,nargs='+',default=[128,512,1024,2048])
    parser.add_argument('--n', type=int,nargs='+')
    parser.add_argument('--bits', type=int,choices=[32,64],default=64)
    parser.add_argument('--binary-bits', type=int,choices=[32,64],default=64)
    parser.add_argument('--fused', action='store_true')
    parser.add_argument('--prepared', action='store_true')
    parser.add_argument('--weight-storage', choices=['limbs','words32'], default='limbs')
    parser.add_argument('--prg', choices=['aes','shake'],default='shake')
    parser.add_argument('--operation', choices=['matmul','trunc','matmul_relu'],default='matmul')
    parser.add_argument('--repeats', type=int,default=5)
    parser.add_argument('--output', type=Path,default=Path('experiments/results/kernels/primitives.json'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    metadata = provenance()
    metadata['source_sha256']['experiments/benchmarks/kernels/primitives.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = dict(provenance=metadata, blas_threadpools=threadpool_info(), torch_threads=torch.get_num_threads(),
                  gpu=torch.cuda.get_device_name(args.device) if torch.device(args.device).type=='cuda' else None, results=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for m in args.m:
        for k in args.k:
            for n in args.n or [k]:
                row=run_case(m,k,n,args.device,args.repeats,args.bits,args.binary_bits,args.fused,args.prg,args.operation,
                             prepared=args.prepared, weight_storage=args.weight_storage)
                report['results'].append(row)
                args.output.write_text(json.dumps(report,indent=2)+'\n')
                summary = {key:row[key] for key in
                           ('operation','m','k','n','ring_bits','binary_bits','prg',
                            'exact_integer_match','accumulator_fits_signed_ring',
                            'maximum_accumulator_before_ring_wrap','max_float_error',
                            'ideal_network_phase_max_ms','rounds','payload_bytes','ideal_slowdowns')}
                print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1,user_api='blas'):
        main()
