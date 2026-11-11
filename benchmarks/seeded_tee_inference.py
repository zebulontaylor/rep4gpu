"""Measure ring64 FFN traffic with a seeded SwiGLU helper simulation.

All four roles and the helper execute in one ordinary CUDA process. There is
no enclave, protected input channel, or activation privacy. Live elapsed times
measure colocated execution and exclude a physical network and TEE transitions.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
from pathlib import Path
import secrets
import statistics
import time

import torch

from benchmarks.device_inference import phase_payloads
from benchmarks.numerical_inference_common import ForwardOracle, fixture, metrics, reference
from benchmarks.seeded_tee_simulation import clear_swiglu, infer, seeded_tee_party


def oracle(integers, fractional_bits=24, tee_down=False):
    """Bound raw projections and propagate the helper's exact rounded output."""
    runtime = ForwardOracle(probabilistic=not tee_down)
    x, gate, gate_bias, down, down_bias, value, value_bias = map(runtime.public, integers)
    scale = 1 << fractional_bits

    def projection(weight, bias):
        raw = runtime.add(runtime.matmul(x, weight), runtime.mul_public(bias, scale))
        result = raw.lo >> fractional_bits
        if bool(((result < -4*scale) | (result > 4*scale)).any()):
            raise ValueError('fixture activation input exceeds [-4,4]')
        return raw.lo

    gate_raw, value_raw = projection(gate, gate_bias), projection(value, value_bias)
    hidden = runtime.public(clear_swiglu(gate_raw, value_raw, fractional_bits))
    output = runtime.add(runtime.trunc(runtime.matmul(hidden, down), fractional_bits), down_bias)
    return output, dict(maximum_pretruncation_abs_bound=runtime.maximum,
                        strict_probabilistic_accumulator_limit=2**62,
                        maximum_rounding_interval_width=int((output.hi-output.lo).max()),
                        activation_input_bound=4,
                        range_validation='public fixture only; not a private range check')


def run_case(batch=1, width=2048, hidden=None, out=None, repeats=3, tee_down=False):
    """Record fresh checked forwards, excluding input sharing and graph setup."""
    from rep4gpu.device_transport import DeviceIdealEndpoint, DeviceRecorder
    from rep4gpu.runtime import make_seeds
    from rep4gpu.transport import IdealNetwork

    if repeats < 1:
        raise ValueError('repeats must be positive')
    hidden, out = hidden or 4*width, out or width
    floats, integers = fixture('swiglu', batch, width, hidden, out)
    interval, proof = oracle(integers, tee_down=tee_down)
    truth = reference(floats, 'swiglu')
    network, seeds = IdealNetwork(timeout=600), make_seeds()
    helper_party = seeded_tee_party(secrets.token_bytes(32))
    options = dict(device='cuda:0', bits=64, binary_bits=64, optimized=True,
                   gpu_open=True, rounding='probabilistic', sparse_products=True)
    parties = [helper_party(i, DeviceIdealEndpoint(network.endpoint(i), 'cuda:0'),
                           {j: seed for j, seed in seeds.items() if j != i}, **options)
               for i in range(4)]

    def check(outputs):
        result = outputs[0].cpu()
        if not all(torch.equal(output.cpu(), result) for output in outputs):
            raise AssertionError('party disagreement')
        if not bool(((result >= interval.lo) & (result <= interval.hi)).all()):
            raise AssertionError('integer interval mismatch')
        return result

    with ThreadPoolExecutor(4) as pool:
        def parallel(function):
            def guarded(party):
                try:
                    return function(party)
                except BaseException as error:
                    network.abort(error)
                    raise
            return list(pool.map(guarded, parties))

        shares = parallel(lambda p: [p.input(0 if i == 0 else 1,
                         v if p.party == (0 if i == 0 else 1) else None, v.shape)
                         for i, v in enumerate(integers)])
        plans = [p.prepare_inference(lambda x, p=p: infer(p, [x, *shares[p.party][1:]], tee_down=tee_down),
                  shares[p.party][0], check_at_send=True) for p in parties]
        execute = lambda p: plans[p.party].run(shares[p.party][0])
        check(parallel(execute))
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            check(parallel(execute))
            torch.cuda.synchronize()
            samples.append((time.perf_counter()-start)*1000)
        records = [DeviceRecorder(p.transport) for p in parties]
        for party, record in zip(parties, records):
            party.transport = record
        output = check(parallel(execute))
        for party, record in zip(parties, records):
            party.transport = record.underlying

    outgoing = [sum(sum(len(v) for v in frame.outgoing.values()) for frame in r.transcript)
                for r in records]
    rounds = len(records[0].transcript)
    if rounds != (9 if tee_down else 10) or any(len(r.transcript) != rounds for r in records):
        raise AssertionError('unexpected FFN communication schedule')
    activation_payload = sum(sum(len(v) for v in frame.outgoing.values())
                             for r in records for frame in r.transcript[2:4])
    if activation_payload != 32*batch*hidden+32:
        raise AssertionError('seeded helper boundary payload differs from its contract')
    result = dict(kind='swiglu', batch=batch, width=width, hidden=hidden, out=out,
                  ring_bits=64, fractional_bits=24, strassen=False,
                  activation_backend='public_fixture_seeded_tee_swiglu',
                  enclave=False, activation_privacy=False,
                  input_and_model_distribution_included=False,
                  helper_input_rounding='floor', helper_output_rounding='nearest_even',
                  down_projection_rounding='floor' if tee_down else 'probabilistic',
                  tee_down=tee_down,
                  party_agreement=True, integer_oracle_passed=True, range_proof=proof,
                  fresh_live_forwards=repeats+2, rounds=rounds,
                  payload_bytes=sum(outgoing), payload_bytes_per_token=sum(outgoing)/batch,
                  outgoing_bytes_per_party=outgoing,
                  activation_boundary_payload_bytes=activation_payload,
                  remaining_payload_bytes=sum(outgoing)-activation_payload,
                  live_colocated_samples_ms=samples,
                  live_colocated_median_ms=statistics.median(samples),
                  **phase_payloads(records), **metrics(output, truth, interval))
    del plans, parties, records, shares
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, nargs='+', default=[1, 8])
    parser.add_argument('--width', type=int, default=2048)
    parser.add_argument('--hidden', type=int)
    parser.add_argument('--out', type=int)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--tee-down', action='store_true',
                        help='also rescale the down projection inside the helper')
    parser.add_argument('--output', type=Path, default=Path('results/seeded_tee_inference.json'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = dict(gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  methodology='Public-fixture simulation in ordinary CUDA; no enclave, protected channels, or activation privacy. Global application payload includes all four roles and tags, excludes initial input/model distribution. Colocated timing is not network or TEE latency.',
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                      Path(__file__), Path('benchmarks/seeded_tee_simulation.py'),
                      Path('benchmarks/numerical_inference_common.py'))}, results=[])
    for batch in args.batch:
        row = run_case(batch, args.width, args.hidden, args.out, args.repeats, args.tee_down)
        baseline_path = Path(f'results/numerical_inference_b{batch}.json')
        if baseline_path.exists():
            baseline = next((r for r in json.loads(baseline_path.read_text())['results']
                             if r['kind'] == 'swiglu' and all(r[k] == row[k]
                             for k in ('batch', 'width', 'hidden', 'out'))), None)
            if baseline is not None:
                row['numerical_baseline_payload_bytes'] = baseline['payload_bytes']
                row['payload_reduction_fraction'] = 1-row['payload_bytes']/baseline['payload_bytes']
        report['results'].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k: row[k] for k in ('batch', 'width', 'hidden', 'out',
                         'rounds', 'payload_bytes', 'max_error', 'party_agreement')}), flush=True)


if __name__ == '__main__':
    main()
