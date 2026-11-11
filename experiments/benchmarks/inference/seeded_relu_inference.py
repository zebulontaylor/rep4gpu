"""Measure ring32 Rep4 FFN traffic with seeded public-fixture helpers.

Run from the repository root with CUDA:
python -m experiments.benchmarks.inference.seeded_relu_inference --batch 1 128 --repeats 3

Both modes reconstruct activations in ordinary CUDA, without an enclave or
protected channels. Input/model distribution and setup are excluded. Replay
reuses a public transcript solely to estimate local processing time.
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

from experiments.benchmarks.inference.device_inference import fixture_bounds, phase_estimate, phase_payloads, validate_outputs
from experiments.benchmarks.inference.inference import checked_add, checked_matmul, fixture, maximum_abs, plain


def oracle(integers, fractional_bits=14, tee_down=False):
    """Exact helper ReLU followed by floor or bounded stochastic rescaling."""
    proof, _, _ = fixture_bounds(integers, 32, fractional_bits, 'floor')
    x, w1, b1, w2, b2 = integers
    raw1 = checked_add(checked_matmul(x, w1), b1 * (1 << fractional_bits))
    hidden = (raw1 >> fractional_bits).clamp_min(0)
    raw2 = checked_matmul(hidden, w2)
    if not tee_down and maximum_abs(raw2) >= 1 << 30:
        raise ValueError('output accumulator exceeds strict ring32 probabilistic bound')
    lower = checked_add(raw2 >> fractional_bits, b2)
    upper = lower if tee_down else checked_add(
        checked_add(raw2, torch.tensor((1 << fractional_bits) - 1)) >> fractional_bits, b2)
    if int(lower.min()) < -(1 << 31) or int(upper.max()) >= 1 << 31:
        raise ValueError('rounded output interval exceeds signed ring32')
    proof.update(activation_rounding='floor', output_rounding='floor' if tee_down else 'probabilistic',
                 maximum_output_rounding_interval_width=int((upper-lower).max()),
                 second_accumulator_max_abs=maximum_abs(raw2))
    return lower, upper, proof


def run_case(batch=1, width=2048, hidden=8192, out=2048, repeats=3, tee_down=False):
    from experiments.benchmarks.activations.seeded_relu_simulation import infer, seeded_relu_party
    from rep4gpu.protocol.device_transport import DeviceIdealEndpoint, DeviceRecorder, DeviceReplay
    from rep4gpu.protocol.runtime import make_seeds
    from rep4gpu.protocol.transport import IdealNetwork

    if min(batch, width, hidden, out, repeats) < 1:
        raise ValueError('dimensions and repeats must be positive')
    device, fractional_bits = torch.device('cuda:0'), 14
    floats, integers, _ = fixture(batch, width, hidden, out, fractional_bits)
    lower, upper, proof = oracle(integers, fractional_bits, tee_down)
    reference = plain(floats)
    network, seeds = IdealNetwork(timeout=600), make_seeds()
    kind = seeded_relu_party(secrets.token_bytes(32))
    options = dict(device=device, bits=32, binary_bits=32, optimized=True, gpu_open=True,
                   rounding='probabilistic', sparse_products=True, narrow_products=True)
    parties = [kind(p, DeviceIdealEndpoint(network.endpoint(p), device),
                    {e: s for e, s in seeds.items() if e != p}, **options) for p in range(4)]
    accuracy = []

    def check(outputs, recorded=None):
        actual = validate_outputs(outputs, lower, upper, recorded if recorded is not None else
                                  (lower if tee_down else None))
        decoded = actual.float() / (1 << fractional_bits)
        accuracy.append(dict(max_logit_error=float((decoded-reference).abs().max()),
                             logit_rmse=float((decoded-reference).square().mean().sqrt()),
                             top1_agreement=float((decoded.argmax(-1)==reference.argmax(-1)).float().mean())))
        return actual.clone()

    def make_plan(party, values):
        return party.prepare_inference(
            lambda x: infer(party, [x, *values[1:]], fractional_bits, tee_down=tee_down),
            values[0], check_at_send=True)

    with ThreadPoolExecutor(4) as pool:
        def parallel(function):
            def guarded(p):
                try:
                    return function(p)
                except BaseException as error:
                    network.abort(error)
                    raise
            return list(pool.map(guarded, parties))

        def distribute(p):
            values = [p.input(owner, value if p.party == owner else None, value.shape)
                      for owner, value in zip((0, 1, 1, 1, 1), integers)]
            for index in (1, 3):
                values[index] = p.prepare_weight_share(values[index], storage='words32')
            return values

        shares = parallel(distribute)
        plans = [make_plan(p, shares[p.party]) for p in parties]
        execute = lambda p: plans[p.party].run(shares[p.party][0])
        check(parallel(execute))
        samples = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = parallel(execute)
            torch.cuda.synchronize()
            samples.append((time.perf_counter()-start)*1000)
            check(outputs)
        starts = [dict(p.counters) for p in parties]
        records = [DeviceRecorder(p.transport) for p in parties]
        for p, record in zip(parties, records):
            p.transport = record
        recorded = check(parallel(execute))
        for p, record in zip(parties, records):
            p.transport = record.underlying

    plans = None
    phase_samples = []
    for party in parties:
        local = kind(party.party, party.transport, party.seeds, **options)
        local._device_keys, local._batch_keys = party._device_keys, party._batch_keys
        plan = make_plan(local, shares[party.party])
        phases = []
        for repeat in range(repeats+1):
            replay = DeviceReplay(records[party.party].transcript, device=device,
                                  check_outgoing=repeat == 0, staging='device',
                                  incoming_buffers=plan.receive_buffers)
            local.transport, local.counters = replay, dict(starts[party.party])
            replay.start(synchronize=torch.cuda.synchronize)
            output = plan.run(shares[party.party][0])
            replay.finish()
            replay.assert_consumed()
            validate_outputs([output], lower, upper, recorded)
            if repeat:
                phases.append(replay.phase_times)
        phase_samples.append(phases)
        del plan, local, replay

    payloads = phase_payloads(records)
    phases = [sum(row) for row in zip(*payloads['outgoing_bytes_per_party_per_phase'])]
    expected_rounds = 8 if tee_down else 9
    if len(phases)-1 != expected_rounds:
        raise AssertionError('unexpected helper FFN exchange count')
    expected_activation = 12*batch*hidden+32
    if sum(phases[1:3]) != expected_activation:
        raise AssertionError('activation payload differs from seeded contract')
    expected_total = (36*batch*(hidden+out)+16*batch*out+576 if tee_down else
                      36*batch*hidden+72*batch*out+800)
    if sum(phases) != expected_total:
        raise AssertionError(f'FFN payload {sum(phases)} differs from {expected_total}')
    if tee_down and sum(phases[4:6]) != 12*batch*out+32:
        raise AssertionError('output rescaling payload differs from seeded contract')
    result = dict(batch=batch, width=width, hidden=hidden, out=out,
                  ring_bits=32, fractional_bits=fractional_bits, tee_down=tee_down,
                  enclave=False, activation_privacy=False, local_component_authentication=False,
                  input_and_model_distribution_included=False,
                  activation_backend='public_fixture_seeded_relu', weight_storage='words32',
                  party_agreement=True, integer_oracle_passed=True,
                  exact_integer_match=tee_down, fresh_live_forwards=repeats+2,
                  range_proof=proof, rounds=expected_rounds,
                  payload_bytes=sum(phases), payload_bytes_per_token=sum(phases)/batch,
                  below_500000_payload_bytes_per_token=sum(phases)/batch < 500000,
                  activation_boundary_payload_bytes=expected_activation,
                  outgoing_bytes_per_party=[sum(row) for row in payloads['outgoing_bytes_per_party_per_phase']],
                  live_colocated_samples_ms=samples, live_colocated_median_ms=statistics.median(samples),
                  ideal_network_phase_max_ms=phase_estimate(phase_samples),
                  replay_phase_samples_seconds=phase_samples, exact_recorded_replay_output_match=True,
                  fresh_accuracy_samples=accuracy,
                  max_logit_error=max(a['max_logit_error'] for a in accuracy),
                  max_logit_rmse=max(a['logit_rmse'] for a in accuracy),
                  minimum_top1_agreement=min(a['top1_agreement'] for a in accuracy),
                  **payloads)
    del parties, records, shares
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, nargs='+', default=[1, 128])
    parser.add_argument('--width', type=int, default=2048)
    parser.add_argument('--hidden', type=int, default=8192)
    parser.add_argument('--out', type=int, default=2048)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--mode', choices=['activation', 'both', 'all'], default='all')
    parser.add_argument('--output', type=Path, default=Path('experiments/results/activations/seeded_relu_inference.json'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    sources = [Path(__file__), Path('experiments/benchmarks/activations/seeded_relu_simulation.py'),
               Path('experiments/benchmarks/inference/inference.py'), Path('experiments/benchmarks/inference/device_inference.py'),
               *sorted(Path('src/rep4gpu').rglob('*.py'))]
    report = dict(gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  methodology='Public-fixture CUDA helper simulation, not a TEE. Global payload counts '
                              'every sent byte once, including protocol tags but excluding setup, '
                              'input/model distribution and future protected-channel overhead. '
                              'Live times are colocated. Isolated-party replay estimates exclude network '
                              'latency/bandwidth, enclave transitions and protected transfers.',
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  results=[])
    modes = [False, True] if args.mode == 'all' else [args.mode == 'both']
    for batch in args.batch:
        for tee_down in modes:
            row = run_case(batch, args.width, args.hidden, args.out, args.repeats, tee_down)
            report['results'].append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps({k: row[k] for k in ('batch', 'tee_down', 'payload_bytes_per_token',
                             'rounds', 'ideal_network_phase_max_ms', 'max_logit_error')}), flush=True)


if __name__ == '__main__':
    main()
