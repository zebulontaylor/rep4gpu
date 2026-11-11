"""Measure witness improvements against preserved source on the same runtime."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch

from rep4gpu.runtime import Party, make_seeds
from rep4gpu.transport import IdealNetwork


BASELINE = Path('results/witness_optimization_baseline_sources')


def implementations(version):
    if version == 'after':
        from rep4gpu import witness_activations as activation, witness_prover as prover
        return activation, prover
    modules = {}
    for name in ('numerical', 'witness_activations', 'witness_prover'):
        module_name = 'rep4gpu._optimization_before_' + name
        spec = importlib.util.spec_from_file_location(module_name, BASELINE / (name+'.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    for name in ('witness_activations', 'witness_prover'):
        modules[name].NumericalFixedPoint = modules['numerical'].NumericalFixedPoint
    return modules['witness_activations'], modules['witness_prover']


def run_case(version, backend, kind, elements=257, rows=2, width=256):
    activation, prover = implementations(version)
    scale = 1 << 24
    shape = (rows, width) if kind in ('rmsnorm', 'softmax') else (elements,)
    x = torch.round(torch.linspace(-4, 4, torch.Size(shape).numel(), dtype=torch.float64)*scale).long().reshape(shape)
    if kind == 'rmsnorm':
        program = activation.rmsnorm_program()
        claims = prover.rmsnorm_witnesses(x)
        truth = x.double()/scale
        truth = truth/(truth.square().mean(-1, keepdim=True)+1e-5).sqrt()
    elif kind == 'softmax':
        program = activation.softmax_program()
        claims = prover.softmax_witnesses(x)
        truth = torch.softmax(x.double()/scale, dim=-1)
    else:
        program = activation.numerical_program(kind)
        claims = prover.numerical_witnesses(kind, {'x': x})
        truth = getattr(torch.nn.functional, kind)(x.double()/scale)
    network, seeds = IdealNetwork(timeout=180), make_seeds()

    def work(role):
        try:
            endpoint = network.endpoint(role)
            if backend == 'cuda':
                from rep4gpu.device_inference import DeviceInferenceParty
                from rep4gpu.device_transport import DeviceIdealEndpoint
                p = DeviceInferenceParty(role, DeviceIdealEndpoint(endpoint, 'cuda:0'),
                    {i:s for i,s in seeds.items() if i != role}, device='cuda:0',
                    bits=64, binary_bits=64, optimized=True, gpu_open=True,
                    rounding='floor', sparse_products=True)
            else:
                p = Party(role, endpoint, {i:s for i,s in seeds.items() if i != role})
            share = p.input(0, x if role == 0 else None, x.shape)
            start = endpoint.bytes_sent
            witnesses = activation.input_witnesses(p, program, 1,
                {name:v.shape for name,v in claims.items()}, claims if role == 1 else None)
            ingestion = endpoint.bytes_sent-start
            profile, depth = defaultdict(lambda: {'bytes':0, 'calls':0}), [0]
            for name in ('mul', 'matmul', 'trunc', 'sign_bit', 'lt_public', 'open'):
                original = getattr(p, name)
                def wrapped(*args, _name=name, _original=original, **kwargs):
                    outer = depth[0] == 0
                    before = endpoint.bytes_sent
                    depth[0] += 1
                    try:
                        return _original(*args, **kwargs)
                    finally:
                        depth[0] -= 1
                        if outer:
                            profile[_name]['bytes'] += endpoint.bytes_sent-before
                            profile[_name]['calls'] += 1
                setattr(p, name, wrapped)
            start, rounds = endpoint.bytes_sent, endpoint.round
            session = activation.WitnessSession(p)
            session.evaluate(program, {'x':share}, witnesses)
            result, = session.finish()
            execution, rounds = endpoint.bytes_sent-start, endpoint.round-rounds
            profile_snapshot = {name: dict(counts) for name, counts in profile.items()}
            # Save accounting before the output opening used only for validation.
            opened = p.open(result).cpu()
            return opened, ingestion, execution, rounds, profile_snapshot
        except BaseException as error:
            network.abort(error)
            raise
    with ThreadPoolExecutor(4) as pool:
        outputs = list(pool.map(work, range(4)))
    actual = outputs[0][0]
    assert all(torch.equal(actual, row[0]) for row in outputs)
    error = actual.double()/scale-truth
    profile = {}
    for name in set().union(*(row[4] for row in outputs)):
        profile[name] = {'bytes':sum(row[4].get(name, {}).get('bytes', 0) for row in outputs),
                         'calls_per_party':outputs[0][4].get(name, {}).get('calls', 0)}
    return dict(version=version, backend=backend, kind=kind, shape=list(shape),
                elements=x.numel(), fractional_bits=24, rounding='floor',
                witness_ingestion_bytes=sum(row[1] for row in outputs),
                execution_bytes=sum(row[2] for row in outputs),
                total_bytes=sum(row[1]+row[2] for row in outputs),
                execution_rounds=outputs[0][3], max_abs_error=float(error.abs().max()),
                rmse=float(error.square().mean().sqrt()), party_agreement=True,
                operation_profile=profile)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', nargs='+', choices=['cpu','cuda'], default=['cpu','cuda'])
    parser.add_argument('--version', nargs='+', choices=['before','after'], default=['before','after'])
    parser.add_argument('--kind', nargs='+', choices=['silu','gelu','rmsnorm','softmax'], default=['silu','gelu','rmsnorm','softmax'])
    parser.add_argument('--elements', type=int, default=257)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--rows', type=int, default=2)
    parser.add_argument('--output', type=Path, default=Path('results/witness_optimization.json'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    sources = [BASELINE/(n+'.py') for n in ('numerical','witness_activations','witness_prover')]
    sources += [Path('rep4gpu')/(n+'.py') for n in ('numerical','witness_activations','witness_prover','runtime','device_inference','device_runtime','conversion','exact_truncation','binary_aggregation','upper_zero','bounded_activation')]
    sources.append(Path(__file__))
    report = dict(methodology='Same four-party runtime before/after, ring64 Q24 exact floor. '
        'Global transmitted bytes include witness input, all constraints and acceptance opening; '
        'exclude already shared x and validation output opening. CUDA uses optimized '
        'DeviceInferenceParty, CPU uses reference Party. No distributed timing claim. '
        'Top-level operation profiles exclude nested operations; direct fused reductions '
        'may appear only in total. RMS mean-square rounding changes with fused reduction.',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},results=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for backend in args.backend:
        for kind in args.kind:
            for version in args.version:
                row = run_case(version,backend,kind,args.elements,args.rows,args.width)
                report['results'].append(row)
                args.output.write_text(json.dumps(report,indent=2)+'\n')
                print(json.dumps(row),flush=True)


if __name__ == '__main__':
    main()
