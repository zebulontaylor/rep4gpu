"""Measure recurring dense private LUT costs and activation grid accuracy.

Run from the repository root. This benchmark begins with already-quantized
secret indices: online quantization and activation range checks are excluded.
Consequently its recurring total is a lower bound for a complete activation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import torch

from rep4gpu.private_lut import prepare_lut
from rep4gpu.runtime import Party, make_seeds
from rep4gpu.transport import IdealNetwork


SIZE = 4096


def table_values(kind, fractional_bits):
    """Encode a public 12-bit activation table with nearest-even outputs."""
    column = torch.arange(SIZE, dtype=torch.float64)
    if kind == 'silu':
        values = torch.nn.functional.silu(-4+column/512)
    elif kind == 'exp':
        values = torch.exp(-column/512)
    else:
        raise ValueError('private activation table must be silu or exp')
    return torch.round(values*(1 << fractional_bits)).long()


def quality(kind, fractional_bits, elements=65537):
    """Measure table quantization on encoded inputs, including both endpoints."""
    scale = 1 << fractional_bits
    bounds = (-4., 4.) if kind == 'silu' else (-8., 0.)
    encoded = torch.round(torch.linspace(*bounds, elements, dtype=torch.float64)*scale).long()
    x = encoded.double()/scale
    if kind == 'silu':
        index = ((encoded+4*scale)*512)//scale
        truth = torch.nn.functional.silu(x)
        mapping = 'floor((x+4)*512), clamped to [0,4095]'
    else:
        index = ((-encoded)*512)//scale
        truth = x.exp()
        mapping = 'floor(-x*512), clamped to [0,4095]'
    index = index.clamp(0, SIZE-1)
    actual = table_values(kind, fractional_bits)[index].double()/scale
    error = actual-truth
    return dict(kind=kind, fractional_bits=fractional_bits, elements=elements,
                domain=list(bounds), index_mapping=mapping,
                max_abs_error=float(error.abs().max()),
                rmse=float(error.square().mean().sqrt()),
                max_relative_error=float((error.abs()/truth.abs().clamp_min(1e-12)).max()),
                endpoint_errors=[float(error[0]),float(error[-1])])


def run_case(kind, bits=64, elements=32, backend='cpu'):
    fractional_bits = 14 if bits == 32 else 24
    public_values = table_values(kind, fractional_bits)
    indices = torch.round(torch.linspace(0, SIZE-1, elements, dtype=torch.float64)).long()
    network, seeds = IdealNetwork(timeout=180), make_seeds()

    def work(role):
        try:
            endpoint = network.endpoint(role)
            selected = {i:s for i,s in seeds.items() if i != role}
            if backend == 'cuda':
                from rep4gpu.device_inference import DeviceInferenceParty
                from rep4gpu.device_transport import DeviceIdealEndpoint
                runtime = DeviceInferenceParty(role, DeviceIdealEndpoint(endpoint, 'cuda:0'),
                    selected, device='cuda:0', bits=bits, binary_bits=bits,
                    optimized=True, gpu_open=True, rounding='floor')
            else:
                runtime = Party(role, endpoint, selected, bits=bits, binary_bits=bits)
            index = runtime.input(0, indices if role == 0 else None, indices.shape)
            start, rounds = endpoint.bytes_sent, endpoint.round
            material = prepare_lut(runtime, public_values, index.shape)
            preparation, preparation_rounds = endpoint.bytes_sent-start, endpoint.round-rounds
            memory = material.local_storage_bytes
            start, rounds = endpoint.bytes_sent, endpoint.round
            output = material.lookup(index)
            online, online_rounds = endpoint.bytes_sent-start, endpoint.round-rounds
            assert material.consumed
            # Validation opening is outside the measured online phase.
            opened = runtime.open(output).cpu()
            return opened, preparation, online, preparation_rounds, online_rounds, memory
        except BaseException as error:
            network.abort(error)
            raise

    with ThreadPoolExecutor(4) as pool:
        outputs = list(pool.map(work, range(4)))
    expected = public_values[indices]
    assert all(torch.equal(row[0], expected) for row in outputs)
    preparation = sum(row[1] for row in outputs)
    online = sum(row[2] for row in outputs)
    assert preparation == 6*elements*SIZE*(bits//8)+192
    assert online == 4*elements*(bits//8)+128
    return dict(kind=kind, backend=backend, elements=elements, ring_bits=bits,
                fractional_bits=fractional_bits, lut_bits=12, lut_entries=SIZE,
                party_agreement=True, preprocessing_recurrence='fresh for every scalar lookup',
                preprocessing_bytes=preparation, online_bytes=online,
                recurring_total_bytes=preparation+online,
                amortized_preprocessing_bytes_per_lookup=preparation/elements,
                amortized_online_bytes_per_lookup=online/elements,
                amortized_recurring_total_bytes_per_lookup=(preparation+online)/elements,
                asymptotic_recurring_bytes_per_lookup=(6*SIZE+4)*(bits//8),
                preprocessing_rounds=outputs[0][3], online_rounds=outputs[0][4],
                local_preprocessing_storage_bytes=outputs[0][5],
                local_preprocessing_storage_bytes_per_lookup=outputs[0][5]/elements,
                public_table_bytes=public_values.numel()*public_values.element_size(),
                public_table_sha256=hashlib.sha256(public_values.numpy().tobytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--elements', type=int, default=32)
    parser.add_argument('--backend', nargs='+', choices=['cpu','cuda'], default=['cpu','cuda'])
    parser.add_argument('--bits', nargs='+', type=int, choices=[32,64], default=[32,64])
    parser.add_argument('--kind', nargs='+', choices=['silu','exp'], default=['silu','exp'])
    parser.add_argument('--output', type=Path, default=Path('results/private_lut.json'))
    args = parser.parse_args()
    if args.elements < 1:
        parser.error('--elements must be positive')
    torch.set_num_threads(1)
    paths = [Path('rep4gpu')/(name+'.py') for name in
             ('private_lut','runtime','device_runtime','device_open','device_transport')]
    paths.append(Path('benchmarks/private_lut.py'))
    report = dict(methodology=(
        'Four live Rep4 roles, no additional dealer. One fresh secret shifted table per '
        'scalar lookup. Public function tables and established seed provisioning are '
        'one-time inputs; six table-sized checked corrections recur for each lookup. '
        'All payload bytes sum the four roles and include consistency hashes. '
        'Lookup begins with secret indices already quantized into the 12-bit grid. '
        'Quantization and input range checks, input sharing, and validation output '
        'opening are excluded; the recurring total is therefore a lower bound for '
        'complete activation communication. Batch amortization divides actual recurring '
        'preprocessing plus online bytes by the lookup count, not by a future reuse count. '
        'Dense truth-table reference, not compact FSS key generation or a network-latency measurement.'),
        source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        torch_version=torch.__version__,
        dense_accuracy=[quality(kind,14 if bits == 32 else 24) for bits in args.bits for kind in args.kind],
        results=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for backend in args.backend:
        for bits in args.bits:
            for kind in args.kind:
                row = run_case(kind,bits,args.elements,backend)
                report['results'].append(row)
                args.output.write_text(json.dumps(report,indent=2)+'\n')
                print(json.dumps(row),flush=True)


if __name__ == '__main__':
    main()
