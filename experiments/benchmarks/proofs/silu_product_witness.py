"""Compare exact SiLU with and without batched product witnesses."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path

import torch

from rep4gpu.protocol.runtime import Party, make_seeds
from rep4gpu.protocol.transport import IdealNetwork
from rep4gpu.proofs.silu_product_witness import silu_product_witness_program, silu_product_witnesses
from rep4gpu.proofs.witness_activations import WitnessSession, input_witnesses, numerical_program
from rep4gpu.proofs.witness_prover import numerical_witnesses


def measure(x, backend, variant):
    if variant == 'rational34':
        from rep4gpu.proofs.silu_rational import rational_silu_program, rational_silu_witnesses
        program, claims = rational_silu_program(), rational_silu_witnesses({'x': x})
    elif variant == 'batched_products':
        program, claims = silu_product_witness_program(), silu_product_witnesses({'x': x})
    else:
        program, claims = numerical_program('silu'), numerical_witnesses('silu', {'x': x})
    network, seeds = IdealNetwork(timeout=300), make_seeds()

    def work(role):
        try:
            ep = network.endpoint(role)
            selected = {i: seed for i, seed in seeds.items() if i != role}
            if backend == 'cuda':
                from rep4gpu.models.device_inference import DeviceInferenceParty
                from rep4gpu.protocol.device_transport import DeviceIdealEndpoint
                rt = DeviceInferenceParty(role, DeviceIdealEndpoint(ep, 'cuda:0'), selected,
                    device='cuda:0', bits=64, binary_bits=64, optimized=True,
                    gpu_open=True, rounding='floor', sparse_products=True)
            else:
                rt = Party(role, ep, selected)
            shared = rt.input(0, x if role == 0 else None, x.shape)
            start = ep.bytes_sent
            witnesses = input_witnesses(rt, program, 1,
                {name: value.shape for name, value in claims.items()},
                claims if role == 1 else None)
            ingestion = ep.bytes_sent-start
            session = WitnessSession(rt)
            session.evaluate(program, {'x': shared}, witnesses)
            output, = session.finish()
            total = ep.bytes_sent-start
            return rt.open(output).cpu(), ingestion, total
        except BaseException as error:
            network.abort(error)
            raise

    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(work, range(4)))
    expected = claims['output']
    assert all(torch.equal(row[0], expected) for row in results)
    actual = results[0][0]
    error = (actual.double()/(1 << 24)-torch.nn.functional.silu(x.double()/(1 << 24))).abs()
    total = sum(row[2] for row in results)
    return dict(program=variant,
        backend=backend, elements=x.numel(), fractional_bits=24,
        witness_input_bytes=sum(row[1] for row in results), total_bytes=total,
        under_10mb=total < 10_000_000, max_abs_error=error.max().item(),
        matches_program_integer_oracle=True,
        exact_output_matches_default=torch.equal(actual, numerical_witnesses('silu', {'x': x})['output']),
        product_false_acceptance_bound=('exact' if variant == 'exact_products' else
            '2^-128 per batch under the challenge PRG assumption'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--elements', type=int, default=14336)
    parser.add_argument('--backend', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--bound', type=float, default=4)
    parser.add_argument('--output', type=Path, default=Path('experiments/results/proofs/silu_product_witness.json'))
    args = parser.parse_args()
    if args.elements < 2 or not 0 < args.bound <= 16:
        parser.error('elements must be at least two and bound must lie in (0,16]')
    torch.set_num_threads(1)
    x = torch.round(torch.linspace(-args.bound, args.bound, args.elements,
                                   dtype=torch.float64)*(1 << 24)).long()
    files = [Path(__file__).relative_to(Path.cwd()), *map(Path, (
        'src/rep4gpu/proofs/product_witness.py', 'src/rep4gpu/proofs/silu_product_witness.py', 'src/rep4gpu/proofs/silu_rational.py',
        'src/rep4gpu/arithmetic/silu_exp.py', 'src/rep4gpu/proofs/witness_activations.py',
        'src/rep4gpu/proofs/witness_prover.py', 'src/rep4gpu/arithmetic/binary_aggregation.py',
        'src/rep4gpu/arithmetic/exact_truncation.py', 'src/rep4gpu/arithmetic/upper_zero.py',
        'src/rep4gpu/protocol/runtime.py', 'src/rep4gpu/protocol/device_runtime.py'))]
    report = dict(methodology='Four live Rep4 roles, Q24 exact floors. Global bytes include '
        'all witness input, verification, public challenge generation, and acceptance. '
        'Already-shared x and validation output opening are excluded. Every role output '
        'must equal its program plaintext oracle. The rational34 variant changes the '
        'approximation; the other variants retain the default numerical formula. No latency claim.',
        fixture_domain=[-args.bound, args.bound], verified_input_domain=[-16, 16],
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, results=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for variant in ('exact_products', 'batched_products', 'rational34'):
        row = measure(x, args.backend, variant)
        report['results'].append(row)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
