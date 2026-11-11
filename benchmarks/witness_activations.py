"""Compare actual Rep4 traffic and numerical error with shared witnesses.

Run from the repository root with ``python -m benchmarks.witness_activations``.
CPU IdealNetwork runs all four protocol roles and counts transmitted payloads,
including hashes. Timings describe this local test harness, not a deployment.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import torch

from benchmarks.numerical_primitives import IntegerOracle, fixture, reference
from rep4gpu.activations import FixedPoint
from rep4gpu.numerical import NumericalFixedPoint
from rep4gpu.runtime import Party, make_seeds
from rep4gpu.transport import IdealNetwork


def prover_witness(kind, x, value, fractional_bits):
    """Generate reciprocal claims from the same quantized forward formula.

    This runs outside MPC. The verifier receives only the shared claim and
    independently reconstructs its denominator from its existing shares.
    """
    from rep4gpu.witness_prover import numerical_witnesses
    inputs = {'x': x}
    if kind == 'swiglu':
        inputs['value'] = value
    return next(iter(numerical_witnesses(kind, inputs, fractional_bits).values()))


def dense_accuracy(kind, elements=65537, fractional_bits=24):
    """Compare formulas on a dense grid independently of protocol traffic."""
    x, value = fixture(kind, elements, fractional_bits)
    truth = reference(kind, x, value, fractional_bits)

    class WitnessOracle(NumericalFixedPoint):
        def reciprocal(self, denominator):
            claim = torch.round(self.scale**2 / denominator.lo.double()).long()
            return self.runtime.public(claim)

    rows = []
    for mode, cls in [('numerical', NumericalFixedPoint), ('witness', WitnessOracle)]:
        if mode == 'witness' and kind in ('silu', 'swiglu'):
            from rep4gpu.witness_prover import numerical_witnesses
            inputs = {'x': x, 'value': value} if kind == 'swiglu' else {'x': x}
            encoded = numerical_witnesses(kind, inputs, fractional_bits)['output']
            if kind == 'swiglu':
                encoded = (encoded*value) >> fractional_bits
            output = encoded.double()/(1 << fractional_bits)
            error = output-truth
            rows.append(dict(kind=kind, mode=mode, elements=elements,
                             fractional_bits=fractional_bits, max_abs_error=float(error.abs().max()),
                             rmse=float(error.square().mean().sqrt())))
            continue
        runtime = IntegerOracle()
        fp = cls(runtime, fractional_bits)
        args = [runtime.public(x)]
        if kind == 'swiglu':
            args.append(runtime.public(value))
        output = getattr(fp, kind)(*args).lo.double() / (1 << fractional_bits)
        error = output-truth
        rows.append(dict(kind=kind, mode=mode, elements=elements,
                         fractional_bits=fractional_bits, max_abs_error=float(error.abs().max()),
                         rmse=float(error.square().mean().sqrt())))
    return rows


def normalization_fixture(kind, width, fractional_bits=24, epsilon=1e-5):
    """Return two encoded rows and honest scalar witnesses for each row."""
    from rep4gpu.witness_prover import rmsnorm_witnesses, softmax_witnesses
    scale = 1 << fractional_bits
    x = torch.round(torch.linspace(-4, 4, 2*width, dtype=torch.float64)*scale).long().reshape(2, width)
    if kind == 'rmsnorm':
        witnesses = rmsnorm_witnesses(x, fractional_bits, epsilon)
        decoded = x.double()/scale
        truth = decoded / (decoded.square().mean(-1, keepdim=True)+epsilon).sqrt()
    elif kind == 'softmax':
        witnesses = softmax_witnesses(x, fractional_bits)
        truth = torch.softmax(x.double()/scale, dim=-1)
    else:
        raise ValueError('normalization must be rmsnorm or softmax')
    return x, witnesses, truth


def run_normalization(kind, width, fractional_bits=24):
    from rep4gpu.witness_activations import WitnessSession, rmsnorm_program, softmax_program
    x, witnesses, truth = normalization_fixture(kind, width, fractional_bits)
    program = rmsnorm_program() if kind == 'rmsnorm' else softmax_program()
    network, seeds = IdealNetwork(timeout=180), make_seeds()
    parties = [Party(i, network.endpoint(i), {j: s for j, s in seeds.items() if j != i}) for i in range(4)]

    def work(p):
        try:
            shared = p.input(0, x if p.party == 0 else None, x.shape)
            before = p.transport.bytes_sent
            claims = {name: p.input(0, value if p.party == 0 else None, value.shape)
                      for name, value in witnesses.items()}
            ingest = p.transport.bytes_sent-before
            before, rounds = p.transport.bytes_sent, p.transport.round
            start = time.perf_counter()
            session = WitnessSession(p, fractional_bits)
            session.evaluate(program, {'x': shared}, claims)
            output, = session.finish()
            elapsed = time.perf_counter()-start
            sent, rounds = p.transport.bytes_sent-before, p.transport.round-rounds
            return p.open(output), ingest, sent, rounds, elapsed
        except BaseException as error:
            network.abort(error)
            raise

    with ThreadPoolExecutor(4) as pool:
        outputs = list(pool.map(work, parties))
    expected = outputs[0][0]
    assert all(torch.equal(row[0], expected) for row in outputs)
    error = expected.double()/(1 << fractional_bits)-truth
    ingest, execution = sum(row[1] for row in outputs), sum(row[2] for row in outputs)
    return dict(kind=kind, mode='witness', rows=2, width=width, elements=x.numel(),
                fractional_bits=fractional_bits, ring_bits=64, rounding='floor',
                party_agreement=True, witness_elements=sum(v.numel() for v in witnesses.values()),
                witness_ingestion_bytes=ingest, evaluation_and_verification_bytes=execution,
                total_bytes=ingest+execution, evaluation_and_verification_rounds=outputs[0][3],
                max_abs_error=float(error.abs().max()), rmse=float(error.square().mean().sqrt()),
                local_four_party_seconds=max(row[4] for row in outputs),
                baseline='No existing RMSNorm/softmax protocol baseline; absolute cost only')


def run_case(kind, elements=17, fractional_bits=24, mode='witness'):
    if mode == 'witness':
        from rep4gpu.witness_activations import WitnessSession, numerical_program

    if kind not in ('reciprocal', 'silu', 'gelu', 'swiglu'):
        raise ValueError('unsupported benchmark function')
    if mode not in ('numerical', 'polynomial', 'mixed_polynomial', 'witness'):
        raise ValueError('unsupported benchmark mode')
    if mode == 'polynomial' and kind == 'reciprocal':
        raise ValueError('there is no FixedPoint reciprocal baseline')
    if mode == 'mixed_polynomial' and kind not in ('silu', 'gelu'):
        raise ValueError('mixed polynomial comparison supports SiLU and GELU')
    ring_bits = 32 if mode == 'mixed_polynomial' else 64
    if mode == 'mixed_polynomial':
        fractional_bits = 14
        from benchmarks.activation_approximation_quality import fit
        from benchmarks.activation_polynomial_protocol import polynomial, precision
        coefficients = fit(kind, 8 if kind == 'silu' else 10, 'mixed')
        internal_bits = precision(coefficients)
    x, value = fixture(kind, elements, fractional_bits)
    witness = prover_witness(kind, x, value, fractional_bits) if mode == 'witness' else None
    seeds, network = make_seeds(), IdealNetwork(timeout=180)
    parties = [Party(i, network.endpoint(i), {j: seed for j, seed in seeds.items() if j != i},
                     bits=ring_bits, binary_bits=ring_bits)
               for i in range(4)]

    def work(party):
        try:
            shared_x = party.input(0, x if party.party == 0 else None, x.shape)
            inputs = {'x': shared_x}
            if kind == 'swiglu':
                inputs['value'] = party.input(0, value if party.party == 0 else None, value.shape)
            start_bytes, start_rounds = party.transport.bytes_sent, party.transport.round
            if mode == 'witness':
                shared_witness = party.input(0, witness if party.party == 0 else None, witness.shape)
            ingest_bytes = party.transport.bytes_sent - start_bytes
            ingest_rounds = party.transport.round - start_rounds
            start_bytes, start_rounds = party.transport.bytes_sent, party.transport.round
            start = time.perf_counter()
            if mode == 'witness':
                session = WitnessSession(party, fractional_bits=fractional_bits)
                program = numerical_program(kind)
                session.evaluate(program, inputs, {next(iter(program.witness_bounds)): shared_witness})
                output, = session.finish()
            elif mode == 'mixed_polynomial':
                output = polynomial(party, shared_x, coefficients, internal_bits)
            else:
                fp = (NumericalFixedPoint if mode == 'numerical' else FixedPoint)(party, fractional_bits)
                if kind == 'swiglu':
                    output = fp.mul(fp.silu(shared_x), inputs['value'])
                else:
                    output = getattr(fp, kind)(shared_x)
            elapsed = time.perf_counter() - start
            execution_bytes = party.transport.bytes_sent - start_bytes
            execution_rounds = party.transport.round - start_rounds
            opened = party.open(output)
            return opened, ingest_bytes, execution_bytes, ingest_rounds, execution_rounds, elapsed
        except BaseException as error:
            network.abort(error)
            raise

    with ThreadPoolExecutor(4) as pool:
        outputs = list(pool.map(work, parties))
    expected = outputs[0][0]
    assert all(torch.equal(output[0], expected) for output in outputs)
    truth = reference(kind, x, value, fractional_bits)
    error = expected.double() / (1 << fractional_bits) - truth
    ingest = sum(row[1] for row in outputs)
    execution = sum(row[2] for row in outputs)
    return dict(kind=kind, mode=mode, elements=elements, fractional_bits=fractional_bits,
                ring_bits=ring_bits, rounding='floor', party_agreement=True,
                witness_elements=0 if witness is None else witness.numel(),
                witness_ingestion_bytes=ingest, evaluation_and_verification_bytes=execution,
                total_bytes=ingest+execution, witness_ingestion_rounds=outputs[0][3],
                evaluation_and_verification_rounds=outputs[0][4],
                max_abs_error=float(error.abs().max()), rmse=float(error.square().mean().sqrt()),
                local_four_party_seconds=max(row[5] for row in outputs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--elements', nargs='+', type=int, default=[17, 257])
    parser.add_argument('--kind', nargs='+', choices=['reciprocal', 'silu', 'gelu', 'swiglu'],
                        default=['reciprocal', 'silu', 'gelu', 'swiglu'])
    parser.add_argument('--fractional-bits', type=int, default=24)
    parser.add_argument('--normalization-widths', nargs='*', type=int, default=[16, 256])
    parser.add_argument('--output', type=Path, default=Path('results/witness_activations.json'))
    args = parser.parse_args()
    if any(n < 2 for n in args.elements):
        parser.error('--elements must be at least 2')
    torch.set_num_threads(1)
    sources = ['rep4gpu/witness_activations.py', 'rep4gpu/numerical.py',
               'rep4gpu/witness_prover.py',
               'rep4gpu/activations.py', 'rep4gpu/runtime.py', 'rep4gpu/conversion.py',
               'benchmarks/witness_activations.py', 'benchmarks/numerical_primitives.py',
               'benchmarks/activation_approximation_quality.py',
               'benchmarks/activation_polynomial_protocol.py']
    report = dict(methodology=(
        'Four live CPU Rep4 Party roles over IdealNetwork, exact floor truncation. '
        'Global bytes sum all outgoing payloads, including runtime consistency hashes. '
        'Inputs are already shared; witness input and all domain/residual verification '
        'and accept-bit opening are included. Final output opening is excluded for every '
        'mode. Witnesses use nearest reciprocal of the independently reproducible encoded '
        'denominator. References use torch float64 on encoded input fixtures. These are '
        'activation-level accuracy and communication measurements, not model accuracy '
        'or distributed latency. Numerical baselines retain caller-established domains; '
        'the witness path includes private domain checks. Polynomial denotes existing degree-ten FixedPoint methods; '
        'numerical denotes the more accurate NumericalFixedPoint formulas. Mixed polynomial '
        'uses the selected mixed degree-eight SiLU or degree-ten GELU schedule in ring32/Q14 '
        'with caller-established input bounds; its precision and range-check policy differ.'),
        source_sha256={path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in sources},
        torch_version=torch.__version__,
        dense_accuracy=[row for kind in args.kind for row in dense_accuracy(kind, fractional_bits=args.fractional_bits)],
        results=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for elements in args.elements:
        for kind in args.kind:
            for mode in ('numerical', 'polynomial', 'mixed_polynomial', 'witness'):
                if kind == 'reciprocal' and mode == 'polynomial':
                    continue
                if mode == 'mixed_polynomial' and kind not in ('silu', 'gelu'):
                    continue
                row = run_case(kind, elements, args.fractional_bits, mode)
                report['results'].append(row)
                args.output.write_text(json.dumps(report, indent=2)+'\n')
                print(json.dumps(row), flush=True)
    for width in args.normalization_widths:
        for kind in ('rmsnorm', 'softmax'):
            row = run_normalization(kind, width, args.fractional_bits)
            report['results'].append(row)
            args.output.write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
