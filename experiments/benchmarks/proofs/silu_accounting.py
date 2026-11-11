"""Attribute the exact-floor witness SiLU transcript to semantic operations."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path

import torch

from experiments.benchmarks.activations.witness_activations import prover_witness
from rep4gpu.models.device_inference import DeviceInferenceParty
from rep4gpu.protocol.device_transport import DeviceIdealEndpoint
from rep4gpu.arithmetic.exact_truncation import trunc_bounded
from rep4gpu.arithmetic.silu_exp import encoded_coefficients
from rep4gpu.protocol.runtime import make_seeds
from rep4gpu.protocol.transport import IdealNetwork
from rep4gpu.proofs.witness_activations import WitnessSession, numerical_program


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('experiments/results/proofs/silu_next_accounting.json'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    n, scale = 14336, 1 << 24
    x = torch.round(torch.linspace(-4,4,n,dtype=torch.float64)*scale).long()
    claim = prover_witness('silu', x, x, 24)
    network, seeds = IdealNetwork(timeout=180), make_seeds()

    def run(role, *, expanded):
        try:
            ep = network.endpoint(role)
            rt = DeviceInferenceParty(role,DeviceIdealEndpoint(ep,'cuda:0'),
                {i:s for i,s in seeds.items() if i != role},device='cuda:0',bits=64,
                binary_bits=64,optimized=True,gpu_open=True,rounding='floor',sparse_products=True)
            shared = rt.input(0,x if role==0 else None,x.shape)
            stages = {}
            def step(name,fn):
                before = ep.bytes_sent
                value = fn()
                stages[name] = stages.get(name,0)+ep.bytes_sent-before
                return value
            start_bytes = ep.bytes_sent
            y = step('witness_input',lambda:rt.input(0,claim if role==0 else None,claim.shape))
            session = WitnessSession(rt)
            fp = session.fp
            if not expanded:
                session.evaluate(numerical_program('silu'), {'x': shared}, {'output': y})
                output, = session.finish()
                total_bytes = ep.bytes_sent - start_bytes
                return total_bytes, rt.open(output).cpu()
            # Mirror production to expose individual products and floors. A second
            # production run below checks both the transcript size and exact output.
            step('witness_upper_zero_width29',lambda:session.upper_zero(rt.add(y,rt.public(scale,like=y)),29))
            sign, absolute = step('input_sign_absolute_and_clamp',lambda:session.signed_absolute_input(shared,16*scale))
            reduced = step('exp_reduce_shift4',lambda:trunc_bounded(rt,fp._negative(absolute),4))
            c0,c1,c2,c3,c4,c5,c6 = encoded_coefficients(24)
            product = step('exp_reduced_square_product',lambda:rt.mul(reduced,reduced))
            square = step('exp_horner_floor_1',lambda:trunc_bounded(rt,product,24))
            linear = step('exp_horner_public_product_c5',lambda:rt.mul_public(reduced,c5))
            quadratic = step('exp_horner_public_product_c6',lambda:rt.mul_public(square,c6))
            q2 = step('exp_horner_floor_2',lambda:trunc_bounded(rt,rt.add(linear,quadratic),24))
            q2 = rt.add(rt.public(c4,like=reduced),q2)
            linear = step('exp_horner_public_product_c3',lambda:rt.mul_public(reduced,c3))
            quadratic = step('exp_horner_product_3',lambda:rt.mul(square,q2))
            q1 = step('exp_horner_floor_3',lambda:trunc_bounded(rt,rt.add(linear,quadratic),24))
            q1 = rt.add(rt.public(c2,like=reduced),q1)
            linear = step('exp_horner_public_product_c1',lambda:rt.mul_public(reduced,c1))
            quadratic = step('exp_horner_product_4',lambda:rt.mul(square,q1))
            exponential = step('exp_horner_floor_4',lambda:trunc_bounded(rt,rt.add(linear,quadratic),24))
            exponential = rt.add(rt.public(c0,like=reduced),exponential)
            for i in range(4):
                product = step('exp_square_product_'+str(i+1),lambda:rt.mul(exponential,exponential))
                product = rt.add(product,rt.public(scale//2,like=product))
                exponential = step('exp_square_nearest_floor_'+str(i+1),lambda:trunc_bounded(rt,product,24))
            denominator = fp.add(fp.constant(1,exponential),exponential)
            multiplier = step('numerator_sign_select',lambda:fp._select(sign,fp._negative(exponential),fp.constant(1,exponential)))
            numerator = step('numerator_raw_product',lambda:rt.mul(absolute,multiplier))
            raw = step('output_denominator_raw_product',lambda:rt.mul(y,denominator))
            error = rt.sub(raw,numerator)
            doubled = rt.mul_public(error,2)
            step('output_lower_upper_zero_width27',lambda:session.upper_zero(rt.add(doubled,denominator),27))
            step('output_upper_sign_width28',lambda:session.bounded_nonnegative(rt.sub(denominator,doubled),28))
            session._outputs.append(y)
            output, = step('aggregate_acceptance',session.finish)
            assert sum(stages.values()) == ep.bytes_sent - start_bytes
            actual = rt.open(output).cpu()
            return stages,actual
        except BaseException as error:
            network.abort(error)
            raise
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda role: run(role, expanded=True),range(4)))
    with ThreadPoolExecutor(4) as pool:
        production = list(pool.map(lambda role: run(role, expanded=False),range(4)))
    assert all(torch.equal(row[1],results[0][1]) for row in results)
    stages = {k:sum(r[0][k] for r in results) for k in results[0][0]}
    total_bytes = sum(stages.values())
    production_bytes = sum(row[0] for row in production)
    assert total_bytes == production_bytes, (total_bytes, production_bytes)
    assert all(torch.equal(row[1],results[0][1]) for row in production)
    error = (results[0][1].double()/scale-torch.nn.functional.silu(x.double()/scale)).abs().max().item()
    baseline = json.loads(Path('experiments/results/proofs/silu_exact_accounting.json').read_text())
    assert baseline['elements'] == n
    assert baseline['total_bytes'] == 60792176
    files = ['experiments/benchmarks/proofs/silu_accounting.py','src/rep4gpu/arithmetic/numerical.py','src/rep4gpu/proofs/witness_activations.py',
             'src/rep4gpu/arithmetic/silu_exp.py','src/rep4gpu/gpu/bits/bounded_activation.py','src/rep4gpu/proofs/witness_prover.py',
             'src/rep4gpu/gpu/bits/carry_tree_swar.py','src/rep4gpu/gpu/bits/compact_conversion_gpu.py',
             'src/rep4gpu/arithmetic/upper_zero.py','src/rep4gpu/arithmetic/binary_aggregation.py',
             'src/rep4gpu/arithmetic/exact_truncation.py','src/rep4gpu/gpu/bits/bounded_exact_truncation_gpu.py']
    report = dict(elements=n,total_bytes=sum(stages.values()),max_abs_error=error,
        production_total_bytes=production_bytes,
        baseline_total_bytes=baseline['total_bytes'],
        saved_bytes=baseline['total_bytes']-total_bytes,
        methodology='Semantic expansion of current numerical_program(silu). Total measured bytes and exact '
        'opened outputs are asserted against a separate full production execution using the same seeds and fixture. '
        'Global payloads include tags; input x sharing and final output opening are excluded.',
        fixture_domain=[-4,4],verified_input_domain=[-16,16],fractional_bits=24,
        counts_per_coordinate=dict(exponential_secret_products=7,
            exponential_public_coefficient_products=4,exponential_shift4_floors=1,
            exponential_q24_floors=8,output_relation_secret_products=2,
            sign_selection_secret_products=2,
            note='Input clamp, sign circuits and aggregate acceptance have additional internal products.'),
        formula='T=floor(-abs(X)/16); U=floor(T*T/S); H=degree6 quadratic Horner with four Q24 floors; '
        'repeat E=floor((E*E+S/2)/S) four times; D=S+E; M=-E if X<0 else S; '
        'require 0<=Y+S<32*S and |X|<=16*S; R=2*(Y*D-|X|*M); '
        'require 0<=R+D<8*S and D-R>=0',
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},
        stages={k:dict(bytes=v,bytes_per_coordinate=v/n,ring64_word_equivalents_per_coordinate=v/n/8)
                for k,v in stages.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
