"""Attribute the exact-floor witness SiLU transcript to semantic operations."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import torch

from benchmarks.witness_activations import prover_witness
from rep4gpu.device_inference import DeviceInferenceParty
from rep4gpu.device_transport import DeviceIdealEndpoint
from rep4gpu.numerical import EXP_COEFFICIENTS
from rep4gpu.runtime import make_seeds
from rep4gpu.transport import IdealNetwork
from rep4gpu.witness_activations import WitnessSession


def main():
    torch.set_num_threads(1)
    n, scale = 14336, 1 << 24
    x = torch.round(torch.linspace(-4,4,n,dtype=torch.float64)*scale).long()
    claim = prover_witness('silu', x, x, 24)
    network, seeds = IdealNetwork(timeout=180), make_seeds()

    def run(role):
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
            w = step('witness_input',lambda:rt.input(0,claim if role==0 else None,claim.shape))
            session = WitnessSession(rt)
            fp = session.fp
            w = step('witness_range_and_clamp',lambda:session.bounded(w,0,1))
            bounded = step('input_range_and_clamp',lambda:session.bounded(shared,-4,4))
            absolute, sign = step('absolute_sign_and_select',lambda:fp._abs_sign(bounded))
            reduced = step('exp_reduce_shift4',lambda:rt.trunc(fp._negative(absolute),4))
            exponential = step('exp_horner_degree10',lambda:fp.polynomial(reduced,EXP_COEFFICIENTS))
            for i in range(4):
                exponential = step('exp_square_'+str(i+1),lambda:fp.mul(exponential,exponential))
            denominator = fp.add(fp.constant(1,exponential),exponential)
            denominator = step('denominator_range_and_clamp',lambda:session.bounded(denominator,1,2))
            raw = step('reciprocal_raw_product',lambda:rt.mul(denominator,w))
            error = rt.sub(raw,rt.public(scale*scale,like=raw))
            doubled = rt.mul_public(error,2)
            step('reciprocal_lower_check',lambda:session.nonnegative(rt.add(doubled,denominator)))
            step('reciprocal_upper_check',lambda:session.nonnegative(rt.sub(denominator,doubled)))
            small = step('negative_sigmoid_product_floor',lambda:fp.mul(exponential,w))
            sigmoid = step('sigmoid_sign_select',lambda:fp._select(sign,small,w))
            output = step('final_x_sigmoid_product_floor',lambda:fp.mul(bounded,sigmoid))
            session._outputs.append(output)
            output, = step('aggregate_acceptance',session.finish)
            actual = rt.open(output).cpu()
            return stages,actual
        except BaseException as error:
            network.abort(error)
            raise
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(run,range(4)))
    assert all(torch.equal(row[1],results[0][1]) for row in results)
    stages = {k:sum(r[0][k] for r in results) for k in results[0][0]}
    reference = next(r for r in json.loads(Path('results/carry_prefix_layer_shapes.json').read_text())['results']
                     if r['kind']=='silu' and r['version']=='after')
    assert sum(stages.values())==reference['total_bytes']
    error = (results[0][1].double()/scale-torch.nn.functional.silu(x.double()/scale)).abs().max().item()
    assert error==reference['max_abs_error']
    files = ['benchmarks/silu_accounting.py','rep4gpu/numerical.py','rep4gpu/witness_activations.py',
             'rep4gpu/carry_tree_swar.py','rep4gpu/compact_conversion_gpu.py','rep4gpu/exact_truncation.py']
    report = dict(elements=n,total_bytes=sum(stages.values()),max_abs_error=error,
        methodology='Manual expansion of current numerical_program(silu), with identical scalar operations, '
        'constraints, exact floor and fixture. Total bytes and output error asserted against production '
        'benchmark. Global payloads include tags; input x sharing and final output opening excluded.',
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},
        stages={k:dict(bytes=v,bytes_per_coordinate=v/n,ring64_word_equivalents_per_coordinate=v/n/8)
                for k,v in stages.items()})
    Path('results/silu_exact_accounting.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
