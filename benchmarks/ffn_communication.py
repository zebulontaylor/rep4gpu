"""Compare measured activation traffic with output-sized FFN protocol costs."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import torch

from rep4gpu.device_inference import DeviceInferenceParty
from rep4gpu.device_transport import DeviceIdealEndpoint
from rep4gpu.exact_truncation import trunc_bounded
from rep4gpu.runtime import make_seeds
from rep4gpu.transport import IdealNetwork


def output_cost(elements):
    network, seeds = IdealNetwork(timeout=120), make_seeds()
    encoded = torch.arange(elements, dtype=torch.int64) % 1024 - 512

    def run(role):
        try:
            endpoint = network.endpoint(role)
            party = DeviceInferenceParty(role, DeviceIdealEndpoint(endpoint, 'cuda:0'),
                {i:s for i,s in seeds.items() if i != role}, device='cuda:0', bits=64,
                binary_bits=64, optimized=True, gpu_open=True, rounding='floor',
                sparse_products=True)
            share = party.input(0, encoded if role == 0 else None, encoded.shape)
            start = endpoint.bytes_sent
            product = party.mul(share, share)
            multiplication = endpoint.bytes_sent-start
            start = endpoint.bytes_sent
            full = party.trunc(product, 24)
            full_truncation = endpoint.bytes_sent-start
            start = endpoint.bytes_sent
            bounded = trunc_bounded(party, product, 24)
            bounded_truncation = endpoint.bytes_sent-start
            expected = (encoded*encoded) >> 24
            assert torch.equal(party.open(full).cpu(), expected)
            assert torch.equal(party.open(bounded).cpu(), expected)
            return multiplication, full_truncation, bounded_truncation
        except BaseException as error:
            network.abort(error)
            raise
    with ThreadPoolExecutor(4) as pool:
        counts = list(pool.map(run, range(4)))
    multiplication, full, bounded = [sum(x[i] for x in counts) for i in range(3)]
    return dict(elements=elements, multiplication_bytes=multiplication,
                fullring_truncation_bytes=full, bounded_truncation_bytes=bounded,
                fullring_total_bytes=multiplication+full,
                bounded_total_bytes=multiplication+bounded,
                fullring_bytes_per_coordinate=(multiplication+full)/elements,
                bounded_bytes_per_coordinate=(multiplication+bounded)/elements)


def scenario(parts):
    total = sum(parts.values())
    activation = parts['silu_with_witness_checks']
    return dict(component_bytes=parts, total_bytes=total,
                activation_fraction=activation/total,
                activation_to_other_ratio=activation/(total-activation))


def main():
    torch.set_num_threads(1)
    hidden, width = 14336, 4096
    source = Path('results/silu_exact_accounting.json')
    activation = json.loads(source.read_text())
    assert activation['elements'] == hidden
    # Refuse stale activation totals after a production or accounting edit.
    for name, expected in activation['source_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    costs = {n:output_cost(n) for n in (hidden, width)}
    dimensions = [('gate_projection', hidden), ('up_projection', hidden),
                  ('gate_product', hidden), ('down_projection', width)]
    bounded_parts = {name:costs[n]['bounded_total_bytes'] for name,n in dimensions}
    generic_parts = {name:costs[n]['bounded_total_bytes' if name == 'gate_product'
                                  else 'fullring_total_bytes'] for name,n in dimensions}
    for parts in (bounded_parts, generic_parts):
        parts['silu_with_witness_checks'] = activation['total_bytes']
    bounded = scenario(bounded_parts)
    generic = scenario(generic_parts)
    paths = [Path(__file__), source, *[Path('rep4gpu')/(name+'.py') for name in
        ('device_inference', 'device_runtime', 'carry_tree_swar', 'exact_truncation',
         'bounded_exact_truncation_gpu', 'compact_conversion_gpu')]]
    report = dict(methodology='One-token SwiGLU FFN width4096 hidden14336, ring64 Q24 exact floor, '
        'optimized CUDA sparse corrections. Directly measures multiplication and both floor '
        'payloads for14336 and4096 outputs. General matmul uses the same six-joint output '
        'resharing schedule as elementwise mul; matrix inner dimension affects local compute, '
        'not these payload sizes. This is a communication decomposition, not a full FFN '
        'execution or latency benchmark. SiLU uses the current measured accounting fixture. '
        'Global payloads include tags. Excludes input/weight distribution, final output '
        'opening, RMSNorm and attention. Neither scenario adds a projection bound check.',
        bounded_projection_contract='Every true signed Q48 projection accumulator must satisfy '
        'abs(accumulator) < 2**62, established before ring overflow by the model/input contract '
        'or an independent proof. Q24 format alone does not establish this bound. The main '
        'totals are conditional on this contract for gate, up and down projections.',
        gate_product_contract='The fixture bounds SiLU output magnitude by16 and the up branch '
        'magnitude by4; their Q48 product magnitude is at most64 * 2**48 < 2**62. '
        'Both scenarios require this up-branch bound. It is not implied by Q24 format.',
        generic_projection_fallback_description='Uses fullring floors for all three projections '
        'and the bounded floor for the gate product, retaining the stated gate-product contract.',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        activation_source=str(source), activation_max_abs_error=activation['max_abs_error'],
        output_costs=costs, **bounded,
        generic_projection_fallback=generic,
        bounded_projection_savings_bytes=generic['total_bytes']-bounded['total_bytes'])
    destination = Path('results/ffn_communication_breakdown.json')
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
