"""Compare measured activation traffic with output-sized FFN protocol costs."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import torch

from rep4gpu.device_inference import DeviceInferenceParty
from rep4gpu.device_transport import DeviceIdealEndpoint
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
            result = party.trunc(product, 24)
            truncation = endpoint.bytes_sent-start
            assert torch.equal(party.open(result).cpu(), (encoded*encoded) >> 24)
            return multiplication, truncation
        except BaseException as error:
            network.abort(error)
            raise
    with ThreadPoolExecutor(4) as pool:
        counts = list(pool.map(run, range(4)))
    return dict(elements=elements, multiplication_bytes=sum(x[0] for x in counts),
                truncation_bytes=sum(x[1] for x in counts),
                total_bytes=sum(sum(x) for x in counts))


def main():
    torch.set_num_threads(1)
    hidden, width = 14336, 4096
    costs = {n:output_cost(n) for n in (hidden,width)}
    source = Path('results/carry_prefix_layer_shapes.json')
    activation = next(row for row in json.loads(source.read_text())['results']
                      if row['kind']=='silu' and row['version']=='after')
    parts = {name:costs[n]['total_bytes'] for name,n in
             [('gate_projection',hidden),('up_projection',hidden),
              ('gate_product',hidden),('down_projection',width)]}
    parts['silu_with_witness_checks'] = activation['total_bytes']
    total = sum(parts.values())
    paths = [Path(__file__),source,*[Path('rep4gpu')/(name+'.py') for name in
        ('device_inference','device_runtime','carry_tree_swar','exact_truncation','compact_conversion_gpu')]]
    report = dict(methodology='One-token SwiGLU FFN width4096 hidden14336, ring64 Q24 exact floor, '
        'optimized CUDA sparse corrections. Directly measures multiplication and truncation '
        'payload for14336 and4096 outputs. General matmul uses the same six-joint output '
        'resharing schedule as elementwise mul; matrix inner dimension affects local compute, '
        'not these payload sizes. This is a communication decomposition, not a full FFN '
        'execution or latency benchmark. Activations use the existing measured witness '
        'fixture. Excludes input/weight distribution, final output opening, RMSNorm and attention.',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        output_costs=costs, component_bytes=parts, total_bytes=total,
        activation_fraction=parts['silu_with_witness_checks']/total,
        activation_to_other_ratio=parts['silu_with_witness_checks']/(total-parts['silu_with_witness_checks']))
    destination = Path('results/ffn_communication_breakdown.json')
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
