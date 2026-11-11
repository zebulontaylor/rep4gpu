import concurrent.futures

import pytest
import torch

from rep4gpu.arithmetic.compact_conversion import weighted_b2a
from rep4gpu.gpu.bits.compact_conversion_gpu import weighted_b2a_gpu
from rep4gpu.protocol.fused_runtime import FusedParty
from rep4gpu.protocol.runtime import Party, Share, make_seeds
from rep4gpu.protocol.transport import IdealNetwork, RecordingTransport, ReplayTransport


@pytest.mark.parametrize('bits', [32,64])
@pytest.mark.parametrize('shape,outputs', [((1,1),1),((3,67),2),((7,3,23),1),((64,17),2)])
def test_weighted_conversion_exact_transcript(bits,shape,outputs):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    generator = torch.Generator().manual_seed(449)
    components = [torch.randint(-(1<<40),1<<40,shape,generator=generator) for _ in range(4)]
    weights = torch.randint(-(1<<30),1<<30,(outputs,shape[0]),generator=generator)
    if outputs == 1:
        weights = weights[0]
    seeds, network = make_seeds(), IdealNetwork()
    records = [RecordingTransport(network.endpoint(p)) for p in range(4)]

    def share(p,device):
        return Share(tuple(components[(p+j+1)%4].to(device) for j in range(3)),True)

    def reference(p):
        rt = Party(p,records[p],{e:s for e,s in seeds.items() if e!=p},bits=bits)
        result = weighted_b2a(rt,share(p,'cpu'),weights)
        counters = dict(rt.counters)
        return result,counters

    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        expected = list(pool.map(reference,range(4)))
    for p in range(4):
        replay = ReplayTransport(records[p].transcript)
        rt = FusedParty(p,replay,{e:s for e,s in seeds.items() if e!=p},bits=bits,device='cuda')
        actual = weighted_b2a_gpu(rt,share(p,'cuda'),weights)
        assert all(torch.equal(a.cpu(),b) for a,b in zip(actual.parts,expected[p][0].parts))
        assert rt.counters == expected[p][1]
        assert rt.joint_inputs == 8 and rt.multiplications == 1
        replay.assert_consumed()
    secret = components[0]&1
    for component in components[1:]:
        secret ^= component&1
    wanted = weights @ secret if secret.ndim==2 else (secret*weights.reshape((-1,1,1))).sum(0)
    reconstructed = sum(expected[(j+3)%4][0].parts[0] for j in range(4))
    if bits == 32:
        reconstructed = reconstructed & 0xffffffff
        wanted = wanted & 0xffffffff
    assert torch.equal(reconstructed,wanted)


def test_cpu_fallback_matches_local_shares():
    seeds,network = make_seeds(),IdealNetwork()
    records = [RecordingTransport(network.endpoint(p)) for p in range(4)]
    parts = [torch.tensor([[v,v+1,v+2],[v+3,v+4,v+5]]) for v in range(4)]
    def work(p):
        rt = Party(p,records[p],{e:s for e,s in seeds.items() if e!=p})
        planes = Share(tuple(parts[(p+j+1)%4] for j in range(3)),True)
        return weighted_b2a(rt,planes,[17,-19]),planes
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        expected = list(pool.map(work,range(4)))
    for p in range(4):
        replay = ReplayTransport(records[p].transcript)
        rt = Party(p,replay,{e:s for e,s in seeds.items() if e!=p})
        actual = weighted_b2a_gpu(rt,expected[p][1],[17,-19])
        assert all(torch.equal(a,b) for a,b in zip(actual.parts,expected[p][0].parts))
        replay.assert_consumed()
