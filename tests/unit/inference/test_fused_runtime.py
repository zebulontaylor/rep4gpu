import concurrent.futures
import pytest
import torch
from rep4gpu.protocol.runtime import Party, Share, make_seeds
from rep4gpu.protocol.fused_runtime import FusedParty
from rep4gpu.protocol.transport import IdealNetwork, RecordingTransport, ReplayTransport


def _fixture(bits, binary, matrix=False, prg='shake', binary_bits=64, broadcast=False):
    generator = torch.Generator().manual_seed(124)
    xs = [torch.randint(-(1 << 62), 1 << 62, (2, 3), generator=generator) for _ in range(4)]
    ys = [torch.randint(-(1 << 62), 1 << 62, (3, 4) if matrix else (1, 3) if broadcast else (2, 3), generator=generator) for _ in range(4)]
    xs[0][0, 0] = -(1 << 63)
    ys[0][0, 0] = (1 << 63) - 1
    seeds = make_seeds()
    network = IdealNetwork()
    transports = [RecordingTransport(network.endpoint(p)) for p in range(4)]
    def shares(party, device):
        return (Share(tuple(xs[(party+j+1)%4].to(device) for j in range(3)), binary),
                Share(tuple(ys[(party+j+1)%4].to(device) for j in range(3)), binary))
    def reference(party):
        runtime = Party(party, transports[party], {e:s for e,s in seeds.items() if e != party}, bits=bits, prg=prg, binary_bits=binary_bits)
        x, y = shares(party, 'cpu')
        result = runtime.matmul(x, y) if matrix else runtime.mul(x, y)
        return result, dict(runtime.counters)
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        expected = list(pool.map(reference, range(4)))
    return seeds, transports, shares, expected


@pytest.mark.parametrize('prg', ['shake', 'aes'])
@pytest.mark.parametrize('bits,binary,binary_bits', [(64, False, 64), (64, True, 64), (32, False, 64), (32, True, 64), (32, True, 32)])
def test_fused_pointwise_exact_transcript(bits, binary, binary_bits, prg):
    if prg == 'aes':
        pytest.importorskip('cryptography')
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    seeds, transports, shares, expected = _fixture(bits, binary, prg=prg, binary_bits=binary_bits)
    for party in range(4):
        replay = ReplayTransport(transports[party].transcript)
        runtime = FusedParty(party, replay, {e:s for e,s in seeds.items() if e != party}, device='cuda', bits=bits, prg=prg, binary_bits=binary_bits)
        actual = runtime.mul(*shares(party, 'cuda'))
        for a, b in zip(actual.parts, expected[party][0].parts):
            assert torch.equal(a.cpu(), b)
        assert runtime.counters == expected[party][1]
        assert runtime.joint_inputs == 6 and runtime.multiplications == 1
        replay.assert_consumed()


@pytest.mark.parametrize('bits', [32, 64])
def test_fused_matmul_exact_transcript(bits):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    seeds, transports, shares, expected = _fixture(bits, False, matrix=True)
    for party in range(4):
        replay = ReplayTransport(transports[party].transcript)
        runtime = FusedParty(party, replay, {e:s for e,s in seeds.items() if e != party}, device='cuda', bits=bits)
        actual = runtime.matmul(*shares(party, 'cuda'))
        for a, b in zip(actual.parts, expected[party][0].parts):
            assert torch.equal(a.cpu(), b)
        assert runtime.counters == expected[party][1]
        replay.assert_consumed()


def test_fused_cpu_fallback():
    seeds, transports, shares, expected = _fixture(64, False)
    runtime = FusedParty(0, ReplayTransport(transports[0].transcript), {e:s for e,s in seeds.items() if e != 0})
    actual = runtime.mul(*shares(0, 'cpu'))
    assert all(torch.equal(a, b) for a, b in zip(actual.parts, expected[0][0].parts))


@pytest.mark.parametrize('bits,binary,binary_bits', [(64, False, 64), (64, True, 64), (32, False, 64), (32, True, 64), (32, True, 32)])
def test_fused_local_operations(bits, binary, binary_bits):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    seeds, _, shares, _ = _fixture(bits, binary)
    for party in range(4):
        owned = {e:s for e,s in seeds.items() if e != party}
        cpu = Party(party, None, owned, bits=bits, binary_bits=binary_bits)
        gpu = FusedParty(party, None, owned, device='cuda', bits=bits, binary_bits=binary_bits)
        x, y = shares(party, 'cpu')
        xg, yg = shares(party, 'cuda')
        operations = [lambda r, a, b: r.add(a, b), lambda r, a, b: r.sub(a, b),
                      lambda r, a, b: r.mul_public(a, -91), lambda r, a, b: r.lshift(a, 17),
                      lambda r, a, b: r.rshift(a, 17), lambda r, a, b: r.public(-91, like=a)]
        for operation in operations:
            actual, expected = operation(gpu, xg, yg), operation(cpu, x, y)
            assert actual.binary == expected.binary
            assert all(torch.equal(a.cpu(), b) for a, b in zip(actual.parts, expected.parts))


def test_fused_broadcast_paths():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    pytest.importorskip('cryptography')
    seeds, transports, shares, expected = _fixture(32, True, prg='aes', binary_bits=32, broadcast=True)
    for party in range(4):
        owned = {e:s for e,s in seeds.items() if e != party}
        cpu = Party(party, None, owned, bits=32, binary_bits=32, prg='aes')
        replay = ReplayTransport(transports[party].transcript)
        gpu = FusedParty(party, replay, owned, device='cuda', bits=32, binary_bits=32, prg='aes')
        x, y = shares(party, 'cpu')
        xg, yg = shares(party, 'cuda')
        actual = gpu.mul(xg, yg)
        assert all(torch.equal(a.cpu(), b) for a, b in zip(actual.parts, expected[party][0].parts))
        replay.assert_consumed()
        for operation in (lambda r,a,b: r.add(a,b), lambda r,a,b: r.sub(a,b),
                          lambda r,a,b: r.mul_public(a,b.parts[0]),
                          lambda r,a,b: r.public(b.parts[0],like=a)):
            actual, wanted = operation(gpu,xg,yg), operation(cpu,x,y)
            assert all(torch.equal(a.cpu(),b) for a,b in zip(actual.parts,wanted.parts))
