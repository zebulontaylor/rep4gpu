import concurrent.futures
import pytest
import torch
from rep4gpu.runtime import Party, Share, make_seeds
from rep4gpu.fused_runtime import FusedParty
from rep4gpu.transport import IdealNetwork, RecordingTransport, ReplayTransport
from rep4gpu import carry_tree, carry_tree_gpu


def _reference(bits, binary_bits, positions, shape=(2, 33), shift=None):
    generator = torch.Generator().manual_seed(349)
    # These are components, not opened operands. Include extreme bit patterns.
    xs = [torch.randint(-(1 << 62), 1 << 62, shape, generator=generator) for _ in range(4)]
    ys = [torch.randint(-(1 << 62), 1 << 62, shape, generator=generator) for _ in range(4)]
    xs[0].reshape(-1)[0] = -(1 << 63)
    ys[0].reshape(-1)[0] = -1
    seeds = make_seeds()
    network = IdealNetwork()
    transports = [RecordingTransport(network.endpoint(p)) for p in range(4)]
    def shares(party, device):
        return (Share(tuple(xs[(party+j+1)%4].to(device) for j in range(3)), True),
                Share(tuple(ys[(party+j+1)%4].to(device) for j in range(3)), True))
    def work(party):
        runtime = Party(party, transports[party], {e:s for e,s in seeds.items() if e != party},
                        bits=bits, binary_bits=binary_bits, prg='aes')
        a, b = shares(party, 'cpu')
        result = carry_tree.carry_bits(runtime, a, b, positions) if shift is None else carry_tree.truncation_bits(runtime, a, b, shift)
        return result, dict(runtime.counters)
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        expected = list(pool.map(work, range(4)))
    return seeds, transports, shares, expected


@pytest.mark.parametrize('bits,binary_bits', [(64, 64), (32, 64), (32, 32)])
@pytest.mark.parametrize('shift', [0, 1, 12])
def test_gpu_truncation_bits_exact_reference(bits, binary_bits, shift):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    pytest.importorskip('cryptography')
    seeds, transports, shares, expected = _reference(bits, binary_bits, (), shift=shift)
    for party in range(4):
        replay = ReplayTransport(transports[party].transcript)
        runtime = FusedParty(party, replay, {e:s for e,s in seeds.items() if e != party},
                             device='cuda', bits=bits, binary_bits=binary_bits, prg='aes')
        actual = carry_tree_gpu.truncation_bits(runtime, *shares(party, 'cuda'), shift)
        for result, reference in zip(actual, expected[party][0]):
            assert result.binary
            assert all(torch.equal(a.cpu(), b) for a, b in zip(result.parts, reference.parts))
        assert runtime.counters == expected[party][1]
        replay.assert_consumed()


@pytest.mark.parametrize('bits,binary_bits', [(64, 64), (32, 32)])
def test_gpu_duplicate_and_uneven_positions(bits, binary_bits):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    pytest.importorskip('cryptography')
    positions = (bits, 0, 3, 1, bits-1, 3, 17)
    seeds, transports, shares, expected = _reference(bits, binary_bits, positions, shape=(65,))
    for party in range(4):
        replay = ReplayTransport(transports[party].transcript)
        runtime = FusedParty(party, replay, {e:s for e,s in seeds.items() if e != party},
                             device='cuda', bits=bits, binary_bits=binary_bits, prg='aes')
        actual = carry_tree_gpu.carry_bits(runtime, *shares(party, 'cuda'), positions)
        assert actual[2] is actual[5]
        for result, reference in zip(actual, expected[party][0]):
            assert all(torch.equal(a.cpu(), b) for a, b in zip(result.parts, reference.parts))
        replay.assert_consumed()


def test_gpu_tree_layout_is_bit_packed():
    count = 1_000_000
    for width, sizes in [(64, (16, 64, 64)), (32, (16, 32, 32))]:
        words, offsets, total = carry_tree_gpu._layout(count, sizes, width)
        assert total == sum((count*size+width-1)//width for size in sizes)
        assert offsets[0] == 0
        # Both state kinds, three replicated components, int64 storage words.
        assert 2*3*total*8 <= 128*count
