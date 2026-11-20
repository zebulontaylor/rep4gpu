import concurrent.futures
import pytest
import torch
from rep4gpu.protocol.runtime import Share, make_seeds
from rep4gpu.protocol.fused_runtime import FusedParty
from rep4gpu.transports.byte import IdealNetwork, RecordingTransport
from rep4gpu.gpu.bits import carry_tree_gpu


def _run_gpu_reference(bits, binary_bits, positions, shape=(2, 33), shift=None, open_outputs=True):
    generator = torch.Generator().manual_seed(349)
    xs = [torch.randint(-(1 << 62), 1 << 62, shape, generator=generator) for _ in range(4)]
    ys = [torch.randint(-(1 << 62), 1 << 62, shape, generator=generator) for _ in range(4)]
    xs[0].reshape(-1)[0] = -(1 << 63)
    ys[0].reshape(-1)[0] = -1
    seeds = make_seeds()
    network = IdealNetwork()
    transports = [RecordingTransport(network.endpoint(p)) for p in range(4)]

    def shares(party, device):
        return (Share(tuple(xs[(party + j + 1) % 4].to(device) for j in range(3)), True),
                Share(tuple(ys[(party + j + 1) % 4].to(device) for j in range(3)), True))

    def work(party):
        runtime = FusedParty(party, transports[party], {e: s for e, s in seeds.items() if e != party},
                             device='cuda', bits=bits, binary_bits=binary_bits, prg='aes')
        a, b = shares(party, 'cuda')
        result = (carry_tree_gpu.carry_bits(runtime, a, b, positions)
                  if shift is None else carry_tree_gpu.truncation_bits(runtime, a, b, shift))
        counters = dict(runtime.counters)
        opened = [runtime.open(r).cpu() for r in result] if open_outputs else None
        return result, opened, counters

    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        results = list(pool.map(work, range(4)))
    a_plain = xs[0] ^ xs[1] ^ xs[2] ^ xs[3]
    b_plain = ys[0] ^ ys[1] ^ ys[2] ^ ys[3]
    return seeds, transports, shares, results, a_plain, b_plain


def _carry_oracle(a: torch.Tensor, b: torch.Tensor, pos: int) -> torch.Tensor:
    if pos == 0:
        return torch.zeros_like(a, dtype=torch.int64)
    mask = (1 << pos) - 1
    out = torch.empty_like(a, dtype=torch.int64)
    for idx in range(a.numel()):
        av = int(a.reshape(-1)[idx].item()) & mask
        bv = int(b.reshape(-1)[idx].item()) & mask
        out.reshape(-1)[idx] = (av + bv) >> pos
    return out


@pytest.mark.parametrize('bits,binary_bits', [(64, 64), (32, 64), (32, 32)])
@pytest.mark.parametrize('shift', [0, 1, 12])
def test_gpu_truncation_bits_exact_oracle(bits, binary_bits, shift):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    pytest.importorskip('cryptography')
    _, _, _, results, a_plain, b_plain = _run_gpu_reference(bits, binary_bits, (), shift=shift)
    want_low = _carry_oracle(a_plain, b_plain, shift)
    want_high = _carry_oracle(a_plain, b_plain, bits)
    want_before = _carry_oracle(a_plain, b_plain, bits - 1)
    want_sign = ((a_plain >> (bits - 1)) & 1) ^ ((b_plain >> (bits - 1)) & 1) ^ want_before
    for _, opened, _ in results:
        for actual, expected in zip(opened, (want_low, want_high, want_sign)):
            assert torch.equal(actual, expected)


@pytest.mark.parametrize('bits,binary_bits', [(64, 64), (32, 32)])
def test_gpu_duplicate_and_uneven_positions(bits, binary_bits):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    pytest.importorskip('cryptography')
    positions = (bits, 0, 3, 1, bits - 1, 3, 17)
    _, _, _, results, a_plain, b_plain = _run_gpu_reference(bits, binary_bits, positions, shape=(65,))
    expected_list = [_carry_oracle(a_plain, b_plain, pos) for pos in positions]
    for shares_out, opened, _ in results:
        assert shares_out[2] is shares_out[5]
        for actual, expected in zip(opened, expected_list):
            assert torch.equal(actual, expected)


def test_gpu_tree_layout_is_bit_packed():
    count = 1_000_000
    for width, sizes in [(64, (16, 64, 64)), (32, (16, 32, 32))]:
        words, offsets, total = carry_tree_gpu._layout(count, sizes, width)
        assert total == sum((count * size + width - 1) // width for size in sizes)
        assert offsets[0] == 0
        assert 2 * 3 * total * 8 <= 128 * count
