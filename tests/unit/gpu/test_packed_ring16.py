"""Packed arithmetic preserves ring16 shares, transcripts, and checked replay."""
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def signed16(value):
    return ((value & 65535) ^ 32768) - 32768


def run_roles(network, fn):
    def guarded(p):
        try:
            return fn(p)
        except BaseException as error:
            network.abort(error)
            raise
    with ThreadPoolExecutor(4) as pool:
        return list(pool.map(guarded, range(4)))


@pytest.mark.parametrize('captured', [False, True])
@pytest.mark.parametrize('prepared', [False, True])
def test_packed_full_ring_matches_wide_shares_and_transcripts(captured, prepared):
    from rep4gpu.protocol.phase_graph import PhaseGraphParty
    from rep4gpu.arithmetic.conversion import a2b
    from rep4gpu.protocol.device_input import DeviceInputMixin
    from rep4gpu.transports.device import DeviceIdealEndpoint, DeviceRecorder
    from rep4gpu.protocol.runtime import Share, make_seeds
    from rep4gpu.transports.byte import IdealNetwork

    class Runtime(DeviceInputMixin, PhaseGraphParty):
        pass

    generator = torch.Generator().manual_seed(1616)
    x = torch.randint(0, 65536, (17, 65), generator=generator)
    x[0, :4] = torch.tensor([0, 32767, 32768, 65535])
    w = torch.randint(0, 65536, (65, 7), generator=generator)
    bias = torch.tensor([0, 1, -1, 32767, -32768, 65535, -65536])
    inputs = (x, (x + 13001) & 65535)
    seeds = make_seeds()

    def execute(packed):
        network = IdealNetwork(timeout=60)
        records = [DeviceRecorder(DeviceIdealEndpoint(network.endpoint(p))) for p in range(4)]
        roles = [Runtime(p, records[p], {e: s for e, s in seeds.items() if e != p},
                         bits=16, binary_bits=32, gpu_open=True,
                         packed_arithmetic=packed) for p in range(4)]
        if not captured:
            for role in roles:
                role.prg_batch = False
        dtype = torch.uint16 if packed else torch.int64

        def distribute(p):
            r = roles[p]
            shared = [r.input(owner, value if owner == p else None, value.shape)
                      for owner, value in ((0, inputs[0]), (0, inputs[1]), (1, w), (1, bias))]
            assert all(t.dtype == dtype for share in shared for t in share.parts)
            return shared

        shares = run_roles(network, distribute)
        initial_parts = [[tuple(t.to(torch.int64).cpu() for t in share.parts)
                          for share in values] for values in shares]
        if prepared:
            for r, values in zip(roles, shares):
                values[2] = r.prepare_weight_share(values[2])

        functions = []
        for r in roles:
            def evaluate(a, weight, b, r=r):
                score = r.sub(r.add(r.matmul(a, weight), b), r.mul_public(b, -3))
                square = r.mul(score, score)
                sign_bin = r.rshift(a2b(r, Share(tuple(t.to(torch.int64) for t in a.parts), False)), 15)
                labels = r.xor(r.public(1, like=sign_bin, binary=True), sign_bin)
                sign, truncated = r.sign_bit(a), r.trunc(a, 3)
                return (score, square, sign, truncated, labels, r.open(score),
                        r.open(square), r.open(sign), r.open(truncated), r.open(labels))
            functions.append(evaluate)
        counters = [dict(r.counters) for r in roles]
        plans = [r.prepare_inference(fn, values[0], values[2], values[3], check_at_send=True)
                 for r, fn, values in zip(roles, functions, shares)] if captured else None
        assert [r.counters for r in roles] == counters
        outputs, counts = [], []
        for index in range(2):
            def infer(p):
                fn = plans[p].run if plans else functions[p]
                result = fn(shares[p][index], shares[p][2], shares[p][3])
                assert all(t.dtype == dtype for share in result[:4] for t in share.parts)
                assert result[4].binary
                assert all(t.dtype == torch.int64 for t in result[4].parts)
                return [tuple(t.to(torch.int64).cpu() for t in share.parts) for share in result[:5]], [t.cpu() for t in result[5:]]
            outputs.append(run_roles(network, infer))
            counts.append([dict(r.counters) for r in roles])
        assert all(any(counts[1][p][e] > counts[0][p][e] for e in counts[0][p]) for p in range(4))
        return initial_parts, outputs, counts, [r.transcript for r in records]

    wide, packed = execute(False), execute(True)
    assert wide[2:] == packed[2:]
    for initial_a, initial_b in zip(wide[0], packed[0]):
        for a, b in zip(initial_a, initial_b):
            for left, right in zip(a, b):
                assert torch.equal(left, right)
    for index, (expected_run, actual_run) in enumerate(zip(wide[1], packed[1])):
        score = signed16(inputs[index] @ w + 4*bias)
        square = signed16(score*score)
        labels = (signed16(inputs[index]) >= 0).long()
        sign = (signed16(inputs[index]) < 0).long()
        truncated = signed16(inputs[index]) >> 3
        for (expected_parts, expected_open), (actual_parts, actual_open) in zip(expected_run, actual_run):
            for expected, actual in zip(expected_parts, actual_parts):
                assert all(torch.equal(a, b) for a, b in zip(expected, actual))
            for expected, actual, reference in zip(expected_open, actual_open, (score, square, sign, truncated, labels)):
                assert torch.equal(actual, expected)
                assert torch.equal(actual, reference)


@pytest.mark.parametrize('operation', ['input', 'product'])
def test_packed_rejects_corrupted_frames(operation):
    from rep4gpu.protocol.ring_inference import DeviceInferenceParty
    from rep4gpu.protocol.device_input import DeviceInputMixin
    from rep4gpu.transports.device import DeviceIdealEndpoint
    from rep4gpu.protocol.runtime import make_seeds
    from rep4gpu.transports.byte import IdealNetwork

    class Runtime(DeviceInputMixin, DeviceInferenceParty):
        pass

    seeds, network = make_seeds(), IdealNetwork(timeout=10)
    roles = [Runtime(p, DeviceIdealEndpoint(network.endpoint(p)),
                     {e: s for e, s in seeds.items() if e != p}, bits=16,
                     packed_arithmetic=True) for p in range(4)]
    original = roles[0]._exchange_device
    poisoned = [False]

    def corrupt(outgoing, sizes):
        outgoing = dict(outgoing)
        if outgoing and not poisoned[0]:
            peer = next(iter(outgoing))
            outgoing[peer] = outgoing[peer].clone()
            outgoing[peer][0] ^= 1
            poisoned[0] = True
        return original(outgoing, sizes)

    roles[0]._exchange_device = corrupt

    def worker(p):
        r = roles[p]
        try:
            if operation == 'input':
                r.input(0, torch.arange(8) if p == 0 else None, (8,))
            else:
                value = r.public(torch.arange(8, device='cuda'))
                r.mul(value, value)
            return None
        except BaseException as error:
            network.abort(error)
            return str(error)

    with ThreadPoolExecutor(4) as pool:
        errors = list(pool.map(worker, range(4)))
    assert poisoned[0]
    assert any(error is not None and 'hash mismatch' in error for error in errors)


def test_packed_storage_rejects_invalid_domains():
    from rep4gpu.protocol.ring_inference import DeviceInferenceParty
    from rep4gpu.transports.device import DeviceIdealEndpoint
    from rep4gpu.protocol.runtime import Share, make_seeds
    from rep4gpu.transports.byte import IdealNetwork

    seed = make_seeds()
    for bits, device in ((32, 'cuda'), (64, 'cuda'), (16, 'cpu')):
        with pytest.raises((ValueError, TypeError)):
            DeviceInferenceParty(0, DeviceIdealEndpoint(IdealNetwork().endpoint(0)),
                                 {e: s for e, s in seed.items() if e}, bits=bits,
                                 device=device, packed_arithmetic=True)
    narrow = torch.zeros(3, device='cuda', dtype=torch.uint16)
    with pytest.raises((ValueError, TypeError)):
        Share((narrow, narrow, narrow), binary=True)
    with pytest.raises((ValueError, TypeError)):
        Share((narrow, narrow.to(torch.int64), narrow))
    cpu = narrow.cpu()
    with pytest.raises((ValueError, TypeError)):
        Share((cpu, cpu, cpu))


@pytest.mark.parametrize('bits', [16, 32, 64])
@pytest.mark.parametrize('gpu_open', [False, True])
def test_disabled_or_wrong_ring_rejects_packed_operands_before_protocol(bits, gpu_open):
    from rep4gpu.protocol.ring_inference import DeviceInferenceParty
    from rep4gpu.transports.device import DeviceIdealEndpoint
    from rep4gpu.protocol.runtime import Share, make_seeds
    from rep4gpu.transports.byte import IdealNetwork

    seeds = make_seeds()
    endpoint = DeviceIdealEndpoint(IdealNetwork(timeout=1).endpoint(0))
    role = DeviceInferenceParty(0, endpoint, {e: s for e, s in seeds.items() if e},
                                bits=bits, gpu_open=gpu_open)
    narrow = Share(tuple(torch.zeros((2, 2), dtype=torch.uint16, device='cuda')
                         for _ in range(3)))
    wide = role.public(torch.ones((2, 2), dtype=torch.int64, device='cuda'))
    before = dict(role.counters)
    for operation in (role.add, role.mul, role.matmul):
        for operands in ((narrow, wide), (wide, narrow)):
            with pytest.raises(ValueError, match='uint16 shares require packed arithmetic'):
                operation(*operands)
    with pytest.raises(ValueError, match='uint16 shares require packed arithmetic'):
        role.open(narrow)
    assert role.counters == before
    assert endpoint.messages_sent == 0


def test_packed_shift_fallbacks_match_wide_components():
    from rep4gpu.protocol.ring_inference import DeviceInferenceParty
    from rep4gpu.transports.device import DeviceIdealEndpoint
    from rep4gpu.protocol.runtime import Share, make_seeds
    from rep4gpu.transports.byte import IdealNetwork

    seeds = make_seeds()
    available = {e: s for e, s in seeds.items() if e}
    endpoint = DeviceIdealEndpoint(IdealNetwork(timeout=1).endpoint(0))
    wide = DeviceInferenceParty(0, endpoint, available, bits=16)
    packed = DeviceInferenceParty(0, endpoint, available, bits=16, packed_arithmetic=True)
    parts = tuple(torch.tensor([0, 1, 32767, 32768, 65535], device='cuda') for _ in range(3))
    a, b = Share(parts), Share(tuple(t.to(torch.uint16) for t in parts))
    for amount in (-1, 0, 15, 16, 63, 64, torch.tensor(3, device='cuda'),
                   torch.tensor([0, 1, 8, 16, 64], device='cuda')):
        for name in ('lshift', 'rshift'):
            expected, actual = getattr(wide, name)(a, amount), getattr(packed, name)(b, amount)
            for want, got in zip(expected.parts, actual.parts):
                assert got.dtype == torch.uint16
                torch.testing.assert_close(got.to(torch.int64), want, rtol=0, atol=0)
    assert endpoint.messages_sent == 0


def test_packed_empty_product():
    from rep4gpu.protocol.phase_graph import PhaseGraphParty
    from rep4gpu.transports.device import DeviceIdealEndpoint
    from rep4gpu.protocol.runtime import make_seeds
    from rep4gpu.transports.byte import IdealNetwork

    network = IdealNetwork(timeout=60)
    seeds = make_seeds()
    roles = [PhaseGraphParty(p, DeviceIdealEndpoint(network.endpoint(p)),
                             {e: s for e, s in seeds.items() if e != p}, bits=16,
                             packed_arithmetic=True, gpu_open=True) for p in range(4)]

    def compute(p):
        role = roles[p]
        empty = role.public(torch.empty((0, 3), dtype=torch.int64, device='cuda'))
        product = role.mul(empty, role.public(torch.ones(3, dtype=torch.int64, device='cuda')))
        assert product.shape == (0, 3)
        assert all(part.dtype == torch.uint16 for part in product.parts)
        return role.open(product).cpu()

    outputs = run_roles(network, compute)
    for out in outputs:
        assert out.shape == (0, 3)
