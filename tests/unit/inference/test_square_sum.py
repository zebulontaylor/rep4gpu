from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from rep4gpu.models.square_sum import signed_square_sum, prepare_square_signs
from rep4gpu.protocol.runtime import Party, Share, make_seeds
from rep4gpu.protocol.transport import IdealNetwork


def signed(value, bits):
    value &= (1 << bits) - 1
    return value - (1 << bits) if value >= 1 << (bits - 1) else value


@pytest.mark.parametrize('bits', [16, 32, 64])
@pytest.mark.parametrize('device,captured', [('cpu', False), ('cuda', False), ('cuda', True)])
def test_private_signed_square_sum_matches_integer_oracle(bits, device, captured):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    generator = torch.Generator().manual_seed(61)
    x = torch.empty((7, 35), dtype=torch.int64).random_(generator=generator)
    x[::2] = -x[::2]-1
    x[0, :4] = torch.tensor([-(1 << 63), -1, 0, (1 << 63)-1])
    signs = tuple(-1 if i % 3 else 1 for i in range(x.shape[1]))
    expected = torch.tensor([[signed(sum(s*int(v)**2 for s, v in zip(signs, row)), bits)]
                             for row in x.tolist()], dtype=torch.int64)
    network, seeds = IdealNetwork(timeout=60), make_seeds()
    if device == 'cuda':
        from rep4gpu.models.phase_graph import PhaseGraphParty
        from rep4gpu.protocol.device_transport import DeviceIdealEndpoint
        roles = [PhaseGraphParty(p, DeviceIdealEndpoint(network.endpoint(p)),
                                {e: s for e, s in seeds.items() if e != p}, bits=bits,
                                binary_bits=32 if bits < 64 else 64, gpu_open=True) for p in range(4)]
    else:
        roles = [Party(p, network.endpoint(p), {e: s for e, s in seeds.items() if e != p},
                       bits=bits, binary_bits=32 if bits < 64 else 64) for p in range(4)]

    def share(p):
        return roles[p].input(2, x if p == 2 else None, x.shape)

    with ThreadPoolExecutor(4) as pool:
        inputs = list(pool.map(share, range(4)))
    functions = [lambda value, rt=rt: rt.open(signed_square_sum(rt, value, signs)) for rt in roles]
    plans = [rt.prepare_inference(fn, value, check_at_send=True)
             for rt, fn, value in zip(roles, functions, inputs)] if captured else None

    def work(p):
        rt = roles[p]
        try:
            before = rt.transport.bytes_sent
            actual = (plans[p].run(inputs[p]) if captured else functions[p](inputs[p])).cpu()
            narrow_bytes = rt.transport.bytes_sent-before
            before = rt.transport.bytes_sent
            squared = rt.mul(inputs[p], inputs[p])
            coefficients = torch.tensor(signs, device=rt.device)
            reference = Share(tuple(rt._normalize((part*coefficients).sum(1, keepdim=True))
                                    for part in squared.parts))
            opened = rt.open(reference).cpu()
            return actual, opened, narrow_bytes, rt.transport.bytes_sent-before
        except BaseException as error:
            network.abort(error)
            raise

    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(work, range(4)))
    for actual, reference, _, _ in results:
        assert torch.equal(actual, expected)
        assert torch.equal(reference, expected)
    assert sum(row[2] for row in results) < sum(row[3] for row in results)


@pytest.mark.parametrize('bits', [16, 32, 64])
@pytest.mark.parametrize('shape', [(3, 2049), (0, 9), (5, 0)])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_local_products_full_word_long_and_empty(bits, shape):
    from rep4gpu.gpu.matmul.square_sum import local_square_sum
    generator = torch.Generator().manual_seed(942)
    parts = [torch.empty(shape, dtype=torch.int64).random_(generator=generator) for _ in range(3)]
    for part in parts:
        part[::2] = -part[::2]-1
    signs = torch.tensor([1 if i % 2 else -1 for i in range(shape[1])], dtype=torch.int64)
    a, b, c = parts
    expected = [((term * signs).sum(1, keepdim=True))
                for term in (a*a+2*a*b, b*b+2*b*c, 2*a*c)]
    actual = local_square_sum(tuple(p.cuda() for p in parts), signs.cuda(), bits)
    for reference, value in zip(expected, actual):
        if bits < 64:
            reference &= (1 << bits)-1
        assert torch.equal(value.cpu(), reference)


def test_public_sign_contract_and_empty_shapes():
    network, seeds = IdealNetwork(), make_seeds()
    rt = Party(0, network.endpoint(0), {e: s for e, s in seeds.items() if e != 0}, bits=16)
    with pytest.raises(ValueError, match='-1 or'):
        prepare_square_signs(rt, [1, 0])
    with pytest.raises(ValueError, match='-1 or'):
        prepare_square_signs(rt, [True])
    x = rt.public(torch.zeros((3, 0), dtype=torch.int64))
    y = signed_square_sum(rt, x, ())
    assert y.shape == (3, 1)
    assert all(torch.count_nonzero(part) == 0 for part in y.parts)
    with pytest.raises(ValueError, match='hidden width'):
        signed_square_sum(rt, x, [1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_contracted_product_rejects_corrupted_message():
    from rep4gpu.models.device_inference import DeviceInferenceParty
    from rep4gpu.protocol.device_transport import DeviceIdealEndpoint
    seeds, network = make_seeds(), IdealNetwork(timeout=10)
    roles = [DeviceInferenceParty(p, DeviceIdealEndpoint(network.endpoint(p)),
                                  {e: s for e, s in seeds.items() if e != p}, bits=16,
                                  binary_bits=32) for p in range(4)]
    original = roles[0]._exchange_device

    def corrupt(outgoing, sizes):
        outgoing = dict(outgoing)
        if outgoing:
            peer = next(iter(outgoing))
            outgoing[peer] = outgoing[peer].clone()
            outgoing[peer][0] ^= 1
        return original(outgoing, sizes)

    roles[0]._exchange_device = corrupt

    def work(p):
        try:
            x = roles[p].public(torch.ones((3, 8), dtype=torch.int64, device='cuda'))
            signed_square_sum(roles[p], x, (1, 1, 1, 1, -1, -1, -1, -1))
            return None
        except Exception as error:
            return str(error)

    with ThreadPoolExecutor(4) as pool:
        errors = list(pool.map(work, range(4)))
    assert any(error is not None and 'hash mismatch' in error for error in errors)
