"""Ring16 device arithmetic, mask streams, and checked inference graphs."""
from concurrent.futures import ThreadPoolExecutor
import struct

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_chacha_halfwords_match_stream_and_batched_graph_masks():
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
    from rep4gpu.gpu.crypto.gpu_prg import DomainKeys, generate, key_bytes
    from rep4gpu.gpu.crypto.gpu_prg_batch import generate_rows
    from rep4gpu.gpu.crypto.gpu_prg_graph import generate_rows_into
    seed, domain, nonce, count = bytes(range(32)), b'ring16-test', 17, 67
    keys = DomainKeys(seed).key(domain)
    raw = Cipher(algorithms.ChaCha20(key_bytes(seed, domain), struct.pack('<QQ', 0, nonce)), mode=None).encryptor().update(bytes(count*2))
    actual = generate(keys, nonce, count, bits=16)
    assert actual.to(torch.uint16).cpu().numpy().tobytes() == raw
    rows = generate_rows((keys, None), (nonce, None), count, bits=16, packed=True)
    assert rows.dtype == torch.uint16
    assert torch.equal(rows[0].to(torch.int64), actual)
    assert torch.count_nonzero(rows[1].to(torch.int64)) == 0
    output = torch.empty((6, count), dtype=torch.int64, device='cuda')
    nonces = torch.tensor([nonce, 0, 0, 0, 0, 0], dtype=torch.uint64, device='cuda')
    generate_rows_into((keys, None, None, None, None, None), nonces, output, bits=16)
    assert torch.equal(output[0], actual)
    assert torch.count_nonzero(output[1:]) == 0


@pytest.mark.parametrize('shape', [(1, 7, 3), (17, 65, 33), (129, 513, 1), (3, 2049, 1)])
@pytest.mark.parametrize('prepared', [False, True])
def test_ring16_local_products_full_word_inputs(shape, prepared):
    from rep4gpu.gpu.matmul.kernels import local_products, prepare_weight
    m, k, n = shape
    g = torch.Generator().manual_seed(87)
    xs = [torch.randint(-(1 << 31), 1 << 31, (m, k), generator=g) for _ in range(3)]
    ys = [torch.randint(-(1 << 31), 1 << 31, (k, n), generator=g) for _ in range(3)]
    expected = ((xs[0]+xs[1])@ys[0]+xs[0]@ys[1],
                (xs[1]+xs[2])@ys[1]+xs[1]@ys[2], xs[0]@ys[2]+xs[2]@ys[0])
    cuda_y = [y.cuda() for y in ys]
    if prepared:
        cuda_y = [prepare_weight(y, bits=16) for y in cuda_y]
    actual = local_products(*(x.cuda() for x in xs), *cuda_y, bits=16)
    for a, e in zip(actual, expected):
        assert torch.equal(a.cpu(), e & 65535)


@pytest.mark.parametrize('captured', [False, True])
@pytest.mark.parametrize('binary_bits', [32, 64])
def test_ring16_checked_products_sign_and_input(captured, binary_bits):
    from rep4gpu.protocol.phase_graph import PhaseGraphParty
    from rep4gpu.protocol.device_input import DeviceInputMixin
    from rep4gpu.transports.device import DeviceIdealEndpoint
    from rep4gpu.protocol.runtime import make_seeds
    from rep4gpu.transports.byte import IdealNetwork

    class Runtime(DeviceInputMixin, PhaseGraphParty):
        pass

    seeds, network = make_seeds(), IdealNetwork(timeout=60)
    roles = [Runtime(p, DeviceIdealEndpoint(network.endpoint(p)),
                     {e: s for e, s in seeds.items() if e != p},
                     bits=16, binary_bits=binary_bits, gpu_open=True) for p in range(4)]
    x = torch.tensor([[-32768, -1, 0, 1, 32767], [200, -300, 400, -500, 600]], dtype=torch.int64)
    w = torch.tensor([[1, -2], [3, 4], [-5, 6], [7, 8], [9, -10]], dtype=torch.int64)
    expected = (((x @ w) & 65535) ^ 32768) - 32768

    def share(p):
        r = roles[p]
        return (r.input(0, x if p == 0 else None, x.shape),
                r.input(1, w if p == 1 else None, w.shape))

    with ThreadPoolExecutor(4) as pool:
        values = list(pool.map(share, range(4)))
    functions = []
    for r in roles:
        def evaluate(a, b, r=r):
            z = r.matmul(a, b)
            sign = r.sign_bit(z)
            labels = r.sub(r.public(1, like=z), sign)
            return (r.open(z), r.open(labels), r.open(sign),
                    r.open(r.mul(z, z)), r.open(r.trunc(z, 3)))
        functions.append(evaluate)
    plans = [r.prepare_inference(fn, *args, check_at_send=True) for r, fn, args in zip(roles, functions, values)] if captured else None

    def work(p):
        try:
            fn = plans[p].run if captured else functions[p]
            return tuple(t.cpu() for t in fn(*values[p]))
        except BaseException as error:
            network.abort(error)
            raise

    with ThreadPoolExecutor(4) as pool:
        result = list(pool.map(work, range(4)))
    for score, labels, sign, square, truncated in result:
        assert torch.equal(score, expected)
        assert torch.equal(labels, (expected >= 0).long())
        assert torch.equal(sign, (expected < 0).long())
        assert torch.equal(square, (((expected*expected)&65535)^32768)-32768)
        assert torch.equal(truncated, expected >> 3)


def test_ring16_product_rejects_corrupted_wire():
    from rep4gpu.protocol.ring_inference import DeviceInferenceParty
    from rep4gpu.transports.device import DeviceIdealEndpoint
    from rep4gpu.protocol.runtime import make_seeds
    from rep4gpu.transports.byte import IdealNetwork
    seeds, network = make_seeds(), IdealNetwork(timeout=10)
    roles = [DeviceInferenceParty(p, DeviceIdealEndpoint(network.endpoint(p)),
                                  {e: s for e, s in seeds.items() if e != p},
                                  bits=16, binary_bits=32) for p in range(4)]
    original = roles[0]._exchange_device
    poisoned = [False]

    def corrupt(outgoing, sizes):
        outgoing = dict(outgoing)
        if not poisoned[0] and outgoing:
            peer = next(iter(outgoing))
            outgoing[peer] = outgoing[peer].clone()
            outgoing[peer][0] ^= 1
            poisoned[0] = True
        return original(outgoing, sizes)

    roles[0]._exchange_device = corrupt

    def worker(p):
        rt = roles[p]
        try:
            value = rt.public(torch.arange(8, dtype=torch.int64, device='cuda'))
            rt.mul(value, value)
            return None
        except BaseException as error:
            network.abort(error)
            return str(error)

    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(worker, range(4)))
    assert poisoned[0]
    assert any(result is not None and 'hash mismatch' in result for result in results)
