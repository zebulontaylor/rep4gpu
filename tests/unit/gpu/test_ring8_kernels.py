"""Ring8 kernels preserve exact residues and the existing ChaCha byte stream."""
import struct

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def stream(key, nonce, count):
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
    cipher = Cipher(algorithms.ChaCha20(key, struct.pack('<QQ', 0, nonce)), None)
    encryptor = cipher.encryptor()
    return encryptor.update(bytes(count)) + encryptor.finalize()


@pytest.mark.parametrize('count', [0, 1, 63, 64, 65, 1025])
@pytest.mark.parametrize('nonce', [0, 2**64-1])
def test_ring8_prg_matches_exact_stream(count, nonce):
    pytest.importorskip('cryptography')
    from rep4gpu.gpu.crypto.gpu_prg import generate
    from rep4gpu.gpu.crypto.gpu_prg_batch import generate_rows

    raw = bytes(range(32))
    key = torch.tensor(struct.unpack('<8I', raw), dtype=torch.uint32, device='cuda')
    expected = stream(raw, nonce, count)
    wide = generate(key, nonce, count, bits=8)
    assert wide.dtype == torch.int64
    assert wide.to(torch.uint8).cpu().numpy().tobytes() == expected
    for packed in (False, True):
        rows = generate_rows((key, None, key), (nonce, None, nonce ^ 1), count,
                             bits=8, packed=packed)
        assert rows.dtype == (torch.uint8 if packed else torch.int64)
        assert rows[0].to(torch.uint8).cpu().numpy().tobytes() == expected
        assert rows[1].to(torch.uint8).cpu().numpy().tobytes() == bytes(count)
        assert rows[2].to(torch.uint8).cpu().numpy().tobytes() == stream(raw, nonce ^ 1, count)
        torch.testing.assert_close(rows[0].to(torch.int64), wide, rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.int64, torch.uint8])
def test_ring8_graph_prg_uses_fresh_nonce_inputs(dtype):
    from rep4gpu.gpu.crypto.gpu_prg_batch import generate_rows
    from rep4gpu.gpu.crypto.gpu_prg_graph import generate_rows_into

    key = torch.arange(8, device='cuda', dtype=torch.int64).to(torch.uint32)
    keys = (key, None, key, key, None, key)
    count = 129
    nonces = torch.zeros(6, dtype=torch.uint64, device='cuda')
    output = torch.empty((6, count), dtype=dtype, device='cuda')
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        generate_rows_into(keys, nonces, output, bits=8)
    capture_stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        generate_rows_into(keys, nonces, output, bits=8)
    capture_stream.synchronize()
    previous = None
    for start in (7, 19):
        values = [start+i if k is not None else None for i, k in enumerate(keys)]
        nonces.copy_(torch.tensor([v or 0 for v in values], dtype=torch.uint64))
        graph.replay()
        expected = generate_rows(keys, values, count, bits=8, packed=dtype == torch.uint8)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        if previous is not None:
            assert not torch.equal(previous, output)
        previous = output.clone()


@pytest.mark.parametrize('shape', [(1, 17, 4), (19, 65, 7), (33, 512, 4), (5, 37, 1)])
@pytest.mark.parametrize('prepared', [False, True])
@pytest.mark.parametrize('dtype', [torch.int64, torch.uint8])
def test_ring8_local_products_match_cpu_full_ring(shape, prepared, dtype):
    from rep4gpu.gpu.matmul.kernels import local_products, prepare_weight

    m, k, n = shape
    generator = torch.Generator().manual_seed(816)
    xs = tuple(torch.randint(-256, 512, (m, k), generator=generator) for _ in range(3))
    ys = tuple(torch.randint(-256, 512, (k, n), generator=generator) for _ in range(3))
    xs[0].flatten()[:4] = torch.tensor([0, 127, 128, 255])
    a, b, c = xs
    d, e, f = ys
    expected = ((a+b)@d+a@e, (b+c)@e+b@f, a@f+c@d)
    inputs = tuple(x.to(device='cuda', dtype=dtype) for x in xs)
    weights = tuple(y.to(device='cuda', dtype=dtype) for y in ys)
    if prepared:
        weights = tuple(prepare_weight(y, bits=8) for y in weights)
        assert all(y.limbs.shape[0] == 1 for y in weights)
    actual = local_products(*inputs, *weights, bits=8)
    for got, want in zip(actual, expected):
        assert got.dtype == dtype
        torch.testing.assert_close(got.to(torch.int64).cpu(), want & 255, rtol=0, atol=0)


@pytest.mark.parametrize('backend,prepared', [('decode', False), ('tensor', False), ('tensor', True)])
def test_ring8_matrix_matches_cpu(backend, prepared):
    from rep4gpu.gpu.matmul.kernels import prepare_weight, ring_matmul

    generator = torch.Generator().manual_seed(81)
    x = torch.randint(0, 256, (5, 35), generator=generator)
    y = torch.randint(0, 256, (35, 7), generator=generator)
    a, b = (t.to(device='cuda', dtype=torch.uint8) for t in (x, y))
    if prepared:
        b = prepare_weight(b, bits=8)
    actual = ring_matmul(a, b, bits=8, backend=backend)
    torch.testing.assert_close(actual.cpu(), (x@y) & 255, rtol=0, atol=0)
