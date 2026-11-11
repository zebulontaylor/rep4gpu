import numpy as np
import pytest
import torch

from rep4gpu.gpu.matmul.kernels import ring_matmul

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def random_ring(shape, seed):
    rng = np.random.default_rng(seed)
    return rng.bit_generator.random_raw(np.prod(shape)).view(np.int64).reshape(shape)


@pytest.mark.parametrize("bits", [32, 64])
@pytest.mark.parametrize("backend", ["decode", "tensor"])
@pytest.mark.parametrize("shape", [(1, 128, 128), (16, 137, 71), (128, 128, 128), (3, 0, 7)])
def test_full_ring(bits, backend, shape):
    m, k, n = shape
    a = random_ring((m, k), 7)
    b = random_ring((k, n), 13)
    expected = a @ b
    if bits == 32:
        expected &= 0xffffffff
    actual = ring_matmul(torch.from_numpy(a).cuda(), torch.from_numpy(b).cuda(),
                         bits, backend=backend)
    np.testing.assert_array_equal(actual.cpu().numpy(), expected)


@pytest.mark.parametrize("bits", [32, 64])
@pytest.mark.parametrize("backend", ["decode", "tensor"])
def test_carries_and_sign(bits, backend):
    values = torch.tensor([0, -1, -2**63, 2**63-1, 127, 128, 255, 256,
                           0x7f7f7f7f7f7f7f7f, -0x7f7f7f7f7f7f7f80], dtype=torch.int64)
    a = values.repeat(17, 7)
    b = values.repeat(23, 7).T.contiguous()
    expected = a @ b
    if bits == 32:
        expected &= 0xffffffff
    actual = ring_matmul(a.cuda(), b.cuda(), bits, backend=backend)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("bits", [32, 64])
@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("m", [1, 17])
def test_fused_local_products(bits, prepared, m):
    from rep4gpu.gpu.matmul.kernels import local_products, prepare_weight

    xs = [torch.from_numpy(random_ring((m, 71), i)).cuda() for i in range(3)]
    ys = [torch.from_numpy(random_ring((71, 39), i+3)).cuda() for i in range(3)]
    xc, yc = [x.cpu() for x in xs], [y.cpu() for y in ys]
    expected = [(xc[0]+xc[1])@yc[0]+xc[0]@yc[1],
                (xc[1]+xc[2])@yc[1]+xc[1]@yc[2],
                xc[0]@yc[2]+xc[2]@yc[0]]
    if bits == 32:
        expected = [x & 0xffffffff for x in expected]
    if prepared:
        ys = [prepare_weight(y, bits) for y in ys]
    for actual, reference in zip(local_products(*xs, *ys, bits), expected):
        torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)


@pytest.mark.parametrize("bits", [32, 64])
def test_prepared_weight(bits):
    from rep4gpu.gpu.matmul.kernels import prepare_weight

    a = random_ring((17, 71), 1)
    b = random_ring((71, 39), 2)
    expected = a @ b
    if bits == 32:
        expected &= 0xffffffff
    packed = prepare_weight(torch.from_numpy(b).cuda(), bits)
    actual = ring_matmul(torch.from_numpy(a).cuda(), packed, bits)
    np.testing.assert_array_equal(actual.cpu().numpy(), expected)


@pytest.mark.parametrize("bits", [32, 64])
def test_prepared_weight_retains_only_packed_storage_by_default(bits):
    from rep4gpu.gpu.matmul.kernels import prepare_weight

    weight = torch.ones((16, 32), dtype=torch.int64, device="cuda")
    packed = prepare_weight(weight, bits)
    assert packed.original is None
    assert [key for key, value in vars(packed).items() if isinstance(value, torch.Tensor)] == ["limbs"]
    assert packed.limbs.numel()*packed.limbs.element_size() == weight.numel()*(bits//8)
    retained = prepare_weight(weight, bits, keep_original=True)
    assert retained.original is weight
    assert set(key for key, value in vars(retained).items() if isinstance(value, torch.Tensor)) == {"limbs", "original"}


@pytest.mark.parametrize("backend", ["auto", "decode", "tensor"])
@pytest.mark.parametrize("shape", [(1,128,128),(16,71,39),(128,128,128)])
def test_compact32_matmul(backend, shape):
    from rep4gpu.gpu.matmul.kernels import CompactWeight, prepare_weight

    m,k,n=shape
    a=random_ring((m,k),31)
    b=random_ring((k,n),43)
    weight=prepare_weight(torch.from_numpy(b).cuda(),32,storage="words32")
    assert isinstance(weight,CompactWeight)
    actual=ring_matmul(torch.from_numpy(a).cuda(),weight,32,backend=backend)
    np.testing.assert_array_equal(actual.cpu().numpy(),(a@b)&0xffffffff)


@pytest.mark.parametrize("m",[1,17])
def test_compact32_local_products(m):
    from rep4gpu.gpu.matmul.kernels import local_products, prepare_weight

    xs=[torch.from_numpy(random_ring((m,71),i+21)).cuda() for i in range(3)]
    ys=[torch.from_numpy(random_ring((71,39),i+31)).cuda() for i in range(3)]
    expected=local_products(*xs,*ys,32)
    compact=[prepare_weight(y,32,storage="words32") for y in ys]
    for actual,want in zip(local_products(*xs,*compact,32),expected):
        torch.testing.assert_close(actual,want,rtol=0,atol=0)


def test_compact32_memory_and_validation():
    from rep4gpu.gpu.matmul.kernels import prepare_weight

    raw=torch.arange(128,device="cuda",dtype=torch.int64).reshape(8,16)
    compact=prepare_weight(raw,32,storage="words32")
    assert compact.words.dtype==torch.int32
    assert compact.words.is_contiguous()
    assert compact.words.numel()*compact.words.element_size()==raw.numel()*4
    assert [name for name,value in vars(compact).items() if isinstance(value,torch.Tensor)]==["words"]
    assert compact.words.data_ptr()!=raw.data_ptr()
    with pytest.raises(ValueError,match="bits=32"):
        prepare_weight(raw,64,storage="words32")
    with pytest.raises(ValueError,match="keep_original"):
        prepare_weight(raw,32,storage="words32",keep_original=True)
    with pytest.raises(ValueError,match="bits=32"):
        ring_matmul(raw,compact,64)


def test_compact32_sign_and_carry_edges():
    from rep4gpu.gpu.matmul.kernels import prepare_weight

    b=torch.tensor([-2**63,-1,2**63-1,0,1,2**32-1,2**31,2**32],dtype=torch.int64).reshape(8,1)
    a=torch.tensor([[1,-1,127,128,255,256,-2**63,2**63-1]],dtype=torch.int64)
    compact=prepare_weight(b.cuda(),32,storage="words32")
    expected=(a@b)&0xffffffff
    for backend in ("decode","tensor"):
        actual=ring_matmul(a.cuda(),compact,32,backend=backend)
        torch.testing.assert_close(actual.cpu(),expected,rtol=0,atol=0)
