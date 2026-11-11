"""Fresh masked tables preserve lookup correctness and one-time material use."""
from concurrent.futures import ThreadPoolExecutor
import copy

import pytest
import torch

from rep4gpu.arithmetic.private_lut import prepare_lut
from rep4gpu.protocol.runtime import Party, make_seeds
from rep4gpu.protocol.transport import IdealNetwork


def four(work, *, bits=64, device='cpu'):
    torch.set_num_threads(1)
    seeds, network = make_seeds(), IdealNetwork(timeout=120)

    def run(role):
        try:
            endpoint = network.endpoint(role)
            selected = {i: seed for i, seed in seeds.items() if i != role}
            if device == 'cpu':
                rt = Party(role, endpoint, selected, bits=bits, binary_bits=bits)
            else:
                from rep4gpu.models.device_inference import DeviceInferenceParty
                from rep4gpu.protocol.device_transport import DeviceIdealEndpoint
                rt = DeviceInferenceParty(role, DeviceIdealEndpoint(endpoint, device=device),
                    selected, bits=bits, binary_bits=bits, device=device,
                    optimized=True, gpu_open=True, rounding='floor')
            return work(rt)
        except BaseException as error:
            network.abort(error)
            raise
    with ThreadPoolExecutor(4) as pool:
        return list(pool.map(run, range(4)))


def measure_lookup(size=4096, elements=3, bits=64, device='cpu'):
    """Count recurring preparation separately from one checked masked opening."""
    values = torch.arange(size, dtype=torch.int64).square()-size
    indices = torch.arange(elements, dtype=torch.int64)*173-31

    def work(rt):
        x = rt.input(0, indices if rt.party == 0 else None, indices.shape)
        start = rt.transport.bytes_sent
        material = prepare_lut(rt, values, x.shape)
        preparation = rt.transport.bytes_sent-start
        memory = material.local_storage_bytes
        opened = []
        original = rt.open

        def checked(value):
            assert all(bool(((part >= 0) & (part < size)).all()) for part in value.parts)
            result = original(value)
            opened.append(result.clone())
            return result
        rt.open = checked
        start = rt.transport.bytes_sent
        result = material.lookup(x, runtime=rt)
        online = rt.transport.bytes_sent-start
        assert material.consumed and len(opened) == 1
        rt.open = original
        with pytest.raises(RuntimeError, match='consumed'):
            material.lookup(x)
        return original(result).cpu(), preparation, online, memory

    rows = four(work, bits=bits, device=device)
    expected = values[indices & (size-1)]
    assert all(torch.equal(row[0], expected) for row in rows)
    preparation, online = sum(row[1] for row in rows), sum(row[2] for row in rows)
    assert preparation == 6*size*elements*(bits//8)+6*32
    assert online == 4*elements*(bits//8)+4*32
    return dict(bits=bits, size=size, elements=elements, device=device,
                preprocessing_bytes=preparation, online_bytes=online,
                total_bytes=preparation+online, local_storage_bytes=rows[0][3])


@pytest.mark.parametrize('bits', [32, 64])
def test_dense_12bit_lut_traffic_and_correctness(bits):
    measure_lookup(bits=bits)


@pytest.mark.parametrize('bits', [32, 64])
def test_all_small_domain_inputs_and_ring_extremes(bits):
    values = torch.tensor([-7, 21, 0, 4, -42, 19, 3, -1], dtype=torch.int64)
    indices = torch.tensor(list(range(8))+[-(1 << (bits-1)), (1 << (bits-1))-1,
                                          -9, 8], dtype=torch.int64).reshape(3, 4)

    def work(rt):
        x = rt.input(0, indices if rt.party == 0 else None, indices.shape)
        material = prepare_lut(rt, values, x.shape)
        return rt.open(material.lookup(x)).cpu()
    for result in four(work, bits=bits):
        assert torch.equal(result, values[indices & 7])


def test_resource_binding_copy_rejection_and_burn_on_failure():
    def work(rt):
        x = rt.public(torch.tensor([1, 2], dtype=torch.int64))
        material = prepare_lut(rt, torch.arange(8, dtype=torch.int64), x.shape)
        with pytest.raises(ValueError, match='another runtime'):
            material.lookup(x, runtime=object())
        with pytest.raises(TypeError, match='copied'):
            copy.copy(material)
        with pytest.raises(TypeError, match='copied'):
            copy.deepcopy(material)
        original = rt.open
        def fail(_):
            raise RuntimeError('opening failed')
        rt.open = fail
        with pytest.raises(RuntimeError, match='opening failed'):
            material.lookup(x)
        rt.open = original
        assert material.consumed
        with pytest.raises(RuntimeError, match='consumed'):
            material.lookup(x)
    four(work)


def test_bad_public_configuration_rejected_before_protocol():
    def work(rt):
        for values in (torch.arange(3), torch.arange(8).double(), torch.zeros((2, 4), dtype=torch.int64)):
            with pytest.raises(ValueError):
                prepare_lut(rt, values, (2,))
        for shape in ((0,), (-1,), (1.5,)):
            with pytest.raises(ValueError):
                prepare_lut(rt, torch.arange(8), shape)
        material = prepare_lut(rt, torch.arange(8), (2,))
        with pytest.raises(ValueError, match='prepared shape'):
            material.lookup(rt.public(torch.tensor(1)))
        with pytest.raises(ValueError, match='arithmetic'):
            material.lookup(rt.public(torch.ones(2, dtype=torch.int64), binary=True))
        assert not material.consumed
    four(work)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('bits', [32, 64])
def test_cuda_private_lookup(bits):
    measure_lookup(size=16, elements=4, bits=bits, device='cuda:0')
