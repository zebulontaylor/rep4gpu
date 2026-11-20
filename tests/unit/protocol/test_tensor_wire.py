"""Wire packing preserves exact ring values and independent storage."""
import numpy as np
import pytest
import torch

from rep4gpu.protocol.runtime import bytes_tensor, tensor_bytes


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('bits', [16, 32, 64])
@pytest.mark.parametrize('empty', [False, True])
def test_wire_roundtrip_noncontiguous_full_width(device, bits, empty):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    base = torch.tensor([-(1 << 63), -(1 << 63) + 1, -1, 0,
                         (1 << 32) - 1, (1 << 32), 1, (1 << 63) - 1], dtype=torch.int64)
    source = base.reshape(2, 4).t().to(device)
    if empty:
        source = source[:0]
    expected = source.cpu().numpy().astype({16: '<u2', 32: '<u4', 64: '<i8'}[bits]).tobytes()
    encoded = tensor_bytes(source, bits)
    assert encoded == expected
    assert len(encoded) == source.numel() * (bits // 8)
    mutable = bytearray(encoded)
    restored = bytes_tensor(mutable, source.shape, device, bits)
    mutable[:] = b'\x00' * len(mutable)
    actual = restored.cpu()
    wanted = source.cpu() & ((1 << bits)-1) if bits < 64 else source.cpu()
    assert torch.equal(actual, wanted)
    assert restored.dtype == torch.int64
    assert restored.device.type == device


@pytest.mark.parametrize('bits', [16, 32, 64])
def test_cpu_wire_owns_input_buffer(bits):
    data = bytearray(np.array([1, 2, 3], dtype={16: '<u2', 32: '<u4', 64: '<i8'}[bits]).tobytes())
    decoded = bytes_tensor(data, (3,), 'cpu', bits)
    data[:] = b'\x00' * len(data)
    assert decoded.tolist() == [1, 2, 3]
