"""Check FP4 rounding, scale fields, padding, and frozen-scale comparisons."""
import pytest
import torch

from experiments.benchmarks.activations.block_fp4 import _encode_e2m1, quantize


def test_e2m1_levels_midpoints_signs_and_saturation():
    positive = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
    assert _encode_e2m1(positive).tolist() == list(range(8))
    assert _encode_e2m1(-positive).tolist() == list(range(8, 16))
    midpoints = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.])
    assert _encode_e2m1(midpoints).tolist() == [0, 2, 2, 4, 4, 6, 6]
    assert _encode_e2m1(torch.tensor([-100., 100.])).tolist() == [15, 7]


def test_mxfp4_known_scale_and_decoded_values():
    value = torch.tensor([[0., .5, 1., 1.5, 2., 3., 4., 6.]], dtype=torch.float64)
    result = quantize(value, 'mxfp4')
    assert result.scale_codes.tolist() == [[127]]
    assert result.global_scale.item() == 1
    assert torch.equal(result.decoded, value)
    assert result.codes.tolist() == [list(range(8))]


def test_mxfp4_power_of_two_boundary():
    lower = torch.nextafter(torch.tensor(4., dtype=torch.float32), torch.tensor(0.)).double()
    values = torch.stack((lower, torch.tensor(4., dtype=torch.float64))).reshape(2, 1)
    result = quantize(values, 'mxfp4')
    assert result.scale_codes.tolist() == [[126], [127]]
    assert result.decoded.tolist() == [[3.], [4.]]


@pytest.mark.parametrize('format,block', [('mxfp4', 32), ('nvfp4', 16)])
def test_rows_and_partial_blocks_do_not_share_scales(format, block):
    values = torch.zeros((2, block+1), dtype=torch.float64)
    values[0, 0] = 6
    values[0, -1] = .5
    values[1, :] = 96
    result = quantize(values, format)
    assert result.codes.shape == values.shape
    assert result.scale_codes.shape == (2, 2)
    assert result.block_size == block
    assert result.scale_codes[0, 0] != result.scale_codes[0, 1]
    assert result.scale_codes[0, 0] != result.scale_codes[1, 0]
    assert result.decoded[1, 0].item() == pytest.approx(96., rel=1e-7)


@pytest.mark.parametrize('format', ['mxfp4', 'nvfp4'])
def test_zero_tensor_retains_sign_and_has_finite_scales(format):
    values = torch.tensor([[0., -0.]], dtype=torch.float64)
    result = quantize(values, format)
    assert result.codes.tolist() == [[0, 8]]
    assert torch.equal(torch.signbit(result.decoded), torch.signbit(values))
    assert result.scale_codes.tolist() == [[0]]
    assert result.global_scale.item() == 1


def test_nvfp4_two_level_scales_and_frozen_comparison():
    values = torch.tensor([[6., 3., 0., -0.]], dtype=torch.float64)
    result = quantize(values, 'nvfp4')
    assert result.scale_codes.item() == 126  # E4M3 encoding of 448.
    assert result.scale_values.item() == 448
    assert result.global_scale == torch.tensor(6./2688., dtype=torch.float32)
    assert result.codes.tolist() == [[7, 5, 0, 8]]
    scaled = quantize(values*2, 'nvfp4')
    assert torch.equal(result.codes, scaled.codes)
    assert torch.equal(result.scale_codes, scaled.scale_codes)
    assert scaled.global_scale == 2*result.global_scale
    assert not torch.equal(result.decoded, scaled.decoded)
    frozen = quantize(values*2, 'nvfp4', scales_from=result)
    assert torch.equal(result.scale_codes, frozen.scale_codes)
    assert result.global_scale == frozen.global_scale
    assert frozen.codes.tolist() == [[7, 7, 0, 8]]


def test_invalid_input_and_unrepresentable_scales():
    for values in (torch.tensor([float('nan')], dtype=torch.float64),
                   torch.tensor([float('inf')], dtype=torch.float64),
                   torch.tensor([1e300], dtype=torch.float64),
                   torch.tensor([1.], dtype=torch.float32),
                   torch.empty(0, dtype=torch.float64)):
        with pytest.raises(ValueError):
            quantize(values, 'nvfp4')
    with pytest.raises(ValueError, match='scale'):
        quantize(torch.tensor([1e-43], dtype=torch.float64), 'nvfp4')
    value = torch.ones((1, 2), dtype=torch.float64)
    with pytest.raises(ValueError):
        quantize(value, 'other')
    with pytest.raises(ValueError):
        quantize(value, 'nvfp4', scales_from=quantize(value, 'mxfp4'))
