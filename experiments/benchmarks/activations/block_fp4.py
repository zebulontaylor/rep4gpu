"""CPU reference output quantization for MXFP4 and 1D NVFP4.

MXFP4 follows OCP MX v1.0 section 6.3's power-of-two scale selection:
https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
NVFP4 uses the documented two-level scale formula:
https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html

Inputs and scale arithmetic use FP32. Element conversion uses deterministic
nearest-even rounding and finite saturation. This is a declared software
quantizer, not an emulator of a native GPU kernel's instruction sequence.
No stochastic rounding, Hadamard transform, or 2D weight scaling is applied.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


_LEVELS = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=torch.float32)
_MIDPOINTS = (_LEVELS[:-1] + _LEVELS[1:]) / 2


@dataclass
class FP4Tensor:
    format: str
    codes: torch.Tensor
    scale_codes: torch.Tensor
    scale_values: torch.Tensor
    global_scale: torch.Tensor
    decoded: torch.Tensor
    block_size: int


def _encode_e2m1(values):
    """Encode signed finite values, retaining negative zero and saturating at six."""
    magnitude = values.abs()
    # Equality goes to the lower code unless that code is odd, in which case
    # the upper code has the even trailing significand bit.
    codes = (magnitude.unsqueeze(-1) > _MIDPOINTS).sum(-1)
    ties_up = ((magnitude.unsqueeze(-1) == _MIDPOINTS)
               & (torch.arange(7) % 2 == 1)).any(-1)
    codes += ties_up.long()
    return (codes.to(torch.uint8) | (torch.signbit(values).to(torch.uint8) << 3))


def quantize(values: torch.Tensor, format: str, scales_from: FP4Tensor | None = None):
    """Quantize CPU float64 values along their last dimension in rowwise blocks.

    Tail blocks are zero-padded for scale selection, then padding is removed
    from codes and decoded values. ``scales_from`` freezes the exact reference
    block and tensor scales instead of selecting new scales. It must have the
    same format and logical shape. Zero blocks use MX exponent -127 or NV block
    scale zero; an all-zero NV tensor uses global scale one. A zero effective
    scale produces signed-zero element codes.

    Reject nonfinite inputs, FP32 overflow, and nonzero NV tensor scales that
    underflow FP32. The decoded result multiplies the stored values in float64,
    exposing the mathematical value of the representation rather than adding
    another output rounding step.
    """
    if (not isinstance(values, torch.Tensor) or values.dtype != torch.float64
            or values.device.type != 'cpu' or values.ndim < 1 or values.numel() == 0
            or not bool(torch.isfinite(values).all())):
        raise ValueError('values must be a nonempty finite CPU float64 tensor with a last dimension')
    if format not in ('mxfp4', 'nvfp4'):
        raise ValueError('format must be mxfp4 or nvfp4')
    source = values.detach().to(torch.float32)
    if not bool(torch.isfinite(source).all()):
        raise ValueError('values overflow FP32 quantizer input')
    if format == 'nvfp4' and bool((values != 0).any()) and not bool((source != 0).any()):
        raise ValueError('nonzero NVFP4 tensor scale is outside supported FP32 range')
    block_size = 32 if format == 'mxfp4' else 16
    width = values.shape[-1]
    rows = math.prod(values.shape[:-1])
    blocks = (width + block_size - 1) // block_size
    padded = torch.nn.functional.pad(source.reshape(rows, width), (0, blocks*block_size-width))
    grouped = padded.reshape(rows, blocks, block_size)
    amax = grouped.abs().amax(-1)
    scale_shape = (*values.shape[:-1], blocks)
    if scales_from is not None:
        if (not isinstance(scales_from, FP4Tensor) or scales_from.format != format
                or scales_from.codes.shape != values.shape or scales_from.block_size != block_size
                or scales_from.scale_codes.shape != scale_shape):
            raise ValueError('frozen scales must match the format and logical tensor shape')
        scale_codes = scales_from.scale_codes.clone()
        scale_values = scales_from.scale_values.clone()
        global_scale = scales_from.global_scale.clone()
    elif format == 'mxfp4':
        # frexp gives an exact exponent classification even next to powers of two.
        exponent = torch.frexp(amax).exponent - 1 - 2
        exponent = torch.where(amax == 0, -127, exponent).clamp(-127, 127)
        scale_codes = (exponent + 127).to(torch.uint8).reshape(scale_shape)
        scale_values = torch.ldexp(torch.ones_like(amax), exponent).reshape(scale_shape)
        global_scale = torch.tensor(1., dtype=torch.float32)
    else:
        tensor_amax = source.abs().max()
        global_scale = (tensor_amax / (448. * 6.) if tensor_amax != 0
                        else torch.tensor(1., dtype=torch.float32))
        if global_scale == 0 or not bool(torch.isfinite(global_scale)):
            raise ValueError('nonzero NVFP4 tensor scale is outside supported FP32 range')
        local = ((amax / 6.) / global_scale).clamp(0., 448.)
        encoded = local.to(torch.float8_e4m3fn)
        scale_codes = encoded.view(torch.uint8).reshape(scale_shape)
        scale_values = encoded.float().reshape(scale_shape)
    effective = scale_values.reshape(rows, blocks) * global_scale
    if bool(((scale_values.reshape(rows, blocks) != 0) & (effective == 0)).any()):
        raise ValueError('effective block scale underflows FP32')
    safe = torch.where(effective == 0, torch.ones_like(effective), effective)
    normalized = grouped / safe.unsqueeze(-1)
    normalized = torch.where(effective.unsqueeze(-1) == 0,
                             torch.copysign(torch.zeros_like(normalized), grouped), normalized)
    codes = _encode_e2m1(normalized)
    magnitude = _LEVELS[(codes & 7).long()].double()
    signed = torch.copysign(magnitude, torch.where((codes & 8) != 0, -1., 1.).double())
    decoded = signed * scale_values.reshape(rows, blocks, 1).double() * global_scale.double()
    return FP4Tensor(format, codes.reshape(rows, -1)[:, :width].reshape(values.shape),
                     scale_codes, scale_values, global_scale,
                     decoded.reshape(rows, -1)[:, :width].reshape(values.shape), block_size)
