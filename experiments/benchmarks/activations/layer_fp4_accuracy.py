"""Apply block FP4 reference quantizers to saved four-party activation outputs.

This measures output encodings, not native FP4 GPU execution. Dynamic scales
and reference-frozen scales are separate experiments. Payload bits alone do
not establish equality when scale metadata differs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.benchmarks.activations.block_fp4 import quantize
from experiments.benchmarks.activations.layer_bit_accuracy import activation, fixture

SOURCES = {
    'mxfp4': 'https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf',
    'nvfp4': 'https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html',
}


def bit_counts(a, b, bits):
    xor = (a.reshape(-1).to(torch.int64) ^ b.reshape(-1).to(torch.int64)) & ((1 << bits)-1)
    per_bit = (xor[:, None] >> torch.arange(bits)) & 1
    counts = per_bit.sum(1)
    return {'words': xor.numel(), 'mismatched_words': int((xor != 0).sum()),
            'exact_fraction': float((xor == 0).double().mean()),
            'hamming_total': int(counts.sum()), 'hamming_mean': float(counts.double().mean()),
            'hamming_max': int(counts.max()), 'per_bit_mismatches_lsb_first': per_bit.sum(0).tolist()}


def compare(actual, reference, fmt, frozen=False):
    ref = quantize(reference, fmt)
    got = quantize(actual, fmt, scales_from=ref if frozen else None)
    payload = bit_counts(got.codes, ref.codes, 4)
    scales = bit_counts(got.scale_codes, ref.scale_codes, 8)
    global_bits = bit_counts(got.global_scale.reshape(1).view(torch.int32),
                             ref.global_scale.reshape(1).view(torch.int32), 32)
    scale_equal = got.scale_codes == ref.scale_codes
    expanded = scale_equal.repeat_interleave(got.block_size, dim=-1)[..., :actual.shape[-1]]
    complete = (got.codes == ref.codes) & expanded & (global_bits['mismatched_words'] == 0)
    decoded_equal = got.decoded == ref.decoded
    zero_sign = (got.decoded == 0) & (ref.decoded == 0) & ((got.codes ^ ref.codes) == 8)
    error = got.decoded-ref.decoded
    return {'elements': actual.numel(), 'block_size': got.block_size,
            'payload': payload, 'block_scales': scales, 'global_scale': global_bits,
            'reference_global_scale': float(ref.global_scale), 'actual_global_scale': float(got.global_scale),
            'full_element_encoding_match_fraction': float(complete.double().mean()),
            'full_tensor_encoding_equal': bool(complete.all()),
            'decoded_equal_fraction': float(decoded_equal.double().mean()),
            'signed_zero_only_payload_mismatches': int(zero_sign.sum()),
            'total_differing_stored_bits': payload['hamming_total'] + scales['hamming_total'] + global_bits['hamming_total'],
            'decoded_max_abs_difference': float(error.abs().max()),
            'decoded_rmse_difference': float(error.square().mean().sqrt()),
            'reference_quantization_max_abs_error': float((ref.decoded-reference).abs().max()),
            'actual_quantization_max_abs_error': float((got.decoded-actual).abs().max())}


def operand_emulation(data, kind, fmt):
    """Quantize reduction-axis operands, then compute with FP32 PyTorch on CPU."""
    values = data['floats']
    x = quantize(values[0], fmt).decoded.float()
    gates = []
    for index in ((1, 3) if kind == 'swiglu' else (1,)):
        # Rowwise quantization of W.T groups the reduction dimension for each output.
        w = quantize(values[index].T.contiguous(), fmt).decoded.T.float()
        gates.append(x @ w + values[index+1].float())
    return activation(kind, gates[0], gates[1] if kind == 'swiglu' else None).double()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('experiments/results/activations/layer_bit_accuracy_q24.json'))
    parser.add_argument('--output', type=Path, default=Path('experiments/results/activations/layer_fp4_accuracy.json'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    old = json.loads(args.input.read_text())
    config = old['configuration']
    archive_path = args.input.with_suffix('.npz')
    archive = np.load(archive_path, allow_pickle=False)
    report = {'sources': SOURCES, 'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                                for p in (args.input, archive_path)},
              'methodology': 'Software deterministic RNE E2M1 block quantization of preserved live CUDA Rep4/TEE-simulation outputs. Quantizer inputs and arithmetic use FP32. MXFP4 uses OCP section 6.3 floor-exponent scaling with groups of 32. NVFP4 uses 1D groups of 16, FP8 E4M3 block scales and amax-derived FP32 tensor scale; no stochastic cast, Hadamard transform, or 2D weight scaling. Groups run along the last dimension per row; final blocks are zero padded, excluded from payload statistics. Dynamic recomputes all scales independently; frozen reuses all reference scales. No native FP4 GPU kernels are measured. Quantized-operand emulation uses FP4-dequantized operands then FP32 CPU matmul, original FP32 bias and FP32 activation, not vendor GEMM emulation.',
              'results': []}
    cache = {}
    for row in old['results']:
        suite, kind, f = row['suite'], row['kind'], row['fractional_bits']
        key = (suite, kind, f)
        if key not in cache:
            data = fixture(suite, f, config['batch'], config['width'], config['hidden'], config['grid_elements'])
            references = {name: activation(kind, values[0], values[1])
                          for name, values in ((n, data[n]) for n in ('original','quantized','floored'))}
            references['tee'] = torch.from_numpy(archive[f'{suite}_{kind}_q{f}_tee_simulation_0']).double()/(1 << f)
            emulated = {fmt: operand_emulation(data, kind, fmt) for fmt in ('mxfp4','nvfp4')} if suite == 'dense' else {}
            cache[key] = references, emulated
        refs, emulated = cache[key]
        entry = {name: row[name] for name in ('suite','kind','fractional_bits','backend','elements_per_repeat')}
        entry['runs'] = []
        for repeat in range(row['repeats']):
            actual = torch.from_numpy(archive[f"{suite}_{kind}_q{f}_{row['backend']}_{repeat}"]).double()/(1 << f)
            result = {'repeat': repeat, 'references': {}}
            for name, reference in refs.items():
                result['references'][name] = {fmt: {mode: compare(actual, reference, fmt, frozen=mode=='frozen')
                                                    for mode in ('dynamic','frozen')}
                                              for fmt in ('mxfp4','nvfp4')}
            result['quantized_operand_emulation'] = {fmt: {
                'max_abs_difference_before_output_quantization': float((actual-reference).abs().max()),
                'dynamic': compare(actual, reference, fmt),
                'frozen': compare(actual, reference, fmt, frozen=True)}
                for fmt, reference in emulated.items()}
            entry['runs'].append(result)
        report['results'].append(entry)
    paths = [Path(__file__), Path('experiments/benchmarks/activations/block_fp4.py'),
             Path('experiments/benchmarks/activations/layer_bit_accuracy.py')]
    report['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f'Wrote {len(report["results"])} cases to {args.output}')


if __name__ == '__main__':
    main()
