"""Distinguish bit changes, numerical distance, and nonfinite comparisons."""
import pytest
import torch

from experiments.benchmarks.activations.float_bit_metrics import FORMATS, compare_formats


def test_adjacent_values_can_flip_many_bits():
    previous = torch.tensor([0x3bff], dtype=torch.int16).view(torch.float16).double()
    actual = torch.tensor([1.], dtype=torch.float64)
    metric = compare_formats(actual, previous)['fp16']
    assert metric['encoded_mismatch_count'] == 1
    assert metric['ulp_distance_max'] == 1
    assert metric['hamming_total'] == 11
    assert metric['hamming_histogram']['11'] == 1
    assert sum(metric['per_bit_mismatch_counts_lsb_first']) == 11


def test_signed_zero_is_a_bit_difference_but_not_numerical_error():
    actual = torch.tensor([-0., 0.], dtype=torch.float64)
    reference = torch.tensor([0., -0.], dtype=torch.float64)
    for metric in compare_formats(actual, reference).values():
        assert metric['exact_match_count'] == 0
        assert metric['signed_zero_only_mismatch_count'] == 2
        assert metric['numerical_mismatch_count'] == 0
        assert metric['hamming_mean'] == 1
        assert metric['ulp_distance_max'] == 0
        assert metric['cast_max_abs_error_finite_pairs'] == 0
        assert metric['per_bit_mismatch_counts_lsb_first'][-1] == 2


def test_negative_to_positive_ulp_ranks_merge_zero():
    value = 2. ** -24  # Smallest positive FP16 subnormal.
    actual = torch.tensor([-value, -0.], dtype=torch.float64)
    reference = torch.tensor([value, value], dtype=torch.float64)
    metric = compare_formats(actual, reference)['fp16']
    assert metric['ulp_distance_max'] == 2
    assert metric['ulp_distance_mean'] == 1.5


def test_fp8_formats_and_identity():
    values = torch.tensor([[-4., -1., 0., 0.5, 1., 4.]], dtype=torch.float64)
    metrics = compare_formats(values, values)
    assert set(metrics) == set(FORMATS)
    for name, metric in metrics.items():
        assert metric['format_bits'] == (8 if name.startswith('fp8') else 32 if name == 'fp32' else 16)
        assert metric['exact_match_fraction'] == 1
        assert metric['finite_pair_count'] == values.numel()
        assert metric['hamming_total'] == 0
        assert metric['ulp_distance_max'] == 0


def test_nonfinite_pairs_are_counted_and_excluded_from_distances():
    actual = torch.tensor([float('inf'), -float('inf'), float('nan'), 1.], dtype=torch.float64)
    reference = torch.tensor([float('inf'), 0., float('nan'), 2.], dtype=torch.float64)
    metrics = compare_formats(actual, reference)
    for metric in metrics.values():
        assert metric['source_actual_counts']['finite'] == 1
        assert metric['source_actual_counts']['positive_infinity'] == 1
        assert metric['source_actual_counts']['negative_infinity'] == 1
        assert metric['source_actual_counts']['nan'] == 1
        assert metric['finite_pair_count'] + metric['nonfinite_pair_count'] == 4
    metric = metrics['fp16']
    assert metric['finite_pair_count'] == 1
    assert metric['nonfinite_pair_count'] == 3
    assert metric['cast_max_abs_error_finite_pairs'] == 1
    assert metric['numerical_mismatch_count'] == 3
    all_nan = compare_formats(actual[2:3], reference[2:3])['fp32']
    assert all_nan['ulp_distance_max'] is None
    assert all_nan['cast_max_abs_error_finite_pairs'] is None


def test_scalar_input_and_invalid_inputs():
    value = torch.tensor(1., dtype=torch.float64)
    assert compare_formats(value, value)['fp16']['elements'] == 1
    for bad in [value.float(), torch.empty(0, dtype=torch.float64), value.reshape(1)]:
        with pytest.raises(ValueError):
            compare_formats(value, bad)
