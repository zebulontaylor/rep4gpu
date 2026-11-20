"""Scale metadata participates in block-format equality."""
import torch

from experiments.benchmarks.activations.layer_fp4_accuracy import compare


def test_identical_mxfp4_nibbles_do_not_hide_changed_scale():
    reference = torch.ones((2, 32), dtype=torch.float64)
    result = compare(reference*2, reference, 'mxfp4')
    assert result['payload']['hamming_total'] == 0
    assert result['block_scales']['mismatched_words'] == 2
    assert result['full_element_encoding_match_fraction'] == 0
    assert result['decoded_equal_fraction'] == 0


def test_nvfp4_global_scale_counts_once_and_can_be_frozen():
    reference = torch.ones((2, 16), dtype=torch.float64)
    actual = reference*(1+2**-20)
    dynamic = compare(actual, reference, 'nvfp4')
    assert dynamic['payload']['hamming_total'] == 0
    assert dynamic['block_scales']['hamming_total'] == 0
    assert dynamic['global_scale']['mismatched_words'] == 1
    assert dynamic['total_differing_stored_bits'] == dynamic['global_scale']['hamming_total']
    assert dynamic['full_tensor_encoding_equal'] is False
    frozen = compare(actual, reference, 'nvfp4', frozen=True)
    assert frozen['full_tensor_encoding_equal'] is True
    assert frozen['total_differing_stored_bits'] == 0


def test_padding_excluded_and_signed_zero_reported():
    reference = torch.zeros((1, 17), dtype=torch.float64)
    actual = -reference
    result = compare(actual, reference, 'nvfp4')
    assert result['payload']['words'] == 17
    assert result['block_scales']['words'] == 2
    assert result['signed_zero_only_payload_mismatches'] == 17
    assert result['payload']['hamming_total'] == 17
    assert result['decoded_equal_fraction'] == 1
