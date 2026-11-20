import numpy as np
import pytest
import torch

from experiments.benchmarks.inference.sparse_classifier_native import pack_sparse, sparse_cuda


def test_packing_preserves_sparse_columns_and_certifies_range():
    w1 = np.array([[1, 0, -1], [0, 2, 1], [-1, 0, 0]], dtype=np.int64)
    b1, w2, b2 = np.array([1, 0, -1]), np.array([[2], [-1], [1]]), np.array([-3])
    packed = pack_sparse(w1, b1, w2, b2)
    restored = np.zeros_like(w1)
    for h in range(w1.shape[1]):
        np.add.at(restored[:, h], packed['index'][h], packed['weight'][h])
    np.testing.assert_array_equal(restored, w1)
    assert packed['bound'] < 32768
    with pytest.raises(ValueError, match='overflow'):
        pack_sparse(w1 * 100000, b1, w2, b2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_sparse_cuda_exact_scores_and_labels_with_padding():
    rng = np.random.default_rng(23)
    x = rng.integers(0, 2, (17, 13), dtype=np.uint8)
    w1 = rng.integers(-1, 2, (13, 7), dtype=np.int64)
    w1[rng.random(w1.shape) > 0.3] = 0
    b1, w2, b2 = rng.integers(-1, 2, 7), rng.integers(-2, 3, (7, 1)), np.array([-7])
    packed = pack_sparse(w1, b1, w2, b2)
    gpu = {k: torch.as_tensor(v, device='cuda') if isinstance(v, np.ndarray) else v for k, v in packed.items()}
    expected = (((x.astype(np.int64) @ w1 + b1)**2) @ w2 + b2).reshape(-1)
    value = torch.as_tensor(x, device='cuda')
    np.testing.assert_array_equal(sparse_cuda(value, gpu).cpu().numpy(), expected)
    np.testing.assert_array_equal(sparse_cuda(value, gpu, labels=True).cpu().numpy(), expected >= 0)
