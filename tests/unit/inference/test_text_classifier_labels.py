from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from rep4gpu.models.text_classifier import nonnegative_label, linear_labels
from rep4gpu.protocol.runtime import Party, make_seeds
from rep4gpu.protocol.transport import IdealNetwork


def run_four(fn, bits, device):
    seeds, network = make_seeds(), IdealNetwork()
    parties = [Party(p, network.endpoint(p), {i: s for i, s in seeds.items() if i != p},
                     bits=bits, binary_bits=bits, device=device) for p in range(4)]
    with ThreadPoolExecutor(4) as pool:
        return list(pool.map(fn, parties))


@pytest.mark.parametrize('bits', [32, 64])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_signed_boundaries_and_private_linear_labels(bits, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    minimum = -(1 << (bits - 1))
    scores = torch.tensor([minimum, minimum + 1, -17, -1, 0, 1, 17, -minimum - 1])
    x = torch.tensor([[0, 0, 1], [2, 1, 1], [1, 4, 1]])
    w = torch.tensor([[3], [-2], [-1]])

    def work(rt):
        secret = rt.input(0, scores if rt.party == 0 else None, scores.shape)
        labels = nonnegative_label(rt, secret)
        assert labels.binary
        boundary = rt.open(labels).cpu()
        inputs = rt.input(0, x if rt.party == 0 else None, x.shape)
        weights = rt.input(1, w if rt.party == 1 else None, w.shape)
        return boundary, rt.open(linear_labels(rt, inputs, weights)).cpu()

    for boundary, labels in run_four(work, bits, device):
        assert torch.equal(boundary, (scores >= 0).long())
        assert torch.equal(labels, (x @ w >= 0).long())
