import hashlib

import numpy as np
import pytest

from experiments.examples.compact_hash_text_classifier import choose_family
from experiments.examples.text_classifier import featurize


@pytest.mark.parametrize('features', [64, 128])
def test_compact_binary_feature_contract(features):
    texts = ['alpha ' * 129 + 'beta', ' '.join(f'word{i}' for i in range(256)), '']
    encoded = featurize(texts, features=features, max_tokens=128)
    counts = encoded[:, :-1]
    binary = (counts > 0).astype(np.int32)
    assert binary.shape == (3, features)
    assert set(np.unique(binary)).issubset({0, 1})
    assert np.all(binary.sum(axis=1) <= min(features, 128))
    assert np.all(counts.sum(axis=1) <= 128)
    alpha = int.from_bytes(hashlib.sha256(b'alpha').digest()[:8], 'little') % features
    beta = int.from_bytes(hashlib.sha256(b'beta').digest()[:8], 'little') % features
    assert alpha != beta
    assert counts[0, alpha] == 128 and counts[0, beta] == 0
    assert binary[0].sum() == 1 and binary[2].sum() == 0


def test_family_selection_ignores_heldout_quality():
    reports = [
        dict(features=64, selected=dict(validation_f1=.9, validation=dict(spam_precision=.95), hidden=4), test=dict(accuracy=.99)),
        dict(features=128, selected=dict(validation_f1=.91, validation=dict(spam_precision=.9), hidden=8), test=dict(accuracy=.1)),
    ]
    assert choose_family(reports)['features'] == 128
    reports[1]['selected'] = reports[0]['selected'].copy()
    assert choose_family(reports)['features'] == 64
