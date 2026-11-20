"""Checkpoint validation and shared inference for text classifier benchmarks."""
import numpy as np

from rep4gpu.models.text_classifier import nonnegative_label


def checked_matmul(a, b, bits=32):
    """Bound every dot product before using int64 arithmetic as an oracle."""
    if int(np.max(np.abs(a), initial=0)) * int(np.max(np.abs(b), initial=0)) * a.shape[1] >= 2**63:
        raise ValueError('integer reference accumulator can overflow int64')
    bound = int(np.max(np.abs(a) @ np.abs(b), initial=0))
    if bound >= 1 << (bits - 1):
        raise ValueError(f'checkpoint exceeds signed ring{bits} accumulator range')
    return a @ b, bound


def load_model(checkpoint, batch, bits=32):
    """Validate all arithmetic used by the selected public evaluation batch."""
    if bits not in (16, 32):
        raise ValueError('classifier ring bits must be 16 or 32')
    limit = 1 << (bits - 1)
    with np.load(checkpoint, allow_pickle=False) as model:
        architecture = str(model['architecture']) if 'architecture' in model.files else ('square' if 'w1' in model.files else 'linear')
        if architecture not in ('linear', 'square', 'square_unscaled', 'fixed_square_head'):
            raise ValueError('unsupported classifier architecture')
        names = (('weights',) if architecture == 'linear' else
                 ('w1', 'b1', 'b2') if architecture == 'fixed_square_head' else ('w1', 'b1', 'w2', 'b2'))
        x0 = model['x_test']
        weights = {name: model[name] for name in names}
        floats = {name: model['float_'+name] for name in names}
        fractional_bits = int(model['fractional_bits']) if architecture == 'square' else 0
        input_divisor = int(model['input_divisor']) if architecture in ('square_unscaled', 'fixed_square_head') else 1 << fractional_bits
        public_signs = ()
        if architecture == 'fixed_square_head':
            signs = model['signs']
            if signs.ndim != 1 or signs.dtype.kind not in 'iu':
                raise ValueError('fixed square signs must be an integer vector')
            public_signs = tuple(int(v) for v in signs)
    if x0.ndim != 2 or not len(x0) or not x0.shape[1] or x0.shape[1] > 1_000_000:
        raise ValueError('checkpoint features must be a nonempty matrix with at most 1000000 columns')
    if architecture != 'linear':
        width = weights['w1'].shape[1] if weights['w1'].ndim == 2 else 0
        expected_shapes = dict(w1=(x0.shape[1], width), b1=(width,), w2=(width, 1), b2=(1,))
        if architecture == 'fixed_square_head':
            del expected_shapes['w2']
            if width % 2 or public_signs != (1,) * (width // 2) + (-1,) * (width // 2):
                raise ValueError('fixed square head requires canonical public half-positive, half-negative signs')
        if (architecture == 'square' and not 0 < fractional_bits < 16) or not 0 < width <= 1_000_000:
            raise ValueError('square classifier requires positive width and fractional_bits in [1,15]')
    else:
        expected_shapes = dict(weights=(x0.shape[1], 1))
    for name, shape in expected_shapes.items():
        value = weights[name]
        if value.shape != shape or floats[name].shape != shape:
            raise ValueError(f'checkpoint {name} shape must be {shape}')
        if value.dtype.kind not in 'iu' or np.any(value < -(limit - 1)) or np.any(value > limit - 1):
            raise ValueError(f'weights must be signed ring{bits} integers other than its minimum')
        if not np.isfinite(floats[name]).all():
            raise ValueError('floating-point weights must be finite')
    if x0.dtype.kind not in 'iu' or np.any(x0 < 0) or np.any(x0 > 65535):
        raise ValueError('features must be integers in [0,65535]')
    if np.any(x0 >= limit):
        raise ValueError(f'features exceed signed ring{bits} range')
    if input_divisor < 1:
        raise ValueError('native input divisor must be positive')
    weights = {name: value.astype(np.int64) for name, value in weights.items()}
    x = x0[np.arange(batch) % len(x0)].astype(np.int64)
    if architecture == 'linear':
        scores, bound = checked_matmul(x, weights['weights'], bits)
        bounds = dict(linear_accumulator=bound)
    else:
        raw, bound1 = checked_matmul(x, weights['w1'], bits)
        first = (raw >> fractional_bits) + weights['b1']
        first_bound = int(np.max(np.abs(first), initial=0))
        if first_bound**2 >= limit:
            raise ValueError(f'square activation exceeds signed ring{bits} accumulator range')
        hidden = first**2 >> fractional_bits
        if architecture == 'fixed_square_head':
            bound2 = int(np.max(hidden.sum(axis=1), initial=0))
            if bound2 >= limit:
                raise ValueError(f'square sum exceeds signed ring{bits} accumulator range')
            raw = (hidden * np.asarray(public_signs)).sum(axis=1, keepdims=True)
        else:
            raw, bound2 = checked_matmul(hidden, weights['w2'], bits)
        scores = (raw >> fractional_bits) + weights['b2']
        if int(np.max(np.abs(scores), initial=0)) >= limit:
            raise ValueError(f'output bias exceeds signed ring{bits} range')
        bounds = dict(first_accumulator=bound1, square_accumulator=first_bound**2,
                      second_accumulator=bound2, output=int(np.max(np.abs(scores), initial=0)))
    sparse_native_eligible = (architecture in ('square_unscaled', 'fixed_square_head') and input_divisor == 1
                              and np.all((x == 0) | (x == 1))
                              and all(np.array_equal(weights[name], floats[name]) for name in names)
                              and np.count_nonzero(weights['w1']) < weights['w1'].size)
    sparse_weights = dict(weights)
    if architecture == 'fixed_square_head':
        sparse_weights['w2'] = np.asarray(public_signs, dtype=np.int64).reshape(-1, 1)
    return dict(sparse_native_weights=sparse_weights, public_signs=public_signs, sparse_native_eligible=bool(sparse_native_eligible), architecture=architecture, fractional_bits=fractional_bits, input_divisor=input_divisor, x=x,
                weights=weights, float_weights=floats, bounds=bounds,
                expected=(scores >= 0).reshape(-1).astype(int).tolist())


def encoder_ring_bounds(model, encoder_ring_bits):
    """Certify the first layer for every binary feature vector, not just fixtures."""
    if encoder_ring_bits != 8:
        raise ValueError('the mixed encoder currently requires ring8')
    if model['architecture'] not in ('fixed_square_head', 'square_unscaled') or model['fractional_bits'] != 0:
        raise ValueError('ring8 encoder requires an unscaled integer square classifier')
    if model['input_divisor'] != 1 or not np.all((model['x'] == 0) | (model['x'] == 1)):
        raise ValueError('ring8 encoder requires binary features and input_divisor=1')
    weights = model['weights']
    if any(not np.array_equal(value, model['float_weights'][name]) for name, value in weights.items()):
        raise ValueError('ring8 encoder requires matching integer and floating-point weights')
    lower = np.minimum(weights['w1'], 0).sum(axis=0) + weights['b1']
    upper = np.maximum(weights['w1'], 0).sum(axis=0) + weights['b1']
    limit = 1 << (encoder_ring_bits-1)
    if np.any(lower < -limit) or np.any(upper >= limit):
        raise ValueError('first encoder exceeds signed ring8 for binary feature inputs')
    return dict(input_domain='binary', minimum=int(lower.min()), maximum=int(upper.max()))


def shared_scores(rt, x, weights, architecture, fractional_bits, public_signs=(), encoder_ring_bits=None):
    if encoder_ring_bits is not None and (architecture not in ('fixed_square_head', 'square_unscaled') or fractional_bits):
        raise ValueError('mixed encoder requires an unscaled integer square classifier')
    if architecture == 'linear':
        return rt.matmul(x, weights['weights'])
    if encoder_ring_bits is not None:
        from rep4gpu.arithmetic.signed_lift import signed_lift
        with rt.arithmetic_ring(encoder_ring_bits):
            first = rt.add(rt.matmul(x, weights['w1']), weights['b1'])
        first = signed_lift(rt, first, source_bits=encoder_ring_bits)
    elif architecture in ('fixed_square_head', 'square_unscaled'):
        first = rt.add(rt.matmul(x, weights['w1']), weights['b1'])
    if architecture == 'fixed_square_head':
        from rep4gpu.models.square_sum import signed_square_sum
        return rt.add(signed_square_sum(rt, first, public_signs), weights['b2'])
    if architecture == 'square_unscaled':
        return rt.add(rt.matmul(rt.mul(first, first), weights['w2']), weights['b2'])
    first = rt.add(rt.trunc(rt.matmul(x, weights['w1']), fractional_bits), weights['b1'])
    hidden = rt.trunc(rt.mul(first, first), fractional_bits)
    score = rt.add(rt.trunc(rt.matmul(hidden, weights['w2']), fractional_bits), weights['b2'])
    return score


def shared_labels(rt, x, weights, architecture, fractional_bits, public_signs=(), encoder_ring_bits=None):
    return nonnegative_label(rt, shared_scores(rt, x, weights, architecture, fractional_bits, public_signs, encoder_ring_bits))


def plain_labels(x, weights, architecture, public_signs=None):
    if architecture == 'linear':
        return x @ weights['weights'] >= 0
    if architecture == 'fixed_square_head':
        hidden = (x @ weights['w1'] + weights['b1']).square()
        return (hidden * public_signs).sum(dim=1, keepdim=True) + weights['b2'] >= 0
    return (x @ weights['w1'] + weights['b1']).square() @ weights['w2'] + weights['b2'] >= 0
