"""Public fixture proofs and comparators for bounded activation experiments."""
from dataclasses import asdict

import torch

from benchmarks.inference import checked_add, checked_matmul, maximum_abs
from rep4gpu.activations import FixedPoint, quadratic_plan


def activation_reference(values, kind):
    x, w1, b1, w2, b2 = values
    hidden = x @ w1 + b1
    if kind == 'relu':
        hidden = torch.relu(hidden)
    elif kind == 'gelu':
        hidden = torch.nn.functional.gelu(hidden)
    elif kind == 'silu':
        hidden = torch.nn.functional.silu(hidden)
    else:
        raise ValueError('activation reference must be relu, gelu or silu')
    return hidden @ w2 + b2


def infer_quadratic(party, values, fractional_bits, kind, bound, profile='uniform'):
    x, w1, b1, w2, b2 = values
    fp = FixedPoint(party, fractional_bits)
    hidden = fp.matmul_quadratic(x, w1, b1, kind=kind, bound=bound, profile=profile)
    return party.open(fp.add(fp.matmul(hidden, w2), b2))


def quadratic_interval(low, high, plan):
    """Bound the convex integer numerator at endpoints and its integer vertex."""
    left, right = plan.encoded_numerator(low), plan.encoded_numerator(high)
    lower, upper = torch.minimum(left, right), torch.maximum(left, right)
    vertex = -(plan.linear_coeff//(2*plan.quadratic_coeff))
    for point in (vertex-1, vertex):
        candidate = torch.full_like(low, point)
        inside = (candidate >= low) & (candidate <= high)
        # The vertex can be outside the declared range; clamp before evaluating.
        candidate = candidate.clamp(-plan.normalized_bound, plan.normalized_bound)
        lower = torch.where(inside, torch.minimum(lower, plan.encoded_numerator(candidate)), lower)
    return lower, upper


def quadratic_fixture_bounds(integers, bits, fractional_bits, rounding, kind, bound, profile='uniform'):
    """Propagate all floor/ceil outcomes without assuming nonnegative activations."""
    if rounding not in ('floor', 'probabilistic'):
        raise ValueError('rounding must be floor or probabilistic')
    plan = quadratic_plan(kind, bound, fractional_bits=fractional_bits, bits=bits, profile=profile)
    x, w1, b1, w2, b2 = integers
    scale = 1 << fractional_bits
    if maximum_abs(b1)*scale > 2**63-1:
        raise ValueError('scaled fixture bias exceeds signed int64')
    raw1 = checked_add(checked_matmul(x, w1), b1*scale)
    if maximum_abs(raw1) > plan.input_bound*scale:
        raise ValueError('fixture exceeds the declared quadratic activation bound')
    first_shift = fractional_bits+plan.prescale
    low_u = raw1 >> first_shift
    high_u = ((checked_add(raw1, torch.tensor((1 << first_shift)-1)) >> first_shift)
              if rounding == 'probabilistic' else low_u)
    low_n, high_n = quadratic_interval(low_u, high_u, plan)
    low_h = low_n >> plan.shift
    high_h = ((checked_add(high_n, torch.tensor((1 << plan.shift)-1)) >> plan.shift)
              if rounding == 'probabilistic' else high_n >> plan.shift)
    positive, negative = w2.clamp_min(0), w2.clamp_max(0)
    raw2lo = checked_add(checked_matmul(low_h, positive), checked_matmul(high_h, negative))
    raw2hi = checked_add(checked_matmul(high_h, positive), checked_matmul(low_h, negative))
    lower = checked_add(raw2lo >> fractional_bits, b2)
    upper = checked_add((checked_add(raw2hi, torch.tensor(scale-1)) >> fractional_bits)
                        if rounding == 'probabilistic' else raw2hi >> fractional_bits, b2)
    maximum = max(maximum_abs(t) for t in (raw1, low_n, high_n, raw2lo, raw2hi))
    if rounding == 'probabilistic' and maximum >= 1 << (bits-2):
        raise ValueError('fixture violates strict probabilistic quarter-ring contract')
    for value in (*integers, raw1, low_u, high_u, low_n, high_n, low_h, high_h,
                  raw2lo, raw2hi, lower, upper):
        if int(value.min()) < -(1 << (bits-1)) or int(value.max()) >= 1 << (bits-1):
            raise ValueError(f'quadratic fixture exceeds signed ring{bits}')
    proof = dict(maximum_accumulator=maximum,
                 first_accumulator_max_abs=maximum_abs(raw1),
                 second_accumulator_max_abs=max(maximum_abs(raw2lo), maximum_abs(raw2hi)),
                 quadratic_numerator_max_abs=max(maximum_abs(low_n), maximum_abs(high_n)),
                 accumulator_fits_signed_ring=True, rounding_bounds_validated=True,
                 maximum_output_rounding_interval_width=int((upper-lower).max()),
                 range_validation='public fixture only; not a private range check',
                 quadratic_plan=asdict(plan), activation_coefficients=list(plan.coefficients),
                 activation_is_approximation=True)
    if rounding == 'probabilistic':
        proof['strict_probabilistic_accumulator_limit'] = 1 << (bits-2)
    return proof, lower, upper
