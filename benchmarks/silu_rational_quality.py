"""Compare floating SiLU rational candidates; no MPC protocol is implemented.

Run from the repository root. The fitted candidate is exploratory least squares,
not minimax or a certified error bound. Its public coefficients still need a
fixed-point scale, accumulator proof, and private verification implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import least_squares
import torch


def fitted_value(coefficients, t):
    a, b, c, d, e, f = coefficients
    numerator = t*(a+t*(b+t*c))
    denominator = 1+t*(d+t*(e+t*f))
    return numerator/denominator, denominator


def fit_zero_preserving():
    x = np.linspace(-4., 4., 8193)
    t = x/4
    truth = x/(1+np.exp(-x))
    initial = np.array([2., 3.2, 1.6, -.4, 1.6, -64/120])

    def residual(coefficients):
        actual, denominator = fitted_value(coefficients, t)
        return np.r_[actual-truth, 100*np.maximum(.25-denominator, 0)]

    result = least_squares(residual, initial, max_nfev=1000,
                           xtol=1e-12, ftol=1e-12, gtol=1e-12)
    if not result.success:
        raise RuntimeError(result.message)
    return result.x, result


def metrics(name, x, actual, denominator, formula, coefficients=None):
    truth = x/(1+np.exp(-x))
    error = actual-truth
    maximum = int(np.abs(error).argmax())
    return dict(name=name, formula=formula, coefficients=coefficients,
                max_abs_error=float(abs(error[maximum])),
                maximum_error_at=float(x[maximum]),
                rmse=float(np.sqrt(np.mean(error**2))),
                sampled_denominator_min=float(denominator.min()),
                sampled_denominator_max=float(denominator.max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path('results/silu_rational_quality.json'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    x = np.linspace(-4., 4., 65537)
    q32 = 12+x*x
    q33 = 120-12*x+12*x*x-x*x*x
    coefficients, fit = fit_zero_preserving()
    fitted, qfit = fitted_value(coefficients, x/4)
    rows = [
        metrics('Maclaurin Pade [3/2]', x, x/2+3*x*x/q32, q32,
                '(6*x+3*x^2+x^3/2)/(12+x^2)'),
        metrics('Maclaurin Pade [3/3]', x, (60*x+24*x*x+3*x*x*x)/q33,
                q33, '(60*x+24*x^2+3*x^3)/(120-12*x+12*x^2-x^3)'),
        metrics('Zero-preserving fitted [3/3]', x, fitted, qfit,
                't*(a+t*(b+t*c))/(1+t*(d+t*(e+t*f))), t=x/4',
                dict(zip(('a','b','c','d','e','f'), map(float, coefficients)))),
    ]
    from benchmarks.witness_activations import dense_accuracy
    numerical = dense_accuracy('silu', elements=65537, fractional_bits=24)
    from rep4gpu.numerical import NumericalFixedPoint
    from rep4gpu.witness_prover import _PlainFloor
    scale = 1 << 24
    encoded = torch.round(torch.from_numpy(x)*scale).long()
    exponential = NumericalFixedPoint(_PlainFloor()).exp(-encoded.abs())
    denominator = scale+exponential
    numerator = torch.where(encoded < 0, encoded*exponential, encoded*scale)
    direct = torch.round(numerator.double()/denominator.double()).long()
    direct_error = direct.double()/scale-torch.nn.functional.silu(encoded.double()/scale)
    direct_output = dict(
        formula='D=S+E; P=where(X<0,X*E,X*S); Y=nearest(P/D)',
        exponential='Existing exact-floor exp(-abs(X)) is still recomputed',
        secure_protocol_implemented=False,
        rounding_contract='Nearest final ratio; differs from existing sequential floors',
        max_abs_error=float(direct_error.abs().max()),
        rmse=float(direct_error.square().mean().sqrt()))
    sources = [Path(__file__), Path('benchmarks/witness_activations.py'),
               Path('rep4gpu/numerical.py'), Path('benchmarks/numerical_primitives.py'),
               Path('rep4gpu/witness_prover.py')]
    report = dict(
        methodology='Floating float64 rational candidates on65537 uniformly spaced '
                    'points over[-4,4]. No secure rational evaluator or fixed-point '
                    'coefficient schedule is implemented. Reported maxima are sampled, '
                    'not proven uniform bounds. Q24 numerical comparison uses the '
                    'existing exact-floor integer oracle on encoded grid inputs.',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        numpy_version=np.__version__, scipy_version=scipy.__version__,
        fit=dict(method='scipy.optimize.least_squares', deterministic=True,
                 random_seed=None, training_grid_points=8193, domain=[-4,4],
                 normalized_argument='t=x/4', numerator_constant=0,
                 denominator_constant=1,
                 initial=[2.,3.2,1.6,-.4,1.6,-64/120],
                 denominator_penalty='100*max(0.25-Q,0) at training points',
                 max_nfev=1000, xtol=1e-12, ftol=1e-12, gtol=1e-12,
                 nfev=fit.nfev, status=fit.status),
        evaluation_grid_points=len(x), candidates=rows,
        existing_q24_numerical_accuracy=numerical,
        direct_output_witness_with_recomputed_exp=direct_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
