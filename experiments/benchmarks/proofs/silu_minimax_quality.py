"""Fit Remez exp polynomials and check their encoded SiLU output witnesses.

Run from the repository root. The default grid/random checks are samples;
--exhaustive visits every Q24 input encoding in [-16,16] for the selected
sixth-degree quadratic Horner polynomial with nearest-rounded squarings. The transcendental
reference uses float64; the candidate arithmetic uses exact signed int64.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import mpmath as mp
import numpy as np

SCALE = 1 << 24
TARGET_ERROR = 6.41e-7
QUADRATIC_Q24 = (16777216, 16777188, 8388203, 2793717, 691720, 128514, 14276)


def remez_exp(degree):
    """Exchange all extrema of exp minus a degree-n polynomial on [-1,0]."""
    mp.mp.dps = 70
    points = [(mp.cos(mp.pi*(degree+1-i)/(degree+1))-1)/2
              for i in range(degree+2)]
    for iteration in range(20):
        matrix = mp.matrix([[x**k for k in range(degree+1)]+[mp.mpf((-1)**i)]
                            for i,x in enumerate(points)])
        solution = mp.lu_solve(matrix, mp.matrix([mp.exp(x) for x in points]))
        coefficients = [solution[k] for k in range(degree+1)]

        def derivative(x):
            x = mp.mpf(x)
            return mp.exp(x)-sum(k*coefficients[k]*x**(k-1)
                                for k in range(1,degree+1))

        scan = [mp.mpf(i)/2000-1 for i in range(2001)]
        values = [derivative(x) for x in scan]
        extrema = []
        for i in range(len(scan)-1):
            if values[i]*values[i+1] < 0:
                lo,hi = scan[i],scan[i+1]
                for _ in range(160):
                    middle = (lo+hi)/2
                    if derivative(lo)*derivative(middle) <= 0:
                        hi = middle
                    else:
                        lo = middle
                extrema.append((lo+hi)/2)
        next_points = [mp.mpf(-1),*extrema,mp.mpf(0)]
        if len(next_points) != degree+2:
            raise AssertionError('Remez error has the wrong number of extrema')
        difference = max(abs(a-b) for a,b in zip(points,next_points))
        points = next_points
        if difference < mp.mpf('1e-40'):
            break
    else:
        raise RuntimeError('Remez exchange did not converge')
    errors = [mp.exp(x)-sum(c*x**k for k,c in enumerate(coefficients)) for x in points]
    if not all(errors[i]*errors[i+1] < 0 for i in range(len(errors)-1)):
        raise AssertionError('Remez error extrema do not alternate')
    magnitudes = [abs(e) for e in errors]
    if max(magnitudes)-min(magnitudes) > mp.mpf('1e-35'):
        raise AssertionError('Remez extrema do not have equal error')
    encoded = [int(mp.nint(c*SCALE)) for c in coefficients]
    if encoded[0] != SCALE or not all(a >= b >= 0 for a,b in zip(encoded,encoded[1:])):
        raise AssertionError('encoded coefficients violate the Horner range proof')
    return dict(degree=degree, iterations=iteration+1,
                real_coefficients=[mp.nstr(c,60) for c in coefficients],
                encoded_coefficients=encoded,
                alternation_points=[mp.nstr(x,60) for x in points],
                alternation_errors=[mp.nstr(e,60) for e in errors],
                real_minimax_max_abs_error=float(max(magnitudes)))


def exponential(encoded_x, coefficients, rounded_squares, schedule="quadratic"):
    """Evaluate the quadratic degree-six schedule or sequential Horner reference."""
    reduced = -np.abs(encoded_x)//16
    if schedule == "quadratic" and len(coefficients) == 7:
        c0, c1, c2, c3, c4, c5, c6 = coefficients
        square = reduced*reduced >> 24
        q2 = c4+((c5*reduced+c6*square) >> 24)
        q1 = c2+((c3*reduced+square*q2) >> 24)
        value = c0+((c1*reduced+square*q1) >> 24)
    else:
        value = np.full_like(encoded_x,coefficients[-1])
        for coefficient in coefficients[-2::-1]:
            value = (value*reduced >> 24)+coefficient
    for rounded in rounded_squares:
        value = (value*value+(SCALE//2 if rounded else 0)) >> 24
    return value


def nearest_quotient(numerator, denominator):
    quotient = numerator//denominator
    remainder = numerator-quotient*denominator
    return quotient+((2*remainder > denominator) |
                     ((2*remainder == denominator) & ((quotient & 1) != 0)))


def metrics(encoded_x, coefficients, rounded_squares):
    exp = exponential(encoded_x,coefficients,rounded_squares)
    denominator = SCALE+exp
    numerator = np.where(encoded_x < 0,encoded_x*exp,encoded_x*SCALE)
    output = nearest_quotient(numerator,denominator)
    real = encoded_x/SCALE
    truth = real/(1+np.exp(-real))
    error = output/SCALE-truth
    maximum = int(np.abs(error).argmax())
    return dict(elements=int(encoded_x.size),max_abs_error=float(abs(error[maximum])),
                max_error_at=float(real[maximum]),rmse=float(np.sqrt(np.mean(error**2))),
                min_exp_encoded=int(exp.min()),max_exp_encoded=int(exp.max()),
                passes_target=bool(np.abs(error).max() <= TARGET_ERROR))


def exhaustive(coefficients, chunk_size=1 << 20):
    """Check both signs for all 268,435,457 nonnegative Q24 magnitudes."""
    maximum = [0.,0.]
    locations = [0,0]
    min_exp,max_exp = SCALE,0
    started = time.monotonic()
    for lo in range(0,16*SCALE+1,chunk_size):
        encoded = np.arange(lo,min(lo+chunk_size,16*SCALE+1),dtype=np.int64)
        exp = exponential(encoded,coefficients,(True,)*4)
        min_exp,max_exp = min(min_exp,int(exp.min())),max(max_exp,int(exp.max()))
        denominator = SCALE+exp
        real = encoded/SCALE
        small_exp = np.exp(-real)
        negative = -real*small_exp/(1+small_exp)
        for branch,numerator,truth in ((0,-encoded*exp,negative),
                                       (1,encoded*SCALE,real+negative)):
            output = nearest_quotient(numerator,denominator)
            error = np.abs(output/SCALE-truth)
            where = int(error.argmax())
            if error[where] > maximum[branch]:
                maximum[branch] = float(error[where])
                locations[branch] = int(encoded[where])*(-1 if branch == 0 else 1)
    return dict(distinct_input_encodings=32*SCALE+1,max_abs_error=max(maximum),
                branch_max_abs_errors=maximum,max_error_at=[x/SCALE for x in locations],
                max_error_encoded_at=locations,min_exp_encoded=min_exp,max_exp_encoded=max_exp,
                accepted_tie_choice_error_upper_bound=max(maximum)+1/SCALE,
                passes_target=max(maximum) <= TARGET_ERROR,
                elapsed_seconds=time.monotonic()-started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exhaustive',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('experiments/results/proofs/silu_minimax_quality.json'))
    args = parser.parse_args()
    rng = np.random.default_rng(831)
    random = rng.integers(-16*SCALE,16*SCALE+1,size=1_000_000,dtype=np.int64)
    report = dict(fractional_bits=24,target_max_abs_error=TARGET_ERROR,
        methodology='True high-precision Remez exchange on [-1,0], followed by nearest Q24 '
        'coefficient encoding. Degree six uses quadratic Horner; degree seven uses sequential '
        'Horner. Argument reduction and polynomial products use exact floor. '
        'Variants change only the four squaring roundings. Output witness is nearest-even '
        'integer to the verified Q48 numerator divided by S+E. Grid and random tests are '
        'samples; exhaustive, when present, covers all Q24 input encodings in [-16,16]. '
        'Float64 references are numerical checks, not a formal real-arithmetic error proof.',
        range_proof='For degree six, 0<=U=floor(T*T/S)<=-T<=S. The coefficients satisfy '
        'C4>=C5, C2>=C3, C2+C4+C6<=C1<=C0=S. Thus 0<=Q2<=C4+C6 and '
        '0<=Q1<=C2+C4+C6<=C1, giving -C1*S<=C1*T+U*Q1<=0 and H in[0,S]. '
        'Nearest-square rescaling preserves[0,S]. Polynomial numerators have absolute '
        'value<=S*S, and square numerators<=S*S+S/2<2**62; denominator lies in[S,2S].',
        source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in (Path(__file__),Path('src/rep4gpu/arithmetic/silu_exp.py'))},
        fits=[],candidates=[])
    for degree in (6,7):
        fit = remez_exp(degree)
        report['fits'].append(fit)
        for label,rounding in [('floor',(False,)*4),('nearest_last',(False,False,False,True)),
                               ('nearest_all',(True,)*4)]:
            row = dict(degree=degree,squaring_rounding=label,
                grid_results={str(bound):metrics(np.rint(np.linspace(-bound,bound,65537)*SCALE).astype(np.int64),
                    fit['encoded_coefficients'],rounding) for bound in (4,16)},
                random_results=metrics(random,fit['encoded_coefficients'],rounding))
            report['candidates'].append(row)
    if args.exhaustive:
        report['nearest_coefficients_exhaustive'] = exhaustive(
            report['fits'][0]['encoded_coefficients'])
        report['selected_exhaustive'] = exhaustive(QUADRATIC_Q24)
    report['selected_candidate'] = dict(degree=6,schedule='quadratic',squaring_rounding='nearest_all',
        encoded_coefficients=list(QUADRATIC_Q24),
        coefficient_adjustments_from_remez=[0,-3,0,0,0,0,0],
        grid_results=metrics(np.rint(np.linspace(-16,16,65537)*SCALE).astype(np.int64),
                             QUADRATIC_Q24,(True,)*4),
        random_results=metrics(random,QUADRATIC_Q24,(True,)*4))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('selected_candidate','candidates')},indent=2))
    if args.exhaustive:
        print(json.dumps(report['selected_exhaustive'],indent=2))


if __name__=='__main__':
    main()
