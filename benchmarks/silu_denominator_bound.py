"""Certify the negative exponential denominator range in encoded arithmetic.

For S=2**f and T in [-S,0], the quantized Taylor coefficients satisfy
C0=C1=S >= C2 >= ... >= Cn>0. If 0<=H<=C[k+1], either floor or ceiling
of H*T/S lies in [-C[k+1],0]. Therefore the next Horner value lies in
[C[k]-C[k+1],C[k]], a subset of [0,C[k]]. The final value is in [0,S],
and four rounded squares preserve [0,S]. Every raw product is at most S*S
in absolute value, below 2**62 for f<=24. This proves S<=S+E<=2*S for
all encoded reduced inputs, independently of the sampled grid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from rep4gpu.numerical import EXP_COEFFICIENTS, NumericalFixedPoint
from rep4gpu.witness_prover import _PlainFloor


def certificate(fractional_bits):
    scale = 1 << fractional_bits
    coefficients = [round(c*scale) for c in EXP_COEFFICIENTS]
    while coefficients[-1] == 0:
        coefficients.pop()
    assert coefficients[0] == coefficients[1] == scale
    assert all(a >= b > 0 for a,b in zip(coefficients,coefficients[1:]))
    assert scale*scale < 1 << 62
    return dict(fractional_bits=fractional_bits, scale=scale,
                coefficients=coefficients, reduced_input_bounds=[-scale,0],
                horner_intermediate_bounds=[[coefficients[k]-coefficients[k+1],coefficients[k]]
                                           for k in range(len(coefficients)-1)],
                exponential_bounds=[0,scale], denominator_bounds=[scale,2*scale],
                raw_product_abs_bound=scale*scale)


def encoded_negative_exp(reduced, fractional_bits=24, rounding='floor'):
    proof = certificate(fractional_bits)
    scale, coefficients = proof['scale'],proof['coefficients']
    assert bool(((reduced >= -scale) & (reduced <= 0)).all())
    upper_round = scale-1 if rounding == 'ceil' else 0
    h = torch.full_like(reduced, coefficients[-1])
    for index in range(len(coefficients)-2,-1,-1):
        h = ((h*reduced+upper_round) >> fractional_bits)+coefficients[index]
        assert bool(((h >= coefficients[index]-coefficients[index+1]) &
                     (h <= coefficients[index])).all())
    for _ in range(4):
        h = (h*h+upper_round) >> fractional_bits
        assert bool(((h >= 0) & (h <= scale)).all())
    return h


class _PlainCeil(_PlainFloor):
    def trunc(self, x, bits):
        return (x+((1 << bits)-1)) >> bits


def grid_case(fractional_bits=24, domain=4, points=65537, rounding='floor'):
    scale = 1 << fractional_bits
    encoded = torch.round(torch.linspace(-domain,domain,points,dtype=torch.float64)*scale).long()
    runtime = _PlainFloor() if rounding == 'floor' else _PlainCeil()
    reduced = runtime.trunc(-encoded.abs(),4)
    expected = encoded_negative_exp(reduced,fractional_bits,rounding)
    actual = NumericalFixedPoint(runtime,fractional_bits).exp(-encoded.abs())
    assert torch.equal(actual,expected)
    denominator = scale+actual
    assert bool(((denominator >= scale) & (denominator <= 2*scale)).all())
    return dict(fractional_bits=fractional_bits,domain=[-domain,domain],points=points,
                rounding=rounding,exponential_min=int(actual.min()),exponential_max=int(actual.max()),
                denominator_min=int(denominator.min()),denominator_max=int(denominator.max()),
                independent_recurrence_matches=True)


def exhaustive_reduced(fractional_bits=24, chunk=1 << 19):
    scale = 1 << fractional_bits
    minimum,maximum = scale,0
    for start in range(-scale,1,chunk):
        reduced = torch.arange(start,min(start+chunk,1),dtype=torch.int64)
        value = encoded_negative_exp(reduced,fractional_bits)
        minimum,maximum = min(minimum,int(value.min())),max(maximum,int(value.max()))
    return dict(fractional_bits=fractional_bits,encoded_reduced_inputs=scale+1,
                rounding='floor',exponential_min=minimum,exponential_max=maximum,
                denominator_min=scale+minimum,denominator_max=scale+maximum)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exhaustive-reduced',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('results/silu_denominator_bound.json'))
    args=parser.parse_args()
    torch.set_num_threads(1)
    files=['benchmarks/silu_denominator_bound.py','rep4gpu/numerical.py']
    report=dict(proof=__doc__,source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},
                certificates=[certificate(f) for f in range(20,25)],
                grids=[grid_case(domain=d,rounding=r) for d in (4,16) for r in ('floor','ceil')])
    if args.exhaustive_reduced:
        report['exhaustive_reduced']=exhaustive_reduced()
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
