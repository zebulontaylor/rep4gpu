"""Measure the integer P3/Q4 SiLU candidate; no MPC evaluator is implemented."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from rep4gpu.proofs.silu_rational import (
    SCALE, POWER_SCALE, POLYNOMIAL_SCALE, A_COEFFICIENTS, B_COEFFICIENTS,
    rational_silu_plain as evaluate,
)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exhaustive',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('experiments/results/proofs/silu_rational34_quality.json'))
    args=parser.parse_args()
    step=1 if args.exhaustive else 512
    chunk_size=1 << 20
    max_error=squared_error=0.;worst=points=0
    first,last=-16*SCALE,16*SCALE
    for begin in range(first,last+1,chunk_size*step):
        encoded=np.arange(begin,min(begin+chunk_size*step,last+1),step,dtype=np.int64)
        x=encoded.astype(np.float64)/SCALE
        reference=x/(1+np.exp(-x))
        error=evaluate(encoded).astype(np.float64)/SCALE-reference
        index=int(np.abs(error).argmax())
        if abs(error[index])>max_error:
            max_error=float(abs(error[index]));worst=int(encoded[index])
        squared_error+=float(np.dot(error,error));points+=len(encoded)
    a_bound=sum(abs(c)*POWER_SCALE for c in A_COEFFICIENTS)+POWER_SCALE//2
    b_bound=sum(abs(c)*POWER_SCALE for c in B_COEFFICIENTS)+POWER_SCALE//2
    assert max(a_bound,b_bound)<1 << 62
    report={
        'source_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (Path('experiments/benchmarks/proofs/silu_rational34_quality.py'),Path('src/rep4gpu/proofs/silu_rational.py'))},
        'domain':[-16,16],'fractional_bits':24,'power_fractional_bits':30,
        'polynomial_fractional_bits':21,'exhaustive_encoded_inputs':args.exhaustive,
        'points':points,'encoded_step':step,'max_abs_error':max_error,
        'worst_encoded_input':worst,'rmse':float(np.sqrt(squared_error/points)),
        'a_coefficients':A_COEFFICIENTS,'b_coefficients':B_COEFFICIENTS,
        'a_accumulator_absolute_bound':a_bound,'b_accumulator_absolute_bound':b_bound,
        'formula':'z=nearest(X^2/2^26); powers zi=nearest(z*zi-1/2^30); '
                  'A=nearest(sum(ai*zi)/2^30); B=nearest(sum(bi*zi)/2^30); '
                  'Y=nearest_even((X*B+2*S*A)/(2*B))',
        'power_products':4,'floors_including_nearest_reductions':6,
        'methodology':'Correlated Q16 rational coefficients found with an integer linear '
                      'feasibility solve for x/2+x^2*P3((x/16)^2)/Q4((x/16)^2). '
                      'Every integer operation here uses ring64-safe ordinary int64 '
                      'arithmetic. Quality reference is float64 SiLU. This experiment '
                      'does not measure protocol communication.'}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
