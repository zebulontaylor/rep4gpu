"""Measure dense-grid integer accuracy for bounded numerical primitives."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from experiments.benchmarks.activations.numerical_primitives import fixture, oracle, reference


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--elements',type=int,default=65537)
    parser.add_argument('--fractional-bits',type=int,nargs='+',default=[20,24])
    parser.add_argument('--output',type=Path,default=Path('experiments/results/activations/numerical_primitives_cpu_quality.json'))
    args=parser.parse_args();torch.set_num_threads(1);rows=[]
    for f in args.fractional_bits:
        for kind in ['exp','reciprocal','erf','silu','gelu','swiglu']:
            x,value=fixture(kind,args.elements,f);truth=reference(kind,x,value,f)
            exact,_=oracle(kind,x,value,f);error=exact.lo.double()/2**f-truth
            interval,maximum=oracle(kind,x,value,f,True)
            bound=torch.maximum((interval.lo.double()/2**f-truth).abs(),(interval.hi.double()/2**f-truth).abs())
            rows.append(dict(kind=kind,fractional_bits=f,elements=len(x),floor_max_error=float(error.abs().max()),floor_rmse=float(error.square().mean().sqrt()),floor_max_relative_error=float((error.abs()/truth.abs().clamp_min(1e-12)).max()),conservative_probabilistic_max_error=float(bound.max()),maximum_rounding_interval_width=int((interval.hi-interval.lo).max()),maximum_pretruncation_abs_bound=maximum))
    sources=['src/rep4gpu/arithmetic/numerical.py','experiments/benchmarks/activations/numerical_primitives.py','experiments/benchmarks/activations/numerical_quality.py']
    report=dict(methodology='Dense uniformly spaced encoded caller-domain inputs, including endpoints and zero for signed domains. Exact integer floor oracle and conservative intervals over every floor/ceil truncation outcome. References are torch float64 exp/reciprocal/erf/exact GELU/SiLU. SwiGLU value spans +4 to -4 opposite gate.',source_sha256={s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources},results=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
