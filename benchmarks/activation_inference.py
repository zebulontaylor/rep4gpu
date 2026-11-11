"""Compare exact ReLU and bounded quadratic substitutes in resident Rep4 FFNs.

Run from the repository root with CUDA available. These experiments use no
TEE. Polynomial cases report error against the original floating activation.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits

from benchmarks.device_inference import run_case
from benchmarks.inference import provenance
from rep4gpu.activations import quadratic_plan


def approximation_grid(kind, fractional_bits=14, profile='uniform'):
    plan = quadratic_plan(kind,4,fractional_bits=fractional_bits,bits=32,profile=profile)
    x = torch.linspace(-4,4,20001,dtype=torch.float64)
    exact = (torch.relu(x) if kind=='relu' else torch.nn.functional.gelu(x)
             if kind=='gelu' else torch.nn.functional.silu(x))
    c, linear, square = plan.coefficients
    polynomial = c+linear*x+square*x*x
    encoded = torch.round(x*(1 << fractional_bits)).to(torch.int64)
    fixed = (plan.encoded_numerator(encoded >> plan.prescale) >> plan.shift).double()/(1 << fractional_bits)
    return dict(kind=kind,plan=asdict(plan),coefficients=list(plan.coefficients),
                interval=[-4,4],samples=x.numel(),
                sampled_real_max_error=float((polynomial-exact).abs().max()),
                sampled_floor_fixed_max_error=float((fixed-exact).abs().max()),
                output_at_zero=c,minimum_real_output=float(polynomial.min()),
                error_kind='sampled grid; not a model quality guarantee')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=int,nargs='+',default=[1,128])
    parser.add_argument('--width',type=int,default=2048)
    parser.add_argument('--repeats',type=int,default=9)
    parser.add_argument('--activation',nargs='+',default=['relu','quadratic_relu','quadratic_gelu','quadratic_silu'],
                        choices=['relu','quadratic_relu','quadratic_gelu','quadratic_silu','simulated_relu'])
    parser.add_argument('--activation-profile',choices=['uniform','local'],default='uniform')
    parser.add_argument('--output',type=Path,default=Path('results/activation_inference.json'))
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    source=provenance()
    for path in (Path(__file__),Path('benchmarks/activation_common.py'),Path('benchmarks/device_inference.py'),Path('benchmarks/simulated_activation.py')):
        source['source_sha256'][str(path.relative_to(Path.cwd()) if path.is_absolute() else path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    report=dict(provenance=source,gpu=torch.cuda.get_device_name(),tee_executed=False,
                approximation_grids=[approximation_grid(k,profile=args.activation_profile)
                                     for k in ('relu','gelu','silu')
                                     if args.activation_profile=='uniform' or k!='relu'],
                methodology='ReLU and quadratic cases use pure Rep4; simulated_relu reconstructs public fixture activations in ordinary helper processes and provides no activation privacy. No TEE is executed or measured. All cases use resident secret weights, fresh masks and checked output opening. GPU-delivered replay excludes network latency/bandwidth. Quadratic substitutes change the network function; each baseline and accuracy reference uses its original ReLU/GELU/SiLU activation. No language-model quality claim.',
                results=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for activation in args.activation:
        for batch in args.batch:
            row=run_case(batch,args.width,4*args.width,args.width,repeats=args.repeats,
                         bits=32,binary_bits=32,fractional_bits=14,prepared=True,
                         weight_storage='words32',optimized=True,gpu_open=True,
                         rounding='probabilistic',phase_graphs=True,bounded_sign=True,
                         graph_check_at_send=True,sparse_products=True,activation=activation,
                         activation_profile=args.activation_profile)
            report['results'].append(row)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            keys=('activation','batch','ideal_network_phase_max_ms','rounds','payload_bytes',
                  'max_logprob_error','max_logit_rmse','minimum_top1_agreement','ideal_slowdowns')
            print(json.dumps({key:row[key] for key in keys}),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):
        main()
