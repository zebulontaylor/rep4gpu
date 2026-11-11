"""Time bounded activation-only Rep4 GPU protocols with fresh live masks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
from pathlib import Path
import statistics
import time

import torch

from experiments.benchmarks.activations.activation_approximation_quality import fit
from experiments.benchmarks.inference.device_inference import phase_estimate, phase_payloads
from rep4gpu.arithmetic.activations import FixedPoint, quadratic_plan
from rep4gpu.protocol.runtime import make_seeds
from rep4gpu.protocol.transport import IdealNetwork


def precision(coefficients):
    # Absolute Horner bounds cover every normalized input in [-1,1].
    for f in range(14, 1, -1):
        total = (sum(abs(c) for c in coefficients[::2])+2)*(1 << f)
        if total*(1 << f) < 1 << 30:
            return f
    raise ValueError('coefficients exceed ring32 bound')


def polynomial(p, x, coefficients, f):
    t = p.trunc(x, 16-f)
    z = p.trunc(p.mul(t,t), f)
    cs = [round(c*(1 << f)) for c in coefficients[::2]]
    y = p.add(p.trunc(p.mul_public(z,cs[-1]),f),p.public(cs[-2],like=x))
    for c in reversed(cs[:-2]):
        y = p.add(p.trunc(p.mul(y,z),f), p.public(c,like=x))
    return p.add(p.mul_public(y,1 << (14-f)), p.trunc(x,1))


def interval_oracle(x, coefficients, f, probabilistic=True):
    """Propagate every floor/ceil outcome using signed int64 intervals."""
    maximum = int(x.abs().max())
    def trunc(lo, hi, shift):
        nonlocal maximum
        maximum = max(maximum,int(lo.abs().max()),int(hi.abs().max()))
        assert maximum < 1 << 30
        return lo >> shift, (hi+((1 << shift)-1 if probabilistic else 0)) >> shift
    def multiply(a,b):
        products = torch.stack([a[0]*b[0],a[0]*b[1],a[1]*b[0],a[1]*b[1]])
        return products.min(0).values, products.max(0).values
    t = trunc(x,x,16-f)
    z = trunc(*multiply(t,t),f)
    cs = [round(c*(1 << f)) for c in coefficients[::2]]
    lo=hi=torch.full_like(x,cs[-1])
    for c in reversed(cs[:-1]):
        lo,hi=trunc(*multiply((lo,hi),z),f)
        lo,hi=lo+c,hi+c
    linear=trunc(x,x,1)
    return (lo << (14-f))+linear[0], (hi << (14-f))+linear[1], maximum


def run_case(kind, degree, batch, repeats, profile='uniform'):
    quadratic_profile = 'uniform' if profile == 'uniform' else 'local'
    from rep4gpu.models.phase_graph import PhaseGraphParty
    from rep4gpu.protocol.device_transport import DeviceIdealEndpoint, DeviceRecorder, DeviceReplay
    coefficients=fit(kind,degree,profile) if degree != 2 else None
    f=precision(coefficients) if coefficients is not None else None
    x=torch.round(torch.linspace(-4,4,batch*8192,dtype=torch.float64)*16384).to(torch.int64).reshape(batch,8192)
    if coefficients is not None:
        lower,upper,maximum=interval_oracle(x,coefficients,f)
        evaluate=lambda p,v:p.open(polynomial(p,v,coefficients,f))
    else:
        from experiments.benchmarks.activations.activation_common import quadratic_interval
        plan=quadratic_plan(kind,4,fractional_bits=14,bits=32,profile=quadratic_profile)
        lo=x >> plan.prescale
        hi=(x+(1 << plan.prescale)-1) >> plan.prescale
        low_n,high_n=quadratic_interval(lo,hi,plan)
        lower=low_n >> plan.shift
        upper=(high_n+(1 << plan.shift)-1) >> plan.shift
        maximum=plan.numerator_abs_bound
        evaluate=lambda p,v:p.open(FixedPoint(p,14).bounded_quadratic(v,kind=kind,bound=4,profile=quadratic_profile))
    network,seeds=IdealNetwork(),make_seeds()
    parties=[PhaseGraphParty(i,DeviceIdealEndpoint(network.endpoint(i),torch.device('cuda:0')),
             {j:s for j,s in seeds.items() if j!=i},device='cuda:0',bits=32,binary_bits=32,
             optimized=True,gpu_open=True,rounding='probabilistic',sparse_products=True,narrow_products=True) for i in range(4)]
    def check(outputs):
        for output in outputs:
            actual=output.cpu()
            assert torch.equal(actual,outputs[0].cpu())
            assert bool(((actual>=lower)&(actual<=upper)).all())
    with ThreadPoolExecutor(4) as pool:
        def parallel(fn):
            def guarded(p):
                try:return fn(p)
                except BaseException as error:
                    network.abort(error)
                    raise
            return list(pool.map(guarded,parties))
        shares=parallel(lambda p:p.input(0,x if p.party==0 else None,x.shape))
        plans=[p.prepare_inference(lambda v,p=p:evaluate(p,v),shares[p.party],check_at_send=True) for p in parties]
        execute=lambda p:plans[p.party].run(shares[p.party])
        check(parallel(execute))
        samples=[]
        for _ in range(repeats):
            start=time.perf_counter(); outputs=parallel(execute); torch.cuda.synchronize()
            samples.append((time.perf_counter()-start)*1000);check(outputs)
        starts=[dict(p.counters) for p in parties]
        records=[DeviceRecorder(p.transport) for p in parties]
        for p,r in zip(parties,records):p.transport=r
        outputs=parallel(execute);check(outputs)
        expected=outputs[0].cpu()
        for p,r in zip(parties,records):p.transport=r.underlying
    phase_samples=[]
    for p in parties:
        local=PhaseGraphParty(p.party,p.transport,p.seeds,device=p.device,bits=32,binary_bits=32,
              optimized=True,gpu_open=True,rounding='probabilistic',sparse_products=True,narrow_products=True)
        local._device_keys=p._device_keys;local._batch_keys=p._batch_keys
        plan=local.prepare_inference(lambda v:evaluate(local,v),shares[p.party],check_at_send=True)
        phases=[]
        for repeat in range(repeats+1):
            replay=DeviceReplay(records[p.party].transcript,device=p.device,check_outgoing=repeat==0,
                                staging='device',incoming_buffers=plan.receive_buffers)
            local.transport=replay;local.counters=dict(starts[p.party])
            replay.start(synchronize=torch.cuda.synchronize)
            output=plan.run(shares[p.party]);replay.finish();replay.assert_consumed()
            assert torch.equal(output.cpu(),expected)
            if repeat:phases.append(replay.phase_times)
        phase_samples.append(phases)
    exact=(torch.nn.functional.gelu(x.double()/16384) if kind=='gelu' else torch.nn.functional.silu(x.double()/16384))
    error=expected.double()/16384-exact
    return dict(recorded_live_max_error=float(error.abs().max()),recorded_live_rmse=float(error.square().mean().sqrt()),narrow_products=True,kind=kind,profile=profile if degree != 2 else quadratic_profile,degree=degree,batch=batch,elements=x.numel(),ring_bits=32,
                input_output_fractional_bits=14,internal_fractional_bits=f,
                coefficients_normalized=None if coefficients is None else list(coefficients),maximum_pretruncation_abs_bound=maximum,
                strict_pretruncation_limit=1 << 30,integer_interval_oracle_passed=True,
                maximum_rounding_interval_width=int((upper-lower).max()),
                live_four_party_median_ms=statistics.median(samples),live_samples_ms=samples,
                ideal_network_phase_max_ms=phase_estimate(phase_samples),
                rounds=len(records[0].transcript),
                payload_bytes=sum(sum(map(len,frame.outgoing.values())) for r in records for frame in r.transcript),
                **phase_payloads(records))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=int,nargs='+',default=[1,128])
    parser.add_argument('--degree',type=int,nargs='+',default=[2,4,6,8,10])
    parser.add_argument('--kind',nargs='+',default=['gelu','silu'])
    parser.add_argument('--profile',nargs='+',default=['uniform','mixed'])
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--output',type=Path,default=Path('experiments/results/activations/activation_polynomial_protocol.json'))
    args=parser.parse_args();torch.set_num_threads(1)
    report=dict(source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['experiments/benchmarks/activations/activation_polynomial_protocol.py','experiments/benchmarks/activations/activation_approximation_quality.py','src/rep4gpu/arithmetic/activations.py','src/rep4gpu/gpu/bits/device_probabilistic.py','src/rep4gpu/models/phase_graph.py']},torch_version=torch.__version__,gpu=torch.cuda.get_device_name(),input_interval=[-4,4],activation_only=True,
      methodology='Pure Rep4 ring32 GPU activation protocols with checked CUDA phase graphs and probabilistic truncation. Live samples draw fresh masks. Device-delivered isolated-party replay reproduces one recorded transcript for timing only; its estimate excludes network latency/bandwidth. Inputs are public bounded fixtures privately shared; range is a caller contract, with no private clipping/check. Quadratic uses the existing uniform or local plan (mixed profile selects local); higher degrees use normalized even Horner with reduced internal precision and Q14 output. Interval integer oracle covers every floor/ceil rounding outcome. No full FFN or language-model quality claim.',results=[])
    for profile in args.profile:
        for kind in args.kind:
            for batch in args.batch:
                for degree in args.degree:
                    row=run_case(kind,degree,batch,args.repeats,profile);report['results'].append(row)
                    args.output.parent.mkdir(parents=True,exist_ok=True)
                    args.output.write_text(json.dumps(report,indent=2)+'\n')
                    print(json.dumps({k:row[k] for k in ('kind','profile','degree','batch','ideal_network_phase_max_ms','rounds','payload_bytes')}),flush=True)

if __name__=='__main__':main()
