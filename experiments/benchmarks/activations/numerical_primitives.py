"""Check and time bounded ring64 numerical protocols with fresh masks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from experiments.benchmarks.inference.device_inference import phase_estimate, phase_payloads


@dataclass
class Interval:
    lo: torch.Tensor
    hi: torch.Tensor

    @property
    def device(self):
        return self.lo.device

    @property
    def shape(self):
        return self.lo.shape


class IntegerOracle:
    """Propagate signed integer intervals through every floor/ceil choice."""
    bits = 64
    device = 'cpu'

    def __init__(self, probabilistic=False):
        self.probabilistic = probabilistic
        self.maximum = 0

    def public(self, value, like=None):
        value = torch.as_tensor(value, dtype=torch.int64)
        return Interval(value, value)

    def add(self, a, b):
        return Interval(a.lo+b.lo, a.hi+b.hi)

    def sub(self, a, b):
        return Interval(a.lo-b.hi, a.hi-b.lo)

    def mul(self, a, b):
        # Check in float64 before int64 multiplication can wrap.
        bound = max(float(a.lo.abs().max()), float(a.hi.abs().max())) * max(float(b.lo.abs().max()), float(b.hi.abs().max()))
        if bound >= 2**62:
            raise AssertionError(f'product bound {bound} exceeds probabilistic ring64 limit')
        values = torch.stack((a.lo*b.lo,a.lo*b.hi,a.hi*b.lo,a.hi*b.hi))
        self.maximum = max(self.maximum, int(values.abs().max()))
        return Interval(values.min(0).values, values.max(0).values)

    def mul_public(self, a, b):
        return self.mul(a, self.public(b))

    def trunc(self, a, shift):
        return Interval(a.lo >> shift, (a.hi+((1 << shift)-1 if self.probabilistic else 0)) >> shift)

    def sign_bit(self, a):
        return Interval((a.hi < 0).long(), (a.lo < 0).long())


def evaluate(runtime, kind, x, value=None, fractional_bits=24):
    from rep4gpu.arithmetic.numerical import NumericalFixedPoint
    fixed = NumericalFixedPoint(runtime, fractional_bits)
    return getattr(fixed, kind)(x, value) if kind == 'swiglu' else getattr(fixed, kind)(x)


def fixture(kind, elements, fractional_bits=24):
    bounds = {'exp':(-9,9), 'reciprocal':(1,2), 'erf':(-3,3)}
    lo,hi = bounds.get(kind,(-4,4))
    real = torch.linspace(lo, hi, elements, dtype=torch.float64)
    x = torch.round(real*(1 << fractional_bits)).long()
    value = torch.round(torch.linspace(4,-4,elements,dtype=torch.float64)*(1 << fractional_bits)).long()
    return x,value


def reference(kind, x, value, fractional_bits=24):
    x=x.double()/(1 << fractional_bits)
    if kind == 'exp': return x.exp()
    if kind == 'reciprocal': return x.reciprocal()
    if kind == 'erf': return x.erf()
    if kind == 'gelu': return torch.nn.functional.gelu(x)
    result=torch.nn.functional.silu(x)
    return result*value.double()/(1 << fractional_bits) if kind=='swiglu' else result


def oracle(kind, x, value, fractional_bits=24, probabilistic=False):
    runtime=IntegerOracle(probabilistic)
    result=evaluate(runtime,kind,runtime.public(x),runtime.public(value),fractional_bits)
    return result,runtime.maximum


def run_case(kind, elements=8192, repeats=3, rounding='probabilistic', fractional_bits=24, replay=True):
    from rep4gpu.models.phase_graph import PhaseGraphParty
    from rep4gpu.protocol.device_transport import DeviceIdealEndpoint, DeviceRecorder, DeviceReplay
    from rep4gpu.protocol.runtime import make_seeds
    from rep4gpu.protocol.transport import IdealNetwork
    torch.cuda.reset_peak_memory_stats()
    x,value=fixture(kind,elements,fractional_bits)
    interval,maximum=oracle(kind,x,value,fractional_bits,rounding=='probabilistic')
    network,seeds=IdealNetwork(timeout=180),make_seeds()
    options=dict(device='cuda:0',bits=64,binary_bits=64,optimized=True,gpu_open=True,rounding=rounding,sparse_products=True)
    parties=[PhaseGraphParty(i,DeviceIdealEndpoint(network.endpoint(i),torch.device('cuda:0')), {j:s for j,s in seeds.items() if j!=i},**options) for i in range(4)]
    def check(outputs):
        expected=outputs[0].cpu()
        for output in outputs:
            actual=output.cpu()
            assert torch.equal(actual,expected), 'parties disagree'
            assert bool(((actual>=interval.lo)&(actual<=interval.hi)).all()), 'integer oracle mismatch'
        return expected
    with ThreadPoolExecutor(4) as pool:
        def parallel(fn):
            def guarded(p):
                try:return fn(p)
                except BaseException as error:
                    network.abort(error)
                    raise
            return list(pool.map(guarded,parties))
        shares=parallel(lambda p:p.input(0,x if p.party==0 else None,x.shape))
        values=parallel(lambda p:p.input(1,value if p.party==1 else None,value.shape)) if kind=='swiglu' else shares
        plans=[p.prepare_inference(lambda a,b,p=p:p.open(evaluate(p,kind,a,b,fractional_bits)),shares[p.party],values[p.party],check_at_send=True) for p in parties]
        execute=lambda p:plans[p.party].run(shares[p.party],values[p.party])
        check(parallel(execute));samples=[]
        for _ in range(repeats):
            start=time.perf_counter();outputs=parallel(execute);torch.cuda.synchronize()
            samples.append((time.perf_counter()-start)*1000);check(outputs)
        starts=[dict(p.counters) for p in parties]
        records=[DeviceRecorder(p.transport) for p in parties]
        for p,r in zip(parties,records):p.transport=r
        expected=check(parallel(execute))
        for p,r in zip(parties,records):p.transport=r.underlying
    phase_samples=[]
    if replay:
        for p in parties:
            local=PhaseGraphParty(p.party,p.transport,p.seeds,**options)
            local._device_keys=p._device_keys;local._batch_keys=p._batch_keys
            plan=local.prepare_inference(lambda a,b:local.open(evaluate(local,kind,a,b,fractional_bits)),shares[p.party],values[p.party],check_at_send=True)
            phases=[]
            for repeat in range(repeats+1):
                recorded=DeviceReplay(records[p.party].transcript,device=p.device,check_outgoing=repeat==0,staging='device',incoming_buffers=plan.receive_buffers)
                local.transport=recorded;local.counters=dict(starts[p.party])
                recorded.start(synchronize=torch.cuda.synchronize)
                output=plan.run(shares[p.party],values[p.party]);recorded.finish();recorded.assert_consumed()
                assert torch.equal(output.cpu(),expected)
                if repeat:phases.append(recorded.phase_times)
            phase_samples.append(phases)
    truth=reference(kind,x,value,fractional_bits)
    error=expected.double()/(1 << fractional_bits)-truth
    relative=error.abs()/truth.abs().clamp_min(1e-12)
    bound_error=torch.maximum((interval.lo.double()/(1 << fractional_bits)-truth).abs(),(interval.hi.double()/(1 << fractional_bits)-truth).abs())
    return dict(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),kind=kind,elements=elements,ring_bits=64,fractional_bits=fractional_bits,rounding=rounding,integer_oracle_passed=True,party_agreement=True,maximum_pretruncation_abs_bound=maximum,maximum_rounding_interval_width=int((interval.hi-interval.lo).max()),max_error=float(error.abs().max()),max_relative_error=float(relative.max()),conservative_rounding_max_error=float(bound_error.max()),rmse=float(error.square().mean().sqrt()),live_samples_ms=samples,live_four_party_median_ms=statistics.median(samples),ideal_network_phase_max_ms=phase_estimate(phase_samples) if replay else None,exact_recorded_replay_passed=replay,rounds=len(records[0].transcript),payload_bytes=sum(sum(map(len,frame.outgoing.values())) for r in records for frame in r.transcript),**phase_payloads(records))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',nargs='+',default=['exp','reciprocal','erf','silu','gelu','swiglu'])
    parser.add_argument('--elements',type=int,nargs='+',default=[8192])
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--rounding',choices=['floor','probabilistic'],default='probabilistic')
    parser.add_argument('--fractional-bits',type=int,default=24)
    parser.add_argument('--output',type=Path,default=Path('experiments/results/activations/numerical_primitives.json'))
    args=parser.parse_args();torch.set_num_threads(1)
    sources=['experiments/benchmarks/activations/numerical_primitives.py','src/rep4gpu/arithmetic/numerical.py','src/rep4gpu/models/phase_graph.py','src/rep4gpu/gpu/bits/device_probabilistic.py']
    report=dict(source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},torch_version=torch.__version__,gpu=torch.cuda.get_device_name(),methodology='Activation-only ring64 protocols. Inputs are bounded public fixtures privately shared; only final outputs are opened. Live runs draw fresh masks. Integer intervals cover all floor/ceil rounding choices and verify accumulator bounds below 2**62. Isolated-party transcript replay checks exact output and outgoing messages; phase estimates exclude network latency and bandwidth.',results=[])
    for elements in args.elements:
        for kind in args.kind:
            row=run_case(kind,elements,args.repeats,args.rounding,args.fractional_bits);report['results'].append(row)
            args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({k:row[k] for k in ('kind','elements','rounding','max_error','live_four_party_median_ms','ideal_network_phase_max_ms','rounds','payload_bytes')}),flush=True)

if __name__=='__main__':main()
