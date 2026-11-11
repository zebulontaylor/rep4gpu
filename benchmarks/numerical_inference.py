"""Run complete secret-weight ring64 numerical FFNs on one colocated GPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
from benchmarks.device_inference import phase_estimate, phase_payloads


def run_case(kind, batch=1, width=2048, hidden=None, out=None, repeats=3,
             rounding='probabilistic', replay=True):
    from benchmarks.numerical_inference_common import fixture, oracle, infer, reference, metrics
    from rep4gpu.phase_graph import PhaseGraphParty
    from rep4gpu.device_transport import DeviceIdealEndpoint, DeviceRecorder, DeviceReplay
    from rep4gpu.runtime import make_seeds
    from rep4gpu.transport import IdealNetwork
    hidden, out = hidden or 4*width, out or width
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    floats,integers=fixture(kind,batch,width,hidden,out)
    interval,proof=oracle(kind,integers,probabilistic=rounding=='probabilistic')
    truth=reference(floats,kind)
    network,seeds=IdealNetwork(timeout=600),make_seeds()
    options=dict(device='cuda:0',bits=64,binary_bits=64,optimized=True,gpu_open=True,
                 rounding=rounding,sparse_products=True)
    parties=[PhaseGraphParty(i,DeviceIdealEndpoint(network.endpoint(i),torch.device('cuda:0')),
              {j:s for j,s in seeds.items() if j!=i},**options) for i in range(4)]
    def check(outputs):
        expected=outputs[0].cpu()
        assert all(torch.equal(o.cpu(),expected) for o in outputs),'party disagreement'
        assert bool(((expected>=interval.lo)&(expected<=interval.hi)).all()),'integer interval mismatch'
        return expected
    with ThreadPoolExecutor(4) as pool:
        def parallel(fn):
            def guarded(p):
                try:return fn(p)
                except BaseException as error:network.abort(error);raise
            return list(pool.map(guarded,parties))
        shares=parallel(lambda p:[p.input(0 if i==0 else 1,v if p.party==(0 if i==0 else 1) else None,v.shape) for i,v in enumerate(integers)])
        start=time.perf_counter()
        plans=[p.prepare_inference(lambda x,p=p:infer(p,[x,*shares[p.party][1:]],kind=kind),shares[p.party][0],check_at_send=True) for p in parties]
        preparation_ms=(time.perf_counter()-start)*1000
        execute=lambda p:plans[p.party].run(shares[p.party][0])
        check(parallel(execute));samples=[]
        for _ in range(repeats):
            start=time.perf_counter();outputs=parallel(execute);torch.cuda.synchronize()
            samples.append((time.perf_counter()-start)*1000);check(outputs)
        starts=[dict(p.counters) for p in parties]
        records=[DeviceRecorder(p.transport) for p in parties]
        for p,r in zip(parties,records):p.transport=r
        expected=check(parallel(execute))
        for p,r in zip(parties,records):p.transport=r.underlying
    peak=torch.cuda.max_memory_allocated()
    # Release live graph pools before constructing isolated replay plans.
    del plans;gc.collect();torch.cuda.empty_cache()
    phase_samples=[]
    if replay:
        for p in parties:
            local=PhaseGraphParty(p.party,p.transport,p.seeds,**options)
            local._device_keys=p._device_keys;local._batch_keys=p._batch_keys
            plan=local.prepare_inference(lambda x:infer(local,[x,*shares[p.party][1:]],kind=kind),shares[p.party][0],check_at_send=True)
            phases=[]
            for repeat in range(repeats+1):
                recorded=DeviceReplay(records[p.party].transcript,device=p.device,check_outgoing=repeat==0,staging='device',incoming_buffers=plan.receive_buffers)
                local.transport=recorded;local.counters=dict(starts[p.party])
                recorded.start(synchronize=torch.cuda.synchronize)
                output=plan.run(shares[p.party][0]);recorded.finish();recorded.assert_consumed()
                assert torch.equal(output.cpu(),expected)
                if repeat:phases.append(recorded.phase_times)
            phase_samples.append(phases)
            del plan,local;gc.collect();torch.cuda.empty_cache()
    baseline=[]
    gpu_values=[v.cuda().half() for v in floats]
    for i in range(repeats+3):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();reference(gpu_values,kind,preserve_dtype=True);end.record();end.synchronize()
        if i>=3:baseline.append(start.elapsed_time(end))
    from benchmarks.inference import baselines
    baseline_times,baseline_errors=baselines(floats,'cuda:0',repeats,forward=lambda v:reference(v,kind,preserve_dtype=True))
    result=dict(baselines=baseline_times,baseline_errors=baseline_errors,kind=kind,batch=batch,width=width,hidden=hidden,out=out,ring_bits=64,fractional_bits=24,
        rounding=rounding,party_agreement=True,integer_oracle_passed=True,range_proof=proof,
        exact_recorded_replay_passed=replay,only_final_output_opened=True,
        resident_raw_secret_weights=True,preparation_ms=preparation_ms,peak_cuda_allocated_bytes=max(peak,torch.cuda.max_memory_allocated()),
        live_samples_ms=samples,live_four_party_median_ms=statistics.median(samples),
        ideal_network_phase_max_ms=phase_estimate(phase_samples) if replay else None,
        fp16_eager_samples_ms=baseline,fp16_eager_median_ms=statistics.median(baseline),
        rounds=len(records[0].transcript),payload_bytes=sum(sum(map(len,f.outgoing.values())) for r in records for f in r.transcript),
        **phase_payloads(records),**metrics(expected,truth,interval))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',nargs='+',default=['silu','gelu','swiglu'])
    parser.add_argument('--batch',type=int,nargs='+',default=[1])
    parser.add_argument('--width',type=int,default=2048)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--rounding',choices=['floor','probabilistic'],default='probabilistic')
    parser.add_argument('--output',type=Path,default=Path('results/numerical_inference.json'))
    args=parser.parse_args();torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    report=dict(gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,methodology='Complete FFN with privately shared input, weights and biases. Fresh masks in live runs; only final output opened. Public integer interval oracle proves accumulator ranges. Checked isolated-party replay excludes network latency and bandwidth. FP16 baseline evaluates the same original floating formulas.',results=[])
    report['source_sha256']={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['benchmarks/numerical_inference.py','benchmarks/numerical_inference_common.py','rep4gpu/numerical.py']}
    for batch in args.batch:
        for kind in args.kind:
            row=run_case(kind,batch,args.width,repeats=args.repeats,rounding=args.rounding)
            report['results'].append(row);args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:row[k] for k in ['kind','batch','width','live_four_party_median_ms','ideal_network_phase_max_ms','max_error','rmse','peak_cuda_allocated_bytes']}),flush=True)
            gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
