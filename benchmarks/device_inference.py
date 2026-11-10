"""Measure the experimental CUDA ChaCha20/BLAKE3 Rep4 inference variant.

Run from the repository root with CUDA available:
python -m benchmarks.device_inference --quick --bits 32 --binary-bits 32
Fixed-fixture replay is only a timing harness; never reuse its masks for requests.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from threadpoolctl import threadpool_info, threadpool_limits

from benchmarks.inference import (baselines, checked_add,
                                  checked_matmul, fixture, infer, maximum_abs,
                                  plain, provenance, synchronize)
from rep4gpu.transport import IdealNetwork
from rep4gpu.runtime import make_seeds


def fixture_bounds(integers, bits, fractional_bits, rounding='floor'):
    """Propagate public fixture ranges with overflow-checked int64 arithmetic."""
    if rounding not in ('floor','probabilistic'):
        raise ValueError('rounding must be floor or probabilistic')
    x,w1,b1,w2,b2=integers
    scale=1 << fractional_bits
    if maximum_abs(b1)*scale > 2**63-1:
        raise ValueError('scaled fixture bias exceeds signed int64')
    raw1=checked_add(checked_matmul(x,w1),b1*scale)
    low_h=(raw1 >> fractional_bits).clamp_min(0)
    if rounding=='probabilistic':
        high_h=(checked_add(raw1,torch.tensor(scale-1)) >> fractional_bits).clamp_min(0)
        positive,negative=w2.clamp_min(0),w2.clamp_max(0)
        raw2lo=checked_add(checked_matmul(low_h,positive),checked_matmul(high_h,negative))
        raw2hi=checked_add(checked_matmul(high_h,positive),checked_matmul(low_h,negative))
        lower=checked_add(raw2lo >> fractional_bits,b2)
        upper=checked_add(checked_add(raw2hi,torch.tensor(scale-1)) >> fractional_bits,b2)
    else:
        high_h=low_h
        raw2lo=raw2hi=checked_matmul(low_h,w2)
        lower=upper=checked_add(raw2lo >> fractional_bits,b2)
    maximum=max(maximum_abs(value) for value in (raw1,raw2lo,raw2hi))
    if rounding=='probabilistic' and maximum >= 1 << (bits-2):
        raise ValueError('fixture violates strict probabilistic abs(accumulator) < 2**(bits-2) contract')
    for value in [*integers,raw1,low_h,high_h,raw2lo,raw2hi,lower,upper]:
        if int(value.min()) < -(1 << (bits-1)) or int(value.max()) >= (1 << (bits-1)):
            raise ValueError(f'fixture exceeds signed ring{bits}')
    proof=dict(maximum_accumulator=maximum,first_accumulator_max_abs=maximum_abs(raw1),
               second_accumulator_max_abs=max(maximum_abs(raw2lo),maximum_abs(raw2hi)),
               accumulator_fits_signed_ring=True,rounding_bounds_validated=True,
               maximum_output_rounding_interval_width=int((upper-lower).max()),
               range_validation='public fixture only; not a private range check')
    if rounding=='probabilistic':
        proof['strict_probabilistic_accumulator_limit']=1 << (bits-2)
    return proof,lower,upper


def fixture_range(integers, bits, fractional_bits):
    """Validate the exact public fixture before any ring operation or sharing."""
    return fixture_bounds(integers,bits,fractional_bits)[0]


def validate_outputs(outputs,lower,upper,expected=None):
    """Require agreement plus either exact reference equality or rounding bounds."""
    actual=outputs[0].cpu()
    if any(not torch.equal(actual,value.cpu()) for value in outputs[1:]):
        raise AssertionError('device inference parties disagree on opened outputs')
    if expected is not None:
        if not torch.equal(actual,expected):
            raise AssertionError('device inference differs from exact recorded or integer reference')
    elif not bool(((actual >= lower) & (actual <= upper)).all()):
        raise AssertionError('device inference exceeds propagated probabilistic rounding bounds')
    return actual


def phase_estimate(samples):
    """Sum phase maxima over per-role sample medians, returning milliseconds."""
    return float(np.max(np.median(np.asarray(samples),axis=1),axis=0).sum()*1000)


def matmul_backend(batch,bits,prepared,storage,optimized,capability):
    """Describe the selected validated local-product route for these inputs."""
    if prepared and storage=='five32':
        return dict(route='five exact ring32 GEMMs with cached combinations')
    if optimized and bits==32 and prepared and storage=='words32' and batch>=16 and tuple(capability)==(8,9):
        return dict(route='SM89 tuned six-product local kernel',tile_mnk=[32,128,64],
                    num_warps=4,num_stages=1)
    if batch==1:
        return dict(route='baseline packed decode' if prepared and storage=='limbs' else 'baseline direct decode')
    return dict(route='baseline six-product local kernel',tile_mnk=[16,32,64],num_warps=4)


def weight_storage_bytes(weights):
    """Count retained tensor storage once, excluding temporary preparation data."""
    seen = set()
    def count(value):
        if value is None:
            return 0
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            identity = (str(value.device), storage.data_ptr())
            if identity in seen:
                return 0
            seen.add(identity)
            return storage.nbytes()
        if hasattr(value, 'parts'):
            return sum(count(part) for part in value.parts)
        return sum(count(getattr(value, name, None)) for name in ('limbs', 'words', 'original'))
    return sum(count(weight) for weight in weights)


def phase_payloads(records):
    """Describe each public exchange; the final local phase sends no payload."""
    frames = [record.transcript for record in records]
    if len({len(transcript) for transcript in frames}) != 1:
        raise ValueError('recorded party transcripts have different round counts')
    kinds = []
    for phase in range(len(frames[0])):
        phase_kinds = {transcript[phase].kind for transcript in frames}
        if len(phase_kinds) != 1:
            raise ValueError('recorded party round kinds disagree')
        kinds.append(phase_kinds.pop())
    return dict(party_order=[0,1,2,3],round_kinds=kinds,phase_kinds=kinds+['local_only'],
                outgoing_bytes_per_party_per_phase=[
                    [sum(map(len,frame.outgoing.values())) for frame in transcript] + [0]
                    for transcript in frames],
                incoming_bytes_per_party_per_phase=[
                    [sum(map(len,frame.incoming.values())) for frame in transcript] + [0]
                    for transcript in frames],
                phase_payload_convention='phase i ends at exchange i; final phase follows the last exchange and has zero payload')


def run_case(batch,width,hidden,out,device='cuda',repeats=5,bits=32,
             binary_bits=32,fractional_bits=12,prepared=False,
             weight_storage='limbs',staging='device',optimized=False,gpu_open=False,rounding='floor'):
    if rounding not in ('floor','probabilistic'):
        raise ValueError('rounding must be floor or probabilistic')
    if rounding=='probabilistic' and not 0 <= fractional_bits < bits-1:
        raise ValueError('probabilistic fractional_bits must be in [0, bits-2]')
    if repeats < 1:
        raise ValueError('repeats must be positive')
    if weight_storage not in ('limbs','words32','five32'):
        raise ValueError('weight_storage must be limbs, words32 or five32')
    if weight_storage == 'five32' and (bits != 32 or not prepared):
        raise ValueError('five32 requires bits=32 and prepared=True')
    if staging not in ('device','host'):
        raise ValueError('staging must be device or host')
    if torch.device(device).type != 'cuda':
        raise ValueError('device inference requires CUDA')
    from rep4gpu.device_inference import DeviceInferenceParty
    from rep4gpu.device_transport import DeviceByteBridge, DeviceIdealEndpoint, DeviceRecorder, DeviceReplay
    device=torch.device(device)
    if device.index is None:
        device=torch.device('cuda',torch.cuda.current_device())
    floats,integers,expected=fixture(batch,width,hidden,out,fractional_bits)
    bounds,lower,upper=fixture_bounds(integers,bits,fractional_bits,rounding)
    backend=matmul_backend(batch,bits,prepared,weight_storage,optimized,torch.cuda.get_device_capability(device))
    reference=plain(floats)
    fresh_accuracy=[]
    party_type=DeviceInferenceParty
    network,seeds=IdealNetwork(),make_seeds()
    adapter=DeviceIdealEndpoint if staging=='device' else DeviceByteBridge
    parties=[party_type(p,adapter(network.endpoint(p),device),{i:s for i,s in seeds.items() if i!=p},
                        device=device,bits=bits,binary_bits=binary_bits,optimized=optimized,gpu_open=gpu_open,rounding=rounding) for p in range(4)]
    def check(outputs):
        actual=validate_outputs(outputs,lower,upper,expected if rounding=='floor' else None)
        decoded=actual.float()/(1 << fractional_bits)
        fresh_accuracy.append(dict(max_logit_error=float((decoded-reference).abs().max()),
                                   max_logprob_error=float((decoded.log_softmax(-1)-reference.log_softmax(-1)).abs().max())))
    with ThreadPoolExecutor(4) as pool:
        def run_parties(function):
            def guarded(party):
                try:
                    return function(party)
                except BaseException as error:
                    network.abort(error)
                    raise
            return list(pool.map(guarded,parties))
        def distribute(p):
            values=[p.input(owner,value if p.party==owner else None,value.shape)
                    for owner,value in zip([0,1,1,1,1],integers)]
            if prepared:
                values[1]=p.prepare_weight_share(values[1],storage=weight_storage)
                values[3]=p.prepare_weight_share(values[3],storage=weight_storage)
            return values
        start=time.perf_counter()
        shares=run_parties(distribute)
        synchronize(device)
        retained_weight_bytes=[weight_storage_bytes((values[1],values[3])) for values in shares]
        setup_ms=(time.perf_counter()-start)*1000
        check(run_parties(lambda p:infer(p,shares[p.party],fractional_bits)))
        live_samples=[]
        for _ in range(repeats):
            start=time.perf_counter()
            outputs=run_parties(lambda p:infer(p,shares[p.party],fractional_bits))
            synchronize(device)
            live_samples.append((time.perf_counter()-start)*1000)
            check(outputs)
        # Record a separate fresh execution, leaving host transcript copies out
        # of every live timing sample. Shared model/input tensors are immutable.
        starts=[dict(p.counters) for p in parties]
        records=[DeviceRecorder(p.transport) for p in parties]
        for p,recorder in zip(parties,records):
            p.transport=recorder
        outputs=run_parties(lambda p:infer(p,shares[p.party],fractional_bits))
        synchronize(device)
        check(outputs)
        recorded_output=outputs[0].cpu().clone()
        for p,recorder in zip(parties,records):
            p.transport=recorder.underlying
    replay_modes={}
    for replay_staging in ('device','host'):
        phase_samples,party_samples=[],[]
        for p in parties:
            phases,times=[],[]
            for repeat in range(repeats+1):
                replay=DeviceReplay(records[p.party].transcript,device=p.device,
                                    check_outgoing=repeat==0,staging=replay_staging)
                local=party_type(p.party,replay,p.seeds,device=p.device,bits=bits,binary_bits=binary_bits,
                                 optimized=optimized,gpu_open=gpu_open,rounding=rounding)
                local.counters=dict(starts[p.party])
                # Domain-key setup belongs to setup, as in the warmed live run.
                # Keys are immutable; ChaCha20 draws still execute each replay.
                local._device_keys=p._device_keys
                local._batch_keys=p._batch_keys
                replay.start(synchronize=lambda:synchronize(device))
                output=infer(local,shares[p.party],fractional_bits)
                replay.finish()
                replay.assert_consumed()
                validate_outputs([output],lower,upper,recorded_output)
                if repeat:
                    phases.append(replay.phase_times)
                    times.append(sum(replay.phase_times)*1000)
            phase_samples.append(phases)
            party_samples.append(times)
        replay_modes[replay_staging]=dict(ideal_network_phase_max_ms=phase_estimate(phase_samples),
                                         replay_party_samples_ms=party_samples,
                                         replay_phase_samples_seconds=phase_samples)
    estimate=replay_modes['device']['ideal_network_phase_max_ms']
    host_estimate=replay_modes['host']['ideal_network_phase_max_ms']
    baseline,baseline_errors=baselines(floats,device,repeats)
    frames=records[0].transcript
    return dict(**bounds,**phase_payloads(records),
                live_transport='colocated IdealNetwork role harness',physical_gpudirect=False,
                batch=batch,width=width,hidden=hidden,out=out,device=str(device),
                ring_bits=bits,binary_bits=binary_bits,fractional_bits=fractional_bits,
                optimized=optimized,
                local_matmul_backend=backend,
                optimized_components=(['batched CUDA ChaCha20','SWAR carry extraction','fused weighted B2A',
                                       'SM89 compact-weight prefill tile dispatch with validated fallback'] if optimized else []) +
                                     (['five-GEMM prepared ring32 matmul'] if weight_storage=='five32' else []) +
                                     (['GPU checked output opening'] if gpu_open else []) +
                                     (['device probabilistic truncation'] if rounding=='probabilistic' else []),
                gpu_open=gpu_open,
                weight_storage_bytes_per_party=retained_weight_bytes,
                weight_storage_bytes_all_parties=sum(retained_weight_bytes),
                rounding=rounding,truncation='compact exact floor' if rounding=='floor' else 'probabilistic floor/ceil',
                prepared_weights=prepared,weight_storage=weight_storage,
                prg='CUDA ChaCha20 with SHA256 domain-key derivation',online_hash='CUDA BLAKE3',
                input_hash='inherited checked BLAKE2b byte protocol',
                output_hash='CUDA BLAKE3 checked device protocol' if gpu_open else 'inherited checked BLAKE2b byte protocol',
                reference_transcript_compatible=False,live_staging=staging,
                exact_integer_match=rounding=='floor',exact_recorded_replay_output_match=True,
                fresh_accuracy_samples=fresh_accuracy,
                fresh_accuracy_sample_order=['warmup']+[f'live_{i}' for i in range(repeats)]+['recording'],
                max_logit_error=max(sample['max_logit_error'] for sample in fresh_accuracy),
                max_logprob_error=max(sample['max_logprob_error'] for sample in fresh_accuracy),
                setup_all_parties_ms=setup_ms,local_four_party_ms=statistics.median(live_samples),
                local_samples_ms=live_samples,recording_included_in_live_timing=False,
                replay_modes=replay_modes,ideal_network_phase_max_ms=estimate,
                host_bridge_ideal_network_phase_max_ms=host_estimate,
                rounds=len(frames),device_rounds=sum(frame.kind=='device' for frame in frames),
                byte_rounds=sum(frame.kind=='bytes' for frame in frames),
                payload_bytes=sum(sum(map(len,frame.outgoing.values())) for r in records for frame in r.transcript),
                baseline_accuracy=baseline_errors,baselines=baseline,
                ideal_slowdowns={name:estimate/ms for name,ms in baseline.items()},
                host_bridge_ideal_slowdowns={name:host_estimate/ms for name,ms in baseline.items()},
                samples=repeats,protocol_warmups=1,replay_warmups_per_party_per_staging=1,
                baseline_samples=max(20,repeats),baseline_warmups=3,baseline_graph_device_replays=100,
                ideal_timing_kind='estimate: sum of per-phase maxima of isolated-party median times',
                timing_boundary='inference entry through checked output opening and final CUDA synchronization; setup excluded',
                methodology='Experimental Rep4 variant; ChaCha20 masks and BLAKE3 integrity checks execute during every online/replay run. Not byte compatible with MP-SPDZ/reference masks or tags. Device replay assumes peer payloads already delivered to GPU; host replay includes outgoing/incoming host staging. Both exclude network latency/bandwidth. Fresh unrecorded live timings and a separate untimed transcript; outgoing bytes checked during replay warmup. Domain keys, input sharing and static weight preparation are setup. Floor outputs match the exact integer fixture; probabilistic outputs agree across parties and obey propagated rounding intervals. Every replay exactly matches the recorded output. Reported errors are worst-case over fresh warmup, live and recorded runs.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--batch',type=int,nargs='+',default=[1,128])
    parser.add_argument('--width',type=int,nargs='+',default=[2048])
    parser.add_argument('--bits',type=int,choices=[32,64],default=32)
    parser.add_argument('--binary-bits',type=int,choices=[32,64],default=32)
    parser.add_argument('--fractional-bits',type=int,default=12)
    parser.add_argument('--prepared',action='store_true')
    parser.add_argument('--optimized',action='store_true',help='enable batched PRG, SWAR carries and fused B2A')
    parser.add_argument('--rounding',choices=['floor','probabilistic'],default='floor')
    parser.add_argument('--gpu-open',action='store_true',help='use the optional BLAKE3 checked GPU output opening')
    parser.add_argument('--weight-storage',choices=['limbs','words32','five32'],default='limbs')
    parser.add_argument('--staging',choices=['device','host'],default='device')
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--output',type=Path,default=Path('results/device_inference.json'))
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    metadata=provenance()
    metadata['source_sha256']['benchmarks/device_inference.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report=dict(torch=torch.__version__,numpy=np.__version__,threads=torch.get_num_threads(),
                gpu=torch.cuda.get_device_name(args.device),blas_threadpools=threadpool_info(),
                provenance=metadata,results=[])
    cases=[(1,8,8,4)] if args.quick else [(b,w,4*w,w) for b in args.batch for w in args.width]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for b,w,h,n in cases:
        row=run_case(b,w,h,n,args.device,args.repeats,args.bits,args.binary_bits,
                     args.fractional_bits,args.prepared,args.weight_storage,args.staging,args.optimized,args.gpu_open,args.rounding)
        report['results'].append(row)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({key:row[key] for key in
                          ('batch','width','ring_bits','binary_bits','live_staging','optimized','optimized_components',
                           'prepared_weights','weight_storage','weight_storage_bytes_all_parties','local_matmul_backend','gpu_open','output_hash','rounding','exact_integer_match',
                           'max_logprob_error','local_four_party_ms','ideal_network_phase_max_ms',
                           'host_bridge_ideal_network_phase_max_ms','ideal_slowdowns')}),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):
        main()
