"""Measure per-party authenticated field inference with isolated replay.

The initial preprocessing provider is an explicit trusted dealer. Its generation
and owner-private input sharing are reported separately from resident online
inference. Replay uses only a fixed fixture and must never handle live inputs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import secrets
import statistics
import time

import numpy as np
import torch
from threadpoolctl import threadpool_info,threadpool_limits

from benchmarks.inference import checked_matmul,provenance,synchronize,timed
from benchmarks.device_inference import phase_estimate,phase_payloads
from benchmarks.spdz_timing import wall_measurement,graph_measurement,device_batch_graph_measurement
from rep4gpu.spdz import PRIME,TrustedDealer
from rep4gpu.spdz_distributed import AuthShare,DistributedParty,split_triple
from rep4gpu.spdz_inputs import make_input_masks,private_input
from rep4gpu.spdz_polynomial import make_square_masks,square
from rep4gpu.transport import IdealNetwork,RecordingTransport,ReplayTransport


class SaltFixture:
    """Benchmark-only entropy recording; live draws always use OS randomness."""
    def _commit_salt(self):
        fresh=secrets.token_bytes(32)
        replay=getattr(self,'_benchmark_replay_salts',None)
        if replay is not None:
            index=self._benchmark_salt_index
            if index >= len(replay):
                raise AssertionError('recorded commitment salts exhausted')
            self._benchmark_salt_index+=1
            return replay[index]
        self._benchmark_salt_log.append(fresh)
        return fresh

    def _coin_draw(self,kind):
        fresh=secrets.token_bytes(32)
        replay=getattr(self,'_benchmark_replay_coins',None)
        if replay is not None:
            index=self._benchmark_coin_indexes.get(kind,0)
            if index>=len(replay.get(kind,[])):
                raise AssertionError('recorded coin entropy exhausted')
            self._benchmark_coin_indexes[kind]=index+1
            return replay[kind][index]
        if not hasattr(self,'_benchmark_coin_logs'):
            self._benchmark_coin_logs={}
        self._benchmark_coin_logs.setdefault(kind,[]).append(fresh)
        return fresh

    def _coin_contribution(self):
        return self._coin_draw('contribution')

    def _coin_salt(self):
        return self._coin_draw('salt')


from benchmarks.spdz_streamed_replay import StreamedDeviceReplay,StreamedIncomingReplay


def sample_stability(times):
    """Report tail spread and drift without selecting a favorable minimum."""
    tail=times[-min(5,len(times)):]
    first=times[:min(5,len(times))]
    tail_median=statistics.median(tail)
    spread=(max(tail)-min(tail))/tail_median
    return dict(sample_count=len(times),last_samples_ms=tail,
                tail_median_ms=tail_median,tail_relative_range=spread,
                first_to_tail_relative_drift=tail_median/statistics.median(first)-1,
                tail_stable=len(tail)>=5 and spread<=.10,
                stability_rule='at least five tail samples with range at most 10% of their median')


def require_checked_boundary(party):
    """A recording or accepted output cannot retain tentative MAC checks."""
    if getattr(party,'pending_values',0) or getattr(party,'_pending',[]):
        raise RuntimeError('benchmark boundary contains pending unauthenticated openings')


def fixture(m,k,n,hidden,workload):
    """Use small integer toy logits; every secret share remains dense and random."""
    if min(m,k,n,hidden)<1:
        raise ValueError('all dimensions must be positive')
    rng=np.random.default_rng(20261110)
    x=torch.zeros((m,k),dtype=torch.int64)
    for row in range(m):
        x[row,int(rng.integers(k))]+=1
        x[row,int(rng.integers(k))]-=1
    if workload=='matmul':
        w=torch.from_numpy(rng.integers(-1,2,(k,n),dtype=np.int64))
        values=[x,w]
        expected=checked_matmul(x,w)
    elif workload=='square_mlp':
        w1=torch.from_numpy(rng.integers(-1,2,(k,hidden),dtype=np.int64))
        b1=torch.from_numpy(rng.integers(-1,2,(1,hidden),dtype=np.int64)).expand(m,hidden).clone()
        w2=torch.from_numpy(rng.integers(-1,2,(hidden,n),dtype=np.int64))
        b2=torch.from_numpy(rng.integers(-1,2,(1,n),dtype=np.int64)).expand(m,n).clone()
        h=checked_matmul(x,w1)+b1
        if int(h.abs().max())>3037000499:
            raise ValueError('square reference would overflow signed int64')
        expected=checked_matmul(h*h,w2)+b2
        values=[x,w1,b1,w2,b2]
    else:
        raise ValueError('workload must be matmul or square_mlp')
    if int(expected.abs().max())>=PRIME//2:
        raise ValueError('fixture logits do not fit the centered prime field')
    return values,expected.remainder(PRIME)


def plain(values,workload):
    if workload=='matmul':
        return values[0]@values[1]
    x,w1,b1,w2,b2=values
    return (x@w1+b1).square()@w2+b2


def baselines(values,workload,device,repeats,return_details=False):
    timings,errors,details={},{},{}
    floats=[value.float() for value in values]
    reference=plain(floats,workload)
    for name,dtype in [('fp32',torch.float32),('fp16',torch.float16)]:
        if name=='fp16' and torch.device(device).type=='cpu':
            continue
        local=[value.to(device=device,dtype=dtype) for value in values]
        compute=lambda:plain(local,workload)
        actual=compute().cpu().float()
        if not bool(torch.isfinite(actual).all()):
            raise ValueError(f'{name} baseline is not finite for this fixture')
        errors[name]=float((actual-reference).abs().max())
        eager=wall_measurement(compute,device,max(20,repeats),20)
        timings[name+'_eager_ms']=eager['median_ms'];details[name+'_eager']=eager
        if torch.device(device).type=='cuda':
            graph=torch.cuda.CUDAGraph()
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(20):compute()
            torch.cuda.current_stream().wait_stream(stream)
            with torch.cuda.graph(graph,stream=stream):output=compute()
            grouped=graph_measurement(graph,device)
            timings[name+'_graph_ms']=grouped['median_ms']
            timings[name+'_graph_fastest_group_ms']=grouped['minimum_ms']
            details[name+'_graph']=grouped
            device_grouped=device_batch_graph_measurement(compute,device,expected_output=output)
            timings[name+'_device_batch_graph_ms']=device_grouped['median_ms']
            timings[name+'_device_batch_graph_fastest_group_ms']=device_grouped['minimum_ms']
            details[name+'_device_batch_graph']=device_grouped
            errors[name+'_graph']=float((output.cpu().float()-reference).abs().max())
            wall=wall_measurement(graph.replay,device,max(20,repeats),20)
            timings[name+'_graph_wall_ms']=wall['median_ms'];details[name+'_graph_wall']=wall
    arrays=[value.numpy().astype(np.float32) for value in values]
    def numpy_compute():
        if workload=='matmul':return arrays[0]@arrays[1]
        x,w1,b1,w2,b2=arrays
        return np.square(x@w1+b1)@w2+b2
    wall=wall_measurement(numpy_compute,'cpu',max(20,repeats),20)
    timings['numpy_cpu_ms']=wall['median_ms'];details['numpy_cpu']=wall
    return (timings,errors,details) if return_details else (timings,errors)


def execute(party,values,correlations,workload):
    if workload=='matmul':
        result=party.matmul(values[0],values[1],correlations[0])
    else:
        x,w1,b1,w2,b2=values
        h=party.add(party.matmul(x,w1,correlations[0]),b1)
        h=square(party,h,correlations[1])
        result=party.add(party.matmul(h,w2,correlations[2]),b2)
    return party.open(result)


def compact_correlations(items):
    """Store already-canonical trusted preprocessing in exact int32 words."""
    def compact(share):
        return AuthShare(share.values.to(torch.int32),share.macs.to(torch.int32))
    result=[]
    for item in items:
        fields=dict(a=compact(item.a))
        if hasattr(item,'a_squared'):
            fields['a_squared']=compact(item.a_squared)
        else:
            fields.update(b=compact(item.b),c=compact(item.c))
        result.append(replace(item,**fields))
    return result


def fresh_correlations(dealer,m,k,n,hidden,workload,compact=False):
    first=split_triple(dealer.triple(m,k,n if workload=='matmul' else hidden))
    if compact:first=compact_correlations(first)
    if workload=='matmul':return [(first[p],) for p in range(4)]
    masks=make_square_masks(dealer,(m,hidden))
    if compact:masks=compact_correlations(masks)
    second=split_triple(dealer.triple(m,hidden,n))
    if compact:second=compact_correlations(second)
    return [(first[p],masks[p],second[p]) for p in range(4)]


def run_case(m=1,k=128,n=128,hidden=None,workload='matmul',device='cuda',
             repeats=3,mac_keys=10,backend='host',stack_macs=True,stream_incoming=False,profile_path=None,replay_warmups=20,prepare_model=False,output_check_threshold=4096,live_repeats=None):
    live_repeats=repeats if live_repeats is None else live_repeats
    if type(live_repeats) is not int or live_repeats<1:
        raise ValueError('live_repeats must be a positive integer')
    hidden=4*k if hidden is None else hidden
    if replay_warmups<1:
        raise ValueError('at least one discarded replay warmup is required')
    if repeats<1 or mac_keys<5:
        raise ValueError('require positive repeats and at least five MAC keys')
    if backend not in ('host','device','compressed','device-deferred'):
        raise ValueError('backend must be host, device, compressed or device-deferred')
    if stream_incoming and backend=='host':
        raise ValueError('stream_incoming requires the device backend')
    if type(output_check_threshold) is not int or output_check_threshold<0:
        raise ValueError('output_check_threshold must be a nonnegative integer')
    device=torch.device(device)
    if device.type=='cuda' and device.index is None:
        device=torch.device('cuda',torch.cuda.current_device())
    values,expected=fixture(m,k,n,hidden,workload)
    network=IdealNetwork(timeout=180)
    session=secrets.token_bytes(32)
    if backend!='host':
        from rep4gpu.spdz_device import DeviceDistributedParty
        from rep4gpu.device_transport import DeviceIdealEndpoint,DeviceRecorder,DeviceReplay
        if backend=='device-deferred':
            from rep4gpu.spdz_deferred import DeviceDeferredCompressedParty
            base=DeviceDeferredCompressedParty
        elif backend=='compressed':
            from rep4gpu.spdz_compression import DeviceCompressedParty
            base=DeviceCompressedParty
        else:
            base=DeviceDistributedParty
        record_type=DeviceRecorder
        endpoints=[DeviceIdealEndpoint(network.endpoint(p),device) for p in range(4)]
    else:
        base,record_type=DistributedParty,RecordingTransport
        endpoints=[network.endpoint(p) for p in range(4)]
    party_type=type('BenchmarkAuthenticatedParty',(SaltFixture,base),{})
    synchronize(device)
    start=time.perf_counter()
    dealer=TrustedDealer(4,mac_keys,device)
    masks=[make_input_masks(dealer,0 if index==0 else 1,tuple(value.shape))
           for index,value in enumerate(values)]
    synchronize(device)
    preprocessing_setup_ms=(time.perf_counter()-start)*1000
    runtime_options=dict(output_check_threshold=output_check_threshold) if backend=='device-deferred' else {}
    parties=[party_type(p,endpoints[p],dealer.alpha_shares[:,p],parties=4,session=session,
                        _allow_insecure_test_transport=True,stack_macs=stack_macs,**runtime_options) for p in range(4)]
    for party in parties:party._benchmark_salt_log=[]
    def check(outputs):
        for party in parties:require_checked_boundary(party)
        if any(not torch.equal(value.cpu(),expected) for value in outputs):
            raise AssertionError('authenticated inference differs from exact field reference')
    online_samples,prep_samples,model_prep_samples=[],[],[]
    with ThreadPoolExecutor(4) as pool:
        def run_parties(function):
            def guarded(p):
                try:return function(p)
                except BaseException as error:
                    network.abort(error)
                    raise
            return list(pool.map(guarded,parties))
        start=time.perf_counter()
        shares=run_parties(lambda p:[private_input(p,mask[p.party],value.to(device).remainder(PRIME)
                                    if p.party==mask[p.party].owner else None)
                                    for mask,value in zip(masks,values)])
        synchronize(device)
        masks.clear()
        if device.type=='cuda':torch.cuda.empty_cache()
        private_input_ms=(time.perf_counter()-start)*1000
        for repeat in range(live_repeats+2):
            start=time.perf_counter()
            correlations=fresh_correlations(dealer,m,k,n,hidden,workload,compact=backend!='host')
            synchronize(device)
            prep_ms=(time.perf_counter()-start)*1000
            query_model=None
            model_prep_ms=0.
            if prepare_model:
                from benchmarks.spdz_prepared_model import prepare_model as prepare
                started=time.perf_counter()
                query_model=run_parties(lambda p:prepare(p,shares[p.party],correlations[p.party],workload))
                synchronize(device)
                model_prep_ms=(time.perf_counter()-started)*1000
                for p in parties:require_checked_boundary(p)
                # The opaque packs own A/F/model snapshots. Original B/C
                # tensors are no longer needed by query or fixture replay.
                correlations=[(None,items[1],None) if workload=='square_mlp' else () for items in correlations]
            recording=repeat==live_repeats+1
            if recording:
                for p in parties:require_checked_boundary(p)
                rounds=[p.round for p in parties]
                flush_before=[getattr(p,'_flush_index',0) for p in parties]
                spent_before=[set(p.spent_triples) for p in parties]
                square_spent_before=[set(getattr(p,'_spent_square_masks',set())) for p in parties]
                canonical_before=[list(getattr(p,'_canonical_versions',[])) for p in parties]
                records=[record_type(p.transport) for p in parties]
                for p,record in zip(parties,records):
                    p.transport=record
                    p._benchmark_salt_log=[]
                    p._benchmark_coin_logs={}
            start=time.perf_counter()
            def online(p):
                if query_model is None:
                    return execute(p,shares[p.party],correlations[p.party],workload)
                from benchmarks.spdz_prepared_model import infer_prepared
                mask=correlations[p.party][1] if workload=='square_mlp' else None
                return p.open(infer_prepared(p,shares[p.party],query_model[p.party],mask,workload))
            outputs=run_parties(online)
            synchronize(device)
            elapsed=(time.perf_counter()-start)*1000
            check(outputs)
            if 0<repeat<=live_repeats:
                online_samples.append(elapsed);prep_samples.append(prep_ms);model_prep_samples.append(model_prep_ms)
            if recording:
                for p,record in zip(parties,records):p.transport=record.underlying
    recorded=outputs[0].cpu().clone()
    modes={}
    selected_profile_mode='device' if backend!='host' else 'host'
    for staging in (('host','device') if backend!='host' else ('host',)):
        phase_samples,party_samples=[],[]
        for p in parties:
            phases,times=[],[]
            for repeat in range(repeats+replay_warmups):
                if backend!='host' and staging=='device' and stream_incoming:
                    replay=StreamedIncomingReplay.create(records[p.party].transcript,device,repeat==0)
                elif backend!='host':
                    replay=DeviceReplay(records[p.party].transcript,device=device,
                                        check_outgoing=repeat==0,staging=staging)
                else:
                    replay=ReplayTransport(records[p.party].transcript,check_outgoing=repeat==0)
                local=party_type(p.party,replay,p.alpha_share,parties=4,session=session,
                                 _allow_insecure_test_transport=True,stack_macs=stack_macs,**runtime_options)
                local.round=rounds[p.party]
                if hasattr(local,'_flush_index'):local._flush_index=flush_before[p.party]
                local.spent_triples=set(spent_before[p.party])
                local._spent_square_masks=set(square_spent_before[p.party])
                if backend!='host':
                    local._canonical_versions=list(canonical_before[p.party])
                local._benchmark_replay_salts=list(p._benchmark_salt_log)
                local._benchmark_salt_index=0
                local._benchmark_salt_log=[]
                local._benchmark_replay_coins={kind:list(draws) for kind,draws in getattr(p,'_benchmark_coin_logs',{}).items()}
                local._benchmark_coin_indexes={}
                fixture_correlations=tuple(replace(item,used=False) if item is not None else None for item in correlations[p.party])
                if query_model is not None:
                    from benchmarks.spdz_prepared_model import replay_model
                    fixture_model=replay_model(local,query_model[p.party])
                profiler=None
                if profile_path is not None and p.party==0 and repeat==min(1,replay_warmups-1) and staging==selected_profile_mode:
                    import cProfile
                    profiler=cProfile.Profile()
                    profiler.enable()
                replay.start(synchronize=lambda:synchronize(device))
                if query_model is None:
                    output=execute(local,shares[p.party],fixture_correlations,workload)
                else:
                    from benchmarks.spdz_prepared_model import infer_prepared
                    mask=fixture_correlations[1] if workload=='square_mlp' else None
                    output=local.open(infer_prepared(local,shares[p.party],fixture_model,mask,workload))
                replay.finish()
                if profiler is not None:
                    profiler.disable()
                    Path(profile_path).parent.mkdir(parents=True,exist_ok=True)
                    profiler.dump_stats(str(profile_path))
                replay.assert_consumed()
                require_checked_boundary(local)
                if local._benchmark_salt_index!=len(local._benchmark_replay_salts):
                    raise AssertionError('replay did not consume every recorded commitment salt')
                if any(local._benchmark_coin_indexes.get(kind,0)!=len(draws) for kind,draws in local._benchmark_replay_coins.items()):
                    raise AssertionError('replay did not consume every recorded coin entropy draw')
                if not torch.equal(output.cpu(),recorded):
                    raise AssertionError('replay output differs from the checked recorded execution')
                if repeat>=replay_warmups:
                    phases.append(replay.phase_times);times.append(sum(replay.phase_times)*1000)
            phase_samples.append(phases);party_samples.append(times)
        modes[staging]=dict(ideal_network_phase_max_ms=phase_estimate(phase_samples),
                            replay_phase_samples_seconds=phase_samples,replay_party_samples_ms=party_samples,
                            sample_stability=[sample_stability(samples) for samples in party_samples])
    selected='device' if backend!='host' else 'host'
    estimate=modes[selected]['ideal_network_phase_max_ms']
    baseline,errors,baseline_details=baselines(values,workload,device,repeats,return_details=True)
    if backend=='host':
        payload=sum(sum(map(len,sent.values())) for record in records for sent,_ in record.transcript)
    else:
        payload=sum(sum(map(len,frame.outgoing.values())) for record in records for frame in record.transcript)
    if backend=='host':
        from types import SimpleNamespace
        profile_records=[SimpleNamespace(transcript=[SimpleNamespace(kind='bytes',outgoing=sent,incoming=received)
                         for sent,received in record.transcript]) for record in records]
    else:
        profile_records=records
    return dict(**phase_payloads(profile_records),workload=workload,m=m,k=k,n=n,hidden=hidden,device=str(device),backend=backend,
                prime=PRIME,mac_keys=mac_keys,stack_macs=stack_macs,
                online_crypto_variant=getattr(parties[0],'crypto_variant','SPDZ31v1/BLAKE2b salted residual commitments'),
                preprocessing_provider='central trusted dealer; not dealer-free',
                fixture_semantics='unscaled integers: two signed one-hot entries, ternary weights/biases; dense random authenticated shares',
                maximum_centered_plaintext_logit_abs=int(torch.where(expected>PRIME//2,expected-PRIME,expected).abs().max()),
                truncation_or_fixed_point_rescaling=False,
                preprocessing_setup_ms=preprocessing_setup_ms,private_input_and_model_sharing_ms=private_input_ms,
                preprocessing_samples_ms=prep_samples,preprocessing_median_ms=statistics.median(prep_samples),
                preprocessing_included_in_online_estimate=False,
                prepared_model=prepare_model,
                model_dependent_preparation_samples_ms=model_prep_samples,
                model_dependent_preparation_median_ms=statistics.median(model_prep_samples),
                model_preparation_contract='Fresh triple/A/F per query; E checked and F=C+A@E prepared before query; no A/B reuse' if prepare_model else None,
                correlation_storage_dtype='int64' if backend=='host' else 'int32',
                exact_field_match=True,exact_recorded_replay_match=True,
                local_four_party_samples_ms=online_samples,local_four_party_ms=statistics.median(online_samples),
                ideal_network_phase_max_ms=estimate,replay_modes=modes,
                stream_incoming=stream_incoming,profile_path=str(profile_path) if profile_path else None,
                warmed_canonical_validation_state_restored=backend!='host',
                local_field_backend=getattr(parties[0],'local_backend','torch-field'),
                authenticated_input_storage_bytes_per_party=[sum(s.values.numel()*s.values.element_size()+s.macs.numel()*s.macs.element_size() for s in local) for local in shares],
                device_incoming_assumption=None if backend=='host' else 'fixed incoming fixture copied and synchronized outside phase clocks' if stream_incoming else 'all fixed incoming device frames preloaded before replay timing',
                ideal_slowdowns={key:estimate/value for key,value in baseline.items()},
                baselines=baseline,baseline_max_absolute_errors=errors,baseline_measurements=baseline_details,
                payload_bytes=payload,rounds=len(records[0].transcript),
                salts_per_inference=len(parties[0]._benchmark_salt_log),
                coin_entropy_draws_per_inference=sum(map(len,getattr(parties[0],'_benchmark_coin_logs',{}).values())),
                independent_coefficient_rows=mac_keys if backend in ('compressed','device-deferred') else None,
                deferred_mac_checks=backend=='device-deferred',
                output_check_threshold=getattr(parties[0],'output_check_threshold',None),
                output_check_mode=getattr(parties[0],'last_output_check_mode',None),
                deferred_flushes_per_inference=getattr(parties[0],'_flush_index',0)-flush_before[0],
                samples=repeats,live_samples=live_repeats,recorded_runs=1,
                warmups=1,replay_warmups_per_party=replay_warmups,
                live_transport='colocated IdealNetwork role harness',physical_gpudirect=False,
                ideal_timing_kind='estimate: sum of per-phase maxima of isolated-party medians',
                timing_boundary='resident private inputs/model through all checked intermediate openings and checked final output; native/trusted preprocessing, recurring per-query model-dependent preparation and private sharing excluded',
                methodology='Four roles execute independent authenticated Beaver arithmetic with ten keys by default. Every live run consumes fresh trusted correlations and fresh OS commitment salts. Separate untimed recording. Replay uses the recorded salts/correlations only for a fixed public fixture, while charging a fresh discarded OS entropy draw per salt; never serve live inputs from replay. Actual arithmetic, canonical validation, residual commitments, hashes, echoes and checks remain timed. Host replay includes host staging; device replay assumes incoming messages already on GPU. Both exclude network latency/bandwidth. Private input sharing recurs for new inputs. Model sharing may be reused; fresh model-dependent correlations are prepared separately for each query and cannot be reused. This is an integer square toy, not a transformer or fixed-point rescaling pipeline.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workload',choices=['matmul','square_mlp'],default='matmul')
    parser.add_argument('--m',type=int,nargs='+',default=[1])
    parser.add_argument('--k',type=int,nargs='+',default=[128])
    parser.add_argument('--n',type=int)
    parser.add_argument('--hidden',type=int)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--backend',choices=['host','device','compressed','device-deferred'],default='host')
    parser.add_argument('--mac-keys',type=int,default=10)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--live-repeats',type=int,default=3,help='fresh live correctness samples, separate from isolated replay samples')
    parser.add_argument('--unstacked-macs',action='store_true')
    parser.add_argument('--prepare-model',action='store_true')
    parser.add_argument('--output-check-threshold',type=int,default=4096,help='deferred final-output full-vector limit;0forces compressed')
    parser.add_argument('--replay-warmups',type=int,default=20)
    parser.add_argument('--profile-path',type=Path,help='cProfile one discarded isolated warmup; timed samples stay unprofiled')
    parser.add_argument('--stream-incoming',action='store_true',help='load one ideal GPU incoming fixture at each untimed transport boundary')
    parser.add_argument('--background-context',default='unspecified',help='concurrent host workload recorded with timing results')
    parser.add_argument('--output',type=Path,default=Path('results/spdz_inference.json'))
    args=parser.parse_args()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    metadata=provenance()
    for source in [Path(__file__),Path('benchmarks/spdz_native_replay.py'),Path('benchmarks/spdz_streamed_replay.py'),Path('benchmarks/spdz_prepared_model.py'),
                   Path('benchmarks/spdz_prepared_fixture.py'),Path('benchmarks/spdz_timing.py'),
                   Path('benchmarks/inference.py'),Path('benchmarks/device_inference.py')]:
        metadata['source_sha256'][str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
    report=dict(cpu_affinity=sorted(os.sched_getaffinity(0)),background_context=args.background_context,provenance=metadata,torch_threads=torch.get_num_threads(),blas_threadpools=threadpool_info(),
                gpu=torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else None,results=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for m in args.m:
        for k in args.k:
            row=run_case(m,k,args.n or k,args.hidden,args.workload,args.device,args.repeats,args.mac_keys,
                         args.backend,not args.unstacked_macs,args.stream_incoming,
                         args.profile_path.with_name(args.profile_path.stem+f'_m{m}_k{k}'+args.profile_path.suffix) if args.profile_path else None,args.replay_warmups,args.prepare_model,args.output_check_threshold,args.live_repeats)
            report['results'].append(row)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({key:row[key] for key in ('workload','m','k','n','hidden','backend','mac_keys',
                              'exact_field_match','prepared_model','model_dependent_preparation_median_ms','local_four_party_ms','ideal_network_phase_max_ms',
                              'preprocessing_median_ms','rounds','payload_bytes','ideal_slowdowns')}),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):
        main()
