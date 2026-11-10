"""Role-local timing helpers for the native-preprocessed authenticated runner.

Only the owning worker receives a runtime/material bundle. The coordinator
combines public timing arrays and never loads role-private preprocessing.
"""
from contextlib import contextmanager
from dataclasses import dataclass,replace
import fcntl
from pathlib import Path
import statistics

import numpy as np
import torch

from benchmarks.device_inference import phase_estimate
from benchmarks.inference import synchronize
from benchmarks.spdz_timing import wall_measurement,graph_measurement,device_batch_graph_measurement
from benchmarks.spdz_inference import SaltFixture,sample_stability,require_checked_boundary
from benchmarks.spdz_streamed_replay import StreamedIncomingReplay
from rep4gpu.transport import RecordingTransport,ReplayTransport


def entropy_fixture_class(runtime_class):
    """Add benchmark recording hooks without changing production entropy APIs."""
    class BenchmarkRuntime(SaltFixture,runtime_class):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self._benchmark_salt_log=[]
            self._benchmark_coin_logs={}
    return BenchmarkRuntime


@dataclass
class RecordingState:
    recorder: object
    round: int
    flush_index: int
    spent_triples: set
    spent_squares: set
    canonical_versions: list
    runtime_class: type
    backend: str


def begin_recording(runtime,backend):
    """Wrap one live execution; this execution is for checking, not live timing."""
    if not isinstance(runtime,SaltFixture):
        raise TypeError('recording requires the benchmark entropy fixture subclass')
    if backend=='host':
        recorder=RecordingTransport(runtime.transport)
    else:
        from rep4gpu.device_transport import DeviceRecorder
        recorder=DeviceRecorder(runtime.transport)
    require_checked_boundary(runtime)
    state=RecordingState(recorder,runtime.round,getattr(runtime,'_flush_index',0),set(runtime.spent_triples),
                         set(getattr(runtime,'_spent_square_masks',set())),
                         list(getattr(runtime,'_canonical_versions',[])),type(runtime),backend)
    runtime._benchmark_salt_log=[]
    runtime._benchmark_coin_logs={}
    runtime.transport=recorder
    return state


def end_recording(runtime,state):
    if runtime.transport is not state.recorder:
        raise ValueError('runtime does not own this recording')
    require_checked_boundary(runtime)
    runtime.transport=state.recorder.underlying


@contextmanager
def exclusive_replay(path):
    """Serialize all workers' GPU replay and staging, outside phase clocks."""
    path=Path(path)
    path.touch(mode=0o600,exist_ok=True)
    with path.open('r+') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)


def run_local_replay(runtime,material,values,state,opened,*,repeats,gpu_lock_path,
                     stream_incoming=False,profile_path=None,replay_warmups=20,prepared_model=None):
    """Replay only this worker's fixture, returning public timings and sizes.

    The live runtime's counters, spent IDs and transport are never reset.
    Restored state exists only in new instances bound to ReplayTransport.
    """
    if repeats<1 or replay_warmups<1:
        raise ValueError('replay repeats and discarded warmups must be positive')
    if runtime.transport is state.recorder:
        raise ValueError('end recording before replay or the completion barrier')
    from examples.spdz_trained_toy import infer_authenticated
    from rep4gpu.device_transport import DeviceReplay
    transcript=state.recorder.transcript
    modes={}
    salts=list(runtime._benchmark_salt_log)
    coins={kind:list(draws) for kind,draws in runtime._benchmark_coin_logs.items()}
    with exclusive_replay(gpu_lock_path):
        synchronize(runtime.device)
        expected=opened.detach().cpu().clone()
        for staging in (('host',) if state.backend=='host' else ('device','host')):
            phases,times,warmup_times=[],[],[]
            for repeat in range(repeats+replay_warmups):
                if state.backend=='host':
                    transport=ReplayTransport(transcript,check_outgoing=repeat==0)
                elif staging=='device' and stream_incoming:
                    transport=StreamedIncomingReplay.create(transcript,runtime.device,repeat==0)
                else:
                    transport=DeviceReplay(transcript,device=runtime.device,check_outgoing=repeat==0,staging=staging)
                if not isinstance(transport,(ReplayTransport,DeviceReplay)):
                    raise TypeError('restored protocol state is restricted to isolated fixture replay')
                options={}
                for name in ('output_check_threshold','max_pending_values'):
                    if hasattr(runtime,name):options[name]=getattr(runtime,name)
                local=state.runtime_class(runtime.party,transport,runtime.alpha_share,
                                          parties=runtime.parties,session=runtime.session,
                                          stack_macs=True,_allow_insecure_test_transport=True,**options)
                local.round=state.round
                if hasattr(local,'_flush_index'):local._flush_index=state.flush_index
                local.spent_triples=set(state.spent_triples)
                local._spent_square_masks=set(state.spent_squares)
                if state.backend!='host':
                    local._canonical_versions=list(state.canonical_versions)
                local._benchmark_replay_salts=salts
                local._benchmark_salt_index=0
                local._benchmark_replay_coins=coins
                local._benchmark_coin_indexes={}
                if prepared_model is None:
                    first=replace(material['matrix_triples']['first'],used=False)
                    second=replace(material['matrix_triples']['second'],used=False)
                else:
                    from benchmarks.spdz_prepared_model import replay_model
                    fixture_model=replay_model(local,prepared_model)
                square_mask=replace(material['square_masks']['hidden'],used=False)
                profiler=None
                if profile_path is not None and repeat==min(1,replay_warmups-1) and staging==('host' if state.backend=='host' else 'device'):
                    import cProfile
                    profiler=cProfile.Profile();profiler.enable()
                transport.start(synchronize=lambda:synchronize(runtime.device))
                if prepared_model is None:
                    result=infer_authenticated(local,values,first,second,square_mask)
                else:
                    from benchmarks.spdz_prepared_model import infer_prepared
                    result=infer_prepared(local,values,fixture_model,square_mask)
                output=local.open(result)
                transport.finish()
                if profiler is not None:
                    profiler.disable()
                    Path(profile_path).parent.mkdir(parents=True,exist_ok=True)
                    profiler.dump_stats(str(profile_path))
                transport.assert_consumed()
                require_checked_boundary(local)
                if local._benchmark_salt_index!=len(salts) or any(
                        local._benchmark_coin_indexes.get(kind,0)!=len(draws) for kind,draws in coins.items()):
                    raise AssertionError('replay did not consume the recorded entropy sequence')
                if not torch.equal(output.cpu(),expected):
                    raise AssertionError('isolated native replay differs from checked live output')
                if repeat>=replay_warmups:
                    phases.append(transport.phase_times)
                    times.append(sum(transport.phase_times)*1000)
                else:
                    warmup_times.append(sum(transport.phase_times)*1000)
            modes[staging]=dict(phase_samples_seconds=phases,samples_ms=times,warmup_samples_ms=warmup_times,median_ms=statistics.median(times),
                                sample_stability=sample_stability(times))
        # Destroy temporary replay contexts/buffers before another worker can
        # acquire the GPU. The live worker remains alive at a CPU barrier.
        del local,transport,result,output,square_mask
        if prepared_model is None:
            del first,second
        else:
            del fixture_model
        synchronize(runtime.device)
    if state.backend=='host':
        outgoing=[sum(map(len,sent.values())) for sent,_ in transcript]
        incoming=[sum(map(len,received.values())) for _,received in transcript]
        kinds=['bytes']*len(transcript)
    else:
        outgoing=[sum(map(len,frame.outgoing.values())) for frame in transcript]
        incoming=[sum(map(len,frame.incoming.values())) for frame in transcript]
        kinds=[frame.kind for frame in transcript]
    return dict(party=runtime.party,exact_recorded_output_match=True,role_local_keys=True,
                mac_keys=runtime.mac_keys,crypto_variant=getattr(runtime,'crypto_variant','SPDZ31v1/BLAKE2b'),
                local_field_backend=getattr(runtime,'local_backend','torch-field'),
                repeats=repeats,warmups=replay_warmups,modes=modes,prepared_model=prepared_model is not None,
                outgoing_bytes_per_phase=outgoing+[0],incoming_bytes_per_phase=incoming+[0],
                phase_kinds=kinds+['local_only'],rounds=len(transcript),
                commitment_salts=len(salts),coin_draws=sum(map(len,coins.values())),
                stream_incoming=stream_incoming,profile_path=str(profile_path) if profile_path else None,
                isolation='Exclusive GPU file lock after all workers complete checked live inference',
                entropy='Live OS entropy; isolated replay charges fresh discarded draws but substitutes the recorded fixture bytes',
                staging='Device mode excludes delivery time; host mode charges payload D2H/H2D. Neither includes network latency/bandwidth.')


def combine_replay_metrics(parties,baselines):
    """Combine public role timings without accepting any preprocessing objects."""
    if len(parties)!=4 or {p['party'] for p in parties}!=set(range(4)):
        raise ValueError('require one public timing record per role')
    ordered=sorted(parties,key=lambda p:p['party'])
    if any(role['phase_kinds']!=ordered[0]['phase_kinds'] for role in ordered[1:]):
        raise ValueError('public role phase schedules disagree')
    modes={}
    for mode in ordered[0]['modes']:
        samples=[p['modes'][mode]['phase_samples_seconds'] for p in ordered]
        estimate=phase_estimate(samples)
        tail=phase_estimate([role_samples[-min(5,len(role_samples)):] for role_samples in samples])
        stable=all(p['modes'][mode]['sample_stability']['tail_stable'] for p in ordered)
        modes[mode]=dict(ideal_network_phase_max_ms=estimate,
                        tail_ideal_network_phase_max_ms=tail,
                        full_to_tail_relative_difference=abs(estimate-tail)/estimate,
                        tails_stable=stable and abs(estimate-tail)/estimate<=.10,
                        ideal_slowdowns={key:estimate/value for key,value in baselines.items()})
    selected='device' if 'device' in modes else 'host'
    return dict(exact_recorded_output_match=all(p['exact_recorded_output_match'] for p in ordered),
                role_local_preprocessing=True,trusted_dealer=False,physical_gpudirect=False,
                prepared_model=all(p.get('prepared_model',False) for p in ordered),
                model_preparation_contract='Fresh model-dependent A/F correlations are prepared separately for each query; model sharing may be reused, correlations cannot',
                ideal_network_phase_max_ms=modes[selected]['ideal_network_phase_max_ms'],modes=modes,
                ideal_timing_kind='sum of per-phase maxima of independent-role median samples',
                timing_boundary='Resident secret inputs/model through every intermediate and final MAC check; native preprocessing, recurring per-query model-dependent preparation and private sharing excluded')


def trained_baselines(model,fixture,device,repeats):
    """Measure the actual trained PyTorch square MLP after workers have exited."""
    import copy
    timings,errors,details={},{},{}
    inputs=fixture['encoded'][0].float()
    with torch.inference_mode():reference=model(inputs).double()
    for label,dtype in [('fp32',torch.float32),('fp16',torch.float16)]:
        if torch.device(device).type=='cpu' and dtype==torch.float16:continue
        local=copy.deepcopy(model).to(device=device,dtype=dtype).eval()
        values=inputs.to(device=device,dtype=dtype)
        @torch.inference_mode()
        def compute():return local(values)
        actual=compute().cpu().double()
        if not bool(torch.isfinite(actual).all()):raise ValueError('clear baseline logits are not finite')
        errors[label]=dict(max_logit_error=float((actual-reference).abs().max()),
                           max_logprob_error=float((actual.log_softmax(-1)-reference.log_softmax(-1)).abs().max()))
        eager=wall_measurement(compute,device,max(20,repeats),20)
        timings[label+'_eager_ms']=eager['median_ms'];details[label+'_eager']=eager
        if torch.device(device).type=='cuda':
            graph=torch.cuda.CUDAGraph();stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(20):compute()
            torch.cuda.current_stream().wait_stream(stream)
            with torch.cuda.graph(graph,stream=stream):output=compute()
            grouped=graph_measurement(graph,device)
            timings[label+'_graph_ms']=grouped['median_ms']
            timings[label+'_graph_fastest_group_ms']=grouped['minimum_ms']
            details[label+'_graph']=grouped
            device_grouped=device_batch_graph_measurement(compute,device,expected_output=output)
            timings[label+'_device_batch_graph_ms']=device_grouped['median_ms']
            timings[label+'_device_batch_graph_fastest_group_ms']=device_grouped['minimum_ms']
            details[label+'_device_batch_graph']=device_grouped
            graph_actual=output.cpu().double()
            errors[label+'_graph']=dict(max_logit_error=float((graph_actual-reference).abs().max()),
                                        max_logprob_error=float((graph_actual.log_softmax(-1)-reference.log_softmax(-1)).abs().max()))
            wall=wall_measurement(graph.replay,device,max(20,repeats),20)
            timings[label+'_graph_wall_ms']=wall['median_ms'];details[label+'_graph_wall']=wall
    arrays=[inputs.numpy(),model[0].weight.detach().numpy().T,model[0].bias.detach().numpy(),
            model[2].weight.detach().numpy().T,model[2].bias.detach().numpy()]
    def numpy_compute():
        x,w1,b1,w2,b2=arrays
        return np.square(x@w1+b1)@w2+b2
    numpy_actual=torch.from_numpy(numpy_compute()).double()
    errors['numpy']=dict(max_logit_error=float((numpy_actual-reference).abs().max()),
                         max_logprob_error=float((numpy_actual.log_softmax(-1)-reference.log_softmax(-1)).abs().max()))
    wall=wall_measurement(numpy_compute,'cpu',max(20,repeats),20)
    timings['numpy_cpu_ms']=wall['median_ms'];details['numpy_cpu']=wall
    return dict(timings=timings,accuracy=errors,measurements=details,repeats=max(20,repeats),minimum_requested_warmups=20,
                minimum_gpu_warmup_ms=100,
                model=f'actual trained Linear({model[0].in_features},{model[0].out_features}), square, Linear({model[2].in_features},{model[2].out_features}) in PyTorch inference mode')
