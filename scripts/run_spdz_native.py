"""Run the trained square toy using role-local native preprocessing.

The coordinator handles public manifests, fixture inputs, TLS identities, and
final outputs. Each worker consumes only its own native preprocessing directory.
The shared-host fixture does not isolate mutually malicious OS users; deployed
roles require separate security domains and privately provisioned credentials.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import ssl
import subprocess
import sys
import tempfile
import time
import traceback

import torch

from benchmarks.socket_inference import certificates, reserve_addresses, MeteredTransport
from rep4gpu.spdz import PRIME
from rep4gpu.spdz_inputs import private_input
from threadpoolctl import threadpool_info,threadpool_limits


NAMES = ('x', 'w1', 'b1', 'w2', 'b2')
OWNERS = (0, 1, 1, 1, 1)


def fixture_api(name):
    if name=='tiny':
        from examples import spdz_trained_toy as module
    elif name=='dense':
        from examples import spdz_trained_dense_toy as module
    elif name=='compact':
        from examples import spdz_compact_toy as module
    elif name=='micro':
        from examples import spdz_micro_toy as module
    elif name=='folded':
        from examples import spdz_folded_dense_toy as module
    else:
        raise ValueError('fixture must be tiny, dense, folded, compact or micro')
    return module


def preprocessing_provider(public, online_mac_keys):
    """Select the role-local loader from the public manifest's exact format."""
    kind = public.get('format')
    if kind == 'mama-prime31-role-v1':
        return dict(provider='mama31', loader_module='rep4gpu.preprocessing_mama',
                    native_mac_keys=10, application_mac_keys=online_mac_keys,
                    privacy_audit_status='failed', security_target_met=False,
                    privacy_audit_failure='Known-mask selective-failure event of order 1/p in small-field MAMA preprocessing.')
    if kind == 'mascot256-prime31-role-v1':
        if online_mac_keys != 7:
            raise ValueError('MASCOT-emulated p31 provider requires seven application MAC keys')
        return dict(provider='mascot256-emulated-p31', loader_module='rep4gpu.preprocessing_mascot',
                    native_mac_keys=1, application_mac_keys=7,
                    internal_protocol='MASCOT', internal_mascot_prime=2**256-189,
                    amplification=4, compiler_security=176, runtime_security=128,
                    privacy_audit_status='pending', security_target_met=False)
    raise ValueError(f'unsupported native preprocessing format: {kind!r}')


def load_private_role(bundle, party, device, storage_dtype, online_mac_keys):
    # Dispatch from public metadata; only this owning worker opens private data.
    from importlib import import_module
    public = json.loads((Path(bundle) / 'public-manifest.json').read_text())
    provider = preprocessing_provider(public, online_mac_keys)
    loader = import_module(provider['loader_module']).load_role
    material = loader(Path(bundle) / f'party{party}', party, device=device,
                      storage_dtype=storage_dtype, online_mac_keys=online_mac_keys)
    return material, provider


def source_fingerprints():
    paths=[Path(__file__),Path('examples/spdz_trained_toy.py'),Path('examples/toy_next_token.py'),Path('examples/spdz_trained_dense_toy.py'),Path('examples/spdz_folded_dense_toy.py'),Path('examples/spdz_compact_toy.py'),Path('examples/spdz_micro_toy.py'),
           Path('benchmarks/spdz_native_replay.py'),Path('benchmarks/spdz_streamed_replay.py'),Path('benchmarks/spdz_prepared_model.py'),
           Path('benchmarks/spdz_prepared_fixture.py'),Path('benchmarks/spdz_timing.py'),Path('benchmarks/spdz_cost.py'),Path('benchmarks/spdz_inference.py'),
           Path('benchmarks/inference.py'),Path('benchmarks/device_inference.py'),
           Path('benchmarks/socket_inference.py'),*sorted(Path('rep4gpu').glob('*.py'))]
    return {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths if path.is_file()}


def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def worker(config_path, output_path):
    config = json.loads(Path(config_path).read_text())
    p, device = config['party'], config['device']
    api=fixture_api(config.get('fixture','tiny'))
    torch.set_num_threads(1)
    transport = None
    runtime = None
    try:
        # Import and consume inside the owning worker. The coordinator never
        # deserializes any role's authenticated shares or MAC-key shares.
        started = time.perf_counter()
        material, provider = load_private_role(config['bundle'], p, device,
                             torch.int64 if config['backend']=='host' else torch.int32,
                             config.get('online_mac_keys',10))
        sync(device)
        load_ms = (time.perf_counter() - started) * 1000
        own = torch.load(config['inputs'], map_location=device, weights_only=True)
        if set(own) != {name for name, owner in zip(NAMES, OWNERS) if owner == p}:
            raise ValueError('worker plaintext input ownership mismatch')
        server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server.verify_mode = ssl.CERT_REQUIRED
        server.load_cert_chain(config['identity']['certificate'], config['identity']['key'])
        server.load_verify_locations(config['ca'])
        client = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=config['ca'])
        client.load_cert_chain(config['identity']['certificate'], config['identity']['key'])
        transport = MeteredTransport(p, [tuple(a) for a in config['addresses']],
                                     timeout=config['timeout'], server_context=server,
                                     client_context=client,
                                     peer_hostnames={q: f'party{q}' for q in range(4)})
        runtime_transport = transport
        backend = config['backend']
        if backend == 'host':
            from rep4gpu.spdz_distributed import DistributedParty as Runtime
        elif backend in ('device','device-compressed','device-deferred'):
            if backend=='device-deferred':
                from rep4gpu.spdz_deferred import DeviceDeferredCompressedParty as Runtime
            elif backend=='device-compressed':
                from rep4gpu.spdz_compression import DeviceCompressedParty as Runtime
            else:
                from rep4gpu.spdz_device import DeviceDistributedParty as Runtime
            from rep4gpu.device_transport import DeviceByteBridge
            runtime_transport = DeviceByteBridge(transport, device)
        else:
            raise ValueError('unsupported runtime backend')
        if config.get('replay_repeats',0):
            from benchmarks.spdz_native_replay import entropy_fixture_class
            Runtime=entropy_fixture_class(Runtime)
        runtime_options={}
        if backend=='device-deferred':
            runtime_options['output_check_threshold']=config.get('output_check_threshold',4096)
        runtime = Runtime(p, runtime_transport, material['alpha_share'],
                          session=material['session'],stack_macs=True,**runtime_options)
        sync(device)
        start = time.perf_counter()
        values=[]
        input_stages={}
        for name in NAMES:
            stage_started=time.perf_counter()
            values.append(private_input(runtime,material['input_masks'][name],own.get(name)))
            sync(device)
            input_stages[name]=(time.perf_counter()-stage_started)*1000
        # Inputs are now authenticated shares. The consumed masks and owner
        # plaintexts are not needed by inference or isolated replay.
        material['input_masks'].clear()
        del own
        sync(device)
        if torch.device(device).type=='cuda':
            torch.cuda.empty_cache()
        input_ms = (time.perf_counter() - start) * 1000
        before_bytes, before_rounds = transport.bytes_sent, transport.rounds
        prepared_model=None
        model_prepare_ms=0.
        if config.get('prepare_model',False):
            from benchmarks.spdz_prepared_model import prepare_model
            started=time.perf_counter()
            triples=(material['matrix_triples']['first'],material['square_masks']['hidden'],
                     material['matrix_triples']['second'])
            prepared_model=prepare_model(runtime,values,triples)
            sync(device)
            model_prepare_ms=(time.perf_counter()-started)*1000
            # Opaque packs own immutable A/F/model snapshots. Burned B/C
            # tensors are not needed by the query or isolated replay.
            material['matrix_triples'].clear()
            del triples
            if torch.device(device).type=='cuda':torch.cuda.empty_cache()
        recording=None
        if config.get('replay_repeats',0):
            from benchmarks.spdz_native_replay import begin_recording
            recording=begin_recording(runtime,backend)
        started = time.perf_counter()
        if prepared_model is None:
            result=api.infer_authenticated(runtime,values,material['matrix_triples']['first'],
                                          material['matrix_triples']['second'],material['square_masks']['hidden'])
        else:
            from benchmarks.spdz_prepared_model import infer_prepared
            result=infer_prepared(runtime,values,prepared_model,material['square_masks']['hidden'])
        opened = runtime.open(result)
        sync(device)
        inference_bytes=transport.bytes_sent-before_bytes
        inference_rounds=transport.rounds-before_rounds
        elapsed = (time.perf_counter() - started) * 1000
        opened_cpu=opened.detach().cpu()
        replay_metrics=None
        if recording is not None:
            from benchmarks.spdz_native_replay import end_recording,run_local_replay
            end_recording(runtime,recording)
            # Every role finishes its checked CUDA work before any role starts
            # isolated replay. This control barrier is outside both timers.
            complete=runtime.broadcast(b'completed',label=b'benchmark-only/live-complete')
            if any(value!=b'completed' for value in complete.values()):
                raise ValueError('benchmark completion barrier disagrees')
            replay_metrics=run_local_replay(runtime,material,values,recording,opened_cpu,
                                           repeats=config['replay_repeats'],
                                           gpu_lock_path=config['replay_lock'],
                                           stream_incoming=config.get('stream_incoming',False),
                                           profile_path=str(Path(output_path).with_suffix('.pstats')) if config.get('profile_replay') and p==0 else None,
                                           replay_warmups=config.get('replay_warmups',20),prepared_model=prepared_model)
            # Keep all CUDA contexts alive until every isolated role finishes.
            # Early worker teardown can otherwise synchronize the next role.
            complete=runtime.broadcast(b'completed',label=b'benchmark-only/replay-complete')
            if any(value!=b'completed' for value in complete.values()):
                raise ValueError('post-replay completion barrier disagrees')
        signed = torch.where(opened_cpu > PRIME // 2, opened_cpu - PRIME, opened_cpu)
        report = dict(party=p, passed=True, backend=backend,
                      authenticated_tls=transport.authenticated,
                      native_preprocessing=True, role_local_loader=True,
                      mac_keys=runtime.mac_keys,native_mac_keys=provider['native_mac_keys'],online_mac_keys=runtime.mac_keys,
                      preprocessing_provider=provider,privacy_audit_status=provider['privacy_audit_status'],
                      security_target_met=False,
                      preprocessing_load_ms=load_ms,
                      input_protocol_ms=input_ms,input_protocol_by_name_ms=input_stages,
                      new_request_input_sharing_ms=input_stages['x'],
                      model_sharing_ms=sum(value for name,value in input_stages.items() if name!='x'),
                      checked_inference_ms=elapsed,prepared_model=prepared_model is not None,
                      model_dependent_preparation_ms=model_prepare_ms,
                      inference_payload_bytes=inference_bytes,
                      inference_rounds=inference_rounds,
                      payload_bytes=transport.bytes_sent, rounds=transport.rounds,
                      output=signed.tolist(), provenance=material.get('provenance', {}),
                      isolated_replay=replay_metrics,
                      live_recording_included=recording is not None)
        Path(output_path).write_text(json.dumps(report, indent=2)+'\n')
    except BaseException as error:
        if runtime is not None:
            runtime.abort(error)
        traceback.print_exc()
        Path(output_path).write_text(json.dumps(dict(party=p, passed=False,
                                                     error=f'{type(error).__name__}: {error}'))+'\n')
        raise
    finally:
        if transport is not None:
            transport.close()


@threadpool_limits.wrap(limits=1,user_api='blas')
def run(bundle, output, *, device='cpu', backend='host', batch=4, timeout=120,
        checkpoint=None,steps=None,replay_repeats=0,stream_incoming=False,profile_replay=False,
        replay_warmups=20,fixture_name='tiny',prepare_model=False,fractional_bits=None,online_mac_keys=10,output_check_threshold=4096):
    """Consume four role bundles once and verify every final public output."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout must be finite and positive')
    if backend not in ('host', 'device','device-compressed','device-deferred'):
        raise ValueError('backend must be host or device')
    if backend != 'host' and torch.device(device).type != 'cuda':
        raise ValueError('device backend requires CUDA')
    if type(replay_repeats) is not int or replay_repeats<0 or type(replay_warmups) is not int or replay_warmups<1:
        raise ValueError('replay repeats must be nonnegative and discarded warmups positive')
    if online_mac_keys not in (7,10):raise ValueError('online_mac_keys must be 7 or 10')
    if type(output_check_threshold) is not int or output_check_threshold<0:
        raise ValueError('output_check_threshold must be nonnegative')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    source_before=source_fingerprints()
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    public = json.loads((bundle/'public-manifest.json').read_text())
    provider=preprocessing_provider(public,online_mac_keys)
    api=fixture_api(fixture_name)
    if checkpoint is not None:
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        architecture={}
        if fixture_name in ('dense','folded'):
            architecture=dict(hidden=state['0.weight'].shape[0],dimension=state['embedding_table'].shape[1])
        model=api.train(1,**architecture)
        model.load_state_dict(state)
    else:
        model=api.train() if steps is None else api.train(steps)
    fixture_options=dict(batch=batch)
    if fractional_bits is not None:fixture_options['fractional_bits']=fractional_bits
    fixture=api.quantized_fixture(model,**fixture_options)
    expected = fixture['expected'].tolist()
    output.parent.mkdir(parents=True, exist_ok=True)
    logs = output.parent / (output.stem+'_logs')
    logs.mkdir(exist_ok=True)
    for p in range(4):
        (logs/f'party{p}.json').unlink(missing_ok=True)
    processes, handles = [], []
    timed_out = False
    with tempfile.TemporaryDirectory(prefix='spdz-native-online-') as temporary:
        directory = Path(temporary)
        os.chmod(directory, 0o700)
        ca, identities = certificates(directory)
        addresses = reserve_addresses()
        try:
            for p in range(4):
                inputs = directory/f'inputs-P{p}.pt'
                torch.save({name: value for name, value, owner in zip(NAMES, fixture['encoded'], OWNERS)
                            if owner == p}, inputs)
                inputs.chmod(0o600)
                config = directory/f'party{p}.json'
                config.write_text(json.dumps(dict(party=p, bundle=str(bundle), device=device,
                                                   backend=backend,fixture=fixture_name,prepare_model=prepare_model,online_mac_keys=online_mac_keys,output_check_threshold=output_check_threshold,
                                                   inputs=str(inputs),ca=ca,
                                                   identity=identities[p], addresses=addresses,
                                                   timeout=timeout,replay_repeats=replay_repeats,
                                                   stream_incoming=stream_incoming,profile_replay=profile_replay,replay_warmups=replay_warmups,
                                                   replay_lock=str(directory/'replay.lock'))))
                config.chmod(0o600)
                handle = (logs/f'party{p}.log').open('w')
                handles.append(handle)
                processes.append(subprocess.Popen([sys.executable, '-m', 'scripts.run_spdz_native',
                                                    '--worker', str(config), '--result', str(logs/f'party{p}.json')],
                                                   stdout=handle, stderr=subprocess.STDOUT))
            deadline = time.monotonic()+timeout
            for process in processes:
                process.wait(timeout=max(.001, deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for handle in handles:
                handle.close()
    parties = []
    for p in range(4):
        path = logs/f'party{p}.json'
        parties.append(json.loads(path.read_text()) if path.exists()
                       else dict(party=p, passed=False, error='no worker result'))
    codes = [process.returncode for process in processes]
    exact = all(role.get('passed') and role.get('output') == expected for role in parties)
    passed = not timed_out and codes == [0]*4 and exact
    ideal=None
    clear=None
    if passed and replay_repeats:
        from benchmarks.spdz_native_replay import combine_replay_metrics,trained_baselines
        # All worker processes have exited; no other role can contend with
        # the clear GPU baseline. No private preprocessing is read here.
        with threadpool_limits(limits=1,user_api='blas'):
            baseline_model=api.baseline_model(model) if hasattr(api,'baseline_model') else model
            clear=trained_baselines(baseline_model,fixture,device,replay_repeats)
        ideal=combine_replay_metrics([role['isolated_replay'] for role in parties],clear['timings'])
    if ideal is not None:
        ideal.update(privacy_audit_status=provider['privacy_audit_status'],security_target_met=False,
                     preprocessing_provider=provider['provider'])
    source_after=source_fingerprints()
    source_stable=source_before==source_after
    passed=passed and source_stable
    report = dict(passed=passed, return_codes=codes, timed_out=timed_out,
                  native_preprocessing=True, trusted_dealer=False,
                  backend=backend,device=device,batch=batch,native_mac_keys=provider['native_mac_keys'],
                  application_mac_keys=online_mac_keys,preprocessing_provider=provider,
                  privacy_audit_status=provider['privacy_audit_status'],security_target_met=False,
                  online_mac_keys=online_mac_keys,online_mac_indices=list(range(online_mac_keys)),
                  output_check_threshold=output_check_threshold if backend=='device-deferred' else None,
                  native_preprocessing_ms=public.get('native_preprocessing_ms'),
                  preprocessing_total_ms=public.get('preprocessing_ms'),
                  preprocessing_included_in_online_estimate=False,
                  public_preprocessing_manifest=public,
                  exact_integer_match=exact, expected=expected, parties=parties,
                  numerical=api.numerical_report(model,fixture),fixture=fixture_name,prepared_model=prepare_model,
                  model_dependent_preparation_ms=max((role.get('model_dependent_preparation_ms',0) for role in parties),default=0),
                  isolated_replay=ideal,clear_baselines=clear,
                  torch_threads=torch.get_num_threads(),blas_threadpools=threadpool_info(),
                  checked_inference_ms=max((role.get('checked_inference_ms', 0) for role in parties), default=0),
                  global_inference_payload_bytes=sum(role.get('inference_payload_bytes', 0) for role in parties),
                  measurement='Four real processes over localhost mutual TLS; shared GPU and OS account. Live inference includes first-use compilation and optional transcript recording; it is not a steady-state or ideal-network estimate. Optional isolated_replay serializes each role after a completion barrier and reports separate perfect-network estimates.',
                  provisioning='Each worker consumes only its own native role bundle. No coordinator reads private preprocessing. Deployment needs separate security domains.',
                  input_protocol_by_name_ms={name:max((role.get('input_protocol_by_name_ms',{}).get(name,0)
                                                       for role in parties),default=0) for name in NAMES},
                  source_stable_during_run=source_stable,source_sha256_before=source_before,
                  source_sha256=source_after)
    from benchmarks.spdz_cost import pipeline_cost
    query_ms=ideal['ideal_network_phase_max_ms'] if ideal is not None else report['checked_inference_ms']
    report['pipeline_cost']=pipeline_cost(query_ms=query_ms,
        model_preparation_ms=report['model_dependent_preparation_ms'],
        input_sharing_ms=report['input_protocol_by_name_ms']['x'],
        cryptographic_preprocessing_ms=public.get('native_preprocessing_ms'),
        preprocessing_query_capacity=1,
        static_model_sharing_ms=max((role.get('model_sharing_ms',0) for role in parties),default=0),
        model_queries=1,baselines=clear['timings'] if clear is not None else None)
    report['pipeline_cost']['security_target_met']=False
    report['pipeline_cost']['timing_context']='Native generation and cold preparation/sharing stages plus isolated query estimate when requested; no correlation reuse or extrapolated batching throughput.'
    output.write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--output', type=Path, default=Path('results/spdz_native_toy.json'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--backend', choices=('host', 'device','device-compressed','device-deferred'), default='host')
    parser.add_argument('--batch',type=int,default=4)
    parser.add_argument('--fixture',choices=['tiny','dense','folded','compact','micro'],default='tiny')
    parser.add_argument('--prepare-model',action='store_true')
    parser.add_argument('--output-check-threshold',type=int,default=4096)
    parser.add_argument('--fractional-bits',type=int)
    parser.add_argument('--steps',type=int)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--online-mac-keys',type=int,choices=[7,10],default=10)
    parser.add_argument('--replay-repeats',type=int,default=0,help='isolated benchmark samples; recommend25 for final measurements')
    parser.add_argument('--stream-incoming',action='store_true')
    parser.add_argument('--replay-warmups',type=int,default=20)
    parser.add_argument('--profile-replay',action='store_true',help='profile party0 discarded replay warmup only')
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--result', type=Path)
    args = parser.parse_args()
    if args.worker:
        with threadpool_limits(limits=1,user_api='blas'):
            worker(args.worker, args.result)
    else:
        if args.bundle is None:
            parser.error('--bundle is required')
        report = run(args.bundle, args.output, device=args.device, backend=args.backend,
                     batch=args.batch, timeout=args.timeout, checkpoint=args.checkpoint,
                     replay_repeats=args.replay_repeats,stream_incoming=args.stream_incoming,profile_replay=args.profile_replay,replay_warmups=args.replay_warmups,
                     fixture_name=args.fixture,prepare_model=args.prepare_model,fractional_bits=args.fractional_bits,steps=args.steps,online_mac_keys=args.online_mac_keys,output_check_threshold=args.output_check_threshold)
        print(json.dumps({k: report[k] for k in ('passed', 'exact_integer_match', 'checked_inference_ms', 'privacy_audit_status', 'security_target_met')}))
        raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
