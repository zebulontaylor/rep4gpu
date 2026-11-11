"""Record one fresh checked stream and replay isolated ideal-channel CPU phases.

Private snapshots stay inside role workers. Fixed replay creates no new query
capacity and excludes local TLS encryption/decryption as well as network wait.
"""
import argparse
import hashlib
import json
from pathlib import Path
import ssl
import subprocess
import sys
import tempfile
import time
import traceback


def fingerprints():
    paths=[Path(__file__), Path('benchmarks/goldilocks_ideal_replay.py'), *Path('rep4gpu').glob('*.py'), Path('examples/spdz_micro_toy.py'),
           Path('scripts/run_spdz_prime256.py'), Path('benchmarks/spdz_autoregressive.py'),
           Path('benchmarks/spdz_timing.py'), Path('benchmarks/socket_inference.py')]
    return {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def request_for(steps):
    if type(steps) is not int or steps<1:raise ValueError('positive step count required')
    return dict(matrices=[dict(name='first',m=steps,k=8,n=2),dict(name='second',m=steps,k=2,n=8)],
                squares=[dict(name='hidden',shape=[steps,2])],
                inputs=[dict(name='seed',owner=0,shape=[1,8]),dict(name='w1',owner=1,shape=[8,2]),
                        dict(name='b1',owner=1,shape=[1,2]),dict(name='w2',owner=1,shape=[2,8]),
                        dict(name='b2',owner=1,shape=[1,8])])


def worker(config_path,result_path):
    c=json.loads(Path(config_path).read_text());p=c['party'];party=None;transport=None
    try:
        from rep4gpu.preprocessing_lowgear_goldilocks_loader import load_role,PRIME
        from rep4gpu.spdz_prime256 import Prime256Party,QShare
        from rep4gpu.spdz_goldilocks_stream import prepare_fixed_weights,matmul_rows,partition_square_rows,square_rows
        from benchmarks.socket_inference import MeteredTransport
        start=time.perf_counter();material=load_role(Path(c['bundle'])/f'party{p}',p)
        load_ms=(time.perf_counter()-start)*1000
        own=json.loads(Path(c['inputs']).read_text())
        if set(own)!=( {'seed'} if p==0 else {'w1','b1','w2','b2'} if p==1 else set()):
            raise ValueError('private plaintext ownership mismatch')
        start=time.perf_counter()
        server=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);server.verify_mode=ssl.CERT_REQUIRED
        server.load_cert_chain(c['identity']['certificate'],c['identity']['key']);server.load_verify_locations(c['ca'])
        client=ssl.create_default_context(ssl.Purpose.SERVER_AUTH,cafile=c['ca'])
        client.load_cert_chain(c['identity']['certificate'],c['identity']['key'])
        transport=MeteredTransport(p,[tuple(a) for a in c['addresses']],timeout=c['timeout'],server_context=server,
            client_context=client,peer_hostnames={q:f'party{q}' for q in range(4)})
        if c.get('defer_masked_checks',False):
            from rep4gpu.spdz_goldilocks_deferred import DeferredGoldilocksParty
            party=DeferredGoldilocksParty(p,transport,material['alpha_share'],session=material['session'],
                                         key_epoch_role_path=Path(c['bundle'])/f'party{p}')
        else:
            party=Prime256Party(p,transport,material['alpha_share'],session=material['session'],field_prime=PRIME)
        tls_ms=(time.perf_counter()-start)*1000
        start=time.perf_counter();model={}
        for name in ('w1','b1','w2','b2'):model[name]=party.input(material['input_masks'][name],own.get(name))
        model_input_ms=(time.perf_counter()-start)*1000
        start=time.perf_counter();counts=(1,)*c['steps']
        h1=prepare_fixed_weights(party,model['w1'],material['matrix_triples']['first'],counts,model_id=bytes.fromhex(c['model_ids'][0]))
        h2=prepare_fixed_weights(party,model['w2'],material['matrix_triples']['second'],counts,model_id=bytes.fromhex(c['model_ids'][1]))
        squares=partition_square_rows(party,material['square_masks']['hidden'],counts)
        prepare_ms=(time.perf_counter()-start)*1000
        start=time.perf_counter();x=party.input(material['input_masks']['seed'],own.get('seed'))
        seed_input_ms=(time.perf_counter()-start)*1000
        from benchmarks.goldilocks_ideal_replay import record_stream,run_local_replay,exclusive_local_replay
        sent=transport.bytes_sent;rounds=transport.rounds
        fixture=record_stream(party,x,model,h1,h2,squares)
        checked_ms=fixture.recorded_live_ms
        logits=[[v-PRIME if v>PRIME//2 else v for v in row] for row in fixture.outputs]
        tokens=list(fixture.tokens)
        recorded_bytes=transport.bytes_sent-sent;recorded_rounds=transport.rounds-rounds
        # No role starts isolated timing while another evaluates the live stream.
        completed=party.broadcast(b'checked-recording-complete',label=b'Goldilocks-replay-boundary-v1')
        if any(value!=b'checked-recording-complete' for value in completed.values()):
            raise ValueError('recorded stream completion boundary differs')
        with exclusive_local_replay(c['replay_lock']):
            isolated=run_local_replay(fixture,repeats=c['replay_repeats'],warmups=c['replay_warmups'])
        report=dict(party=p,passed=True,logits=logits,tokens=tokens,checked_sequence_ms=checked_ms,
                    step_samples_ms=[],isolated_replay=isolated,preprocessing_load_ms=load_ms,tls_setup_ms=tls_ms,
                    model_input_sharing_ms=model_input_ms,model_preparation_ms=prepare_ms,seed_input_sharing_ms=seed_input_ms,
                    online_payload_bytes=recorded_bytes,online_rounds=recorded_rounds,
                    rows_consumed_per_layer=c['steps'],all_row_handles_used=all(h.used for h in (*h1,*h2)),
                    all_square_handles_used=all(h.used for h in squares),
                    all_native_correlations_used=all(t.used for t in material['matrix_triples'].values()) and all(s.used for s in material['square_masks'].values()),
                    all_input_masks_used=all(mask.used for mask in material['input_masks'].values()),authenticated_tls=transport.authenticated)
        Path(result_path).write_text(json.dumps(report,indent=2)+'\n')
    except BaseException as error:
        if party is not None:party.abort(error)
        from rep4gpu.lowgear_key_cache import retire_epoch
        retire_epoch(Path(c['bundle'])/f'party{p}','failed dependent Goldilocks stream worker')
        traceback.print_exc();Path(result_path).write_text(json.dumps(dict(party=p,passed=False,error=str(error)))+'\n');raise
    finally:
        if transport is not None:transport.close()


def _run(output,bundle,*,steps=128,seed=0,checkpoint='results/spdz_micro_toy/model.pt',timeout=120,gpu_device=None,repeats=5,defer_masked_checks=True,replay_repeats=7,replay_warmups=2):
    import math
    import torch
    from examples import spdz_micro_toy as api
    from benchmarks.spdz_autoregressive import integer_stream,clear_baselines
    from benchmarks.socket_inference import certificates,reserve_addresses
    from rep4gpu.preprocessing_lowgear_goldilocks_loader import PRIME
    from rep4gpu.lowgear_key_cache import retire_epoch
    if not math.isfinite(timeout) or timeout<=0 or type(seed) is not int or seed not in range(8):raise ValueError('invalid timeout or seed')
    if defer_masked_checks is not True:raise ValueError('ideal replay requires the deferred runtime')
    if type(replay_repeats)is not int or replay_repeats<1 or type(replay_warmups)is not int or replay_warmups<0:raise ValueError('invalid replay sample counts')
    expected_request=request_for(steps)
    torch.set_num_threads(1);before=fingerprints();bundle=Path(bundle).resolve();output=Path(output).resolve()
    m=json.loads((bundle/'public-manifest.json').read_text())
    if m.get('request')!=expected_request or not m.get('ready'):raise ValueError('fresh native pool must exactly match dependent stream request')
    owner_fixture_started=time.perf_counter()
    checkpoint=Path(checkpoint)
    model=torch.nn.Sequential(torch.nn.Linear(8,2),api.Square(),torch.nn.Linear(2,8)).eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
    fixture=api.quantized_fixture(model,fractional_bits=8,batch=1)
    if fixture['proof']['output_abs_bound']>=PRIME//2:raise ValueError('Goldilocks signed field bound exceeded')
    parameters=[v.tolist() for v in fixture['encoded'][1:]]
    expected=integer_stream(parameters,seed,steps)
    model_ids=[hashlib.sha256(checkpoint.read_bytes()+layer).hexdigest() for layer in (b'first',b'second')]
    owner_fixture_setup_ms=(time.perf_counter()-owner_fixture_started)*1000
    logs=output.parent/(output.stem+'_logs');logs.mkdir(parents=True,exist_ok=True)
    processes=[];handles=[];timed_out=False
    wall_start=time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix='gold-stream-owner-inputs-') as temporary:
            private=Path(temporary);private.chmod(0o700)
            ca,identities=certificates(private);addresses=reserve_addresses()
            for p in range(4):
                own={'seed':[int(i==seed) for i in range(8)]} if p==0 else dict(zip(('w1','b1','w2','b2'),([int(x)%PRIME for row in value for x in row] for value in parameters))) if p==1 else {}
                inputs=private/f'inputs{p}.json';inputs.write_text(json.dumps(own));inputs.chmod(0o600)
                config=private/f'config{p}.json';config.write_text(json.dumps(dict(party=p,bundle=str(bundle),steps=steps,model_ids=model_ids,
                       inputs=str(inputs),ca=ca,identity=identities[p],addresses=addresses,timeout=timeout,defer_masked_checks=True,replay_repeats=replay_repeats,replay_warmups=replay_warmups,replay_lock=str(private/'role-replay.lock'))));config.chmod(0o600)
                result=logs/f'party{p}.json';result.unlink(missing_ok=True)
                handle=(logs/f'party{p}.log').open('w');handles.append(handle)
                processes.append(subprocess.Popen([sys.executable,'-m','scripts.run_spdz_goldilocks_replay','--worker',str(config),'--result',str(result)],stdout=handle,stderr=subprocess.STDOUT))
            deadline=time.monotonic()+timeout
            for process in processes:process.wait(timeout=max(.001,deadline-time.monotonic()))
    except subprocess.TimeoutExpired:
        timed_out=True;retire_epoch(bundle,'timed out dependent Goldilocks stream coordinator')
    except BaseException:
        retire_epoch(bundle,'failed dependent Goldilocks stream coordinator')
        raise
    finally:
        for process in processes:
            if process.poll() is None:process.kill()
        for process in processes:process.wait()
        for handle in handles:handle.close()
    process_inclusive_ms=(time.perf_counter()-wall_start)*1000
    parties=[json.loads((logs/f'party{p}.json').read_text()) if (logs/f'party{p}.json').exists() else dict(party=p,passed=False) for p in range(4)]
    codes=[p.returncode for p in processes]
    exact=all(p.get('passed') and p.get('logits')==expected['logits'] and p.get('tokens')==expected['tokens'][1:] and p.get('all_row_handles_used') and p.get('all_square_handles_used') and p.get('all_native_correlations_used') and p.get('all_input_masks_used') for p in parties)
    after=fingerprints();source_stable=before==after
    clear=clear_baselines(model,seed,steps,repeats=repeats,gpu_device=gpu_device)
    clear_match=all(value['tokens']==expected['tokens'] for value in clear['accuracy'].values())
    passed=codes==[0]*4 and exact and source_stable and clear_match
    if not passed:retire_epoch(bundle,'failed dependent Goldilocks stream verification')
    maxima={key:max((p.get(key,0) for p in parties),default=0) for key in ('preprocessing_load_ms','tls_setup_ms','model_input_sharing_ms','model_preparation_ms','seed_input_sharing_ms','checked_sequence_ms')}
    native_ms=m['preprocessing_ms'];staged_ms=native_ms+owner_fixture_setup_ms+sum(maxima.values())
    epoch=Path(m.get('key_epoch_root',bundle)).resolve()
    epoch_manifest=json.loads((epoch/'public-manifest.json').read_text())
    cold_pool_ms=epoch_manifest['preprocessing_ms']
    warm_consumer_ms=native_ms+owner_fixture_setup_ms+process_inclusive_ms
    conservative_cold_ms=warm_consumer_ms+(cold_pool_ms if epoch!=bundle else 0)
    report=dict(passed=passed,return_codes=codes,exact_integer_match=exact,source_stable_during_run=source_stable,
                public_preprocessing_manifest=m,deferred_masked_checks=bool(defer_masked_checks),steps_served=steps,seed=seed,expected_stream=expected,parties=parties,
                source_sha256_before=before,source_sha256=after,clear_baselines=clear,clear_trajectories_match_integer=clear_match,
                fresh_native_pool_ms=native_ms,stages_max_role_ms=maxima,checked_sequence_ms=maxima['checked_sequence_ms'],
                checked_per_dependent_token_ms=maxima['checked_sequence_ms']/steps,observed_staged_sequence_ms=staged_ms,
                staged_per_served_token_ms=staged_ms/steps,consumer_process_inclusive_ms=process_inclusive_ms,
                key_epoch_root=str(epoch),cold_epoch_full_pool_preprocessing_ms=cold_pool_ms,
                cold_epoch_setup_isolated=False,inclusive_cost_complete=False,
                warm_epoch_native_plus_process_inclusive_ms=warm_consumer_ms,
                conservative_cold_plus_stream_sequence_ms=conservative_cold_ms,
                conservative_cold_plus_stream_per_token_ms=conservative_cold_ms/steps,
                cold_setup_contract='The warm-epoch costs exclude initial HE-key setup. The separately reported cold-epoch full-pool cost is a conservative upper bound on that setup, because it also generated a separate cold pool. No independent-batch denominator is credited to this dependent stream.',
                owner_fixture_setup_ms=owner_fixture_setup_ms,
                native_plus_process_inclusive_ms=native_ms+owner_fixture_setup_ms+process_inclusive_ms,
                inclusive_per_served_token_ms=(native_ms+owner_fixture_setup_ms+process_inclusive_ms)/steps,
                source_fingerprint_contract='Every rep4gpu module plus runner, fixture and baseline sources; frozen throughout four-role consumption.',
                amortization='Exactly T dependent tokens consume every prepared row, square row and input mask; no independent-batch or replay denominator.',
                timing_contract='Native preprocessing_ms includes native compile/provisioning. Owner checkpoint/fixture setup is included once. Staged sum uses per-stage role maxima. Separate process-inclusive stopwatch includes online certificate generation, process startup/imports, loading, TLS, model/seed sharing, checked fixed-weight preparation and the actual dependent sequence.',
                next_token_contract='Checked public signed logits determine lowest-index argmax; next input is that public one-hot token.',
                trusted_dealer=False,security_target_met=False,statistical_security=40,computational_security=128,
                privacy_audit_status='conditional_native_Goldilocks_experiment',timed_out=timed_out)
    from benchmarks.goldilocks_ideal_replay import phase_estimate
    report['ideal_confidential_channel']=phase_estimate([p['isolated_replay'] for p in parties]) if passed else None
    report['recorded_live_timing_contract']='The fresh successful TLS execution was instrumented to copy exact byte fixtures. Its live wall time includes that recording overhead; it is diagnostic, not an uninstrumented latency sample.'
    report['replay_accounting']='Only steps_served distinct fresh rows count. Repeats and warmups reproduce one fixed private fixture and never increase preprocessing capacity. Fresh native costs retain this recording pool denominator.'
    report['benchmark_process_wall_ms']=report.pop('consumer_process_inclusive_ms')
    report['benchmark_process_wall_includes_replay_and_lock_wait']=True
    for key in ('warm_epoch_native_plus_process_inclusive_ms','conservative_cold_plus_stream_sequence_ms',
                'conservative_cold_plus_stream_per_token_ms','native_plus_process_inclusive_ms','inclusive_per_served_token_ms'):
        report.pop(key,None)
    report['timing_contract']='Isolated replay retains all local protocol arithmetic, serialization, hashes, MAC checks and fresh entropy draw cost. Exchange dispatch/wait and local TLS cryptography are omitted. Per-phase role maxima model the current barrier schedule on ideal confidential channels; no physical perfect-network or complete MPC latency claim.'
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')
    return report


def run(output,bundle,**kwargs):
    try:
        return _run(output,bundle,**kwargs)
    except BaseException:
        root=Path(bundle)
        if any((root/f'party{p}'/'consumed').exists() for p in range(4)):
            from rep4gpu.lowgear_key_cache import retire_epoch
            retire_epoch(root,'failed dependent Goldilocks consumer after consumption')
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',type=Path);p.add_argument('--result',type=Path)
    p.add_argument('--bundle',type=Path);p.add_argument('--output',type=Path,default=Path('results/goldilocks_ideal128.json'))
    p.add_argument('--steps',type=int,default=128);p.add_argument('--seed',type=int,default=0);p.add_argument('--timeout',type=float,default=120)
    p.add_argument('--replay-repeats',type=int,default=7);p.add_argument('--replay-warmups',type=int,default=2)
    p.add_argument('--checkpoint',default='results/spdz_micro_toy/model.pt');p.add_argument('--gpu-baseline-device');p.add_argument('--baseline-repeats',type=int,default=5)
    args=p.parse_args()
    if args.worker:worker(args.worker,args.result)
    else:
        report=run(args.output,args.bundle,steps=args.steps,seed=args.seed,checkpoint=args.checkpoint,timeout=args.timeout,gpu_device=args.gpu_baseline_device,repeats=args.baseline_repeats,defer_masked_checks=True,replay_repeats=args.replay_repeats,replay_warmups=args.replay_warmups)
        print(json.dumps({k:report[k] for k in ('passed','exact_integer_match','steps_served','checked_sequence_ms','ideal_confidential_channel')}))
        raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__':main()
