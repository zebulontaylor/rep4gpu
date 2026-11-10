"""Run the trained micro model with native prime-field preprocessing over TLS.

This is a CPU bigint reference, not a GPU timing result. Each worker loads only
its own one-use preprocessing and owner-assigned plaintext fixture inputs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import ssl
import subprocess
import sys
import tempfile
import time
import traceback

import torch
from threadpoolctl import threadpool_limits
from benchmarks.socket_inference import certificates,reserve_addresses,MeteredTransport

NAMES=('x','w1','b1','w2','b2')
OWNERS=(0,1,1,1,1)
PRIME=2**256-189


def native_backend(name):
    if name=='mascot':
        from rep4gpu.preprocessing_mascot_direct import load_role,PRIME
    elif name=='lowgear':
        from rep4gpu.preprocessing_lowgear import load_role,PRIME
    elif name=='lowgear-goldilocks-s40':
        from rep4gpu.preprocessing_lowgear_goldilocks_loader import load_role,PRIME
    else:raise ValueError('unknown native preprocessing backend')
    return load_role,PRIME


def fingerprints():
    paths=[Path(__file__),*Path('rep4gpu').glob('*.py'),*Path('benchmarks').glob('spdz*.py'),Path('benchmarks/socket_inference.py'),Path('examples/spdz_micro_toy.py')]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def integer_reference(values):
    """Independent signed Python-integer oracle, with no runtime field helper."""
    x,w1,b1,w2,b2=values
    def product(a,b):
        return [[sum(a[i][k]*b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]
    hidden=product(x,w1)
    hidden=[[(v+b1[0][j])**2 for j,v in enumerate(row)] for row in hidden]
    output=product(hidden,w2)
    return [[v+b2[0][j] for j,v in enumerate(row)] for row in output]


def worker(config_path,result_path):
    config=json.loads(Path(config_path).read_text());p=config['party'];transport=None;party=None
    try:
        from rep4gpu.spdz_prime256 import Prime256Party,QShare
        load_role,prime=native_backend(config.get('preprocessing_backend','mascot'))
        started=time.perf_counter();material=load_role(Path(config['bundle'])/f'party{p}',p)
        load_ms=(time.perf_counter()-started)*1000
        own=json.loads(Path(config['inputs']).read_text())
        if set(own)!={name for name,owner in zip(NAMES,OWNERS) if owner==p}:raise ValueError('plaintext ownership mismatch')
        server=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);server.verify_mode=ssl.CERT_REQUIRED
        server.load_cert_chain(config['identity']['certificate'],config['identity']['key']);server.load_verify_locations(config['ca'])
        client=ssl.create_default_context(ssl.Purpose.SERVER_AUTH,cafile=config['ca'])
        client.load_cert_chain(config['identity']['certificate'],config['identity']['key'])
        transport=MeteredTransport(p,[tuple(a) for a in config['addresses']],timeout=config['timeout'],
            server_context=server,client_context=client,peer_hostnames={q:f'party{q}' for q in range(4)})
        party=Prime256Party(p,transport,material['alpha_share'],session=material['session'],field_prime=prime)
        values=[];input_times={}
        for name in NAMES:
            started=time.perf_counter();values.append(party.input(material['input_masks'][name],own.get(name)))
            input_times[name]=(time.perf_counter()-started)*1000
        before_bytes=transport.bytes_sent;before_rounds=transport.rounds;started=time.perf_counter()
        hidden=party.matmul(values[0],values[1],material['matrix_triples']['first'])
        bias=values[2]
        if bias.shape[0]==1 and hidden.shape[0]!=1:bias=QShare(bias.values*hidden.shape[0],bias.macs*hidden.shape[0],hidden.shape)
        hidden=party.add(hidden,bias);hidden=party.square(hidden,material['square_masks']['hidden'])
        result=party.matmul(hidden,values[3],material['matrix_triples']['second']);bias=values[4]
        if bias.shape[0]==1 and result.shape[0]!=1:bias=QShare(bias.values*result.shape[0],bias.macs*result.shape[0],result.shape)
        result=party.add(result,bias)
        opened=party.open(result);online_ms=(time.perf_counter()-started)*1000
        signed=[v-prime if v>prime//2 else v for v in opened]
        report=dict(party=p,passed=True,output=signed,preprocessing_load_ms=load_ms,input_protocol_by_name_ms=input_times,
                    checked_online_ms=online_ms,online_payload_bytes=transport.bytes_sent-before_bytes,
                    online_rounds=transport.rounds-before_rounds,authenticated_tls=transport.authenticated,
                    role_local_loader=True,provenance=material.get('provenance',{}))
        Path(result_path).write_text(json.dumps(report,indent=2)+'\n')
    except BaseException as error:
        if party is not None:party.abort(error)
        traceback.print_exc();Path(result_path).write_text(json.dumps(dict(party=p,passed=False,error=str(error)))+'\n');raise
    finally:
        if transport is not None:transport.close()


def run(bundle,output,checkpoint='results/spdz_micro_toy/model.pt',batch=1,timeout=120,preprocessing_backend='mascot'):
    from examples import spdz_micro_toy as api
    if type(batch) is not int or batch<1:raise ValueError('batch must be positive')
    _,prime=native_backend(preprocessing_backend)
    torch.set_num_threads(1);bundle=Path(bundle).resolve();output=Path(output).resolve()
    public=json.loads((bundle/'public-manifest.json').read_text());before=fingerprints()
    model=api.train(1);model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
    fixture=api.quantized_fixture(model,fractional_bits=8,batch=batch)
    values=[value.tolist() for value in fixture['encoded']];expected=integer_reference(values)
    if expected!=fixture['expected'].tolist():raise AssertionError('independent Python oracle disagrees with fixture')
    output.parent.mkdir(parents=True,exist_ok=True);logs=output.parent/(output.stem+'_logs');logs.mkdir(exist_ok=True)
    handles=[];processes=[];timed_out=False
    with tempfile.TemporaryDirectory(prefix='q256-owner-online-') as temporary:
        directory=Path(temporary);directory.chmod(0o700);ca,identities=certificates(directory);addresses=reserve_addresses()
        try:
            for p in range(4):
                own={name:[int(v)%prime for row in value for v in row] for name,value,owner in zip(NAMES,values,OWNERS) if owner==p}
                inputs=directory/f'inputs-P{p}.json';inputs.write_text(json.dumps(own));inputs.chmod(0o600)
                config=directory/f'party{p}.json';config.write_text(json.dumps(dict(party=p,bundle=str(bundle),inputs=str(inputs),
                   ca=ca,identity=identities[p],addresses=addresses,timeout=timeout,preprocessing_backend=preprocessing_backend)));config.chmod(0o600)
                result=logs/f'party{p}.json';result.unlink(missing_ok=True);handle=(logs/f'party{p}.log').open('w');handles.append(handle)
                processes.append(subprocess.Popen([sys.executable,'-m','scripts.run_spdz_prime256','--worker',str(config),'--result',str(result)],stdout=handle,stderr=subprocess.STDOUT))
            deadline=time.monotonic()+timeout
            for process in processes:process.wait(timeout=max(.001,deadline-time.monotonic()))
        except subprocess.TimeoutExpired:timed_out=True
        finally:
            for process in processes:
                if process.poll() is None:process.terminate()
            for process in processes:
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait()
            for handle in handles:handle.close()
    parties=[json.loads((logs/f'party{p}.json').read_text()) if (logs/f'party{p}.json').exists() else dict(party=p,passed=False) for p in range(4)]
    expected_flat=[v for row in expected for v in row];codes=[p.returncode for p in processes]
    exact=all(p.get('passed') and p.get('output')==expected_flat for p in parties);after=fingerprints()
    online=max((p.get('checked_online_ms',0) for p in parties),default=0)
    report=dict(passed=codes==[0]*4 and exact and before==after,return_codes=codes,timed_out=timed_out,
       exact_integer_match=exact,expected=expected,parties=parties,prime=prime,mac_keys=1,device='cpu',batch=batch,
       preprocessing_backend=preprocessing_backend,
       runtime='CPU bigint Prime256Party; four real TLS roles',trusted_dealer=False,privacy_audit_status='pending',security_target_met=False,
       checked_online_ms=online,public_preprocessing_manifest=public,numerical=api.numerical_report(model,fixture),
       source_stable_during_run=before==after,source_sha256_before=before,source_sha256=after,
       measurement='Actual shared-host four-process CPU/TLS latency including all checks and final opening; not a GPU or perfect-network estimate.')
    from benchmarks.spdz_native_replay import trained_baselines
    from benchmarks.spdz_cost import pipeline_cost
    clear=trained_baselines(model,fixture,'cpu',25)
    report['clear_baselines']=clear
    report['checkpoint_sha256']=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    report['provisioning']='Shared-host functional fixture; deployment requires private execution domains and authenticated confidential channels. Each worker reads only its own claimed native role bundle and owner-assigned plaintext inputs.'
    inputs={name:max((p.get('input_protocol_by_name_ms',{}).get(name,0) for p in parties),default=0) for name in NAMES}
    report['input_protocol_by_name_ms']=inputs
    report['pipeline_cost']=pipeline_cost(query_ms=online,model_preparation_ms=0,input_sharing_ms=inputs['x'],
        cryptographic_preprocessing_ms=public.get('native_preprocessing_ms'),preprocessing_query_capacity=1,
        static_model_sharing_ms=sum(inputs[name] for name in NAMES if name!='x'),model_queries=1,baselines=clear['timings'])
    report['pipeline_cost']['security_target_met']=False
    report['pipeline_cost']['timing_context']='Actual native fresh generation plus CPU/TLS sharing and checked query; compilation separate; one fresh correlation pack, no replay amortization.'
    after=fingerprints()
    report.update(source_stable_during_run=before==after,source_sha256=after)
    report['passed']=report['passed'] and before==after
    output.write_text(json.dumps(report,indent=2)+'\n');return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--bundle',type=Path)
    parser.add_argument('--checkpoint',default='results/spdz_micro_toy/model.pt');parser.add_argument('--batch',type=int,default=1)
    parser.add_argument('--timeout',type=float,default=120);parser.add_argument('--output',type=Path,default=Path('results/spdz_prime256_micro.json'))
    parser.add_argument('--preprocessing-backend',choices=('mascot','lowgear','lowgear-goldilocks-s40'),default='mascot')
    parser.add_argument('--worker',type=Path);parser.add_argument('--result',type=Path);args=parser.parse_args()
    if args.worker:worker(args.worker,args.result)
    else:
        if args.bundle is None:parser.error('--bundle is required')
        report=run(args.bundle,args.output,args.checkpoint,args.batch,args.timeout,args.preprocessing_backend)
        print(json.dumps({k:report[k] for k in ('passed','exact_integer_match','checked_online_ms','security_target_met')}));raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
