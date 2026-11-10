"""Fresh fixed-model native q256 preparation and four-role CPU/TLS toy query.

The checkpoint is a public test fixture. Native workers receive plaintext model
parameters only at the assigned model owner; online workers load only their own
claimed shares. Same-user processes are functional isolation, not security domains.
"""
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
from threadpoolctl import threadpool_limits
from benchmarks.socket_inference import certificates,reserve_addresses,MeteredTransport
from scripts.run_spdz_prime256 import integer_reference
from rep4gpu.spdz_prime256 import PRIME


def _fingerprints():
    paths=[Path(__file__),Path('scripts/preprocess_mascot_prepared.py'),
           Path('scripts/run_spdz_prime256.py'),Path('scripts/build_mascot_secure.py'),
           Path('rep4gpu/preprocessing_mascot_prepared.py'),Path('rep4gpu/spdz_prime256_prepared.py'),
           Path('rep4gpu/preprocessing_mascot_direct.py'),Path('rep4gpu/preprocessing_mama.py'),
           Path('rep4gpu/spdz_prime256.py'),Path('rep4gpu/spdz_distributed.py'),Path('rep4gpu/transport.py'),
           Path('benchmarks/socket_inference.py'),Path('examples/spdz_micro_toy.py')]
    return {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def worker(config_path,result_path):
    config=json.loads(Path(config_path).read_text());p=config['party'];transport=None;party=None
    try:
        from rep4gpu.preprocessing_mascot_prepared import load_role
        from rep4gpu.spdz_prime256 import Prime256Party
        from rep4gpu.spdz_prime256_prepared import import_prepared_rows,matmul_prepared_rows
        started=time.perf_counter();material=load_role(Path(config['bundle'])/f'party{p}',p)
        first=material['prepared_batches']['first'];second=material['prepared_batches']['second']
        server=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);server.verify_mode=ssl.CERT_REQUIRED
        server.load_cert_chain(config['identity']['certificate'],config['identity']['key']);server.load_verify_locations(config['ca'])
        client=ssl.create_default_context(ssl.Purpose.SERVER_AUTH,cafile=config['ca'])
        client.load_cert_chain(config['identity']['certificate'],config['identity']['key'])
        transport=MeteredTransport(p,[tuple(a) for a in config['addresses']],timeout=config['timeout'],
            server_context=server,client_context=client,peer_hostnames={q:f'party{q}' for q in range(4)})
        party=Prime256Party(p,transport,material['alpha_share'],session=material['session'])
        h1=import_prepared_rows(party,first,(1,),model_id=first.model_id)[0]
        h2=import_prepared_rows(party,second,(1,),model_id=second.model_id)[0]
        setup_ms=(time.perf_counter()-started)*1000
        own=json.loads(Path(config['query_input']).read_text()) if p==0 else None
        started=time.perf_counter();x=party.input(material['input_masks']['query'],own)
        input_ms=(time.perf_counter()-started)*1000
        before_bytes=transport.bytes_sent;before_rounds=transport.rounds;started=time.perf_counter()
        hidden=matmul_prepared_rows(party,x,h1,model_id=first.model_id)
        hidden=party.add(hidden,material['static_shares']['b1'])
        hidden=party.square(hidden,material['square_masks']['hidden'])
        output=matmul_prepared_rows(party,hidden,h2,model_id=second.model_id)
        output=party.add(output,material['static_shares']['b2'])
        opened=party.open(output)
        elapsed=(time.perf_counter()-started)*1000
        report=dict(party=p,passed=True,output=[v-PRIME if v>PRIME//2 else v for v in opened],
                    checked_query_ms=elapsed,input_sharing_ms=input_ms,role_setup_ms=setup_ms,
                    online_payload_bytes=transport.bytes_sent-before_bytes,online_rounds=transport.rounds-before_rounds,
                    authenticated_tls=transport.authenticated)
        Path(result_path).write_text(json.dumps(report,indent=2)+'\n')
    except BaseException as error:
        if party is not None:party.abort(error)
        traceback.print_exc();Path(result_path).write_text(json.dumps(dict(party=p,passed=False,error=str(error)))+'\n');raise
    finally:
        if transport is not None:transport.close()


def run(output,bundle,*,native_root='work/mascot256-fresh-build',checkpoint='results/spdz_micro_toy/model.pt',timeout=120):
    from examples import spdz_micro_toy as api
    from scripts.preprocess_mascot_prepared import run_native
    if not math.isfinite(timeout) or timeout<=0:raise ValueError('timeout must be positive and finite')
    torch.set_num_threads(1);before=_fingerprints()
    output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True);bundle=Path(bundle).resolve()
    model=api.train(1);model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
    fixture=api.quantized_fixture(model,fractional_bits=8,batch=1)
    values=[value.tolist() for value in fixture['encoded']];expected=integer_reference(values)
    if expected!=fixture['expected'].tolist():raise AssertionError('independent integer oracle disagrees')
    model_ids=[hashlib.sha256(Path(checkpoint).read_bytes()+layer.encode()).hexdigest() for layer in ('first','second')]
    request=dict(models=[dict(name='first',owner=1,k=8,n=2,rows=1,model_id=model_ids[0]),
                         dict(name='second',owner=1,k=2,n=8,rows=1,model_id=model_ids[1])],
                 static_inputs=[dict(name='b1',owner=1,shape=[1,2]),dict(name='b2',owner=1,shape=[1,8])],
                 squares=[dict(name='hidden',shape=[1,2])],inputs=[dict(name='query',owner=0,shape=[1,8])])
    logs=output.parent/(output.stem+'_logs');logs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='q256-fixed-public-fixture-') as temporary:
        directory=Path(temporary);directory.chmod(0o700)
        def flatten(value):return [int(v)%PRIME for row in value for v in row]
        owner=directory/'model-owner1.json';owner.write_text(json.dumps(dict(first=flatten(values[1]),second=flatten(values[3]),b1=flatten(values[2]),b2=flatten(values[4]))));owner.chmod(0o600)
        native=run_native(request,bundle,owner_input_paths={1:owner},native_root=native_root,timeout=timeout)
        if not native.get('ready'):
            report=dict(passed=False,native=native,security_target_met=False);output.write_text(json.dumps(report,indent=2)+'\n');return report
        query=directory/'query-owner0.json';query.write_text(json.dumps(flatten(values[0])));query.chmod(0o600)
        ca,identities=certificates(directory);addresses=reserve_addresses();processes=[];handles=[];timed_out=False
        try:
            for p in range(4):
                config=directory/f'online{p}.json';config.write_text(json.dumps(dict(party=p,bundle=str(bundle),query_input=str(query) if p==0 else None,
                   ca=ca,identity=identities[p],addresses=addresses,timeout=timeout)));config.chmod(0o600)
                result=logs/f'party{p}.json';result.unlink(missing_ok=True);handle=(logs/f'party{p}.log').open('w');handles.append(handle)
                processes.append(subprocess.Popen([sys.executable,'-m','scripts.run_spdz_prime256_prepared','--worker',str(config),'--result',str(result)],stdout=handle,stderr=subprocess.STDOUT))
            deadline=time.monotonic()+timeout
            for process in processes:process.wait(timeout=max(.001,deadline-time.monotonic()))
        except subprocess.TimeoutExpired:timed_out=True
        finally:
            for process in processes:
                if process.poll()is None:process.kill()
            for process in processes:process.wait()
            for handle in handles:handle.close()
    parties=[json.loads((logs/f'party{p}.json').read_text()) if (logs/f'party{p}.json').exists() else dict(party=p,passed=False) for p in range(4)]
    expected_flat=expected[0];codes=[process.returncode for process in processes]
    exact=all(p.get('passed') and p.get('output')==expected_flat for p in parties)
    online=max((p.get('checked_query_ms',0) for p in parties),default=0)
    inputs=max((p.get('input_sharing_ms',0) for p in parties),default=0)
    after=_fingerprints()
    report=dict(passed=codes==[0]*4 and exact and before==after,return_codes=codes,timed_out=timed_out,exact_integer_match=exact,
                expected=expected,parties=parties,native=native,checked_query_ms=online,new_input_sharing_ms=inputs,
                fresh_preprocessing_ms=native['native_preprocessing_ms'],fresh_preprocessing_query_capacity=1,
                inclusive_per_query_ms=native['native_preprocessing_ms']+inputs+online,
                inclusive_cost_complete=False,
                inclusive_cost_scope='Fresh native generation and owner provisioning plus input sharing and checked query; no native compilation, certificate provisioning, or online role loading/TLS setup.',
                max_role_setup_ms=max((p.get('role_setup_ms',0) for p in parties),default=0),
                compile_ms=native['compile_ms'],source_stable=before==after,source_sha256=after,
                trusted_dealer=False,device='cpu',security_target_met=False,privacy_audit_status='conditional_native_fixed_model_experiment',
                measurement='Staged shared-host contended functional run; fresh native fixed-W preparation plus four-process CPU bigint/TLS checks; no GPU, perfect-network, or latency-target claim.',
                provisioning='Public checkpoint fixture. Each native owner reads only its assigned private model file; each online worker loads only its own role bundle. Same-user processes are not separate security domains.')
    output.write_text(json.dumps(report,indent=2)+'\n');return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',type=Path);parser.add_argument('--result',type=Path)
    parser.add_argument('--output',type=Path,default=Path('results/spdz_prime256_prepared_micro.json'))
    parser.add_argument('--bundle',type=Path,default=Path('work/mascot_prepared/micro1'));parser.add_argument('--native-root',default='work/mascot256-fresh-build')
    parser.add_argument('--checkpoint',default='results/spdz_micro_toy/model.pt');parser.add_argument('--timeout',type=float,default=120)
    args=parser.parse_args()
    if args.worker:worker(args.worker,args.result)
    else:
        report=run(args.output,args.bundle,native_root=args.native_root,checkpoint=args.checkpoint,timeout=args.timeout)
        print(json.dumps({key:report.get(key) for key in ('passed','exact_integer_match','fresh_preprocessing_ms','checked_query_ms','inclusive_per_query_ms')}));raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
