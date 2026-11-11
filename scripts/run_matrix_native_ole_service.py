"""Experimental four-role matrix-MAC integration with persistent native OLE/TLS.

Every role samples its own key and candidate shares, runs native peer endpoints,
and loads only its local endpoint outputs. No trusted fixture oracle or central
preprocessing key/share holder is used. The OLE backend remains experimental,
privacy-unreviewed and TEST-ONLY; successful algebra is not a security claim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import ssl
import subprocess
import sys
import time
import traceback

from benchmarks.socket_inference import certificates,reserve_addresses


def fingerprints(binary):
    paths=[Path(__file__),Path('rep4gpu/matrix_native_ole.py'),Path('rep4gpu/matrix_native_ole_service.py'),Path('rep4gpu/matrix_spdz_reference.py'),
           Path('rep4gpu/transport.py'),Path('rep4gpu/spdz_distributed.py'),
           *Path('native/matrix_vope').glob('direct_ole*'),*Path('native/matrix_vope').glob('tls_io*'),*Path('native/matrix_vope').glob('ole_service*'),Path('native/matrix_vope/build_ole_service.py'),
           Path('native/matrix_vope/wolverine_base.hpp'),Path('native/matrix_vope/seeded_vole.hpp'),
           Path('native/matrix_vope/fp127.h'),Path('native/matrix_vope/fp127ext4.h'),
           Path('native/matrix_vope/build_direct_ole_tls.py'),*Path(binary).parent.glob('*manifest*.json'),*Path(binary).parent.glob('*lock*.json')]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}|{'native_binary':hashlib.sha256(Path(binary).read_bytes()).hexdigest()}


def worker(config_path,result_path):
    from rep4gpu.matrix_spdz_reference import MatrixParty,Matrix,random_matrix
    from rep4gpu.matrix_native_ole import NativeOleConfig
    from rep4gpu.matrix_native_ole_service import ExperimentalNativeOleService
    from rep4gpu.transport import SocketTransport
    config=json.loads(Path(config_path).read_text());role=config['role'];party=None;provider=None;transport=None
    started=time.perf_counter()
    try:
        identity=config['identity'];server=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);server.verify_mode=ssl.CERT_REQUIRED
        server.load_cert_chain(identity['certificate'],identity['key']);server.load_verify_locations(config['ca'])
        client=ssl.create_default_context(ssl.Purpose.SERVER_AUTH,cafile=config['ca']);client.load_cert_chain(identity['certificate'],identity['key'])
        transport=SocketTransport(role,[tuple(x) for x in config['addresses']],timeout=config['timeout'],
            server_context=server,client_context=client,peer_hostnames={p:f'party{p}' for p in range(4)})
        # Key generation happens only inside this assigned role, never in the coordinator.
        key=random_matrix(config['width'],config['columns'])
        party=MatrixParty(role,transport,key,session=bytes.fromhex(config['epoch']),
            statistical_security_bits=config['statistical_security_bits'],statistical_check_budget=config['statistical_check_budget'])
        provider=ExperimentalNativeOleService(party,NativeOleConfig(Path(config['binary']),Path(config['private_directory']),tuple(tuple(p) for p in config['ports']),
            Path(config['ca']),Path(identity['certificate']),Path(identity['key']),config['timeout']))
        setup_ms=(time.perf_counter()-started)*1000
        stage=time.perf_counter();candidate=provider.candidate(bytes.fromhex(config['candidate_id']));candidate_ms=(time.perf_counter()-stage)*1000
        stage=time.perf_counter();sextuple=party.prepare_sextuple(candidate,provider);authenticate_ms=(time.perf_counter()-stage)*1000
        stage=time.perf_counter();masks=[provider.input_mask(owner,bytes.fromhex(identifier)) for owner,identifier in enumerate(config['input_mask_ids'])]
        input_preparation_ms=(time.perf_counter()-stage)*1000
        own=json.loads(Path(config['inputs']).read_text())
        expected_names={0:{'x'},1:{'y'},2:set(),3:set()}[role]
        if set(own)!=expected_names:raise ValueError('matrix plaintext ownership mismatch')
        stage=time.perf_counter()
        x=party.input(masks[0],Matrix.from_rows(own['x']) if role==0 else None)
        y=party.input(masks[1],Matrix.from_rows(own['y']) if role==1 else None)
        input_sharing_ms=(time.perf_counter()-stage)*1000
        before_bytes=transport.bytes_sent;before_messages=transport.messages_sent;stage=time.perf_counter()
        output=party.open(party.multiply(x,y,sextuple));online_ms=(time.perf_counter()-stage)*1000
        stage=time.perf_counter();provider.close();close_ms=(time.perf_counter()-stage)*1000
        stats=party.statistical_accounting()
        report=dict(role=role,passed=True,output=output.to_rows(),setup_ms=setup_ms,candidate_generation_ms=candidate_ms,
            authentication_sacrifice_transpose_ms=authenticate_ms,input_mask_preparation_ms=input_preparation_ms,
            private_input_sharing_ms=input_sharing_ms,checked_multiply_and_output_ms=online_ms,
            all_correlations_consumed=sextuple.used and all(mask.used for mask in masks),
            control_bytes_sent=transport.bytes_sent,control_messages_sent=transport.messages_sent,
            online_control_bytes=transport.bytes_sent-before_bytes,online_control_messages=transport.messages_sent-before_messages,
            statistical_accounting={k:(dict(numerator=v.numerator,denominator=v.denominator) if k=='bound' else v) for k,v in stats.items()},
            native_ole_batches=provider.metrics,transport_authenticated=transport.authenticated,
            native_service_setup_wall_ms=provider.service_setup_wall_ms,native_service_setup=provider.setup_metrics,
            native_service_close_ms=close_ms,native_service_close=provider.close_metrics,
            all_native_contexts_retired=len(provider.close_metrics)==6 and all(m['poisoned']is True for m in provider.close_metrics),
            evidence=dict(test_only=True,privacy_reviewed=False,security_target_met=False,central_trusted_fixture_used=False))
        Path(result_path).write_text(json.dumps(report,indent=2)+'\n')
    except BaseException as error:
        if provider is not None:provider.abort(error)
        if party is not None:party.abort(error)
        traceback.print_exc();Path(result_path).write_text(json.dumps(dict(role=role,passed=False,error=str(error)))+'\n');raise
    finally:
        if provider is not None and not provider._closed:provider.abort(RuntimeError('worker exited before native context retirement'))
        if provider is not None:provider._cleanup()
        if transport is not None:transport.close()


def _ports():
    sockets=[]
    try:
        for _ in range(12):
            connection=socket.socket();connection.bind(('127.0.0.1',0));sockets.append(connection)
        numbers=[connection.getsockname()[1] for connection in sockets]
        return [[sender,receiver,port] for (sender,receiver),port in zip(((a,b) for a in range(4) for b in range(4) if a!=b),numbers)]
    finally:
        for connection in sockets:connection.close()


def run(output,work,width=2,statistical_security_bits=40,budget=1<<32,binary='work/ole_service/ole_service',timeout=90):
    from rep4gpu.matrix_spdz_reference import PRIME
    run_started=time.perf_counter()
    if type(width)is not int or width<1:raise ValueError('positive matrix width required')
    columns=1
    while PRIME**columns<(1<<statistical_security_bits)*2*budget:columns+=1
    work=Path(work).resolve();work.mkdir(parents=True,exist_ok=False,mode=0o700);work.chmod(0o700)
    output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    binary=Path(binary).resolve();before=fingerprints(binary)
    # Public known test matrices; only assigned owners receive plaintext files.
    x=[[i*width+j+1 for j in range(width)] for i in range(width)]
    y=[[(i+3)*(j+1)+int(i==j) for j in range(width)] for i in range(width)]
    expected=[[sum(x[i][k]*y[k][j] for k in range(width))%PRIME for j in range(width)] for i in range(width)]
    certs=work/'credentials';certs.mkdir(mode=0o700);ca,identities=certificates(certs)
    ports=_ports();addresses=reserve_addresses();epoch=secrets.token_hex(32)
    candidate=secrets.token_hex(32);mask_ids=[secrets.token_hex(32) for _ in range(2)]
    children=[];handles=[];timed_out=False;static_setup_ms=(time.perf_counter()-run_started)*1000;started=time.perf_counter()
    try:
        for role in range(4):
            private=work/f'party{role}';private.mkdir(mode=0o700)
            inputs=private/'inputs.json';inputs.write_text(json.dumps({'x':x} if role==0 else {'y':y} if role==1 else {}));inputs.chmod(0o600)
            config=private/'config.json';config.write_text(json.dumps(dict(role=role,width=width,columns=columns,
                statistical_security_bits=statistical_security_bits,statistical_check_budget=budget,epoch=epoch,candidate_id=candidate,
                input_mask_ids=mask_ids,ports=ports,addresses=addresses,ca=ca,identity=identities[role],inputs=str(inputs),
                binary=str(binary),private_directory=str(private/'native'),timeout=timeout)));config.chmod(0o600)
            log=(private/'worker.log').open('w');handles.append(log)
            children.append(subprocess.Popen([sys.executable,'-m','scripts.run_matrix_native_ole_service','--worker',str(config),'--result',str(private/'result.json')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        deadline=time.monotonic()+timeout
        for child in children:child.wait(timeout=max(.001,deadline-time.monotonic()))
    except subprocess.TimeoutExpired:timed_out=True
    finally:
        import signal
        for child in children:
            if child.poll()is None or child.returncode!=0:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
        for child in children:child.wait()
        for handle in handles:handle.close()
    elapsed=(time.perf_counter()-started)*1000
    roles=[json.loads((work/f'party{p}/result.json').read_text()) if (work/f'party{p}/result.json').exists() else dict(role=p,passed=False,error='missing result') for p in range(4)]
    after=fingerprints(binary);codes=[child.returncode for child in children]
    exact=all(role.get('passed') and role.get('output')==expected and role.get('all_correlations_consumed') and role.get('all_native_contexts_retired') for role in roles)
    stage_names=('setup_ms','candidate_generation_ms','authentication_sacrifice_transpose_ms','input_mask_preparation_ms','private_input_sharing_ms','checked_multiply_and_output_ms','native_service_close_ms')
    stage_maxima={name:max((role.get(name,0) for role in roles),default=0) for name in stage_names}
    import torch
    from threadpoolctl import threadpool_limits
    from benchmarks.spdz_timing import wall_measurement
    import numpy as np
    torch.set_num_threads(1)
    tx,ty=torch.tensor(x,dtype=torch.float32),torch.tensor(y,dtype=torch.float32)
    ax,ay=np.array(x,dtype=np.float32),np.array(y,dtype=np.float32)
    def exact_python():return [[sum(x[i][k]*y[k][j] for k in range(width))%PRIME for j in range(width)] for i in range(width)]
    with threadpool_limits(limits=1,user_api='blas'):
        clear={name:wall_measurement(compute,'cpu',25,3) for name,compute in [('exact_python_field',exact_python),('fp32_torch',lambda:tx@ty),('fp32_numpy',lambda:ax@ay)]}
    if (tx@ty).tolist()!=expected or (ax@ay).tolist()!=expected:raise AssertionError('clear same-operation reference mismatch')
    after=fingerprints(binary)
    report=dict(passed=codes==[0]*4 and exact and before==after,return_codes=codes,timed_out=timed_out,exact_integer_match=exact,
        width=width,prime=PRIME,columns=columns,statistical_security_bits=statistical_security_bits,statistical_check_budget=budget,
        expected_output=expected,roles=roles,actual_worker_process_wall_ms=elapsed,coordinator_static_setup_ms=static_setup_ms,
        observed_static_plus_worker_ms=static_setup_ms+elapsed,stage_maximum_ms=stage_maxima,observed_stage_sum_ms=static_setup_ms+sum(stage_maxima.values()),
        clear_same_operation_measurements=clear,
        provisional_worker_wall_slowdowns={name:elapsed/value['median_ms'] for name,value in clear.items()},
        source_stable=before==after,source_sha256_before=before,source_sha256=after,
        experimental=True,test_only=True,privacy_reviewed=False,security_target_met=False,central_trusted_fixture_used=False,
        measurement='Four actual CPU/TLS role processes and genuine native two-party OLE endpoints; no central key/share holder or trusted oracle. Persistent contexts reused across phases with fresh raw correlations/ROT on every request. Functional experimental run, not qualified private preprocessing or performance target evidence.',
        provisioning='Public test matrices are known to the coordinator; plaintext files are assigned only to owners0/1. Keys, random candidate shares, masks and native endpoint outputs are generated/read only inside each role. Same-user host is not hostile-party OS isolation.')
    if not report['passed']:(work/'RETIRED').write_text('failed experimental native matrix service epoch\n')
    output.write_text(json.dumps(report,indent=2)+'\n');return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',type=Path);parser.add_argument('--result',type=Path)
    parser.add_argument('--output',type=Path,default=Path('results/matrix_native_ole_service_m2.json'));parser.add_argument('--work',type=Path,default=Path('work/matrix_native_ole_service/m2'))
    parser.add_argument('--width',type=int,default=2);parser.add_argument('--statistical-security-bits',type=int,default=40)
    parser.add_argument('--statistical-check-budget',type=int,default=1<<32);parser.add_argument('--binary',default='work/ole_service/ole_service');parser.add_argument('--timeout',type=float,default=90)
    args=parser.parse_args()
    if args.worker:worker(args.worker,args.result)
    else:
        report=run(args.output,args.work,args.width,args.statistical_security_bits,args.statistical_check_budget,args.binary,args.timeout)
        print(json.dumps({k:report[k] for k in ('passed','return_codes','exact_integer_match','source_stable','actual_worker_process_wall_ms')}));raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__':main()
