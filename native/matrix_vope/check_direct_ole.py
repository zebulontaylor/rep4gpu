"""Run real native OLE peers; publish only parity status, counts, and timings."""
from pathlib import Path
import argparse
import json
import os
import secrets
import socket
import subprocess
import time

from direct_ole_io import PRIME,ScalarOleRequest,VectorOleRequest,LocalOleBatch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'work/direct_ole'
BIN=OUT/'direct_ole'


def run(name,scalars,vectors,*,fault='',mismatch=False,timeout=90):
    folder=OUT/name
    folder.mkdir(parents=True,exist_ok=True)
    length=len(vectors[0]);calls=len(scalars)
    ScalarOleRequest(length+int(mismatch),tuple(scalars)).write(folder/'role1.bin')
    VectorOleRequest(tuple(tuple(vector) for vector in vectors)).write(folder/'role2.bin')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    processes=[];logs=[]
    start=time.monotonic()
    try:
        for role in (1,2):
            log=(folder/f'role{role}.log').open('w');logs.append(log)
            processes.append(subprocess.Popen([str(BIN),str(role),str(port),str(folder/f'role{role}.bin'),str(folder/f'role{role}.json'),fault],stdout=log,stderr=subprocess.STDOUT))
        for process in processes:process.wait(timeout=timeout)
    finally:
        for process in processes:
            if process.poll() is None:process.kill();process.wait()
        for log in logs:log.close()
    wall=time.monotonic()-start
    codes=[process.returncode for process in processes]
    rejected=mismatch or bool(fault and fault!='cope-high-zero')
    assert codes==([1,1] if rejected else [0,0]),(name,codes)
    if rejected:return {'name':name,'codes':codes,'rejected':True},None
    roles=[LocalOleBatch.read(folder/f'role{role}.json',role=role,length=length,calls=calls) for role in (1,2)]
    outgoing,incoming=roles[1].outgoing_u(),roles[0].incoming_w()
    for call in range(calls):
        for row in range(length):
            assert (roles[0].shares[call][row]+roles[1].shares[call][row])%PRIME==scalars[call]*vectors[call][row]%PRIME
            # The reference provider consumes u-w = delta*x.
            assert (outgoing[call][row]-incoming[call][row])%PRIME==scalars[call]*vectors[call][row]%PRIME
    raw=[json.loads((folder/f'role{role}.json').read_text()) for role in (1,2)]
    assert all(len({tuple(row) for row in role.shares})==calls for role in roles)
    for role in raw:
        counters=[call['entropy_substreams'] for call in role['sessions']]
        assert all(counters[i]<counters[i+1] for i in range(calls-1))
    report={'name':name,'codes':codes,'length':length,'calls':calls,'checked_products':length*calls,
            'role_private_requests':True,'fresh_raw_base_per_call':True,'fresh_rot_choices_per_call':508,
            'process_wall_seconds':wall,'process_wall_products_per_second':length*calls/wall,
            'setup_inclusive_role_seconds':[role['context_setup_seconds']+sum(call['seconds'] for call in role['sessions']) for role in raw],
            'context_setup_seconds':[role['context_setup_seconds'] for role in raw],
            'session_seconds':[[call['seconds'] for call in role['sessions']] for role in raw],
            'total_sent_bytes':[role['context_sent_bytes']+sum(call['sent_bytes'] for call in role['sessions']) for role in raw],
            'session_sent_bytes':[[call['sent_bytes'] for call in role['sessions']] for role in raw]}
    report['setup_inclusive_products_per_second']=[length*calls/seconds for seconds in report['setup_inclusive_role_seconds']]
    return report,roles


def random_inputs(length,calls):
    vector=tuple(secrets.randbelow(PRIME) for _ in range(length))
    return tuple(secrets.randbelow(PRIME) for _ in range(calls)),tuple(vector for _ in range(calls))


def matrix_case():
    rows,inner,columns=7,3,2
    a=[[secrets.randbelow(PRIME) for _ in range(inner)] for _ in range(rows)]
    b=[[secrets.randbelow(PRIME) for _ in range(columns)] for _ in range(inner)]
    scalars=tuple(b[k][j] for k in range(inner) for j in range(columns))
    vectors=tuple(tuple(a[i][k] for i in range(rows)) for k in range(inner) for j in range(columns))
    report,roles=run('matrix_7x3x2',scalars,vectors)
    local=[]
    for role in roles:
        local.append([[sum(role.shares[k*columns+j][i] for k in range(inner))%PRIME for j in range(columns)] for i in range(rows)])
    assert all((local[0][i][j]+local[1][i][j])%PRIME==sum(a[i][k]*b[k][j] for k in range(inner))%PRIME for i in range(rows) for j in range(columns))
    report.update(matrix_shape=[rows,inner,columns],matrix_entries_checked=rows*columns,matrix_parity=True)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--throughput',action='store_true');args=parser.parse_args()
    vector=(0,1,PRIME-1,(1<<126)+17)
    checks=[run('zero_one_minusone',(0,1,PRIME-1),(vector,vector,vector))[0],matrix_case()]
    scalars,vectors=random_inputs(8,2)
    checks.append(run('mismatch',scalars,vectors,mismatch=True)[0])
    for fault in ('split-input','response-x','response-z','cope-high-one'):
        checks.append(run('fault_'+fault,scalars,vectors,fault=fault)[0])
    checks.append(run('raw_zero_accept',scalars,vectors,fault='cope-high-zero')[0])
    folder=OUT/'invalid_input';folder.mkdir(parents=True,exist_ok=True)
    ScalarOleRequest(1,(0,)).write(folder/'noncanonical.bin')
    malformed=(folder/'noncanonical.bin').read_bytes()[:-16]+PRIME.to_bytes(16,'little')
    (folder/'noncanonical.bin').write_bytes(malformed)
    bad=subprocess.run([str(BIN),'1','1',str(folder/'noncanonical.bin'),str(folder/'forbidden.json')],capture_output=True,text=True)
    assert bad.returncode!=0 and 'noncanonical' in bad.stderr
    env=dict(os.environ,EMP_TEST_MODE='1')
    blocked=subprocess.run([str(BIN),'1','1','absent','absent'],env=env,capture_output=True,text=True)
    assert blocked.returncode!=0 and 'deterministic test mode' in blocked.stderr
    checks.extend([{'name':'noncanonical_request','rejected_before_network':True},{'name':'EMP_TEST_MODE','rejected_before_network':True}])
    if args.throughput:
        for length in (128,1024,8192):
            scalars,vectors=random_inputs(length,10)
            checks.append(run(f'throughput_{length}',scalars,vectors)[0])
    report={'security_reviewed':False,'transport_authenticated':False,'full_preprocessing':False,
            'timing_context':'shared host functional measurements; no plaintext or inference speed claim','checks':checks}
    (OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':len(checks),'public_report':str(OUT/'results.json')}))


if __name__=='__main__':main()
