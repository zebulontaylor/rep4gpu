"""Run disposable direct-COPE roles and verify their private outputs offline."""
from pathlib import Path
import argparse
import json
import os
import socket
import subprocess

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'work/direct_seeded_cope'
BIN=OUT/'direct_seeded_cope'
Q=(1<<127)-1

def run(name,length=128,sessions=3,fault='',fault_role=2,mismatch=False):
    folder=OUT/name
    folder.mkdir(parents=True,exist_ok=True)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    procs=[];logs=[]
    try:
        for role in (1,2):
            log=(folder/f'role{role}.log').open('w');logs.append(log)
            procs.append(subprocess.Popen([str(BIN),str(role),str(port),str(folder/f'role{role}.json'),
                                          str(sessions),str(length+1 if mismatch and role==2 else length),fault if role==fault_role or fault=='cope-high' else ''],
                                          stdout=log,stderr=subprocess.STDOUT))
        for proc in procs:proc.wait(timeout=45)
    finally:
        for proc in procs:
            if proc.poll() is None:proc.kill();proc.wait()
        for log in logs:log.close()
    codes=[p.returncode for p in procs]
    if fault or mismatch:
        assert codes==[1,1],(name,codes)
        return {'name':name,'codes':codes,'fault_detected':True}
    assert codes==[0,0],(name,codes)
    scalar=json.loads((folder/'role1.json').read_text())
    programmer=json.loads((folder/'role2.json').read_text())
    assert scalar['regular_noise_assumption'] is False
    assert all(s['value']==programmer['sessions'][0]['value'] for s in programmer['sessions'])
    assert len({s['delta'] for s in scalar['sessions']})==sessions
    assert len({tuple(s['share']) for s in scalar['sessions']})==sessions
    assert any(x>1<<64 for x in programmer['sessions'][0]['value'])
    assert any(s['delta']>1<<64 for s in scalar['sessions'])
    for party in (scalar,programmer):
        assert not party['network_debug_disclosure'] and not party['security_reviewed']
        counts=[s['entropy_substreams_consumed'] for s in party['sessions']]
        assert all(a<b for a,b in zip(counts,counts[1:]))
    for a,b in zip(scalar['sessions'],programmer['sessions']):
        assert len(a['share'])==len(b['value'])==len(b['share'])==length
        assert all(0<=x<Q for x in a['share']+b['share']+b['value'])
        assert all((k+a['delta']*v)%Q==m for k,v,m in zip(a['share'],b['value'],b['share']))
    return {'name':name,'codes':codes,'checked_correlations':length*sessions,
            'same_seed':True,'fresh_delta':True,'full_width':True,
            'context_setup_seconds':[scalar['context_setup_seconds'],programmer['context_setup_seconds']],
            'session_seconds':[[s['seconds'] for s in p['sessions']] for p in (scalar,programmer)],
            'session_sent_bytes':[[s['sent_bytes'] for s in p['sessions']] for p in (scalar,programmer)]}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fault',action='append',default=[])
    args=parser.parse_args()
    checks=[run('honest128'),run('honest1',length=1),run('honest257',length=257)]
    checks.append(run('mismatched_length',mismatch=True))
    for fault in args.fault:checks.append(run('fault_'+fault,fault=fault))
    env=dict(os.environ,EMP_TEST_MODE='1')
    blocked=subprocess.run([str(BIN),'1','1',str(OUT/'forbidden.json'),'1'],env=env,capture_output=True,text=True)
    assert blocked.returncode==2 and 'forbidden' in blocked.stderr
    checks.append({'name':'EMP_TEST_MODE','rejected_before_connection':True})
    (OUT/'results.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))
