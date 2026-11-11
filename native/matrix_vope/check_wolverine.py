"""Check disposable two-role base-sVOLE fixtures without printing private values."""
from pathlib import Path
import json
import os
import socket
import subprocess

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'work/wolverine_base'
BIN=OUT/'wolverine_base'
Q=(1<<127)-1


def port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        return sock.getsockname()[1]


def peers(name,arguments):
    folder=OUT/name
    folder.mkdir(parents=True,exist_ok=True)
    number=port()
    processes=[]
    logs=[]
    try:
        for role in (1,2):
            log=(folder/f'role{role}.log').open('w')
            logs.append(log)
            command=[str(BIN),*arguments(role,number,folder)]
            processes.append(subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT))
        for process in processes:
            process.wait(timeout=60)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
        for log in logs:
            log.close()
    return folder,[process.returncode for process in processes]


def run(name,length=128,sessions=3,mode='random',fault='',mismatch=False):
    folder,codes=peers(name,lambda role,number,path:[str(role),str(number),str(path/f'role{role}.json'),str(sessions),str(length+(1 if mismatch and role==2 else 0)),mode,fault])
    reject=bool(mismatch or (fault and fault!='cope-high-zero'))
    assert codes==([1,1] if reject else [0,0]),(name,codes)
    if reject:
        assert 'check failed' in (folder/'role1.log').read_text() or mismatch
        return {'name':name,'codes':codes,'rejected':True}
    alice=json.loads((folder/'role1.json').read_text())
    bob=json.loads((folder/'role2.json').read_text())
    degree=2 if mode.startswith('paired-') else 1
    for result in (alice,bob):
        assert result['output_degree']==degree
        assert result['rho']==(128 if degree==1 else 120)
        assert result['security_reviewed'] is False and result['full_preprocessing'] is False
        assert len(result['sessions'])==sessions
        assert [item['index'] for item in result['sessions']]==list(range(sessions))
        assert all(item['fresh_rot_choices']==508 for item in result['sessions'])
        counters=[item['entropy_substreams'] for item in result['sessions']]
        assert all(counters[i]<counters[i+1] for i in range(sessions-1))
    programmed=mode.endswith('programmed')
    if programmed:
        assert all(item['value']==bob['sessions'][0]['value'] for item in bob['sessions'])
    elif sessions>1:
        assert len({tuple(item['value']) for item in bob['sessions']})==sessions
    for coordinate in range(degree):
        gamma='gamma' if coordinate==0 else 'gamma_second'
        share='share' if coordinate==0 else 'share_second'
        assert len({item[gamma] for item in alice['sessions']})==sessions
        assert any(item[gamma]>(1<<64) for item in alice['sessions'])
        for index in range(sessions):
            a,b=alice['sessions'][index],bob['sessions'][index]
            assert len(a[share])==len(b[share])==len(b['value'])==length
            assert 0<=a[gamma]<Q
            assert all(0<=value<Q for value in a[share]+b[share]+b['value'])
            for row in range(length):
                assert (a[share][row]+a[gamma]*b['value'][row])%Q==b[share][row]
    if degree==2:
        assert all(item['gamma']!=item['gamma_second'] for item in alice['sessions'])
    assert any(value>(1<<64) for value in bob['sessions'][0]['value'])
    return {'name':name,'codes':codes,'checked_correlations':length*sessions*degree,
            'output_degree':degree,'same_programmed_vector':programmed,
            'fresh_keys':True,'fresh_rot_choices_per_session':508,
            'context_setup_seconds':[result['context_setup_seconds'] for result in (alice,bob)],
            'session_seconds':[[item['seconds'] for item in result['sessions']] for result in (alice,bob)],
            'session_sent_bytes':[[item['sent_bytes'] for item in result['sessions']] for result in (alice,bob)],
            'session_phase_seconds':[[item['phase_seconds'] for item in result['sessions']] for result in (alice,bob)]}


def reflection(size):
    folder,codes=peers(f'reflection_{size}',lambda role,number,path:['reflection',str(role),str(number),str(size)])
    assert codes==[1,0],codes
    assert 'reflection rejected; entropy retired' in (folder/'role1.log').read_text()
    return {'name':f'reflection_{size}','rejected':True,'entropy_retired':True}


def main():
    checks=[run('random1',1,sessions=1),run('random128'),run('programmed128',mode='programmed'),
            run('paired_random128',mode='paired-random'),run('paired_programmed128',mode='paired-programmed'),
            run('paired_programmed257',257,mode='paired-programmed'),run('mismatch',mismatch=True)]
    for fault in ('split-input','response-x','response-z','cope-high-one'):
        checks.append(run('fault_'+fault,mode='paired-programmed',fault=fault))
    # Raw selective-failure is permitted inside the checked precompression
    # subprotocol. The retained scalar is drawn from its later fresh extractor.
    checks.append(run('raw_zero_accept',mode='paired-programmed',fault='cope-high-zero'))
    checks.extend([reflection(16),reflection(32)])
    env=dict(os.environ,EMP_TEST_MODE='1')
    rejected=subprocess.run([str(BIN),'1','1',str(OUT/'forbidden.json'),'1','1'],env=env,capture_output=True,text=True)
    assert rejected.returncode!=0 and 'deterministic test mode' in rejected.stderr
    assert not (OUT/'forbidden.json').exists()
    checks.append({'name':'EMP_TEST_MODE','rejected_before_connection':True})
    report={'security_reviewed':False,'performance_context':'functional tests on shared host; no speed claim','checks':checks}
    (OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':len(checks),'security_reviewed':False,'report':str(OUT/'results.json')}))


if __name__=='__main__':
    main()
