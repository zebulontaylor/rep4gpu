"""Build role-private p31 correlations inside large-prime MASCOT.

Only each assigned worker samples its input randomness and reads its output.
The coordinator handles public layout, process status, and build provenance.
"""
from __future__ import annotations
import hashlib,json,math,os,re,secrets,shutil,struct,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from .spdz import PRIME
from .spdz_distributed import AuthShare,MatrixTriple
from .spdz_inputs import InputMask
from .spdz_polynomial import SquareMask
from .preprocessing_mama import _layout,_sha,_port_base

MAC_KEYS=7
INTERNAL_PRIME=2**256-189
FORMAT='mascot256-prime31-role-v1'
MAX_ELEMENTS=16384
_TEST_FULL_FIELD_NONBIT=102611540537317685572622297990872707316008220491675340097760747618177963390522


def _program(request):
    """Return source and the public, ordered private-input allocation plan."""
    records,total,_=_layout(request,'00'*32)
    if total>MAX_ELEMENTS:raise ValueError('bridge export exceeds bounded prototype size')
    for item in request.get('matrices',[]):
        if item['k']>65536:raise ValueError('matrix accumulator exceeds78-bit contract')
    lines=['from Compiler.types import sint,regint,Matrix',
           'from Compiler.library import check_point,runtime_error_if,print_ln',
           'program.security=176','program.bit_length=78',f'P={PRIME}',
           'def invalid_bit_flags(flag):',
           '    zero=regint(1,size=flag.size)',
           '    for shift in (0,64,128,192):',
           '        zero=zero*((flag>>shift).to_regint(n_bits=64,sync=False)==0)',
           '    return (1-zero).sum()!=0',
           'def bit_product(bits):',
           '    while len(bits)>1:',
           '        bits=[bits[i]*bits[i+1] if i+1<len(bits) else bits[i] for i in range(0,len(bits),2)]',
           '    return bits[0]',
           'def reduce_p(x,k):',
           '    assert 1<=k<=78',
           '    bits=x.bit_decompose(k)',
           '    y=sum(b*(1<<(i%31)) for i,b in enumerate(bits))',
           '    for unused in range((k+30)//31):',
           '        y=y-(1-y.less_than(P,bit_length=34))*P',
           '    return y',
           'def output_to(x,owner):',
           '    x.reveal_to(owner)._v.to_regint(n_bits=0,sync=False).binary_output(owner)']
    header_length=len(lines)
    input_lines=[];validation=[]
    plan=[];serial=0
    def random(owner,n):
        nonlocal serial
        name=f'r{serial}';serial+=1
        plan.append(dict(owner=owner,count=n))
        input_lines.extend([f'{name}bits=[sint.get_input_from({owner},size={n}) for unused in range(31)]',
                            f'{name}checks=[(b*(1-b)).reveal() for b in {name}bits]'])
        validation.extend([f'for flag in {name}checks:',
                           "    runtime_error_if(invalid_bit_flags(flag),'non-bit preprocessing contribution')"])
        # Validated31-bit inputs have only one noncanonical encoding: p.
        lines.append(f'{name}=sum(b*(1<<i) for i,b in enumerate({name}bits))-P*bit_product({name}bits)')
        return name
    def uniform(n):
        values=[random(owner,n) for owner in range(4)]
        name=f'u{len(plan)}'
        lines.append(f'{name}=reduce_p('+ '+'.join(values)+',33)')
        return name
    alpha_parts=[random(owner,MAC_KEYS) for owner in range(4)]
    lines.append('alpha=reduce_p('+ '+'.join(alpha_parts)+',33)')
    outputs=[[alpha_parts[p]] for p in range(4)]
    values={};clear=[[] for unused in range(4)]
    for p in range(4):
        outputs[p]+=[f'sint({1 if p==0 else 0})']+[f'{alpha_parts[p]}[{j}]' for j in range(MAC_KEYS)]
    for record in records[1:]:
        kind,name,component=record['kind'],record['name'],record['component'];n=record['count']
        value=f'v{record["offset"]}'
        if component in ('a','b','share'):
            original=uniform(n);lines.append(f'{value}={original}')
        elif kind=='matrices':
            item=next(x for x in request['matrices'] if x['name']==name)
            m,k,cols=item['m'],item['k'],item['n']
            a=values[(kind,name,'a')];b=values[(kind,name,'b')]
            lines += [f'ma=Matrix({m},{k},sint)',f'mb=Matrix({k},{cols},sint)',
                      f'ma.assign_vector({a})',f'mb.assign_vector({b})',
                      f'{value}=reduce_p(ma.dot(mb).get_vector(),{62+(k-1).bit_length()})']
        else:
            a=values[(kind,name,'a')];lines.append(f'{value}=reduce_p({a}*{a},62)')
        values[(kind,name,component)]=value
        for coord in range(MAC_KEYS+1):
            field=value if coord==0 else f'{value}*alpha[{coord-1}]'
            masks=[random(p,n) for p in range(3)]
            canonical=masks
            for p,mask in enumerate(masks):outputs[p].append(mask)
            residual=f'z{record["offset"]}_{coord}'
            width=33 if coord==0 else 62
            lines.append(f'{residual}=reduce_p({field}+3*P-('+ '+'.join(canonical)+f'),{width})')
            outputs[3].append(residual)
        if kind=='inputs':clear[record['owner']].append(value)
    lines.append('check_point()')
    for p in range(4):
        for expression in outputs[p]+clear[p]:lines.append(f'output_to({expression},{p})')
    lines += ['check_point()',"print_ln('MASCOT p31 export checks complete')"]
    # Each scalar reduction performs one bit decomposition and at most three
    # signed comparisons. Each exported scalar uses at most9 reductions;
    # combining the four key contributions uses one per application key.
    masked_operations=4*(9*(total-1)+MAC_KEYS)
    if masked_operations>=2**32:raise ValueError('masked-operation budget exceeded')
    bounds=dict(max_intermediate_bits=78,compiler_statistical_security=176,
                masked_integer_operations_upper_bound=masked_operations,
                input_field_elements=sum(x['count'] for x in plan),
                input_bits=31*sum(x['count'] for x in plan))
    source=lines[:header_length]+input_lines+['check_point()']+validation+lines[header_length:]
    return '\n'.join(source)+'\n',plan,bounds


def _sample_inputs(role,party,plan,test_invalid_bit=None):
    """Run only in the assigned role subprocess; never return randomness."""
    path=Path(role)/'Player-Data'/f'Input-P{party}-0'
    injected=False
    with path.open('x') as stream:
        os.chmod(path,0o600)
        for allocation in plan:
            if allocation['owner']!=party:continue
            values=[secrets.randbelow(PRIME) for unused in range(allocation['count'])]
            for bit in range(31):
                words=[(x>>bit)&1 for x in values]
                if test_invalid_bit is not None and party==0 and not injected and bit==0:
                    words[1]=test_invalid_bit;injected=True
                stream.write(' '.join(map(str,words))+'\n')


def _verify_circuit(role,metadata):
    source,plan,bounds=_program(metadata['request'])
    if (hashlib.sha256(source.encode()).hexdigest()!=metadata.get('export_source_sha256')
            or plan!=metadata.get('input_plan') or bounds!=metadata.get('bounds')):
        raise ValueError('large-prime export source or bounds mismatch')
    artifacts=metadata.get('compiled_artifacts',{})
    if (artifacts.get('Programs/Source/mascot_p31_export.mpc')!=metadata['export_source_sha256']
            or not any(name.startswith('Programs/Bytecode/') for name in artifacts)
            or not any(name.startswith('Programs/Schedules/') for name in artifacts)):
        raise ValueError('missing compiled circuit fingerprints')
    for name,digest in artifacts.items():
        if not name.startswith('Programs/') or '..' in Path(name).parts or _sha(Path(role)/name)!=digest:
            raise ValueError('compiled circuit fingerprint mismatch')


def _worker(role,party,command):
    metadata=json.loads((Path(role)/'public-manifest.json').read_text())
    _verify_circuit(role,metadata)
    root=Path(metadata['native_root']).resolve()
    from scripts.build_mascot_secure import verify_policy
    verify_policy(root)
    if (_sha(root/'mascot256-preprocessing-policy.json')!=metadata['native_policy_sha256']
            or _sha(root/'mascot-party.x')!=metadata['native_binary_sha256']):
        raise ValueError('worker native build differs from public provenance')
    expected=[str(root/'mascot-party.x'),'-p',str(party),'-N','4','-P',str(INTERNAL_PRIME),
              '-lgp','256','-S','128','-e','-pn',str(metadata['native_port_base']),
              '-b',str(metadata['native_batch_size']),'mascot_p31_export']
    if command!=expected or Path(role).name!=f'party{party}':raise ValueError('worker native command mismatch')
    _sample_inputs(role,party,metadata['input_plan'],metadata.get('test_invalid_bit'))
    environment={k:v for k,v in os.environ.items() if not k.startswith('LD_')}
    environment.update(LD_LIBRARY_PATH=f'{root}/local/lib:{root}',OMP_NUM_THREADS='1')
    os.chdir(role)
    os.execve(command[0],command,environment)


def load_role(bundle_path,party,device='cpu',*,storage_dtype=torch.int64,online_mac_keys=MAC_KEYS):
    """Consume one complete role bundle, then import only its own outputs."""
    role=Path(bundle_path).resolve();metadata=json.loads((role/'public-manifest.json').read_text())
    if metadata.get('format')!=FORMAT or not metadata.get('ready') or metadata.get('return_codes')!=[0]*4:
        raise ValueError('large-prime export did not complete')
    if metadata.get('retired'):raise ValueError('retired preprocessing cannot be imported')
    if metadata.get('test_invalid_bit') is not None:raise ValueError('fault-injection bundles cannot be imported')
    required=dict(parties=4,prime=PRIME,application_mac_keys=MAC_KEYS,native_mac_keys=1,
                  internal_mascot_prime=INTERNAL_PRIME,amplification=4,
                  compiler_security=176,compiler_bit_length=78,runtime_security=128)
    if any(metadata.get(k)!=v for k,v in required.items()):raise ValueError('large-prime security parameters mismatch')
    _verify_circuit(role,metadata)
    policy_path=role/'native-policy.json'
    if metadata.get('native_policy_sha256')!=_sha(policy_path):raise ValueError('native role policy changed')
    policy=json.loads(policy_path.read_text())
    if policy.get('post_authentication_sacrifice_coin')is not True:
        raise ValueError('native sacrifice lacks a fresh post-authentication coin')
    if policy!=metadata.get('native_policy') or policy.get('binary_sha256')!=metadata.get('native_binary_sha256'):
        raise ValueError('native role policy mismatch')
    from scripts.build_mascot_secure import verify_policy
    root=Path(metadata['native_root'])
    verify_policy(root)
    if _sha(root/'mascot256-preprocessing-policy.json')!=metadata['native_policy_sha256']:
        raise ValueError('role policy differs from the reviewed build')
    if party not in range(4) or role.name!=f'party{party}':raise ValueError('incorrect role path')
    if online_mac_keys!=MAC_KEYS or storage_dtype not in (torch.int32,torch.int64):raise ValueError('unsupported online representation')
    fd=os.open(role/'consumed',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'w') as f:f.write(metadata['session']);f.flush();os.fsync(f.fileno())
    fd=os.open(role,os.O_DIRECTORY|os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    records,total,clear_counts=_layout(metadata['request'],metadata['session'])
    if records!=metadata['records'] or total!=metadata['share_count']:raise ValueError('export layout mismatch')
    raw=(role/'Player-Data'/f'Binary-Output-P{party}-0').read_bytes()
    expected=MAC_KEYS+total*(MAC_KEYS+1)+clear_counts[party]
    if len(raw)!=expected*8:raise ValueError('role output length mismatch')
    words=np.frombuffer(raw,dtype='<i8')
    if np.any(words<0) or np.any(words>=PRIME):raise ValueError('noncanonical p31 output')
    key=words[:MAC_KEYS].copy();offset=MAC_KEYS;data=[]
    for record in records:
        n=record['count'];data.append(words[offset:offset+n*(MAC_KEYS+1)].reshape(MAC_KEYS+1,n).T.copy());offset+=n*(MAC_KEYS+1)
    if data[0][0,0]!=(1 if party==0 else 0) or not np.array_equal(data[0][0,1:],key):raise ValueError('key calibration failed')
    clear=words[offset:];matrices={};squares={};inputs={}
    dtype=np.int32 if storage_dtype==torch.int32 else np.int64
    for record,local in zip(records[1:],data[1:]):
        shape=record['shape'];kind,name,component=record['kind'],record['name'],record['component']
        share=AuthShare(torch.from_numpy(local[:,0].astype(dtype,copy=True)).reshape(shape).to(device),torch.from_numpy(local[:,1:].T.astype(dtype,copy=True,order='C')).reshape(MAC_KEYS,*shape).to(device))
        target=matrices if kind=='matrices' else squares if kind=='squares' else inputs
        target.setdefault(name,{})[component]=share
    for kind,target in [('matrices',matrices),('squares',squares)]:
        for name,parts in list(target.items()):
            record=next(r for r in records if r['kind']==kind and r['name']==name)
            identifier=bytes.fromhex(record['identifier'])
            target[name]=MatrixTriple(parts['a'],parts['b'],parts['c'],identifier=identifier) if kind=='matrices' else SquareMask(parts['a'],parts['a_squared'],identifier)
    for record in records:
        if record['kind']!='inputs':continue
        n=record['count'];start=record['clear_offset']
        plain=torch.from_numpy(clear[start:start+n].copy()).reshape(record['shape']).to(device) if record['owner']==party else None
        inputs[record['name']]=InputMask(inputs[record['name']]['share'],plain,record['owner'],bytes.fromhex(record['identifier']))
    provenance={k:metadata[k] for k in ('format','internal_mascot_prime','native_mac_keys','application_mac_keys','amplification','compiler_security','compiler_bit_length','runtime_security','native_binary_sha256','native_policy_sha256','export_source_sha256','privacy_audit_status','bounds','compiled_primitive_counts','preprocessing_ms','native_preprocessing_ms')}
    session=hashlib.sha256(b'MASCOT256-p31-online7-v1\0'+bytes.fromhex(metadata['session'])).digest()
    return dict(alpha_share=torch.from_numpy(key).to(device),matrix_triples=matrices,square_masks=squares,input_masks=inputs,session=session,provenance=provenance)


def run_native(manifest_path,output_dir,*,native_root,timeout=900,batch_size=1000,_test_invalid_bit=None):
    """Create fresh role-private bundles using live large-prime MASCOT."""
    request=manifest_path if isinstance(manifest_path,dict) else json.loads(Path(manifest_path).read_text())
    if not math.isfinite(timeout) or timeout<=0 or type(batch_size)is not int or batch_size<1:raise ValueError('invalid native limits')
    if _test_invalid_bit not in (None,2,2**64,_TEST_FULL_FIELD_NONBIT):raise ValueError('unsupported test-only bit injection')
    native_root=Path(native_root).resolve();binary=native_root/'mascot-party.x'
    policy_path=native_root/'mascot256-preprocessing-policy.json'
    policy=json.loads(policy_path.read_text())
    if policy.get('post_authentication_sacrifice_coin')is not True:
        raise ValueError('native sacrifice lacks a fresh post-authentication coin')
    if policy.get('binary_sha256')!=_sha(binary):raise ValueError('native executable policy mismatch')
    # The builder supplies the reviewed source/dependency lock. The producer
    # refuses a missing policy instead of inferring security from a file name.
    from scripts.build_mascot_secure import verify_policy
    verify_policy(native_root)
    environment={k:v for k,v in os.environ.items() if not k.startswith('LD_')}
    environment.update(LD_LIBRARY_PATH=f'{native_root}/local/lib:{native_root}',OMP_NUM_THREADS='1')
    source,plan,bounds=_program(request);session=secrets.token_hex(32)
    records,total,clear_counts=_layout(request,session)
    output=Path(output_dir).resolve();output.mkdir(parents=True,exist_ok=False,mode=0o700)
    compiler=output/'compiler'
    for part in ('Source','Bytecode','Schedules'):(compiler/'Programs'/part).mkdir(parents=True,exist_ok=True)
    program='mascot_p31_export';source_path=compiler/'Programs/Source'/f'{program}.mpc';source_path.write_text(source)
    metadata=dict(format=FORMAT,ready=False,parties=4,prime=PRIME,application_mac_keys=MAC_KEYS,
                  native_mac_keys=1,internal_mascot_prime=INTERNAL_PRIME,amplification=4,
                  compiler_security=176,compiler_bit_length=78,runtime_security=128,session=session,request=request,
                  records=records,share_count=total,clear_counts=clear_counts,input_plan=plan,bounds=bounds,
                  native_binary_sha256=_sha(binary),native_policy_sha256=_sha(policy_path),native_policy=policy,native_root=str(native_root),
                  export_source_sha256=_sha(source_path),privacy_audit_status='pending_post_authentication_validation',trusted_dealer=False)
    if _test_invalid_bit is not None:metadata['test_invalid_bit']=_test_invalid_bit
    start=time.monotonic()
    with (compiler/'compile.log').open('w') as log:
        code=subprocess.run([sys.executable,'-B',str(native_root/'compile.py'),'-F','78',program],cwd=compiler,
                            env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdout=log,stderr=subprocess.STDOUT,timeout=timeout).returncode
    if code:
        metadata.update(return_codes=[],compile_return_code=code,passed=False)
        (output/'public-manifest.json').write_text(json.dumps(metadata,indent=2));return metadata
    compiled_counts={name:int(count) for count,name in re.findall(r'(\d+) integer (triples|bits|opens|simple multiplications)',(compiler/'compile.log').read_text())}
    if not {'triples','bits','opens'}<=compiled_counts.keys() or sum(compiled_counts.values())>=2**32:
        raise ValueError('compiled primitive budget is missing or exceeds2^32')
    metadata['compiled_primitive_counts']=compiled_counts
    metadata['compiled_artifacts']={str(path.relative_to(compiler)):_sha(path)
                                    for part in ('Source','Bytecode','Schedules')
                                    for path in (compiler/'Programs'/part).iterdir() if path.is_file()}
    metadata['compile_ms']=(time.monotonic()-start)*1000
    certs=output/'public-certificates';certs.mkdir();roles=[]
    def openssl(*args):subprocess.run(['openssl',*map(str,args)],check=True,capture_output=True,timeout=60)
    for p in range(4):
        role=output/f'party{p}';role.mkdir(mode=0o700);player=role/'Player-Data';player.mkdir(mode=0o700)
        (role/'Programs').symlink_to(compiler/'Programs',target_is_directory=True)
        openssl('req','-x509','-newkey','ec','-pkeyopt','ec_paramgen_curve:prime256v1','-nodes',
                '-keyout',player/f'P{p}.key','-out',certs/f'P{p}.pem','-days','1','-subj',f'/CN=P{p}')
        (player/f'P{p}.key').chmod(0o600);roles.append(role)
    base=_port_base()
    metadata.update(native_port_base=base,native_batch_size=batch_size)
    for role in roles:
        for cert in certs.glob('*.pem'):shutil.copy2(cert,role/'Player-Data'/cert.name)
        openssl('rehash',role/'Player-Data')
        (role/'public-manifest.json').write_text(json.dumps(metadata))
    jobs=[];logs=[];native_start=time.monotonic()
    worker_script=Path(__file__).resolve()
    try:
        for p,role in enumerate(roles):
            command=[str(binary),'-p',str(p),'-N','4','-P',str(INTERNAL_PRIME),'-lgp','256','-S','128','-e','-pn',str(base),'-b',str(batch_size),program]
            log=(role/'native.log').open('w');logs.append(log)
            jobs.append(subprocess.Popen([sys.executable,'-m','rep4gpu.preprocessing_mascot','--worker',str(p),
                                          '--role',str(role),'--command',json.dumps(command)],
                                         cwd=Path(__file__).resolve().parents[1],stdout=log,stderr=subprocess.STDOUT))
        deadline=time.monotonic()+timeout
        for job in jobs:job.wait(timeout=max(.001,deadline-time.monotonic()))
    except BaseException as error:metadata['error']=f'{type(error).__name__}: {error}'
    finally:
        for job in jobs:
            if job.poll()is None:job.kill()
        for job in jobs:job.wait()
        for log in logs:log.close()
    codes=[job.returncode for job in jobs];passed=codes==[0]*4 and _test_invalid_bit is None
    metadata.update(ready=passed,passed=passed,return_codes=codes,native_preprocessing_ms=(time.monotonic()-native_start)*1000,
                    preprocessing_ms=(time.monotonic()-start)*1000)
    encoded=json.dumps(metadata,indent=2)+'\n';(output/'public-manifest.json').write_text(encoded)
    for role in roles:
        (role/'public-manifest.json').write_text(encoded)
        (role/'native-policy.json').write_text(policy_path.read_text())
        for path in (role/'Player-Data').glob('Binary-Output-*'):path.chmod(0o600)
    return metadata


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=int);parser.add_argument('--role');parser.add_argument('--command');parser.add_argument('--environment')
    parser.add_argument('--native-root');parser.add_argument('--manifest');parser.add_argument('--output');parser.add_argument('--timeout',type=float,default=900)
    args=parser.parse_args()
    if args.worker is not None:raise SystemExit(_worker(args.role,args.worker,json.loads(args.command)))
    report=run_native(args.manifest,args.output,native_root=args.native_root,timeout=args.timeout)
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','input_plan','native_policy')},indent=2))
    raise SystemExit(0 if report['passed'] else 1)
