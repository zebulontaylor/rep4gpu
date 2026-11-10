"""Independent algebra, generated-program, and role-private format checks."""
import ast
import hashlib
import json
import random
import stat

import numpy as np
import pytest
import torch

from rep4gpu import preprocessing_mascot as bridge

P = 2**31-1


@pytest.fixture(autouse=True)
def _synthetic_native_verifier(monkeypatch):
    # These are loader-format fixtures, not claims about a native binary.
    def verify(root):
        policy = json.loads((root/'mascot256-preprocessing-policy.json').read_text())
        assert policy['format'] == 'synthetic-loader-test-only'
        return policy
    monkeypatch.setattr('scripts.build_mascot_secure.verify_policy',verify)


class _ClearSecret:
    """Exact integer stand-in for the generated reduction circuit only."""
    def __init__(self,value):
        self.value = int(value)
    @staticmethod
    def _value(other):
        return other.value if isinstance(other,_ClearSecret) else int(other)
    def __add__(self,other):return _ClearSecret(self.value+self._value(other))
    __radd__ = __add__
    def __sub__(self,other):return _ClearSecret(self.value-self._value(other))
    def __rsub__(self,other):return _ClearSecret(self._value(other)-self.value)
    def __mul__(self,other):return _ClearSecret(self.value*self._value(other))
    __rmul__ = __mul__
    def bit_decompose(self,k):
        assert 0 <= self.value < 2**k
        return [_ClearSecret((self.value>>i)&1) for i in range(k)]
    def less_than(self,value,*,bit_length):
        assert 0 <= self.value < 2**bit_length
        return _ClearSecret(self.value < value)


def _request():
    return dict(matrices=[dict(name='first',m=2,k=3,n=2)],
                squares=[dict(name='hidden',shape=[2,2])],
                inputs=[dict(name='x',owner=0,shape=[2,3]),
                        dict(name='w',owner=2,shape=[3,2])])


def _reduction():
    source,_,_ = bridge._program(_request())
    function = next(node for node in ast.parse(source).body
                    if isinstance(node,ast.FunctionDef) and node.name=='reduce_p')
    namespace = dict(P=P)
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<generated-reduction>','exec'),namespace)
    return namespace['reduce_p']


@pytest.mark.parametrize('bits',[31,33,62,78])
def test_generated_secret_reduction_matches_independent_integer_modulo(bits):
    reduce_p = _reduction()
    rng = random.Random(831+bits)
    candidates = [0,1,P-1,P,P+1,2*P-1,2*P,2**bits-1]
    candidates += [rng.randrange(2**bits) for _ in range(100)]
    for value in candidates:
        if value >= 2**bits:continue
        actual = reduce_p(_ClearSecret(value),bits).value
        assert actual == value % P
        assert 0 <= actual < P


def test_all_public_intermediate_bounds_fit_claimed_decomposition_widths():
    assert 4*(P-1) < 2**33
    assert (P-1)**2+3*P < 2**62
    assert 65536*(P-1)**2+3*P < 2**78
    assert 65537*(P-1)**2+3*P >= 2**78
    for field in (0,1,P-1):
        for masks in (0,3*(P-1)):
            residual = field+3*P-masks
            assert 0 <= residual < 2**33
            assert _reduction()(_ClearSecret(residual),33).value == (field-masks)%P


def test_generated_program_validates_bits_before_reconstructing_contributions():
    source,plan,bounds = bridge._program(_request())
    body = ast.parse(source).body
    assert 'program.security=176' in source
    assert bounds['max_intermediate_bits'] == 78
    assert bounds['compiler_statistical_security'] == 176
    assert bounds['input_bits'] == 31*sum(item['count'] for item in plan)
    for index,node in enumerate(body):
        if not isinstance(node,ast.Assign) or not isinstance(node.targets[0],ast.Name):continue
        name = node.targets[0].id
        if not (name.startswith('r') and name.endswith('bits')):continue
        stem = name[:-4]
        following = body[index+1:]
        check_index = next(i for i,x in enumerate(following) if isinstance(x,ast.Assign)
                           and isinstance(x.targets[0],ast.Name) and x.targets[0].id==stem+'checks')
        guard_index = next(i for i,x in enumerate(following) if isinstance(x,ast.For)
                           and isinstance(x.iter,ast.Name) and x.iter.id==stem+'checks')
        combine_index = next(i for i,x in enumerate(following) if isinstance(x,ast.Assign)
                             and isinstance(x.targets[0],ast.Name) and x.targets[0].id==stem)
        assert check_index < guard_index < combine_index
        check = ast.unparse(following[check_index])
        assert '(b * (1 - b)).reveal()' in check
        assert 'runtime_error_if' in ast.unparse(following[guard_index])
    reveals = [node for node in ast.walk(ast.parse(source)) if isinstance(node,ast.Call)
               and isinstance(node.func,ast.Attribute) and node.func.attr=='reveal']
    assert reveals and all(isinstance(node.func.value,ast.BinOp) for node in reveals)


def test_every_private_residual_is_secretly_reduced_before_output():
    source,_,_ = bridge._program(_request())
    tree = ast.parse(source)
    residuals = {node.targets[0].id:node.value for node in tree.body if isinstance(node,ast.Assign)
                 and isinstance(node.targets[0],ast.Name) and node.targets[0].id.startswith('z')}
    assert residuals
    for name,expression in residuals.items():
        assert isinstance(expression,ast.Call) and isinstance(expression.func,ast.Name)
        assert expression.func.id == 'reduce_p'
        coordinate = int(name.rsplit('_',1)[1])
        assert expression.args[1].value == (33 if coordinate == 0 else 62)
        assert '3 * P' in ast.unparse(expression.args[0])
    output = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='output_to')
    assert '.reveal_to(owner)' in ast.unparse(output)
    assert '.binary_output(owner)' in ast.unparse(output)


def test_compiler_rejects_oversized_public_requests_without_private_io(monkeypatch):
    monkeypatch.setattr(bridge.secrets,'randbelow',lambda *args:pytest.fail('compiler sampled private entropy'))
    bridge._program(_request())
    with pytest.raises(ValueError,match='bounded prototype'):
        bridge._program(dict(inputs=[dict(name='large',owner=0,shape=[1,bridge.MAX_ELEMENTS])]))
    monkeypatch.setattr(bridge,'MAX_ELEMENTS',2**31)
    with pytest.raises(ValueError,match='78-bit'):
        bridge._program(dict(matrices=[dict(name='too-wide',m=1,k=65537,n=1)]))


def _role_fixture(root,party=0):
    role = root/f'party{party}'
    (role/'Player-Data').mkdir(parents=True)
    request,session = _request(),'5a'*32
    records,total,clear_counts = bridge._layout(request,session)
    key = np.arange(7,dtype=np.int64)+1+party*7
    blocks = []
    for record in records:
        count = record['count']
        if record['kind']=='calibration':
            block = np.concatenate(([int(party==0)],key)).reshape(8,1)
        else:
            block = (np.arange(8*count,dtype=np.int64).reshape(8,count)+record['offset']*19+party*127)%P
            block[-1,-1] = P-1
        blocks.append(block)
    clear = (np.arange(clear_counts[party],dtype=np.int64)+71)%P
    words = np.concatenate([key,*[block.reshape(-1) for block in blocks],clear])
    (role/'Player-Data'/f'Binary-Output-P{party}-0').write_bytes(words.astype('<i8').tobytes())
    native = root/'public-native-build'
    native.mkdir(exist_ok=True)
    binary = b'synthetic native binary - not executable'
    (native/'mascot-party.x').write_bytes(binary)
    policy = dict(format='synthetic-loader-test-only',binary_sha256=hashlib.sha256(binary).hexdigest(),
                  post_authentication_sacrifice_coin=True)
    public_policy = json.dumps(policy,sort_keys=True).encode()
    (role/'native-policy.json').write_bytes(public_policy)
    (native/'mascot256-preprocessing-policy.json').write_bytes(public_policy)
    source,plan,bounds = bridge._program(request)
    public_artifacts = {'Programs/Source/mascot_p31_export.mpc':source.encode(),
                        'Programs/Schedules/mascot_p31_export.sch':b'synthetic schedule',
                        'Programs/Bytecode/mascot_p31_export-0.bc':b'synthetic bytecode'}
    for name,data in public_artifacts.items():
        path = role/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
    metadata = dict(format=bridge.FORMAT,ready=True,return_codes=[0]*4,request=request,session=session,
                    parties=4,prime=P,native_root=str(native),native_policy=policy,
                    native_policy_sha256=hashlib.sha256(public_policy).hexdigest(),
                    records=records,share_count=total,clear_counts=clear_counts,input_plan=plan,
                    compiled_artifacts={name:hashlib.sha256(data).hexdigest() for name,data in public_artifacts.items()},
                    internal_mascot_prime=bridge.INTERNAL_PRIME,native_mac_keys=1,application_mac_keys=7,
                    amplification=4,compiler_security=176,compiler_bit_length=78,runtime_security=128,
                    native_binary_sha256=policy['binary_sha256'],native_port_base=16321,native_batch_size=100,
                    compiled_primitive_counts=dict(triples=3,bits=4,opens=5),
                    preprocessing_ms=123.0,native_preprocessing_ms=100.0,
                    export_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    privacy_audit_status='synthetic-test-only',bounds=bounds)
    (role/'public-manifest.json').write_text(json.dumps(metadata))
    return role,metadata,key,blocks,clear


@pytest.mark.parametrize('party',range(4))
@pytest.mark.parametrize('storage_dtype',[torch.int32,torch.int64])
def test_role_loader_preserves_coordinate_order_owner_clear_and_one_use(tmp_path,party,storage_dtype):
    role,metadata,key,blocks,clear = _role_fixture(tmp_path,party)
    material = bridge.load_role(role,party,storage_dtype=storage_dtype)
    assert torch.equal(material['alpha_share'],torch.from_numpy(key))
    records = metadata['records']
    for index,record in enumerate(records[1:],start=1):
        if record['kind']=='matrices':share = getattr(material['matrix_triples'][record['name']],record['component'])
        elif record['kind']=='squares':share = getattr(material['square_masks'][record['name']],record['component'])
        else:share = material['input_masks'][record['name']].share
        assert share.values.dtype == share.macs.dtype == storage_dtype
        assert torch.equal(share.values.long().reshape(-1),torch.from_numpy(blocks[index][0]))
        assert torch.equal(share.macs.long().reshape(7,-1),torch.from_numpy(blocks[index][1:]))
    for record in records:
        if record['kind']!='inputs':continue
        mask = material['input_masks'][record['name']]
        if record['owner']==party:
            start,count = record['clear_offset'],record['count']
            assert torch.equal(mask.clear.reshape(-1),torch.from_numpy(clear[start:start+count]))
        else:assert mask.clear is None
    assert len(material['session']) == 32 and material['session'] != bytes.fromhex(metadata['session'])
    with pytest.raises(FileExistsError):bridge.load_role(role,party)


@pytest.mark.parametrize('bad',[-1,P,2**40])
def test_noncanonical_private_output_is_rejected_after_bundle_retirement(tmp_path,bad):
    role,_,_,_,_ = _role_fixture(tmp_path)
    path = role/'Player-Data/Binary-Output-P0-0'
    words = np.frombuffer(path.read_bytes(),dtype='<i8').copy()
    words[-1] = bad
    path.write_bytes(words.tobytes())
    with pytest.raises(ValueError,match='noncanonical'):bridge.load_role(role,0)
    assert (role/'consumed').exists()
    with pytest.raises(FileExistsError):bridge.load_role(role,0)


def test_worker_samples_only_its_assigned_private_input_allocations(tmp_path,monkeypatch):
    role = tmp_path/'party2'
    (role/'Player-Data').mkdir(parents=True)
    draws = []
    def random_word(bound):
        assert bound == P
        draws.append(len(draws))
        return [0,P-1,7][(len(draws)-1)%3]
    monkeypatch.setattr(bridge.secrets,'randbelow',random_word)
    plan = [dict(owner=0,count=100),dict(owner=2,count=2),dict(owner=3,count=100),dict(owner=2,count=1)]
    bridge._sample_inputs(role,2,plan)
    assert len(draws) == 3
    path = role/'Player-Data/Input-P2-0'
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    lines = path.read_text().splitlines()
    assert len(lines) == 62
    for start,count,expected in [(0,2,[0,P-1]),(31,1,[7])]:
        reconstructed = [sum(int(lines[start+bit].split()[item])<<bit for bit in range(31)) for item in range(count)]
        assert reconstructed == expected
    with pytest.raises(FileExistsError):bridge._sample_inputs(role,2,plan)


@pytest.mark.parametrize('field,bad',[('parties',3),('prime',P-2),('internal_mascot_prime',P),
                                     ('amplification',1),('compiler_security',40),('compiler_bit_length',256),('runtime_security',40),
                                     ('application_mac_keys',10),('native_mac_keys',10)])
def test_unreviewed_security_parameters_rejected_before_private_output_read(tmp_path,monkeypatch,field,bad):
    from pathlib import Path
    role,metadata,_,_,_ = _role_fixture(tmp_path)
    metadata[field] = bad
    (role/'public-manifest.json').write_text(json.dumps(metadata))
    original = Path.read_bytes
    def forbid_private(path):
        if path.name.startswith('Binary-Output'):
            pytest.fail('security-policy failure still imported private role output')
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',forbid_private)
    with pytest.raises(ValueError,match='security parameters'):bridge.load_role(role,0)
    assert not (role/'consumed').exists()


@pytest.mark.parametrize('change',['role-policy','root-policy','binary-binding'])
def test_public_build_policy_binding_rejected_before_private_import(tmp_path,change):
    from pathlib import Path
    role,metadata,_,_,_ = _role_fixture(tmp_path)
    if change == 'role-policy':
        (role/'native-policy.json').write_text('{}')
    elif change == 'root-policy':
        root = Path(metadata['native_root'])
        (root/'mascot256-preprocessing-policy.json').write_text(json.dumps(dict(format='synthetic-loader-test-only',binary_sha256='changed')))
    else:
        metadata['native_binary_sha256'] = 'different-binary'
        (role/'public-manifest.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='policy'):bridge.load_role(role,0)
    assert not (role/'consumed').exists()


@pytest.mark.parametrize('change',['truncated','extra-word','bad-calibration','layout'])
def test_corrupt_private_format_or_layout_burns_the_bundle(tmp_path,change):
    role,metadata,_,_,_ = _role_fixture(tmp_path)
    path = role/'Player-Data/Binary-Output-P0-0'
    raw = path.read_bytes()
    if change == 'truncated':path.write_bytes(raw[:-1])
    elif change == 'extra-word':path.write_bytes(raw+bytes(8))
    elif change == 'bad-calibration':
        words = np.frombuffer(raw,dtype='<i8').copy()
        words[7] = 0
        path.write_bytes(words.tobytes())
    else:
        metadata['records'][1]['offset'] += 1
        (role/'public-manifest.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError):bridge.load_role(role,0)
    assert (role/'consumed').exists()
    with pytest.raises(FileExistsError):bridge.load_role(role,0)


def test_validated_word_canonicalization_handles_allones_without_integer_carry():
    source,_,_ = bridge._program(_request())
    tree = ast.parse(source)
    product = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='bit_product')
    reconstruct = next(node for node in tree.body if isinstance(node,ast.Assign)
                       and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='r0')
    code = compile(ast.Module(body=[product,reconstruct],type_ignores=[]),'<generated-bit-word>','exec')
    rng = random.Random(922)
    for value in [0,1,P-1,P,*[rng.randrange(P+1) for _ in range(100)]]:
        namespace = dict(P=P,r0bits=[_ClearSecret((value>>i)&1) for i in range(31)])
        exec(code,namespace)
        assert namespace['r0'].value == value%P
        assert namespace['bit_product']([_ClearSecret(1) for _ in range(31)]).value == 1
        for position in range(31):
            bits = [_ClearSecret(int(index!=position)) for index in range(31)]
            assert namespace['bit_product'](bits).value == 0


def test_all_bit_validation_guards_precede_any_secret_word_composition():
    source,_,bounds = bridge._program(_request())
    body = ast.parse(source).body
    guard_indices = [i for i,node in enumerate(body) if isinstance(node,ast.For)
                     and isinstance(node.iter,ast.Name) and node.iter.id.endswith('checks')]
    compose_indices = [i for i,node in enumerate(body) if isinstance(node,ast.Assign)
                       and isinstance(node.targets[0],ast.Name) and node.targets[0].id.startswith('r')
                       and node.targets[0].id[1:].isdigit()]
    assert guard_indices and compose_indices and max(guard_indices)<min(compose_indices)
    assert bounds['masked_integer_operations_upper_bound'] < 2**32
    assert bounds['compiler_statistical_security']-32 >= 128


class _ClearFieldVector:
    def __init__(self,values):
        self.values = [int(value) for value in values]
        self.size = len(self.values)
    def __rshift__(self,bits):
        return _ClearFieldVector([value>>bits for value in self.values])
    def to_regint(self,*,n_bits,sync):
        assert n_bits == 64 and sync is False
        mask,sign = 2**64-1,2**63
        return np.array([((value&mask)^sign)-sign for value in self.values],dtype=np.int64)


def test_scalar_bit_guard_rejects_nonzero_second_lane_and_high_field_bits():
    source,_,_ = bridge._program(_request())
    assert 'program.bit_length=78' in source
    tree = ast.parse(source)
    helper = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='invalid_bit_flags')
    guard = next(node for node in tree.body if isinstance(node,ast.For)
                 and isinstance(node.iter,ast.Name) and node.iter.id.endswith('checks'))
    code = compile(ast.Module(body=[helper,*guard.body],type_ignores=[]),'<generated-bit-guard>','exec')
    def reject(condition,*args):
        if bool(condition):raise ValueError('invalid contribution')
    def run(values):
        exec(code,dict(flag=_ClearFieldVector(values),runtime_error_if=reject,
                       regint=lambda value,*,size:np.full(size,value,dtype=np.int64)))
    q = bridge.INTERNAL_PRIME
    b = 102611540537317685572622297990872707316008220491675340097760747618177963390522
    check = b*(1-b)%q
    assert b not in (0,1) and check == 2**64 and check%(2**64)==0
    run([0,0,0])
    rng = random.Random(256)
    invalid_flags = [2,check,q-1,*[1<<bit for bit in (0,63,64,127,128,191,192,255)]]
    invalid_flags += [rng.randrange(1,q) for _ in range(100)]
    for invalid in invalid_flags:
        for lane in range(3):
            flags = [0,0,0]
            flags[lane] = invalid
            with pytest.raises(ValueError,match='invalid'):
                run(flags)


def test_direct_mac_residual_reduction_matches_independent_integer_sharing():
    reduce_p = _reduction()
    rng = random.Random(4217)
    cases = [(alpha,x,masks) for alpha in (0,1,P-1) for x in (0,1,P-1)
             for masks in ((0,0,0),(P-1,P-1,P-1),(0,P-1,1))]
    cases += [(rng.randrange(P),rng.randrange(P),tuple(rng.randrange(P) for _ in range(3))) for _ in range(200)]
    for alpha,x,masks in cases:
        expression = alpha*x+3*P-sum(masks)
        assert 3 <= expression <= (P-1)**2+3*P < 2**62
        residual = reduce_p(_ClearSecret(expression),62).value
        assert 0 <= residual < P
        assert (sum(masks)+residual)%P == alpha*x%P
    assert 63+176+1 <= 256
    assert 34+176+1 <= 256


@pytest.mark.parametrize('change',['export-source','input-plan','bounds','source-file','artifact-hash',
                                   'source-rebound','missing-bytecode','missing-schedule','path-escape'])
def test_circuit_or_compile_artifact_tampering_is_rejected_before_private_import(tmp_path,monkeypatch,change):
    from pathlib import Path
    role,metadata,_,_,_ = _role_fixture(tmp_path)
    source_name = 'Programs/Source/mascot_p31_export.mpc'
    if change == 'export-source':metadata['export_source_sha256'] = '0'*64
    elif change == 'input-plan':metadata['input_plan'][0]['count'] += 1
    elif change == 'bounds':metadata['bounds']['max_intermediate_bits'] = 79
    elif change in ('source-file','source-rebound'):
        path = role/source_name
        path.write_bytes(path.read_bytes()+b'\n# changed circuit\n')
        if change == 'source-rebound':metadata['compiled_artifacts'][source_name] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif change == 'artifact-hash':metadata['compiled_artifacts'][source_name] = '0'*64
    elif change == 'missing-bytecode':
        metadata['compiled_artifacts'] = {name:digest for name,digest in metadata['compiled_artifacts'].items()
                                          if not name.startswith('Programs/Bytecode/')}
    elif change == 'missing-schedule':
        metadata['compiled_artifacts'] = {name:digest for name,digest in metadata['compiled_artifacts'].items()
                                          if not name.startswith('Programs/Schedules/')}
    else:metadata['compiled_artifacts']['Programs/../private'] = '0'*64
    (role/'public-manifest.json').write_text(json.dumps(metadata))
    original = Path.read_bytes
    def forbid_private(path):
        if path.name.startswith('Binary-Output'):
            pytest.fail('invalid circuit imported private role output')
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',forbid_private)
    with pytest.raises(ValueError):bridge.load_role(role,0)
    assert not (role/'consumed').exists()


def test_coordinator_neither_samples_private_words_nor_reads_role_output_contents():
    from pathlib import Path
    tree = ast.parse(Path(bridge.__file__).read_text())
    coordinator = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='run_native')
    calls = [node for node in ast.walk(coordinator) if isinstance(node,ast.Call)]
    names = {ast.unparse(node.func) for node in calls}
    assert '_sample_inputs' not in names
    assert 'secrets.randbelow' not in names
    assert 'secrets.randbits' not in names
    assert not any(isinstance(node.func,ast.Attribute) and node.func.attr=='read_bytes' for node in calls)
    reads = [ast.unparse(node.func.value) for node in calls if isinstance(node.func,ast.Attribute)
             and node.func.attr=='read_text']
    assert reads and all('Binary-Output' not in expression and 'Input-P' not in expression for expression in reads)
    private_calls = [node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef)
                     and any(isinstance(call,ast.Call) and isinstance(call.func,ast.Name)
                             and call.func.id=='_sample_inputs' for call in ast.walk(node))]
    assert [node.name for node in private_calls] == ['_worker']


def test_loader_cannot_import_another_assigned_role_path(tmp_path):
    role,_,_,_,_ = _role_fixture(tmp_path,0)
    with pytest.raises(ValueError,match='role path'):
        bridge.load_role(role,1)
    assert not (role/'consumed').exists()


def _worker_command(metadata,party):
    from pathlib import Path
    return [str(Path(metadata['native_root'])/'mascot-party.x'),'-p',str(party),'-N','4',
            '-P',str(bridge.INTERNAL_PRIME),'-lgp','256','-S','128','-e',
            '-pn',str(metadata['native_port_base']),'-b',str(metadata['native_batch_size']),'mascot_p31_export']


@pytest.mark.parametrize('change',['executable','party','parties','prime','field-bits','security','tls',
                                   'port','batch','program','duplicate-override','omit-tls','role-path'])
def test_worker_rejects_every_unpinned_command_component_before_entropy(tmp_path,monkeypatch,change):
    role,metadata,_,_,_ = _role_fixture(tmp_path)
    party = 1 if change == 'role-path' else 0
    command = _worker_command(metadata,party)
    replacements = {'executable':(0,'/bin/true'),'party':(2,'3'),'parties':(4,'3'),
                    'prime':(6,str(P)),'field-bits':(8,'31'),'security':(10,'40'),
                    'tls':(11,'-unsupported'),'port':(13,'16322'),'batch':(15,'101'),
                    'program':(16,'other-circuit')}
    if change in replacements:
        index,value = replacements[change]
        command[index] = value
    elif change == 'duplicate-override':command += ['-S','0']
    elif change == 'omit-tls':command.remove('-e')
    monkeypatch.setattr(bridge,'_sample_inputs',lambda *args:pytest.fail('unvalidated worker command sampled entropy'))
    monkeypatch.setattr(bridge.os,'execve',lambda *args:pytest.fail('unvalidated worker command executed'))
    with pytest.raises(ValueError,match='command mismatch'):
        bridge._worker(role,party,command)


@pytest.mark.parametrize('change',['policy-file','binary-file','policy-digest','binary-digest','circuit'])
def test_worker_rejects_changed_build_or_circuit_before_entropy(tmp_path,monkeypatch,change):
    from pathlib import Path
    role,metadata,_,_,_ = _role_fixture(tmp_path)
    root = Path(metadata['native_root'])
    if change == 'policy-file':
        path = root/'mascot256-preprocessing-policy.json'
        policy = json.loads(path.read_text())
        policy['public-marker'] = 'changed'
        path.write_text(json.dumps(policy))
    elif change == 'binary-file':(root/'mascot-party.x').write_bytes(b'changed native executable')
    elif change == 'policy-digest':metadata['native_policy_sha256'] = '0'*64
    elif change == 'binary-digest':metadata['native_binary_sha256'] = '0'*64
    else:metadata['export_source_sha256'] = '0'*64
    (role/'public-manifest.json').write_text(json.dumps(metadata))
    monkeypatch.setattr(bridge,'_sample_inputs',lambda *args:pytest.fail('unvalidated worker sampled entropy'))
    monkeypatch.setattr(bridge.os,'execve',lambda *args:pytest.fail('unvalidated worker executed'))
    with pytest.raises(ValueError):bridge._worker(role,0,_worker_command(metadata,0))


def test_valid_worker_checks_then_samples_only_assigned_role_and_pins_loader_environment(tmp_path,monkeypatch):
    role,metadata,_,_,_ = _role_fixture(tmp_path,2)
    command = _worker_command(metadata,2)
    events = []
    def sample(actual_role,party,plan,fault):
        assert actual_role == role and party == 2 and plan == metadata['input_plan'] and fault is None
        events.append('sample')
    def chdir(actual):
        assert actual == role
        events.append('chdir')
    class Executed(Exception):pass
    def execute(binary,argv,environment):
        assert binary == command[0] and argv == command
        assert not any(name.startswith('LD_') for name in environment if name!='LD_LIBRARY_PATH')
        assert environment['LD_LIBRARY_PATH'] == f"{metadata['native_root']}/local/lib:{metadata['native_root']}"
        assert environment['OMP_NUM_THREADS'] == '1'
        events.append('exec')
        raise Executed()
    monkeypatch.setenv('LD_PRELOAD','untrusted-library.so')
    monkeypatch.setenv('LD_AUDIT','untrusted-audit.so')
    monkeypatch.setattr(bridge,'_sample_inputs',sample)
    monkeypatch.setattr(bridge.os,'chdir',chdir)
    monkeypatch.setattr(bridge.os,'execve',execute)
    with pytest.raises(Executed):bridge._worker(role,2,command)
    assert events == ['sample','chdir','exec']


def test_native_provenance_keeps_internal_field_and_compiler_width_distinct(tmp_path):
    role,metadata,_,_,_ = _role_fixture(tmp_path)
    provenance = bridge.load_role(role,0)['provenance']
    assert provenance['internal_mascot_prime'] == 2**256-189
    assert provenance['compiler_bit_length'] == 78
    assert provenance['compiler_security'] == 176
    for name in ('native_binary_sha256','native_policy_sha256','compiled_primitive_counts',
                 'preprocessing_ms','native_preprocessing_ms'):
        assert provenance[name] == metadata[name]


def test_fault_injection_bundle_never_becomes_importable_even_if_marked_ready(tmp_path):
    role,metadata,_,_,_ = _role_fixture(tmp_path)
    metadata['test_invalid_bit'] = dict(party=0,lane=1,value=2)
    (role/'public-manifest.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='fault-injection'):bridge.load_role(role,0)
    assert not (role/'consumed').exists()


@pytest.mark.parametrize('flag',[None,False,1])
def test_pre_authentication_sacrifice_policy_cannot_import_outputs(tmp_path,monkeypatch,flag):
    from pathlib import Path
    role,metadata,_,_,_ = _role_fixture(tmp_path)
    policy = dict(metadata['native_policy'])
    if flag is None:policy.pop('post_authentication_sacrifice_coin')
    else:policy['post_authentication_sacrifice_coin'] = flag
    raw = json.dumps(policy,sort_keys=True).encode()
    (role/'native-policy.json').write_bytes(raw)
    (Path(metadata['native_root'])/'mascot256-preprocessing-policy.json').write_bytes(raw)
    metadata['native_policy'] = policy
    metadata['native_policy_sha256'] = hashlib.sha256(raw).hexdigest()
    (role/'public-manifest.json').write_text(json.dumps(metadata))
    original = Path.read_bytes
    def forbid_private(path):
        if path.name.startswith('Binary-Output'):
            pytest.fail('pre-authentication sacrifice policy imported private outputs')
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',forbid_private)
    with pytest.raises(ValueError,match='sacrifice'):bridge.load_role(role,0)
    assert not (role/'consumed').exists()


def test_retired_bundle_is_rejected_even_if_other_ready_flags_look_valid(tmp_path):
    role,metadata,_,_,_ = _role_fixture(tmp_path)
    metadata['retired'] = True
    (role/'public-manifest.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='retired'):bridge.load_role(role,0)
    assert not (role/'consumed').exists()


def test_predictable_pre_authentication_sacrifice_challenge_has_adaptive_error_counterexample():
    # Authentication accepts chosen field inputs; no MAC forgery is assumed.
    # This algebraic model explains why challenge timing is a policy gate.
    q = bridge.INTERNAL_PRIME
    a0,a1,b,t,delta,alpha = 2,5,3,7,4,11
    c0,c1 = (a0*b+delta)%q,(a1*b+t*delta)%q
    mac0,mac1 = alpha*c0%q,alpha*c1%q
    assert (mac0-alpha*c0)%q == (mac1-alpha*c1)%q == 0
    assert c0 != a0*b%q
    rho = (t*a0-a1)%q
    assert (t*c0-c1-rho*b)%q == 0
    fresh = t+1
    fresh_rho = (fresh*a0-a1)%q
    assert (fresh*c0-c1-fresh_rho*b)%q == delta
