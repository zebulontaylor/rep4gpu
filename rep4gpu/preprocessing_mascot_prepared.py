"""Source and role-local import for model-dependent native q256 batches.

The launcher must provision model inputs privately to their assigned native
roles. Compilation and native execution are external to this module. Native
MASCOT authenticates model inputs and computes F=A@W; no Python coordinator
samples A or reconstructs W, F, masks, or MAC keys.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path

from .preprocessing_mascot_direct import (PRIME,QShare,QSquare,QInputMask,_native_file,
                                        _decode_owner_clear,_verify_policy,_sha)

FORMAT='mascot256-prepared-right-role-v1'


@dataclass
class QPreparedBatch:
    weights: QShare
    a: QShare
    factor: QShare
    identifier: bytes
    model_id: bytes
    used: bool=False


def program(request,session):
    """Return native source, ordered export records, and owner-input plan.

    ``models`` entries specify name, owner, k, n, rows, and a 32-byte hex model
    ID. ``static_inputs`` export additional reusable model values such as biases.
    ``inputs`` request ordinary owner-only query masks; ``squares`` request fresh
    square masks. Owner model input files follow the returned per-owner plan.
    """
    session_bytes=bytes.fromhex(session)
    if len(session_bytes)!=32:raise ValueError('session must be 32 bytes')
    lines=['from Compiler.types import sint,cint,regint,Matrix',
           'from Compiler.instructions import inputmaskreg',
           'from Compiler.library import check_point,print_ln',
           'program.security=128','check_point()','sint.write_to_file(sint(1))']
    records=[];offset=1;plan=[[] for _ in range(4)];clear_counts=[0]*4;names=set()
    def record(kind,item,component,shape):
        nonlocal offset
        count=math.prod(shape)
        identifier=hashlib.blake2b(session_bytes+kind.encode()+item['name'].encode(),digest_size=32).hexdigest()
        entry=dict(kind=kind,name=item['name'],component=component,shape=list(shape),count=count,
                   offset=offset,identifier=identifier)
        if kind=='prepared':entry['model_id']=item['model_id']
        if kind=='inputs':
            owner=item['owner'];entry.update(owner=owner,clear_offset=clear_counts[owner]);clear_counts[owner]+=count
        records.append(entry);offset+=count
    def checked(item,shape):
        if not isinstance(item.get('name'),str) or not item['name'] or item['name'] in names:raise ValueError('unique nonempty names required')
        names.add(item['name'])
        if any(type(n) is not int or n<=0 for n in shape):raise ValueError('positive dimensions required')
    def owner_input(item,shape,variable):
        owner=item['owner']
        if type(owner) is not int or owner not in range(4):raise ValueError('invalid model owner')
        plan[owner].append(dict(name=item['name'],shape=list(shape),count=math.prod(shape)))
        lines.append(f'{variable}=sint.get_input_from({owner},size={math.prod(shape)})')
    for index,item in enumerate(request.get('models',[])):
        k,n,rows=item['k'],item['n'],item['rows'];checked(item,(rows,k,n))
        if len(bytes.fromhex(item['model_id']))!=32:raise ValueError('model ID must be 32 bytes')
        owner_input(item,(k,n),f'w{index}')
        lines += [f'wm{index}=Matrix({k},{n},sint)',f'wm{index}.assign_vector(w{index})',
                  f'am{index}=Matrix({rows},{k},sint)',f'am{index}.assign_vector(sint.get_random(size={rows*k}))',
                  f'fm{index}=am{index}.dot(wm{index})','check_point()',
                  f'sint.write_to_file(w{index})',f'sint.write_to_file(am{index}.get_vector())',f'sint.write_to_file(fm{index}.get_vector())']
        for component,shape in [('weights',(k,n)),('a',(rows,k)),('factor',(rows,n))]:record('prepared',item,component,shape)
    for index,item in enumerate(request.get('static_inputs',[])):
        shape=tuple(item['shape']);checked(item,shape)
        if len(shape)!=2:raise ValueError('static inputs must be matrices')
        owner_input(item,shape,f'static{index}')
        lines+=['check_point()',f'sint.write_to_file(static{index})'];record('static',item,'share',shape)
    for index,item in enumerate(request.get('squares',[])):
        shape=tuple(item['shape']);checked(item,shape)
        if len(shape)!=2:raise ValueError('square masks must be matrices')
        lines += [f'sq{index},sc{index}=sint.get_random_square(size={math.prod(shape)})','check_point()',
                  f'sint.write_to_file(sq{index})',f'sint.write_to_file(sc{index})']
        record('squares',item,'a',shape);record('squares',item,'a_squared',shape)
    for index,item in enumerate(request.get('inputs',[])):
        shape=tuple(item['shape']);checked(item,shape);owner=item['owner']
        if len(shape)!=2 or type(owner)is not int or owner not in range(4):raise ValueError('invalid query mask')
        count=math.prod(shape)
        lines += [f'im{index}=sint(size={count})',f'ic{index}=cint(size={count})',
                  f'inputmaskreg(im{index},ic{index},regint({owner},size={count}))','check_point()',f'sint.write_to_file(im{index})']
        for shift in (0,64,128,192):lines.append(f'(ic{index}>>{shift}).to_regint(n_bits=0,sync=False).binary_output({owner})')
        record('inputs',item,'share',shape)
    lines += ['check_point()',"print_ln('Native prepared q256 export complete')"]
    return '\n'.join(lines)+'\n',records,offset,plan,clear_counts


def write_owner_inputs(role_path,party,request,session,values):
    """Run inside the assigned native role to provision its own model inputs.

    ``values`` maps only that role's names to canonical flat q256 lists. No
    query plaintext enters preprocessing. Existing input files are never reused.
    """
    role=Path(role_path)
    if role.name!=f'party{party}' or party not in range(4):raise ValueError('assigned role mismatch')
    _,_,_,plan,_=program(request,session)
    if set(values)!={item['name'] for item in plan[party]}:raise ValueError('private model input ownership mismatch')
    path=role/'Player-Data'/f'Input-P{party}-0'
    descriptor=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(descriptor,'w') as stream:
        for item in plan[party]:
            row=values[item['name']]
            if len(row)!=item['count'] or any(type(v)is not int or not 0<=v<PRIME for v in row):raise ValueError('noncanonical private model input')
            stream.write(' '.join(map(str,row))+'\n')
        stream.flush();os.fsync(stream.fileno())


def load_role(bundle_path,party):
    """Atomically consume one ready native role bundle; never read peer files."""
    role=Path(bundle_path).resolve();m=json.loads((role/'public-manifest.json').read_text())
    if role.name!=f'party{party}' or party not in range(4):raise ValueError('assigned role mismatch')
    if m.get('format')!=FORMAT or not m.get('ready') or m.get('return_codes')!=[0]*4:raise ValueError('prepared native export incomplete')
    if m.get('prime')!=PRIME or m.get('mac_keys')!=1 or m.get('security_parameter')!=128:raise ValueError('native q256 parameters mismatch')
    source,records,total,plan,clear_counts=program(m['request'],m['session'])
    if (m.get('records')!=records or m.get('share_count')!=total or m.get('owner_input_plan')!=plan
            or m.get('clear_counts')!=clear_counts or m.get('export_source_sha256')!=hashlib.sha256(source.encode()).hexdigest()):
        raise ValueError('prepared native source or layout mismatch')
    binary=Path(m['native_binary']);policy,digest=_verify_policy(binary.parent,binary)
    role_policy=role/'native-policy.json'
    if (policy!=m.get('native_policy') or digest!=m.get('native_policy_sha256') or _sha(binary)!=m.get('native_binary_sha256')
            or _sha(role_policy)!=digest or json.loads(role_policy.read_text())!=policy):raise ValueError('native provenance mismatch')
    descriptor=os.open(role/'consumed',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(descriptor,'w') as stream:stream.write(m['session']);stream.flush();os.fsync(stream.fileno())
    directory=os.open(role,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(directory)
    finally:os.close(directory)
    rows,key=_native_file(role/'Persistence'/f'Transactions-P{party}.data',total)
    if rows[0]!=(int(party==0),key):raise ValueError('native constant calibration failed')
    path=role/'Player-Data'/f'Binary-Output-P{party}-0'
    clear=_decode_owner_clear(path.read_bytes() if path.exists() else b'',[r['count'] for r in records if r['kind']=='inputs' and r['owner']==party])
    prepared={};static={};squares={};inputs={}
    for record in records:
        local=rows[record['offset']:record['offset']+record['count']]
        share=QShare([v for v,g in local],[g for v,g in local],tuple(record['shape']))
        kind=record['kind'];name=record['name']
        if kind=='static':static[name]=share
        else:
            target={'prepared':prepared,'squares':squares,'inputs':inputs}[kind]
            target.setdefault(name,{})[record['component']]=share
    for record in records:
        kind,name=record['kind'],record['name'];identifier=bytes.fromhex(record['identifier'])
        if kind=='prepared' and isinstance(prepared[name],dict):
            parts=prepared[name];prepared[name]=QPreparedBatch(parts['weights'],parts['a'],parts['factor'],identifier,bytes.fromhex(record['model_id']))
        if kind=='squares' and isinstance(squares[name],dict):
            parts=squares[name];squares[name]=QSquare(parts['a'],parts['a_squared'],identifier)
        if kind=='inputs':
            own=clear[record['clear_offset']:record['clear_offset']+record['count']] if record['owner']==party else None
            inputs[name]=QInputMask(inputs[name]['share'],own,record['owner'],identifier)
    session=hashlib.sha256(b'MASCOT-prepared-q256-v1\0'+bytes.fromhex(m['session'])).digest()
    return dict(alpha_share=key,prepared_batches=prepared,static_shares=static,square_masks=squares,input_masks=inputs,session=session,provenance=m)
