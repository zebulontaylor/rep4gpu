"""Compile explicit authenticated matrix operations for the published HE fork.

This public circuit bridge does not approve the fork's key generation or OT
implementation. Its bytecode must use VMATMULS; ordinary Matrix.dot retains the
older scalar-triple path in that fork.
"""
import hashlib,json,math,os,subprocess,sys
from pathlib import Path

SOURCE_COMMIT='618301c19599e024194f6e442717e717ef7ecc2b'
FORMAT='lowgear-matrix-public-circuit-v1'
def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _program(request,security=40):
    if type(security)is not int or security not in (40,128):raise ValueError('unsupported configured security')
    from .preprocessing_mama import _layout
    _layout(request,'00'*32)
    lines=['from Compiler.types import sint,cint,regint',
           'from Compiler.instructions import vmatmuls,matmul_desc,inputmaskreg',
           'from Compiler.library import check_point,print_ln',f'program.security={security}',
           'def persist(v):',
           '    sint.write_to_file([v.get_vector(i,1) for i in range(v.size)])',
           'check_point()','persist(sint(1))']
    for i,item in enumerate(request.get('matrices',[])):
        m,k,n=(item[x]for x in ('m','k','n'))
        lines.extend([f'a{i}=sint.get_random(size={m*k})',f'b{i}=sint.get_random(size={k*n})',
                      f'c{i}=sint(size={m*n})',f'vmatmuls(matmul_desc(c{i},a{i},b{i},{m},{k},{n}))',
                      'check_point()',f'persist(a{i})',f'persist(b{i})',f'persist(c{i})'])
    for i,item in enumerate(request.get('squares',[])):
        count=math.prod(item['shape'])
        lines.extend([f'sq{i},sqc{i}=sint.get_random_square(size={count})','check_point()',f'persist(sq{i})',f'persist(sqc{i})'])
    for i,item in enumerate(request.get('inputs',[])):
        count=math.prod(item['shape']);owner=item['owner']
        lines.extend([f'im{i}=sint(size={count})',f'ic{i}=cint(size={count})',
                      f'inputmaskreg(im{i},ic{i},regint({owner},size={count}))','check_point()',
                      f'persist(im{i})',f'ic{i}.to_regint(n_bits=0).binary_output({owner})'])
    lines.extend(['check_point()',"print_ln('Native matrix export checks complete')"])
    return '\n'.join(lines)+'\n'


def compile_public(request,output_dir,*,native_root='work/mpspdz-convolution-triples',security=40,timeout=120):
    """Compile public code and reject a fallback to scalar matrix multiplication.

    No role files or native protocol processes are created. The output records
    the source, compiler files, bytecode opcodes and schedules used by this
    diagnostic build. A separate reviewed native policy is still required for
    secret preprocessing.
    """
    request=json.loads(Path(request).read_text())if not isinstance(request,dict)else request
    root=Path(native_root).resolve();out=Path(output_dir).resolve();out.mkdir(parents=True,exist_ok=False)
    for directory in ('Source','Bytecode','Schedules'):(out/'Programs'/directory).mkdir(parents=True)
    name='matrix_export';source=out/'Programs/Source'/f'{name}.mpc';source.write_text(_program(request,security))
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONHASHSEED':'0'}
    with (out/'compile.log').open('w')as log:
        result=subprocess.run([sys.executable,'-B',str(root/'compile.py'),'-F','32',name],cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    if result.returncode:raise RuntimeError(f'matrix public compilation failed; see {out}/compile.log')
    inspect='''import sys,json,collections
sys.path.insert(0,sys.argv[1])
from Compiler.program import Program,Tape
from Compiler import instructions
from Compiler.GC import instructions as gc_instructions
from Compiler.instructions_base import ParsedInstruction
import inspect
for module in (instructions,gc_instructions):
 for value in vars(module).values():
  if inspect.isclass(value) and value.__name__[0]!='v':
   try:ParsedInstruction.reverse_opcodes[value.code]=value
   except AttributeError:pass
ParsedInstruction.reverse_opcodes[instructions.vmatmuls.code]=instructions.vmatmuls
ops=collections.Counter();text=[];dimensions=[];squares=0
for tape in Program.read_tapes(sys.argv[2]):
 for inst in Tape.read_instructions(tape):
  name=inst.type.__name__;ops[name]+=1;text.append(str(inst))
  if name=='vmatmuls':dimensions.extend([inst.args[j+3:j+6] for j in range(0,len(inst.args),6)])
  if name=='square':squares+=inst.size or 1
print(json.dumps(dict(opcodes=dict(ops),assembly=text,matrix_dimensions=dimensions,scalar_squares=squares)))
'''
    decoded=subprocess.run([sys.executable,'-B','-c',inspect,str(root),name],cwd=out,env=env,check=True,capture_output=True,text=True,timeout=timeout)
    instructions=json.loads(decoded.stdout);ops=instructions['opcodes']
    expected=sorted([[item[k] for k in ('m','k','n')] for item in request.get('matrices',[])])
    if sorted(instructions['matrix_dimensions'])!=expected:raise ValueError('matrix circuit VMATMULS dimensions/count differ')
    if instructions['scalar_squares']!=sum(math.prod(item['shape']) for item in request.get('squares',[])):raise ValueError('matrix circuit square usage differs')
    if any(ops.get(key,0)for key in ('matmuls','matmulsm','dotprods','muls')):raise ValueError('matrix export used a scalar multiplication fallback')
    (out/'matrix_export.asm').write_text('\n'.join(instructions['assembly'])+'\n')
    files=[source,*sorted((out/'Programs/Bytecode').glob('*.bc')),*sorted((out/'Programs/Schedules').glob('*.sch'))]
    report=dict(format=FORMAT,source_commit=SOURCE_COMMIT,request=request,configured_security=security,
                native_root=str(root),source_sha256=_sha(source),artifacts={str(p.relative_to(out)):_sha(p)for p in files},
                compiler_sha256={str(p.relative_to(root)):_sha(p)for p in sorted((root/'Compiler').rglob('*.py'))},
                matrix_dimensions=instructions['matrix_dimensions'],scalar_squares=instructions['scalar_squares'],
                compile_entry_sha256=_sha(root/'compile.py'),opcodes=ops,used_native_matrix_opcode=True,
                native_protocol_executed=False,security_target_met=False,privacy_audit_status='raw_fork_unapproved_public_compile_only')
    (out/'public-circuit.json').write_text(json.dumps(report,indent=2)+'\n');return report
