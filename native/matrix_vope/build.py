"""Build the short-vector experiment against the existing vole-labs build."""

from pathlib import Path
import subprocess
import hashlib
import json

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
WORK = REPO / 'work/matrix_vope'
WORK.mkdir(parents=True, exist_ok=True)
BACKEND = REPO / 'work/vole-labs'
BUILD = BACKEND / 'build'
lock = json.loads((HERE / 'source-lock.json').read_text())
for dependency in lock['dependencies']:
    directory = BACKEND / dependency['path']
    head = subprocess.check_output(['git', '-C', str(directory), 'rev-parse', 'HEAD'], text=True).strip()
    if head != dependency['commit']:
        raise SystemExit(f"Dependency commit mismatch: {directory}")
    dirty = subprocess.check_output(['git', '-C', str(directory), 'diff', 'HEAD', '--name-only'], text=True)
    if dirty.strip():
        raise SystemExit(f"Dependency has modified tracked files: {directory}")
for name, expected in lock['source_files_sha256'].items():
    path = BACKEND / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"Dependency source hash mismatch: {path}")
required = [BUILD / 'emp-ot/libemp-ot.a', BUILD / 'emp-tool/libemp-tool.a']
if not all(path.exists() for path in required):
    raise SystemExit('Build work/vole-labs first; see native/matrix_vope/README.md.')
command = [
    'c++', '-std=c++20', '-O2', '-Wall', '-Wno-ignored-attributes',
    '-march=native', '-maes', '-mrdseed', '-DENABLE_RDSEED', '-pthread',
    '-I' + str(BACKEND), '-I' + str(BACKEND / 'thirdparty/emp-ot'),
    '-I' + str(BACKEND / 'thirdparty/emp-tool'), '-I' + str(BUILD / 'emp-tool'),
    str(HERE / 'seeded_vole.cpp'), *map(str, required),
    '-lssl', '-lcrypto', '-o', str(WORK / 'seeded_vole'),
]
with (WORK / 'build.log').open('w') as log:
    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
if result.returncode:
    raise SystemExit(f'Compile failed; see {WORK / "build.log"}')
print(WORK / 'seeded_vole')
