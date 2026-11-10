"""Build the pinned MAMA backend in a fresh directory, without copying party data.

Example from the repository root:
python -m scripts.build_mama_secure --source /path/to/MP-SPDZ \
    --sodium-prefix /path/to/libsodium-1.0.20-prefix --build-dir work/mama-secure

The source lock checks selected MP-SPDZ 0.4.4 files. The build fetches pinned
SoftSpoken and base-OT sources, applies the independent-sacrifice patch, and
uses LONG_BIT_MAC_KEY. Supply the libsodium prefix used by this installation;
its include directory must contain sodium.h and sodium/version.h. Build logs
and a binary/dependency manifest are written inside the new build directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT/'native/mama-source-lock.json'
PATCH_PATH = ROOT/'native/patches/mama-independent-sacrifice.patch'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_source(source, lock):
    for name, expected in lock['source_files_sha256'].items():
        path = source/name
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'source does not match the pinned MP-SPDZ file: {name}')
    if digest(PATCH_PATH) != lock['patch_sha256']:
        raise ValueError('native sacrifice patch differs from the source lock')


def copy_sodium(prefix, target, lock):
    include = prefix/'include'
    version = include/'sodium/version.h'
    if not version.is_file() or f'"{lock["sodium_version"]}"' not in version.read_text():
        raise ValueError(f'libsodium {lock["sodium_version"]} headers are required')
    library_dir = next((prefix/name for name in ('lib', 'lib64')
                        if list((prefix/name).glob('libsodium.*'))), None)
    if library_dir is None:
        raise ValueError('the libsodium prefix has no lib/libsodium.* or lib64/libsodium.*')
    (target/'include').mkdir(parents=True)
    (target/'lib').mkdir()
    shutil.copy2(include/'sodium.h', target/'include/sodium.h')
    shutil.copytree(include/'sodium', target/'include/sodium')
    hashes = {}
    for path in library_dir.glob('libsodium.*'):
        if path.suffix == '.la':
            continue
        resolved = path.resolve(strict=True)
        # Keep both the linker's filename and the library's versioned filename
        # so the dynamic loader can resolve its SONAME inside the private prefix.
        for name in {path.name, resolved.name}:
            destination = target/'lib'/name
            shutil.copy2(resolved, destination)
            hashes[name] = digest(destination)
    if not (target/'lib/libsodium.so').is_file():
        raise ValueError('the libsodium prefix must include the shared library libsodium.so')
    return hashes


def write_policy(build):
    """Bind the reviewed configuration to actual source and dependency bytes."""
    sources = ['CONFIG', 'CONFIG.mine', 'Machines/mama-party.cpp',
               'Protocols/MamaPrep.hpp', 'Protocols/ShuffleSacrifice.hpp',
               'Protocols/MAC_Check.hpp', 'OT/NPartyTripleGenerator.hpp',
               'OT/OTExtensionWithMatrix.cpp', 'OT/OTMultiplier.hpp',
               'OT/MascotMacKey.hpp', 'GC/TinierShare.h', 'Makefile']
    dependencies = ['local/lib/liblibOTe.a', 'local/lib/libcryptoTools.a',
                    'libSPDZ.so', 'local/lib/libsodium.so']
    policy = dict(format='mama-prime31-reviewed-build-v1', prime=2147483647,
                  mac_keys=10, security_parameter=128, ot='SoftSpokenMalOt',
                  binary_sha256=digest(build/'mama-party.x'),
                  source_sha256={name:digest(build/name) for name in sources},
                  dependency_sha256={name:digest(build/name) for name in dependencies},
                  source_lock_sha256=digest(LOCK_PATH), reviewed_patch_sha256=digest(PATCH_PATH),
                  build_properties=dict(LONG_BIT_MAC_KEY=True, USE_KOS=0,
                                        USE_LIBOTE2=0, INSECURE=False))
    path = build/'native-preprocessing-policy.json'
    path.write_text(json.dumps(policy, indent=2)+'\n')
    return path


def record_functional_smoke(build, public_report):
    """Record a successful native execution without mutating its build policy."""
    report = json.loads(public_report.read_text())
    policy = build/'native-preprocessing-policy.json'
    expected = dict(prime=2147483647, mac_keys=10, security_parameter=128)
    if not report.get('ready') or not report.get('passed'):
        raise ValueError('native functional smoke did not complete successfully')
    if report.get('return_codes') != [0, 0, 0, 0] or any(report.get(k) != v for k,v in expected.items()):
        raise ValueError('native functional smoke has different roles or security parameters')
    if report.get('native_binary_sha256') != digest(build/'mama-party.x'):
        raise ValueError('native functional smoke used a different binary')
    if report.get('native_policy_sha256') != digest(policy):
        raise ValueError('native functional smoke used a different policy')
    record = dict(format='mama-native-functional-smoke-v1', passed=True,
                  native_binary_sha256=report['native_binary_sha256'],
                  native_policy_sha256=digest(policy), public_report=str(public_report),
                  public_report_sha256=digest(public_report), return_codes=report['return_codes'],
                  **expected,
                  claim='Functional execution evidence; not an independent security proof.')
    path = build/'build-functional-smoke.json'
    path.write_text(json.dumps(record, indent=2)+'\n')
    return path


def clean_build_environment():
    """Prevent ambient flag overrides from changing the reviewed configuration."""
    flag_variables = {'CFLAGS', 'CXXFLAGS', 'CPPFLAGS', 'MY_CFLAGS', 'SECURE',
                      'OPTIM', 'GDEBUG', 'USE_KOS', 'USE_LIBOTE2', 'MAKEFLAGS', 'MFLAGS',
                      'MAKEOVERRIDES', 'MAKEFILES', 'DEBUG', 'MOD', 'PROF', 'BREW_CFLAGS'}
    return {key:value for key,value in os.environ.items() if key not in flag_variables}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--sodium-prefix', type=Path, required=True)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--jobs', type=int, choices=(1, 2), default=2)
    parser.add_argument('--cxx', default='clang++')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args(argv)
    source, build = args.source.resolve(), args.build_dir.resolve()
    lock = json.loads(LOCK_PATH.read_text())
    validate_source(source, lock)
    if args.validate_only:
        print('Pinned source files and patch match.')
        return 0
    if build.exists():
        raise ValueError(f'build directory already exists: {build}')
    if build == source or source in build.parents:
        raise ValueError('build directory must be outside the source tree')
    compiler = shutil.which(args.cxx)
    if compiler is None or any(c.isspace() for c in compiler):
        raise ValueError('C++ compiler must resolve to an executable path without spaces')
    def ignore(directory, names):
        excluded = {'Player-Data', '.git', 'deps', 'local', 'bin', 'Programs', '__pycache__'}
        outputs = ('.o', '.d', '.a', '.so', '.x', '.map')
        return [name for name in names if name in excluded or name.endswith(outputs)]
    shutil.copytree(source, build, ignore=ignore, symlinks=False)
    (build/'Player-Data').mkdir(mode=0o700)
    (build/'deps').mkdir()
    sodium_hashes = copy_sodium(args.sodium_prefix.resolve(), build/'local', lock)
    (build/'CONFIG.mine').write_text(lock['config'])
    commands = []
    environment = clean_build_environment()
    def run(command, name, cwd=build):
        commands.append(dict(command=list(map(str, command)), cwd=str(cwd), log=name+'.log'))
        with (build/(name+'.log')).open('w') as log:
            subprocess.run(list(map(str, command)), cwd=cwd, stdout=log,
                           stderr=subprocess.STDOUT, check=True, env=environment)
    try:
        run(['patch', '-p1', '--batch', '--forward', '-i', PATCH_PATH], 'apply-patch')
        for dep in lock['dependencies']:
            destination = build/'deps'/dep['name']
            run(['git', 'clone', '--no-checkout', dep['url'], destination], 'clone-'+dep['name'])
            run(['git', '-C', destination, 'checkout', '--detach', dep['commit']], 'checkout-'+dep['name'])
            run(['git', '-C', destination, 'submodule', 'update', '--init', '--recursive',
                 '--jobs', str(args.jobs)], 'submodules-'+dep['name'])
        ote = build/'deps/libOTe'
        ote_build = ote/'out/mama128'
        run(['cmake', '-S', ote, '-B', ote_build, '-G', 'Ninja',
             f'-DCMAKE_CXX_COMPILER={compiler}', '-DCMAKE_BUILD_TYPE=Release',
             f'-DCMAKE_INSTALL_PREFIX={build}/local', f'-DCMAKE_PREFIX_PATH={build}/local',
             '-DCMAKE_INSTALL_LIBDIR=lib', '-DENABLE_SOFTSPOKEN_OT=ON',
             '-DENABLE_SILENTOT=OFF', '-DENABLE_SIMPLESTOT=OFF',
             '-DENABLE_AVX=ON', '-DENABLE_SSE=ON', '-DENABLE_PORTABLE_AES=ON'], 'libote-configure')
        run(['cmake', '--build', ote_build, '--parallel', args.jobs], 'libote-build')
        run(['cmake', '--install', ote_build], 'libote-install')
        run(['make', '-C', build/'deps/SimpleOT', '-j'+str(args.jobs), 'libsimpleot',
             f'CFLAGS=-O3 -Wall -Wextra -I{build}/local/include'], 'simpleot-build')
        simplest = build/'deps/SimplestOT_C/ref10'
        run(['cmake', '-S', simplest, '-B', simplest,
             f'-DOC_THIRDPARTY_HINT={build}/local'], 'simplestot-configure')
        run(['cmake', '--build', simplest, '--parallel', args.jobs], 'simplestot-build')
        run(['make', '-j'+str(args.jobs), f'CXX={compiler}', 'mama-party.x'], 'mama-build')
        binary = build/'mama-party.x'
        policy_path = write_policy(build)
        report = dict(passed=True, source_lock_sha256=digest(LOCK_PATH),
                      patch_sha256=digest(PATCH_PATH), policy_sha256=digest(policy_path),
                      native_binary=str(binary),
                      native_binary_sha256=digest(binary),
                      compiler=subprocess.check_output([compiler, '--version'], text=True).splitlines()[0],
                      dependencies=lock['dependencies'], sodium_libraries_sha256=sodium_hashes,
                      config=(build/'CONFIG.mine').read_text(), commands=commands)
        (build/'build-report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(dict(native_binary=str(binary), native_binary_sha256=digest(binary))))
        return 0
    except subprocess.CalledProcessError as error:
        (build/'build-failure.json').write_text(json.dumps(dict(return_code=error.returncode,
                                               commands=commands), indent=2)+'\n')
        return error.returncode if error.returncode > 0 else 128-error.returncode


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
