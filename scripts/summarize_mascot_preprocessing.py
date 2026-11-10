"""Summarize public native preprocessing costs without reading role secrets.

Native communication counters use decimal megabytes and rounded log output.
They count protocol traffic between the colocated roles, not NIC packet bytes.
"""
import argparse
import json
from pathlib import Path
import re


def summarize(root):
    root=Path(root)
    manifest=json.loads((root/'public-manifest.json').read_text())
    roles=[]
    for party in range(4):
        log=(root/f'party{party}/native.log').read_text()
        def number(pattern):
            match=re.search(pattern,log)
            return float(match.group(1)) if match else None
        roles.append(dict(party=party,native_seconds=number(r'Time = ([0-9.e+]+) seconds'),
                          sent_decimal_MB=number(r'Data sent = ([0-9.e+]+) MB'),
                          global_sent_decimal_MB=number(r'Global data sent = ([0-9.e+]+) MB')))
    complete=all(role['native_seconds'] is not None for role in roles)
    report=dict(provider=manifest['format'],ready=manifest['ready'],return_codes=manifest['return_codes'],
                privacy_audit_status=manifest['privacy_audit_status'],fresh_bundle_count=1,
                preprocessing_ms=manifest['preprocessing_ms'],static_compile_ms=manifest['compile_ms'],
                fresh_preprocessing_ms=manifest['preprocessing_ms']-manifest['compile_ms'],
                native_orchestration_ms=manifest['native_preprocessing_ms'],role_statistics=roles,
                maximum_native_process_seconds=max(role['native_seconds'] for role in roles) if complete else None,
                compiled_primitive_counts=manifest['compiled_primitive_counts'],
                authenticated_p31_scalars=manifest['share_count']-1,
                native_binary_sha256=manifest['native_binary_sha256'],native_policy_sha256=manifest['native_policy_sha256'],
                decimal_MB_bytes=1000000,communication_values_rounded=True,
                communication_context='Logical native protocol traffic over encrypted colocated peer channels; NIC packet overhead is not counted.',
                amortization_contract='One fresh bundle; replaying its correlations does not increase the count of fresh inferences.')
    if (root/'runtime-context.json').exists():report['runtime_context']=json.loads((root/'runtime-context.json').read_text())
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();encoded=json.dumps(summarize(args.root),indent=2)+'\n'
    if args.output:args.output.write_text(encoded)
    else:print(encoded,end='')
