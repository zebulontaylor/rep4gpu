"""Generate functional-only MAMA bundles; small-field privacy review failed."""
import argparse
import json
from pathlib import Path
from rep4gpu.preprocessing_mama import run_native


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--native-root',type=Path,required=True)
    parser.add_argument('--timeout',type=float,default=900)
    parser.add_argument('--batch-size',type=int,default=1000)
    args=parser.parse_args()
    report=run_native(args.manifest,args.output,native_root=args.native_root,
                      timeout=args.timeout,batch_size=args.batch_size)
    print(json.dumps({name:report.get(name) for name in ('passed','return_codes','preprocessing_ms','native_binary_sha256',
                                                       'privacy_audit_status','security_target_met','functional_only')},indent=2))
    return 0 if report.get('passed') else 1


if __name__=='__main__':raise SystemExit(main())
