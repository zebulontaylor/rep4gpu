"""Generate and consume disposable native MAMA correlations in four TLS roles.

Run from the repository root with ``python -m scripts.check_mama_preprocessing
--native-root work/mama128-build --output work/native-check-FRESH``. Each worker
loads only its assigned role. Opened masks and triples are retired test material.
"""
import argparse,json,subprocess,sys,time,ssl
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def worker(root,p,addresses,online_mac_keys=10):
 import torch
 from rep4gpu.preprocessing_mama import load_role
 from rep4gpu.spdz_distributed import DistributedParty
 from rep4gpu.spdz_inputs import private_input
 from rep4gpu.spdz import PRIME
 from rep4gpu.transport import SocketTransport
 role=root/f'party{p}';player=role/'Player-Data'
 addresses=[tuple(address) for address in addresses]
 # Only public certificates are concatenated; this process loads its own key.
 ca=role/'all-public-certs.pem';ca.write_bytes(b''.join((player/f'P{i}.pem').read_bytes() for i in range(4)))
 server=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);server.verify_mode=ssl.CERT_REQUIRED
 server.load_cert_chain(player/f'P{p}.pem',player/f'P{p}.key');server.load_verify_locations(cafile=ca)
 client=ssl.create_default_context(ssl.Purpose.SERVER_AUTH,cafile=ca)
 client.load_cert_chain(player/f'P{p}.pem',player/f'P{p}.key')
 transport=None
 try:
  bundle=load_role(role,p,online_mac_keys=online_mac_keys)
  transport=SocketTransport(p,addresses,timeout=30,server_context=server,client_context=client,
                            peer_hostnames={i:f'P{i}' for i in range(4)})
  party=DistributedParty(p,transport,bundle['alpha_share'],session=bundle['session'])
  triple=bundle['matrix_triples']['probe'];triple.used=True
  a,b,c=party.open_many([triple.a,triple.b,triple.c],label=b'disposable-preprocessing-calibration')
  expected=torch.tensor([[sum(int(a[i,k])*int(b[k,j]) for k in range(a.shape[1]))%PRIME
                           for j in range(b.shape[1])] for i in range(a.shape[0])],dtype=torch.int64)
  assert torch.equal(c,expected)
  sq=bundle['square_masks']['sq'];sq.used=True
  a,aa=party.open_many([sq.a,sq.a_squared],label=b'disposable-square-calibration')
  assert torch.equal(aa,(a*a)%PRIME)
  value=torch.tensor([[1,-2,3],[-4,5,-6]],dtype=torch.int64) if p==0 else None
  x=private_input(party,bundle['input_masks']['x'],value)
  opened=party.open(x)
  assert torch.equal(opened,torch.tensor([[1,-2,3],[-4,5,-6]])%PRIME)
  report=dict(party=p,passed=True,mac_keys=party.mac_keys,native_mac_keys=10,online_mac_keys=party.mac_keys,online_mac_indices=list(range(party.mac_keys)),bytes_sent=transport.bytes_sent,
              matrix_relation_checked=True,square_relation_checked=True,owner_input_checked=True)
  (role/'probe-check.json').write_text(json.dumps(report))
 finally:
  if transport is not None:transport.close()


def validate_bundle(root,online_mac_keys=10):
 from benchmarks.socket_inference import reserve_addresses
 addresses=reserve_addresses();jobs=[];logs=[];start=time.monotonic()
 for p in range(4):
  role=root/f'party{p}';log=(role/'probe-check.log').open('w');logs.append(log)
  jobs.append(subprocess.Popen([sys.executable,__file__,'--worker',str(p),'--bundle-root',str(root),'--addresses',json.dumps(addresses),'--online-mac-keys',str(online_mac_keys)],stdout=log,stderr=subprocess.STDOUT))
 try:
  for job in jobs:job.wait(timeout=45)
 finally:
  for job in jobs:
   if job.poll() is None:job.kill()
  for job in jobs:job.wait()
  for log in logs:log.close()
 report=dict(return_codes=[j.returncode for j in jobs],elapsed=time.monotonic()-start)
 report['passed']=report['return_codes']==[0]*4
 (root/'probe-validation.json').write_text(json.dumps(report,indent=2)+'\n')
 print(report)
 if not report['passed']:
  for p in range(4):print(p,(root/f'party{p}/probe-check.log').read_text()[-1500:])
 return report


def run_fresh(native_root,output,online_mac_keys=10):
 from rep4gpu.preprocessing_mama import run_native
 output=Path(output).resolve()
 if output.exists():raise ValueError('fresh validation output directory must not exist')
 request={'matrices':[{'name':'probe','m':2,'k':3,'n':2}],
          'squares':[{'name':'sq','shape':[2,2]}],
          'inputs':[{'name':'x','owner':0,'shape':[2,3]}]}
 native=run_native(request,output/'preprocessing',native_root=native_root,timeout=90)
 checked=validate_bundle(output/'preprocessing',online_mac_keys) if native['passed'] else {'passed':False,'return_codes':[]}
 report=dict(passed=native['passed'] and checked['passed'],native_return_codes=native['return_codes'],
             privacy_audit_status='failed_small_field_selective_failure',security_target_met=False,functional_only=True,
             checked_role_return_codes=checked['return_codes'],mac_keys=online_mac_keys,native_mac_keys=10,
             online_mac_keys=online_mac_keys,online_mac_indices=list(range(online_mac_keys)),prime=2147483647,
             native_preprocessing_ms=native.get('native_preprocessing_ms'),
             preprocessing_ms=native.get('preprocessing_ms'),trusted_dealer=False,
             disposable_correlations=True,native_policy_sha256=native.get('native_policy_sha256'),
             native_binary_sha256=native['native_binary_sha256'],
             checks='Native local constant-one calibration, independent Python biginteger matrix relation, square relation, owner-private input, full MAC checks over mutual TLS.')
 (output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
 return report


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--native-root',type=Path)
 parser.add_argument('--output',type=Path)
 parser.add_argument('--worker',type=int)
 parser.add_argument('--bundle-root',type=Path)
 parser.add_argument('--addresses')
 parser.add_argument('--online-mac-keys',type=int,choices=(7,10),default=10)
 args=parser.parse_args()
 if args.worker is not None:
  worker(args.bundle_root,args.worker,json.loads(args.addresses),args.online_mac_keys);return 0
 if args.native_root is None or args.output is None:parser.error('--native-root and --output are required')
 report=run_fresh(args.native_root,args.output,args.online_mac_keys)
 print(json.dumps(report,indent=2))
 return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
