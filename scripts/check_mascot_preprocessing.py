"""Consume disposable large-prime bridge output in four isolated TLS roles."""
import argparse,json,ssl,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def worker(root,p,addresses):
 import torch
 from rep4gpu.preprocessing_mascot import load_role
 from rep4gpu.spdz_distributed import DistributedParty
 from rep4gpu.transport import SocketTransport
 from rep4gpu.spdz import PRIME
 role=root/f'party{p}';player=role/'Player-Data'
 ca=role/'all-public-certs.pem';ca.write_bytes(b''.join((player/f'P{i}.pem').read_bytes() for i in range(4)))
 server=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);server.verify_mode=ssl.CERT_REQUIRED
 server.load_cert_chain(player/f'P{p}.pem',player/f'P{p}.key');server.load_verify_locations(cafile=ca)
 client=ssl.create_default_context(ssl.Purpose.SERVER_AUTH,cafile=ca);client.load_cert_chain(player/f'P{p}.pem',player/f'P{p}.key')
 bundle=load_role(role,p)
 transport=SocketTransport(p,[tuple(a) for a in addresses],timeout=60,server_context=server,client_context=client,peer_hostnames={i:f'P{i}' for i in range(4)})
 try:
  party=DistributedParty(p,transport,bundle['alpha_share'],session=bundle['session'])
  for name,triple in bundle['matrix_triples'].items():
   triple.used=True
   a,b,c=party.open_many([triple.a,triple.b,triple.c],label=b'disposable-native-matrix')
   expected=torch.tensor([[sum(int(a[i,k])*int(b[k,j]) for k in range(a.shape[1]))%PRIME for j in range(b.shape[1])] for i in range(a.shape[0])])
   assert torch.equal(c,expected)
  for name,square in bundle['square_masks'].items():
   square.used=True;a,c=party.open_many([square.a,square.a_squared],label=b'disposable-native-square')
   assert torch.equal(c,a*a%PRIME)
  (role/'probe-check.json').write_text(json.dumps(dict(party=p,passed=True,mac_keys=7,matrix_relations=True,square_relations=True)))
 finally:transport.close()


def validate_bundle(root):
 from benchmarks.socket_inference import reserve_addresses
 root=Path(root).resolve();addresses=reserve_addresses();jobs=[];logs=[];start=time.monotonic()
 for p in range(4):
  log=(root/f'party{p}/probe-check.log').open('w');logs.append(log)
  jobs.append(subprocess.Popen([sys.executable,__file__,'--worker',str(p),'--root',str(root),'--addresses',json.dumps(addresses)],stdout=log,stderr=subprocess.STDOUT))
 try:
  for job in jobs:job.wait(timeout=90)
 finally:
  for job in jobs:
   if job.poll()is None:job.kill()
  for job in jobs:job.wait()
  for log in logs:log.close()
 result=dict(return_codes=[j.returncode for j in jobs],elapsed_ms=(time.monotonic()-start)*1000,disposable=True)
 result['passed']=result['return_codes']==[0]*4
 (root/'probe-validation.json').write_text(json.dumps(result,indent=2)+'\n')
 if not result['passed']:
  for p in range(4):print(p,(root/f'party{p}/probe-check.log').read_text()[-1800:])
 return result


if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--worker',type=int);parser.add_argument('--addresses')
 args=parser.parse_args()
 if args.worker is not None:worker(args.root,args.worker,json.loads(args.addresses))
 else:
  report=validate_bundle(args.root);print(json.dumps(report,indent=2));raise SystemExit(0 if report['passed'] else 1)
