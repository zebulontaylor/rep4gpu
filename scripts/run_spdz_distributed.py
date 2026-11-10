"""Run four isolated authenticated Beaver roles over mutual TLS.

The coordinator is an explicit trusted preprocessing dealer. Each temporary
role file contains that role's additive values, MAC shares, MAC-key shares,
and its own plaintext input if it is an input owner. Workers receive no other
owner's plaintext and cannot reconstruct complete triples or global MAC keys
from their assigned bundles. The shared-account fixture is not an OS security
boundary against malicious worker code.
"""
import argparse
import hashlib
import math
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import ssl
import torch
from benchmarks.socket_inference import certificates,reserve_addresses,MeteredTransport
from rep4gpu.spdz import TrustedDealer,PRIME,field_matmul


class DiagnosticTransport(MeteredTransport):
    def __init__(self,*args,**kwargs):
        self.exchange_ms=[]
        super().__init__(*args,**kwargs)

    def exchange(self,outgoing):
        started=time.perf_counter()
        result=super().exchange(outgoing)
        self.exchange_ms.append((time.perf_counter()-started)*1000)
        return result


def _fingerprints():
    repository=Path(__file__).resolve().parents[1]
    sources=['scripts/run_spdz_distributed.py','rep4gpu/spdz.py','rep4gpu/spdz_distributed.py',
             'rep4gpu/spdz_inputs.py','rep4gpu/transport.py','rep4gpu/kernels.py','rep4gpu/field_kernels.py',
             'benchmarks/socket_inference.py']
    return {name:hashlib.sha256((repository/name).read_bytes()).hexdigest() for name in sources}


def _pack(share):
    return dict(values=share.values.cpu(),macs=share.macs.cpu())


def _unpack(data,device):
    from rep4gpu.spdz_distributed import AuthShare
    return AuthShare(data['values'].to(device),data['macs'].to(device))


def _synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


def worker(config_path,result_path):
    from rep4gpu.spdz_distributed import DistributedParty,MatrixTriple
    from rep4gpu.spdz_inputs import InputMask,private_input
    config=json.loads(Path(config_path).read_text())
    p,device=config['party'],config['device']
    torch.set_num_threads(1)
    transport=None
    try:
        local=torch.load(config['preprocessing'],map_location='cpu',weights_only=True)
        assert local['party']==p and set(local)=={'party','alpha_share','input_x','input_y','triple'}
        inputs=[]
        for name in ('input_x','input_y'):
            data=local[name]
            inputs.append((InputMask(_unpack(data['share'],device),
                                     None if data['clear'] is None else data['clear'].to(device),
                                     data['owner'],data['identifier']),
                           None if data['value'] is None else data['value'].to(device)))
        triple=MatrixTriple(*[_unpack(local['triple'][name],device) for name in ('a','b','c')],
                            identifier=local['triple']['identifier'])
        alpha=local['alpha_share'].to(device)
        server=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server.verify_mode=ssl.CERT_REQUIRED
        server.load_cert_chain(config['identity']['certificate'],config['identity']['key'])
        server.load_verify_locations(config['ca'])
        client=ssl.create_default_context(ssl.Purpose.SERVER_AUTH,cafile=config['ca'])
        client.load_cert_chain(config['identity']['certificate'],config['identity']['key'])
        transport=DiagnosticTransport(p,[tuple(a) for a in config['addresses']],timeout=config['timeout'],
                                   server_context=server,client_context=client,
                                   peer_hostnames={q:f'party{q}' for q in range(4)})
        party=DistributedParty(p,transport,alpha,parties=4,session=bytes.fromhex(config['session']),stack_macs=config.get('stack_macs',False))
        input_started=time.perf_counter()
        x,y=[private_input(party,mask,value) for mask,value in inputs]
        _synchronize(device)
        input_ms=(time.perf_counter()-input_started)*1000
        input_bytes=transport.bytes_sent
        # Compile/initialize public-shape arithmetic before online timing.
        if str(device).startswith('cuda'):
            party._matmul_local(triple,torch.zeros_like(x.values),torch.zeros_like(y.values))
        _synchronize(device)
        party.broadcast(b'ready',label=b'public-arithmetic-warmup-complete')
        online_bytes=transport.bytes_sent
        online_exchange_start=len(transport.exchange_ms)
        started=time.perf_counter()
        result=party.matmul(x,y,triple)
        opened=party.open(result)
        _synchronize(device)
        online_ms=(time.perf_counter()-started)*1000
        report=dict(party=p,passed=True,online_ms=online_ms,authenticated_tls=transport.authenticated,
                    payload_bytes=transport.bytes_sent,input_payload_bytes=input_bytes,
                    online_payload_bytes=transport.bytes_sent-online_bytes,input_protocol_ms=input_ms,
                    online_exchange_ms=sum(transport.exchange_ms[online_exchange_start:]),
                    local_arithmetic_serialization_staging_ms=max(0,online_ms-sum(transport.exchange_ms[online_exchange_start:])),
                    messages=transport.messages_sent,rounds=transport.rounds,
                    output=opened.cpu().tolist(),local_value_shape=list(x.values.shape),
                    local_mac_shape=list(x.macs.shape),alpha_share_shape=list(alpha.shape),
                    other_owners_plaintext_supplied=False,local_only_preprocessing=True,triple_consumed=triple.used)
        Path(result_path).write_text(json.dumps(report,indent=2)+'\n')
    except BaseException as error:
        Path(result_path).write_text(json.dumps(dict(party=p,passed=False,error=repr(error)))+'\n')
        traceback.print_exc()
        raise
    finally:
        if transport is not None:
            transport.close()


def run(output,*,device='cpu',m=8,k=16,n=8,mac_keys=5,timeout=90,stack_macs=False):
    from rep4gpu.spdz_distributed import split_triple
    from rep4gpu.spdz_inputs import make_input_masks
    if not math.isfinite(timeout) or timeout<=0:
        raise ValueError('timeout must be finite and positive')
    if min(m,k,n)<1:
        raise ValueError('matrix dimensions must be positive')
    if mac_keys<5:
        raise ValueError('at least five MAC keys are required')
    source_before=_fingerprints()
    output=Path(output).resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    logs=output.parent/(output.stem+'_logs')
    logs.mkdir(exist_ok=True)
    for p in range(4):
        (logs/f'party{p}.json').unlink(missing_ok=True)
    torch.set_num_threads(1)
    generator=torch.Generator().manual_seed(2301)
    x=torch.randint(-8,9,(m,k),generator=generator)
    y=torch.randint(-8,9,(k,n),generator=generator)
    prep_started=time.perf_counter()
    dealer=TrustedDealer(parties=4,mac_keys=mac_keys,device='cpu')
    masks_x,masks_y=make_input_masks(dealer,0,tuple(x.shape)),make_input_masks(dealer,1,tuple(y.shape))
    triples=split_triple(dealer.triple(m,k,n))
    preprocessing_ms=(time.perf_counter()-prep_started)*1000
    expected=field_matmul(x.remainder(PRIME),y.remainder(PRIME))
    torch.backends.cuda.matmul.allow_tf32=False
    xf,yf=x.to(device=device,dtype=torch.float32),y.to(device=device,dtype=torch.float32)
    for _ in range(3):
        xf@yf
    _synchronize(device)
    samples=[]
    for _ in range(10):
        start=time.perf_counter()
        floating=xf@yf
        _synchronize(device)
        samples.append((time.perf_counter()-start)*1000)
    assert torch.equal(floating.cpu().round().long().remainder(PRIME),expected)
    processes,handles=[],[]
    timed_out=False
    with tempfile.TemporaryDirectory(prefix='spdz-preprocessing-') as temporary:
        directory=Path(temporary)
        ca,identities=certificates(directory)
        addresses=reserve_addresses()
        session=os.urandom(32).hex()
        for p in range(4):
            private=directory/f'party{p}.pt'
            def packed_input(mask,value):
                return dict(share=_pack(mask.share),clear=mask.clear,owner=mask.owner,
                            identifier=mask.identifier,value=value if p==mask.owner else None)
            triple_data={name:_pack(getattr(triples[p],name)) for name in ('a','b','c')}
            triple_data['identifier']=triples[p].identifier
            torch.save(dict(party=p,alpha_share=dealer.alpha_shares[:,p].clone(),
                            input_x=packed_input(masks_x[p],x),input_y=packed_input(masks_y[p],y),
                            triple=triple_data),private)
            private.chmod(0o600)
            config=directory/f'party{p}.json'
            config.write_text(json.dumps(dict(party=p,device=device,preprocessing=str(private),ca=ca,
                                              identity=identities[p],addresses=addresses,timeout=timeout,session=session,stack_macs=stack_macs)))
            config.chmod(0o600)
            handle=(logs/f'party{p}.log').open('w')
            handles.append(handle)
            processes.append(subprocess.Popen([sys.executable,'-m','scripts.run_spdz_distributed','--worker',str(config),
                                                '--result',str(logs/f'party{p}.json')],stdout=handle,stderr=subprocess.STDOUT))
        deadline=time.monotonic()+timeout
        try:
            for process in processes:
                process.wait(timeout=max(.001,deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out=True
            for process in processes:
                if process.poll() is None:
                    process.terminate()
        finally:
            for process in processes:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill();process.wait()
            for handle in handles:
                handle.close()
        parties=[]
        for p in range(4):
            path=logs/f'party{p}.json'
            parties.append(json.loads(path.read_text()) if path.exists() else dict(party=p,passed=False,error='no worker result'))
    codes=[process.returncode for process in processes]
    passed=not timed_out and codes==[0]*4 and all(p.get('passed') and p.get('output')==expected.tolist() for p in parties)
    expected_digest=hashlib.sha256(expected.numpy().astype('<i8',copy=False).tobytes()).hexdigest()
    if expected.numel()>4096:
        for p in parties:
            if 'output' in p:
                value=torch.tensor(p.pop('output'),dtype=torch.int64)
                p.update(output_sha256=hashlib.sha256(value.numpy().astype('<i8',copy=False).tobytes()).hexdigest(),
                         output_shape=list(value.shape),exact_field_match=torch.equal(value,expected))
                (logs/f"party{p['party']}.json").write_text(json.dumps(p,indent=2)+'\n')
    online=max((p.get('online_ms',0) for p in parties),default=0)
    fp32=statistics.median(samples)
    fingerprints=_fingerprints()
    source_stable=source_before==fingerprints
    passed=passed and source_stable
    report=dict(passed=passed,return_codes=codes,timed_out=timed_out,device=device,
                completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=source_before,
                source_sha256_after=fingerprints,source_stable_during_run=source_stable,
                torch_version=str(torch.__version__),cuda_version=torch.version.cuda,
                gpu_name=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None,
                reproduction_command=[sys.executable,'-m','scripts.run_spdz_distributed','--device',device,
                                      '--m',str(m),'--k',str(k),'--n',str(n),'--mac-keys',str(mac_keys),
                                      '--timeout',str(timeout),'--output',str(output)]+(['--stack-macs'] if stack_macs else []),
                fp32_samples_ms=samples,
                matrix_shape=[m,k,n],field_prime=PRIME,mac_keys=mac_keys,stack_macs=stack_macs,fp32_tf32_enabled=False,fixture_seed=2301,
                preprocessing='Explicit trusted dealer; each worker receives only its local additive/MAC/key shares.',
                preprocessing_ms=preprocessing_ms,online_critical_path_ms=online,fp32_matmul_median_ms=fp32,
                online_to_fp32_ratio=online/fp32 if fp32 else None,
                global_payload_bytes=sum(p.get('payload_bytes',0) for p in parties),
                global_online_payload_bytes=sum(p.get('online_payload_bytes',0) for p in parties),
                input_critical_path_ms=max((p.get('input_protocol_ms',0) for p in parties),default=0),
                global_messages=sum(p.get('messages',0) for p in parties),parties=parties,
                expected=expected.tolist() if expected.numel()<=4096 else None,expected_sha256=expected_digest,expected_shape=list(expected.shape),
                exchange_diagnostic='Exchange wall time includes real TLS and peer synchronization. Remaining local time includes arithmetic, serialization, host/device staging and colocated GPU contention; it is not an ideal-network projection.',
                physical_network='Four real processes over localhost mutual TLS; includes wire transfer and synchronization.',
                wire_accounting='Counts application protocol bytes including broadcast framing, excluding TCP/TLS framing. Timings include actual TLS transport.',
                gpu_placement='All workers use the selected local GPU; not a four-GPU measurement.' if str(device).startswith('cuda') else 'CPU workers.',
                methodology='Owner-private masked inputs are timed separately. One fresh preprocessed matrix triple per role; checked masked openings and checked final output. Preprocessing, input sharing, TLS setup, arithmetic warmup and independent FP32 baseline are outside matmul online timing. The trusted dealer must not observe the input transcript.')
    output.write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path)
    parser.add_argument('--result',type=Path)
    parser.add_argument('--output',type=Path,default=Path('results/spdz_distributed.json'))
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--m',type=int,default=8)
    parser.add_argument('--k',type=int,default=16)
    parser.add_argument('--n',type=int,default=8)
    parser.add_argument('--mac-keys',type=int,default=5)
    parser.add_argument('--timeout',type=float,default=90)
    parser.add_argument('--stack-macs',action='store_true')
    args=parser.parse_args()
    if args.worker:
        worker(args.worker,args.result)
    else:
        report=run(args.output,device=args.device,m=args.m,k=args.k,n=args.n,mac_keys=args.mac_keys,timeout=args.timeout,stack_macs=args.stack_macs)
        print(json.dumps(report),flush=True)
        if not report['passed']:
            raise SystemExit(1)


if __name__=='__main__':
    main()
