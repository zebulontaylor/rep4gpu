"""Measure every client and MPC payload in the native SiLU proof bridge."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import torch

from rep4gpu.proofs.proof_activation import (
    CLIENT_ROLE, ProofClientNetwork, serve_proof_client, verify_proof_activation,
)
from rep4gpu.protocol.runtime import Party,make_seeds
from rep4gpu.proofs.silu_rational import rational_silu_plain
from rep4gpu.protocol.transport import IdealNetwork


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir',type=Path,default=Path('.local/work/silu_proof_full'))
    parser.add_argument('--backend',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--output',type=Path,default=Path('experiments/results/proofs/silu_proof.json'))
    args=parser.parse_args()
    torch.set_num_threads(1)
    public_path=args.artifact_dir/'public.json'
    public=json.loads(public_path.read_text())
    private=json.loads((args.artifact_dir/'private-blindings.json').read_text())
    fixture=json.loads((args.artifact_dir/'input.json').read_text())
    x=torch.tensor(fixture['x'],dtype=torch.int64)
    expected=torch.tensor(private['output_y'],dtype=torch.int64)
    if x.shape!=expected.shape or x.numel()!=public['count']:
        raise ValueError('native proof artifacts disagree on the input shape')
    if not torch.equal(expected,rational_silu_plain(x)):
        raise ValueError('native candidate differs from the production rational oracle')
    binary=Path('src/native/silu_proof/target/release/siv-silu-proof').resolve()
    if args.backend=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA backend requested but unavailable')
    network=IdealNetwork(timeout=900)
    clients=ProofClientNetwork(timeout=900)
    seeds=make_seeds()
    run_directory=args.artifact_dir/'verifier-run'
    def abort(error):
        network.abort(error)
        clients.abort(error)
    def helper():
        try:
            started=time.perf_counter()
            serve_proof_client(clients.endpoint(CLIENT_ROLE),public,private)
            return {'seconds':time.perf_counter()-started}
        except BaseException as error:
            abort(error)
            raise
    def role(party):
        try:
            endpoint=network.endpoint(party)
            known={i:seed for i,seed in seeds.items() if i!=party}
            if args.backend=='cuda':
                from rep4gpu.models.device_inference import DeviceInferenceParty
                from rep4gpu.protocol.device_transport import DeviceIdealEndpoint
                rt=DeviceInferenceParty(party,DeviceIdealEndpoint(endpoint,'cuda:0'),known,
                    device='cuda:0',bits=64,binary_bits=64,optimized=True,
                    gpu_open=True,rounding='floor',sparse_products=True)
            else:
                rt=Party(party,endpoint,known,bits=64,binary_bits=64)
            shared=rt.input(0,x if party==0 else None,x.shape)
            before=endpoint.bytes_sent
            started=time.perf_counter()
            metrics={}
            output=verify_proof_activation(rt,shared,clients.endpoint(party),binary=binary,
                                          workdir=run_directory/f'role{party}',metrics=metrics)
            if args.backend=='cuda':torch.cuda.synchronize()
            seconds=time.perf_counter()-started
            payload=endpoint.bytes_sent-before
            actual=rt.open(output).cpu()
            if not torch.equal(actual,expected):
                raise AssertionError(f'role {party} output differs from the native candidate')
            return {'role':party,'mpc_bytes':payload,'verification_seconds':seconds,**metrics}
        except BaseException as error:
            abort(error)
            raise
    started=time.perf_counter()
    with ThreadPoolExecutor(5) as pool:
        client_future=pool.submit(helper)
        futures=[pool.submit(role,party) for party in range(4)]
        roles=[future.result() for future in futures]
        helper_result=client_future.result()
    elapsed=time.perf_counter()-started
    mpc=sum(row['mpc_bytes'] for row in roles)
    client=clients.bytes_sent
    reference=torch.nn.functional.silu(x.double()/(1 << 24))
    error=expected.double()/(1 << 24)-reference
    source_paths=[Path('src/rep4gpu/proofs/proof_activation.py'),Path('src/rep4gpu/proofs/field_group_runtime.py'),
        Path('src/rep4gpu/proofs/ristretto.py'),Path('src/rep4gpu/proofs/silu_rational.py'),
        Path('experiments/benchmarks/proofs/silu_proof.py'),Path('src/native/silu_proof/src/main.rs'),
        Path('src/native/silu_proof/Cargo.toml'),Path('src/native/silu_proof/Cargo.lock'),
        Path('src/native/silu_proof/rust-toolchain.toml'),binary]
    source_paths.extend(sorted(Path('src/native/silu_proof/vendor').rglob('*.rs')))
    report={
        'elements':x.numel(),'backend':args.backend,'fractional_bits':24,
        'mpc_bytes':mpc,'client_bytes':client,'total_bytes':mpc+client,
        'under_decimal_10mb':mpc+client<10_000_000,
        'client_endpoints':[{'role':role,'bytes_sent':clients.endpoint(role).bytes_sent,
                            'messages_sent':clients.endpoint(role).messages_sent} for role in range(5)],
        'roles':roles,'all_party_outputs_match':True,'matches_native_candidate':True,
        'matches_production_rational_oracle':True,
        'max_abs_error':float(error.abs().max()),'rmse':float(error.square().mean().sqrt()),
        'raw_commitment_bytes':64*x.numel(),'raw_proof_bytes':sum(len(bytes.fromhex(chunk['proof'])) for chunk in public['chunks']),
        'proof_chunks':len(public['chunks']),'proof_chunk_size':public['chunk_size'],
        'proof_generation_included_in_timing':False,
        'colocated_elapsed_seconds':elapsed,'helper_seconds':helper_result['seconds'],
        'public_artifact_sha256':hashlib.sha256(public_path.read_bytes()).hexdigest(),
        'source_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        'methodology':'Actual immutable client frames plus four-role MPC transport payloads. '
                      'Includes public commitments/proofs sent separately to all four roles, '
                      'plaintext candidate delivery to owner zero, checked candidate sharing, '
                      'fresh public challenge, masked blinding shares, ring lifts and private '
                      'group binding. Initial activation sharing and final validation opening '
                      'are excluded. Timings measure colocated proof verification and MPC '
                      'execution; they are not WAN latency. Private blindings are never reported.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    # Omit the potentially long vendored-source list from console output.
    print(json.dumps({key:value for key,value in report.items() if key!='source_sha256'},indent=2))


if __name__=='__main__':main()
