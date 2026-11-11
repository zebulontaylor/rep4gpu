"""External-client accounting counts framed bytes on every delivered link."""
import pytest

from rep4gpu.proof_activation import CLIENT_ROLE, ProofClientNetwork


def test_client_channels_count_actual_frames_in_both_directions():
    network=ProofClientNetwork()
    client=network.endpoint(CLIENT_ROLE)
    for role in range(4): client.send(role,b'publicproof')
    for role in range(4):
        endpoint=network.endpoint(role)
        assert endpoint.receive(CLIENT_ROLE)==b'publicproof'
        endpoint.send(CLIENT_ROLE,b'seed')
    assert [client.receive(role) for role in range(4)]==[b'seed']*4
    assert client.bytes_sent==4*(12+len(b'publicproof'))
    assert network.bytes_sent==4*(12+len(b'publicproof'))+4*(12+len(b'seed'))


def test_client_channels_reject_nonbytes_and_party_to_party_links():
    network=ProofClientNetwork()
    with pytest.raises(TypeError): network.endpoint(4).send(0,bytearray(b'bad'))
    with pytest.raises(ValueError): network.endpoint(0).send(1,b'bad')
    with pytest.raises(ValueError): network.endpoint(4).receive(4)


def test_public_bundle_roundtrip_counts_binary_points():
    from rep4gpu.proof_activation import encode_public_bundle,decode_public_bundle
    public={'version':'v1','count':1,'chunk_size':1,
            'commitments_x':['11'*32],'commitments_y':['22'*32],
            'chunks':[{'start':0,'count':1,'proof':'abcd'}]}
    encoded=encode_public_bundle(public)
    assert decode_public_bundle(encoded)==public
    assert len(encoded)==1+2+24+64+24+2
    with pytest.raises(ValueError):decode_public_bundle(encoded+b'bad')


@pytest.mark.parametrize('device',['cpu','cuda:0'])
@pytest.mark.parametrize('tamper',['none','input','output','input_high','output_high','blindings'])
def test_native_proof_activation_private_binding(tmp_path,tamper,device):
    import json
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    import subprocess
    import torch
    from rep4gpu.proof_activation import serve_proof_client,verify_proof_activation
    from test_silu_output_witness import run_four,secret
    if device.startswith('cuda') and not torch.cuda.is_available():pytest.skip('CUDA required')
    binary=Path('native/silu_proof/target/release/siv-silu-proof').resolve()
    if not binary.exists():pytest.skip('native SiLU proof binary required')
    candidate=torch.tensor([-16*(1 << 24),0,25231595,16*(1 << 24)])
    input_path=tmp_path/'input.json';input_path.write_text(json.dumps({'x':candidate.tolist()}))
    artifacts=tmp_path/'prover'
    subprocess.run([str(binary),'prove',str(input_path),str(artifacts),'4'],check=True,capture_output=True)
    public=json.loads((artifacts/'public.json').read_text())
    private=json.loads((artifacts/'private-blindings.json').read_text())
    expected=torch.tensor(private['output_y'])
    actual=candidate.clone()
    if tamper=='input':actual[1]+=1
    if tamper=='input_high':actual[1]=-(1 << 63)
    if tamper=='output':private['output_y'][1]+=1
    if tamper=='output_high':private['output_y'][1]=-(1 << 63)
    if tamper=='blindings':private['blindings_x'][0]='00'*32
    network=ProofClientNetwork(timeout=30)
    def work(party):
        x=secret(party,actual)
        if tamper!='none':
            with pytest.raises(ValueError,match='binding failed'):
                verify_proof_activation(party,x,network.endpoint(party.party),binary=binary,
                                        workdir=tmp_path/f'role{party.party}')
            return None
        y=verify_proof_activation(party,x,network.endpoint(party.party),binary=binary,
                                  workdir=tmp_path/f'role{party.party}')
        return party.open(y).cpu()
    with ThreadPoolExecutor(1) as pool:
        client=pool.submit(serve_proof_client,network.endpoint(CLIENT_ROLE),public,private)
        results=run_four(work,device)
        client.result()
    if tamper=='none':assert all(torch.equal(result,expected) for result in results)
    assert network.bytes_sent>4*64*len(candidate)


def test_abort_wakes_waiting_client_immediately():
    from concurrent.futures import ThreadPoolExecutor
    from rep4gpu.proof_activation import ProofClientAborted
    network=ProofClientNetwork(timeout=120)
    with ThreadPoolExecutor(1) as pool:
        waiting=pool.submit(network.endpoint(CLIENT_ROLE).receive,0)
        network.abort(ValueError('invalid proof'))
        with pytest.raises(ProofClientAborted):waiting.result(timeout=1)


@pytest.mark.parametrize('failure',['invalid_proof','bundle_disagreement'])
def test_proof_failure_aborts_client_without_seed_timeout(tmp_path,failure):
    import json
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    import subprocess
    import time
    import torch
    from rep4gpu.proof_activation import (
        ProofClientAborted,encode_public_bundle,serve_proof_client,verify_proof_activation,
    )
    from test_silu_output_witness import run_four,secret
    from rep4gpu.transport import NetworkAbortedError
    binary=Path('native/silu_proof/target/release/siv-silu-proof').resolve()
    if not binary.exists():pytest.skip('native SiLU proof binary required')
    input_path=tmp_path/'input.json';input_path.write_text(json.dumps({'x':[0]}))
    artifacts=tmp_path/'prover'
    subprocess.run([str(binary),'prove',str(input_path),str(artifacts),'1'],check=True,capture_output=True)
    public=json.loads((artifacts/'public.json').read_text())
    private=json.loads((artifacts/'private-blindings.json').read_text())
    if failure=='invalid_proof':
        proof=bytearray.fromhex(public['chunks'][0]['proof']);proof[0]^=1
        public['chunks'][0]['proof']=proof.hex()
    network=ProofClientNetwork(timeout=120)
    def client_work():
        endpoint=network.endpoint(CLIENT_ROLE)
        if failure=='invalid_proof':return serve_proof_client(endpoint,public,private)
        packet=encode_public_bundle(public)
        for role in range(4):
            if role==3:
                changed=dict(public);changed['version']=public['version']+'bad'
                endpoint.send(role,encode_public_bundle(changed))
            else:endpoint.send(role,packet)
        endpoint.send(0,bytes(8))
        endpoint.receive(0)
    def work(party):
        x=secret(party,torch.tensor([0]))
        return verify_proof_activation(party,x,network.endpoint(party.party),binary=binary,
                                       workdir=tmp_path/f'role{party.party}')
    began=time.monotonic()
    with ThreadPoolExecutor(1) as pool:
        client=pool.submit(client_work)
        with pytest.raises((ValueError,subprocess.CalledProcessError,NetworkAbortedError,ProofClientAborted)):
            run_four(work)
        with pytest.raises(ProofClientAborted):client.result(timeout=2)
    assert time.monotonic()-began<10


def test_public_codec_rejects_projection_overflow_count_without_allocation():
    import struct
    from rep4gpu.proof_activation import encode_public_bundle,decode_public_bundle
    public={'version':'v1','count':1 << 35,'chunk_size':1,
            'commitments_x':[],'commitments_y':[],'chunks':[]}
    with pytest.raises(ValueError,match='metadata'):encode_public_bundle(public)
    packet=b'\x02v1'+struct.pack('<QQQ',1 << 35,1,0)
    with pytest.raises(ValueError,match='count'):decode_public_bundle(packet)
