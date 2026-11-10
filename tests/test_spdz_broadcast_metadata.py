"""Struct oracles for the two-stage broadcast allocation."""
import hashlib
import struct
import pytest
import torch
from rep4gpu.spdz_metadata import upload_bytes
from rep4gpu.spdz_broadcast_metadata import prepare_broadcast


@pytest.mark.parametrize('size',[0,1,32,128,927,1025,65539])
def test_prepared_broadcast_headers_and_owned_frames(size):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    session=hashlib.blake2b(b'paired-session',digest_size=32).digest()
    label=hashlib.blake2b(b'paired-label',digest_size=32).digest()
    round_id=2**64-1
    data=bytes((i*19+73)%256 for i in range(size))
    payload=torch.tensor(list(data),dtype=torch.uint8,device='cuda')
    session_gpu=upload_bytes(session,'cuda')
    def header(p,stage,n):
        return b'SPDZ31G1'+session+struct.pack('<QI',round_id,p)+stage+label+struct.pack('<Q',n)
    for p in range(4):
        vh,vf,eh,ef=prepare_broadcast(session_gpu,label,round_id,p,payload)
        assert vh.cpu().numpy().tobytes()==b''.join(header(j,b'V',size) for j in range(4))
        assert eh.cpu().numpy().tobytes()==b''.join(header(j,b'E',128) for j in range(4))
        assert vf.cpu().numpy().tobytes()==header(p,b'V',size)+data
        assert ef[:93].cpu().numpy().tobytes()==header(p,b'E',128)
        # This slice is the digest output destination; filling it preserves
        # both stages' headers and the independently owned input frame.
        ef[93:].fill_(37)
        expected_echo=header(p,b'E',128)+bytes([37])*128
        assert ef.cpu().numpy().tobytes()==expected_echo
        prepare_broadcast(session_gpu,label,0,p,torch.empty(0,dtype=torch.uint8,device='cuda'))
        assert ef.cpu().numpy().tobytes()==expected_echo
        assert vf.cpu().numpy().tobytes()==header(p,b'V',size)+data
        assert vf.untyped_storage().data_ptr()!=payload.untyped_storage().data_ptr()
