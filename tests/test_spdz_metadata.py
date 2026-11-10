import hashlib
import struct
import gc
import pytest
import torch
from rep4gpu.spdz_metadata import upload_bytes,headers


@pytest.mark.parametrize('size',[0,1,32,123,492,1024])
def test_async_metadata_upload_exact(size):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    data=bytes((i*71+19)%256 for i in range(size))
    actual=upload_bytes(data,'cuda')
    assert actual.cpu().numpy().tobytes()==data


def test_async_metadata_storage_survives_pending_copy():
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    stream=torch.cuda.Stream()
    values=[]
    with torch.cuda.stream(stream):
        torch.cuda._sleep(5_000_000)
        for i in range(32):
            values.append(upload_bytes(bytes([i])*1024,'cuda'))
        gc.collect()
    stream.synchronize()
    assert all(value.cpu().numpy().tobytes()==bytes([i])*1024 for i,value in enumerate(values))


@pytest.mark.parametrize('round_id,size',[(0,0),(1,32),(7,123456),(2**32+3,2**32+17),(2**64-1,2**64-1)])
@pytest.mark.parametrize('stage',[b'V',b'E'])
def test_gpu_header_matches_struct_oracle(round_id,size,stage):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    session=hashlib.blake2b(b'public-test-session',digest_size=32).digest()
    label=hashlib.blake2b(b'public-test-label'+stage,digest_size=32).digest()
    actual=headers(upload_bytes(session,'cuda'),label,round_id,size,stage).cpu().numpy().tobytes()
    expected=b''.join(b'SPDZ31G1'+session+struct.pack('<QI',round_id,p)+stage+label+struct.pack('<Q',size) for p in range(4))
    assert actual==expected


def test_metadata_rejects_bulk_or_cpu_payloads():
    with pytest.raises(ValueError,match='4096'):upload_bytes(bytes(4097),'cuda')
    with pytest.raises(ValueError,match='CUDA'):upload_bytes(b'metadata','cpu')


@pytest.mark.parametrize('size',[0,1,32,128,1025,65539])
@pytest.mark.parametrize('stage',[b'V',b'E'])
def test_fused_frame_struct_bytes_and_owned_storage(size,stage):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from rep4gpu.spdz_metadata import header_frame
    session=hashlib.blake2b(b'frame-session',digest_size=32).digest()
    label=hashlib.blake2b(b'frame-label',digest_size=32).digest()
    session_gpu=upload_bytes(session,'cuda')
    data=bytes((i*37+11)%256 for i in range(size))
    payload=torch.tensor(list(data),dtype=torch.uint8,device='cuda')
    expected_headers=b''.join(b'SPDZ31G1'+session+struct.pack('<QI',19,p)+stage+label+struct.pack('<Q',size) for p in range(4))
    for party in range(4):
        actual,frame=header_frame(session_gpu,label,19,stage,party,payload)
        assert actual.cpu().numpy().tobytes()==expected_headers
        assert frame.cpu().numpy().tobytes()==expected_headers[party*93:(party+1)*93]+data
        old_headers,old_frame=actual.clone(),frame.clone()
        replacement=torch.full_like(payload,91)
        header_frame(session_gpu,label,20,stage,party,replacement)
        assert torch.equal(actual,old_headers) and torch.equal(frame,old_frame)
        if size:
            assert frame.untyped_storage().data_ptr()!=payload.untyped_storage().data_ptr()
