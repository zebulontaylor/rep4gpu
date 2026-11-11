import struct

import pytest
import torch

from rep4gpu.gpu.crypto.gpu_prg import DomainKeys, generate, key_bytes
from rep4gpu.gpu.crypto.gpu_prg_batch import generate_rows
from rep4gpu.protocol.runtime import SCHEDULE

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


@pytest.mark.parametrize("bits",[32,64])
@pytest.mark.parametrize("party",range(4))
@pytest.mark.parametrize("count",[0,1,17,1025])
@pytest.mark.parametrize("packed",[False,True])
def test_mask_schedule_matches_individual_draws(bits,party,count,packed):
    caches={excluded:DomainKeys(bytes([excluded])*32) for excluded in range(4) if excluded!=party}
    counters={excluded:2**64-3 for excluded in caches}
    keys,nonces,expected=[],[],[]
    for _,_,receiver,_ in SCHEDULE:
        if receiver==party:
            keys.append(None)
            nonces.append(None)
            expected.append(torch.zeros(count,dtype=torch.int64,device="cuda"))
        else:
            key=caches[receiver].key(bytes((bits,receiver)))
            nonce=counters[receiver]
            keys.append(key)
            nonces.append(nonce)
            expected.append(generate(key,nonce,count,bits))
            counters[receiver]+=1
    result=generate_rows(keys,nonces,count,bits,packed=packed)
    assert result.dtype == (torch.uint32 if packed and bits == 32 else torch.int64)
    torch.testing.assert_close(result.to(torch.int64),torch.stack(expected),rtol=0,atol=0)
    for receiver in counters:
        draws=sum(receiver==r for _,_,r,_ in SCHEDULE)
        assert counters[receiver]==2**64-3+draws


@pytest.mark.parametrize("bits",[32,64])
def test_rows_against_cryptography(bits):
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives.ciphers import Cipher,algorithms

    cache=DomainKeys(bytes(range(32)))
    domains=(b"ring",b"binary",b"wire",b"outside")
    keys=[cache.key(d) for d in domains]
    nonces=[0,1,0x123456789abcdef,2**64-1]
    actual=generate_rows(keys,nonces,1003,bits)
    for row,(domain,nonce) in enumerate(zip(domains,nonces)):
        enc=Cipher(algorithms.ChaCha20(key_bytes(bytes(range(32)),domain),struct.pack("<QQ",0,nonce)),None).encryptor()
        expected=enc.update(bytes(1003*(bits//8)))+enc.finalize()
        value=actual[row].cpu().numpy().astype("<u4" if bits==32 else "<i8",copy=False).tobytes()
        assert value==expected


def test_row_validation():
    key=DomainKeys(bytes(32)).key(b"test")
    for keys,nonces in (([],[]),([None],[None]),([key],[None]),([None,key],[1,0])):
        with pytest.raises((ValueError,TypeError)):
            generate_rows(keys,nonces,1)
    with pytest.raises(ValueError,match="64 bits"):
        generate_rows([key],[2**64],1)
