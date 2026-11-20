import hashlib
import struct

import pytest
import torch

from rep4gpu.gpu.crypto.gpu_prg import DOMAIN_TAG, DomainKeys, _generate, generate, generate_bytes, key_bytes


def crypto_stream(key,nonce,size,block=0):
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms

    encryptor=Cipher(algorithms.ChaCha20(key,struct.pack("<QQ",block,nonce)),None).encryptor()
    return encryptor.update(bytes(size))+encryptor.finalize()


def device_key(key):
    return torch.tensor(struct.unpack("<8I",key),device="cuda",dtype=torch.uint32)


def encoded_words(value,bits):
    data=value.cpu().numpy()
    return data.astype("<u4" if bits==32 else "<i8",copy=False).tobytes()


def test_domain_key_framing():
    seed=bytes(range(32))
    domain=bytes((32,32,1,32,3))
    expected=hashlib.sha256(seed+DOMAIN_TAG+struct.pack("<Q",len(domain))+domain).digest()
    assert key_bytes(seed,domain)==expected
    assert key_bytes(seed,"arithmetic")==key_bytes(seed,b"arithmetic")
    assert key_bytes(seed,b"arithmetic")!=key_bytes(seed,b"boolean")
    assert key_bytes(seed,b"arithmetic")!=key_bytes(bytes(reversed(seed)),b"arithmetic")
    with pytest.raises(ValueError,match="256 bits"):
        key_bytes(bytes(31),domain)
    with pytest.raises(TypeError,match="bytes"):
        key_bytes(32,domain)


requires_gpu=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


@requires_gpu
@pytest.mark.parametrize("bits",[32,64])
@pytest.mark.parametrize("count",[0,1,7,8,9,15,16,17,31,32,33,129,1025])
@pytest.mark.parametrize("nonce",[0,1,0x0123456789abcdef,2**64-1])
def test_cryptography_words(bits,count,nonce):
    pytest.importorskip("cryptography")
    key=bytes(range(32))
    result=generate(device_key(key),nonce,count,bits)
    assert result.dtype==torch.int64
    assert result.shape==(count,)
    assert encoded_words(result,bits)==crypto_stream(key,nonce,count*(bits//8))
    if bits==32:
        assert torch.all((result>=0)&(result<2**32))


@requires_gpu
@pytest.mark.parametrize("size",[0,1,3,4,7,8,9,63,64,65,127,128,129,1003])
def test_byte_boundaries(size):
    pytest.importorskip("cryptography")
    key=bytes(reversed(range(32)))
    result=generate_bytes(device_key(key),2**64-1,size)
    assert result.dtype==torch.uint8
    assert result.cpu().numpy().tobytes()==crypto_stream(key,2**64-1,size)


@requires_gpu
def test_rfc8439_appendix_a1_vector1():
    # https://www.rfc-editor.org/rfc/rfc8439.html#appendix-A.1
    expected=bytes.fromhex(
        "76b8e0ada0f13d90405d6ae55386bd28"
        "bdd219b8a08ded1aa836efcc8b770dc7"
        "da41597c5157488d7724e03fb8d84a37"
        "6a43b8f41518a11cc387b669b2ee6586")
    key=device_key(bytes(32))
    assert generate_bytes(key,0,64).cpu().numpy().tobytes()==expected


@requires_gpu
@pytest.mark.parametrize("block",[2**32-1,2**32,2**64-2])
def test_original_counter_high_word_and_carry(block):
    pytest.importorskip("cryptography")
    key=bytes(range(31,-1,-1))
    nonce=0xfedcba9876543210
    # One CPU block per cipher also works with cryptography's IETF overflow guard.
    expected=b"".join(crypto_stream(key,nonce,64,block+i) for i in range(2))
    out=_generate(device_key(key),nonce,16,64,block_start=block)
    assert encoded_words(out,64)==expected


@requires_gpu
def test_multiple_domain_keys_and_fresh_draws():
    pytest.importorskip("cryptography")
    seed=bytes(range(32))
    cache=DomainKeys(seed,"cuda")
    streams=[]
    for domain in (b"ring32",b"ring64",b"boolean32",b"boolean64"):
        key=cache.key(domain)
        assert key is cache.key(domain)
        assert key.dtype==torch.uint32 and key.shape==(8,)
        for nonce in (0,1,2):
            actual=generate_bytes(key,nonce,139).cpu().numpy().tobytes()
            assert actual==crypto_stream(key_bytes(seed,domain),nonce,139)
            streams.append(actual)
    assert len(set(streams))==len(streams)


@requires_gpu
def test_invalid_parameters_and_counter_limits():
    key=device_key(bytes(32))
    for nonce in (-1,2**64):
        with pytest.raises(ValueError,match="nonce"):
            generate(key,nonce,1)
    with pytest.raises(ValueError,match="nonnegative"):
        generate(key,0,-1)
    with pytest.raises(ValueError,match="bits"):
        generate(key,0,1,24)
    with pytest.raises(ValueError,match=r"2\*\*32"):
        generate(key,0,2**32*8,64)
    with pytest.raises(ValueError,match="counter"):
        _generate(key,0,16,64,block_start=2**64-1)
    with pytest.raises(ValueError,match="uint32"):
        generate(torch.zeros(8,dtype=torch.int64,device="cuda"),0,1)
