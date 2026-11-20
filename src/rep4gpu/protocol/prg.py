"""Cryptographic correlated-mask generators with explicit draw counters."""
import hashlib
import struct
from functools import lru_cache


@lru_cache(maxsize=16)
def _aes_key(seed):
    return hashlib.sha256(b'Rep4GPU-AES256-key-v1' + seed).digest()


def random_bytes(seed, counter, count, *, ring_bits=64, binary=False, binary_bits=64, backend='shake'):
    """Generate one mask draw; the caller must never reuse a seed/counter pair.

    SHAKE preserves the reference runtime's byte stream. AES uses a 64-bit
    draw nonce followed by a 64-bit domain/count field. Its counter blocks
    cannot cross into another draw nonce within the accepted count range.
    """
    if ring_bits not in (16, 32, 64):
        raise ValueError('ring bits must be 16, 32, or 64')
    if not 0 <= counter < (1 << 64):
        raise ValueError('PRG draw counter exhausted')
    if not 0 <= count < (1 << 60):
        raise ValueError('PRG element count must fit in 60 bits')
    if len(seed) < 32:
        raise ValueError('PRG seed must contain at least 256 bits')
    if binary_bits not in (32, 64) or (binary_bits == 32 and ring_bits not in (16, 32)):
        raise ValueError('32-bit binary words require a 16-bit or 32-bit ring')
    width = binary_bits if binary else ring_bits
    size = count * (width // 8)
    if backend == 'shake':
        context = b'Rep4GPU-mask-v1' + struct.pack('<QQ', counter, count)
        if ring_bits == 32:
            context += bytes([width])
        if binary and binary_bits == 32:
            context += b'B32'
        if ring_bits == 16:
            context += b'R16' + bytes((width, int(binary)))
        return hashlib.shake_256(seed + context).digest(size)
    if backend != 'aes':
        raise ValueError('PRG backend must be shake or aes')
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    domain = 4 if binary and binary_bits == 32 else (2 if ring_bits == 32 else 0) + int(binary)
    if ring_bits == 16:
        # Domains 0..4 retain the established ring32/ring64 byte streams.
        domain = (6 if binary_bits == 32 else 7) if binary else 5
    iv = struct.pack('>QQ', counter, (domain << 60) | count)
    encryptor = Cipher(algorithms.AES(_aes_key(seed)), modes.CTR(iv)).encryptor()
    return encryptor.update(bytes(size)) + encryptor.finalize()
