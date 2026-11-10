"""Independent per-key random projections for authenticated field openings.

The ordering follows MP-SPDZ's MAMA MAC check: fix the opened values, generate
fresh public randomness, then commit and reveal the local compressed checks.
Each key gets its own coefficient vector. Reusing one vector across all keys
would let one projection cancellation bypass every MAC.

For a fixed nonzero opening error and independent uniform coefficients, a key
passes with probability at most 2/p - 1/p**2: its error projection vanishes with
probability 1/p; otherwise the adversary must guess an unknown uniform MAC-key
share. With K independent keys the bound is (2/p - 1/p**2)**K. A session with T
checks has the union bound T times this quantity, in addition to the assumptions
on commitments, SHAKE256 expansion, broadcast, and authenticated preprocessing.
This calculation is conditional on independent hidden keys supplied by secure
preprocessing; it is not a proof of a preprocessing protocol or a full MPC stack.
Aborted sessions must retire their keys and preprocessing.

Seven online coordinates give an online statistical union term of approximately
2**-146 for fewer than 2**64 checks under one key set. Native preprocessing and
the default configuration retain ten coordinates; the optional first-seven
selection is fixed before the session. See ``docs/online-mac-bounds.md``.

MP-SPDZ samples independent coordinates through ``FixedVec::almost_randomize``:
https://github.com/data61/MP-SPDZ/blob/master/Protocols/MAC_Check.hpp
https://github.com/data61/MP-SPDZ/blob/master/Math/FixedVec.h
"""

import hashlib
import json
import secrets
import struct

import numpy as np
import torch

from .spdz import PRIME, AuthenticationError
from .spdz_distributed import DistributedParty, ProtocolError, _hash, _pack


MIN_MAC_KEYS = 7
DOMAIN = b'SPDZ31-independent-MAC-v1'


def coefficient_row(seed, context, key, count):
    """Expand one public coefficient row, rejecting the non-field 31-bit word.

    SHAKE256 acts as the public pseudorandom stream. Rejection gives uniform
    field coefficients in the ideal-stream analysis; reducing biased words
    modulo the field would require a different soundness bound.
    """
    if not isinstance(seed, bytes) or len(seed) != 32:
        raise ValueError('coefficient seed must contain 32 bytes')
    if not isinstance(context, bytes):
        raise TypeError('coefficient context must be bytes')
    if not isinstance(key, int) or not 0 <= key < 2**32:
        raise ValueError('MAC key index must fit in 32 bits')
    if not isinstance(count, int) or not 0 <= count < 2**32:
        raise ValueError('coefficient count must fit in 32 bits')
    prefix = (DOMAIN + b'/coefficients' + struct.pack('<Q', len(context)) + context
              + seed + struct.pack('<IQ', key, count))
    result = np.empty(count, dtype=np.int64)
    filled, block = 0, 0
    while filled < count:
        # A separate block domain avoids reusing a SHAKE prefix on retries.
        needed = count - filled
        raw = hashlib.shake_256(prefix + struct.pack('<Q', block)).digest(4 * (needed + 16))
        candidates = np.frombuffer(raw, dtype='<u4') & PRIME
        accepted = candidates[candidates < PRIME][:needed]
        result[filled:filled + accepted.size] = accepted
        filled += accepted.size
        block += 1
    return torch.from_numpy(result)


def _entropy(party, method_name):
    method = getattr(party, method_name, None)
    value = secrets.token_bytes(32) if method is None else method()
    if not isinstance(value, bytes) or len(value) != 32:
        raise ValueError('compression entropy draws must contain 32 bytes')
    return value


def joint_seed(party, context):
    """Commit all seed contributions before revealing any of them.

    An honest 256-bit contribution makes the XOR unpredictable when corrupt
    parties commit. A participant can abort after seeing the coin; it cannot
    choose a replacement contribution for this operation.
    """
    contribution = _entropy(party, '_coin_contribution')
    salt = _entropy(party, '_coin_salt')
    binding = DOMAIN + b'/coin/' + context
    own = _hash(binding + struct.pack('<I', party.party) + salt + contribution)
    commitments = party.broadcast(own, label=binding + b'/commit')
    if any(len(value) != 32 for value in commitments.values()):
        raise ProtocolError('invalid coin commitment length')
    reveals = party.broadcast(salt + contribution, label=binding + b'/reveal')
    seed = bytearray(32)
    for sender in range(party.parties):
        reveal = reveals[sender]
        if len(reveal) != 64 or not secrets.compare_digest(
                _hash(binding + struct.pack('<I', sender) + reveal), commitments[sender]):
            raise ProtocolError('coin commitment opening failed')
        for i, byte in enumerate(reveal[32:]):
            seed[i] ^= byte
    return bytes(seed)


def compress_residual(residual, seed, context):
    """Return one canonical field sum per key without overflowing int64."""
    if residual.dtype != torch.int64 or residual.ndim != 2:
        raise ValueError('residual must be an int64 matrix indexed by key and value')
    keys, count = residual.shape
    if keys < MIN_MAC_KEYS:
        raise ValueError('compressed checks require at least seven independent MAC keys')
    if count >= 2**32:
        raise ValueError('compressed checks support fewer than 2**32 values')
    result = []
    for key in range(keys):
        coefficients = coefficient_row(seed, context, key, count).to(residual.device)
        # Each product fits in int64; reduction before summing keeps the total
        # below 2**63 for every supported count.
        products = (residual[key] * coefficients).remainder(PRIME)
        result.append(products.sum().remainder(PRIME))
    return torch.stack(result)


def check_opened(party, opened, macs, *, context):
    """Check values already fixed by a completed consistent broadcast.

    Call inside ``party.operation()``. ``context`` must bind the session,
    operation counter, shapes, field, number of keys, and public operation label.
    All parties commit their compressed residuals before any are revealed.
    """
    if party.mac_keys < MIN_MAC_KEYS:
        raise ValueError('compressed checks require at least seven independent MAC keys')
    if opened.dtype != torch.int64 or opened.ndim != 1 or macs.shape != (party.mac_keys, opened.numel()):
        raise ValueError('opened values and MAC dimensions mismatch')
    seed = joint_seed(party, context)
    residual = (macs - party.alpha_share[:, None] * opened).remainder(PRIME)
    compressed = compress_residual(residual, seed, context)
    raw, salt = _pack(compressed), _entropy(party, '_commit_salt')
    binding = DOMAIN + b'/residual/' + context
    commitment = _hash(binding + struct.pack('<I', party.party) + salt + raw)
    commitments = party.broadcast(commitment, label=binding + b'/commit')
    if any(len(value) != 32 for value in commitments.values()):
        raise ProtocolError('invalid compressed residual commitment length')
    reveals = party.broadcast(salt + raw, label=binding + b'/reveal')
    total = torch.zeros_like(compressed)
    for sender in range(party.parties):
        reveal = reveals[sender]
        if len(reveal) != 32 + 4 * party.mac_keys or not secrets.compare_digest(
                _hash(binding + struct.pack('<I', sender) + reveal), commitments[sender]):
            raise ProtocolError('compressed residual commitment opening failed')
        total = (total + party._unpack(reveal[32:], party.mac_keys)).remainder(PRIME)
    if torch.any(total != 0).item():
        raise AuthenticationError('compressed field MAC check failed')


def compressed_open_many(party, shares, *, label=b''):
    """Open matrices with independent per-key compressed authentication checks."""
    with party.operation():
        shares = tuple(shares)
        if not shares:
            raise ValueError('open_many requires at least one matrix')
        if party.mac_keys < MIN_MAC_KEYS:
            raise ValueError('compressed checks require at least seven independent MAC keys')
        for share in shares:
            party._validate(share)
        layout = json.dumps({'field': PRIME, 'dtype': 'uint32-le', 'keys': party.mac_keys,
                             'shapes': [list(x.values.shape) for x in shares]},
                            sort_keys=True, separators=(',', ':')).encode()
        context = (DOMAIN + party._session_hash + struct.pack('<Q', party.round)
                   + _hash(layout) + _hash(label))
        values = torch.cat([x.values.reshape(-1) for x in shares])
        opened = torch.zeros_like(values)
        received = party.broadcast(_pack(values), label=context + b'/values')
        for sender in range(party.parties):
            opened = (opened + party._unpack(received[sender], values.numel())).remainder(PRIME)
        macs = torch.cat([x.macs.reshape(party.mac_keys, -1) for x in shares], dim=1)
        check_opened(party, opened, macs, context=context)
        results, offset = [], 0
        for share in shares:
            size = share.values.numel()
            results.append(opened[offset:offset + size].reshape(share.values.shape))
            offset += size
        return results


class CompressedParty(DistributedParty):
    """Optional seven-key-or-more party retaining the full-vector golden path."""

    def open_many(self, shares, *, label=b''):
        return compressed_open_many(self, shares, label=label)


CHACHA_BACKEND = b'chacha20-per-position-rejection-v1'
MAX_CHACHA_ATTEMPTS = 64


def _coefficient_chacha_key(seed, context, key, count):
    if not isinstance(seed, bytes) or len(seed) != 32:
        raise ValueError('coefficient seed must contain 32 bytes')
    if not isinstance(context, bytes):
        raise TypeError('coefficient context must be bytes')
    if not isinstance(key, int) or not 0 <= key < 2**32:
        raise ValueError('MAC key index must fit in 32 bits')
    if not isinstance(count, int) or not 0 <= count < 2**32:
        raise ValueError('coefficient count must fit in 32 bits')
    return hashlib.sha256(DOMAIN + b'/' + CHACHA_BACKEND + struct.pack('<Q', len(context))
                          + context + seed + struct.pack('<IQ', key, count)).digest()


def coefficient_row_chacha(seed, context, key, count):
    """CPU oracle for the GPU's ChaCha20 per-position rejection stream.

    Position j uses word j from the original ChaCha20 layout with a 64-bit zero
    block counter and attempt number as the 64-bit nonce. A rejected position
    retries the same word in the next nonce's stream. Sixty-four rejections at
    one position abort instead of wrapping or substituting a biased coefficient.
    ``CHACHA_BACKEND`` must also be included in the public opening context.
    """
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
    key_bytes = _coefficient_chacha_key(seed, context, key, count)
    result = np.empty(count, dtype=np.int64)
    pending = np.ones(count, dtype=bool)
    for attempt in range(MAX_CHACHA_ATTEMPTS):
        if not pending.any():
            return torch.from_numpy(result)
        stream = Cipher(algorithms.ChaCha20(key_bytes, struct.pack('<QQ', 0, attempt)), mode=None).encryptor()
        raw = stream.update(bytes(4 * count)) + stream.finalize()
        candidate = np.frombuffer(raw, dtype='<u4') & PRIME
        accepted = pending & (candidate < PRIME)
        result[accepted] = candidate[accepted]
        pending &= ~accepted
    if pending.any():
        raise ProtocolError('coefficient rejection attempts exhausted')
    return torch.from_numpy(result)


def coefficient_rows_gpu(seed, context, keys, count, device, *, return_status=False):
    """Generate independent canonical CUDA coefficient rows with exact rejection.

    Returns int32 coefficients shaped ``(keys,count)``. With ``return_status``,
    also return a one-element CUDA int32 failure flag; the caller must include
    that flag in its fail-closed check before releasing any opened result. The
    default checks the flag here and raises on exhaustion. The public opening
    context must bind ``CHACHA_BACKEND`` at all roles.
    """
    import triton
    if not isinstance(keys, int) or keys < MIN_MAC_KEYS or keys >= 2**32:
        raise ValueError('provide at least seven independent MAC keys')
    device = torch.device(device)
    if device.type != 'cuda':
        raise ValueError('coefficient rows require a CUDA device')
    from .spdz_metadata import upload_bytes
    key_data = b''.join(_coefficient_chacha_key(seed, context, key, count) for key in range(keys))
    key_chunks = [upload_bytes(key_data[start:start + 4096], device)
                  for start in range(0, len(key_data), 4096)]
    key_tensor = (key_chunks[0] if len(key_chunks) == 1 else torch.cat(key_chunks)).view(torch.uint32).reshape(keys, 8)
    output = torch.empty((keys, count), dtype=torch.int32, device=device)
    failure = torch.zeros(1, dtype=torch.int32, device=device)
    if count:
        kernel = _coefficient_gpu_kernel()
        with torch.cuda.device(device):
            kernel[(triton.cdiv(triton.cdiv(count, 16), 128), keys)](
                key_tensor, output, failure, count, MAX_CHACHA_ATTEMPTS, 128, num_warps=4)
    if return_status:
        return output, failure
    if failure.item():
        raise ProtocolError('coefficient rejection attempts exhausted')
    return output


_GPU_COEFFICIENT_KERNEL = None


def _coefficient_gpu_kernel():
    """Build and cache the CUDA coefficient kernel on first use."""
    global _GPU_COEFFICIENT_KERNEL
    if _GPU_COEFFICIENT_KERNEL is not None:
        return _GPU_COEFFICIENT_KERNEL
    import triton
    import triton.language as tl
    from .gpu_prg_batch import _block_words

    @triton.jit
    def coefficients(KEY, OUT, FAILURE, COUNT: tl.constexpr, ATTEMPTS: tl.constexpr,
                     BLOCK: tl.constexpr):
        row = tl.program_id(1)
        block = tl.program_id(0).to(tl.uint64)*BLOCK + tl.arange(0, BLOCK).to(tl.uint64)
        pending = tl.full((BLOCK,), 0xffff, tl.uint32)
        attempt = tl.full((), 0, tl.uint64)
        while (tl.sum((pending != 0).to(tl.int32), 0) != 0) & (attempt < ATTEMPTS):
            words = _block_words(KEY + row*8, attempt, block, BLOCK)
            for word in tl.static_range(16):
                position = block*16 + word
                value = words[word] & 0x7fffffff
                accepted = (value < 0x7fffffff) | (position >= COUNT)
                active = ((pending >> word) & 1) != 0
                tl.store(OUT + row*COUNT + position, value.to(tl.int32),
                         (position < COUNT) & active & accepted)
                pending &= ~(accepted.to(tl.uint32) << word)
            attempt += 1
        if tl.sum((pending != 0).to(tl.int32), 0) != 0:
            # Define every output even on exhaustion; callers can combine the
            # failure flag with later GPU validation without reading old memory.
            for word in tl.static_range(16):
                position = block*16 + word
                failed = ((pending >> word) & 1) != 0
                tl.store(OUT + row*COUNT + position, 0, (position < COUNT) & failed)
            tl.atomic_max(FAILURE, 1)

    _GPU_COEFFICIENT_KERNEL = coefficients
    return coefficients


from .spdz_device import DeviceDistributedParty


class DeviceCompressedParty(DeviceDistributedParty):
    """GPU authenticated openings with seven or more independent MAC projections.

    Only the committed public coin uses CPU byte broadcasts. Value shares,
    challenge coefficients, weighted checks, commitments, and residual reveals
    remain on the GPU. The underlying transport's device/host staging policy
    still determines how frames move between roles.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with self.operation():
            if self.mac_keys < MIN_MAC_KEYS:
                raise ValueError('compressed checks require at least seven independent MAC keys')
            self.crypto_variant = 'SPDZ31G1/BLAKE3/independent-ChaCha-MAC'

    def _coin_contribution(self):
        return secrets.token_bytes(32)

    def _coin_salt(self):
        return secrets.token_bytes(32)

    def open_many(self, shares, *, label=b''):
        from .spdz_device import _sum_values
        import triton
        with self.operation():
            shares = tuple(shares)
            if not shares:
                raise ValueError('open_many requires at least one matrix')
            if self.mac_keys < MIN_MAC_KEYS:
                raise ValueError('compressed checks require at least seven independent MAC keys')
            self._ensure_canonical(shares)
            layout = json.dumps({'field': PRIME, 'dtype': 'uint32-le', 'keys': self.mac_keys,
                                 'shapes': [list(x.values.shape) for x in shares]},
                                sort_keys=True, separators=(',', ':')).encode()
            context = (b'SPDZ31G-independent-MAC-v1' + self._session_hash
                       + struct.pack('<Q', self.round) + _hash(layout) + _hash(label)
                       + _hash(CHACHA_BACKEND))
            count = sum(x.values.numel() for x in shares)
            if count >= 2**31:
                raise ValueError('device compressed checks support fewer than 2**31 values')
            values = self.broadcast_device(self._packed_values(shares), label=context + b'/values',
                                           canonical_byte_offset=0)
            opened = torch.empty(count, dtype=torch.int64, device=self.device)
            if count:
                _sum_values[(triton.cdiv(count, 256),)](
                    *(values[p] for p in range(4)), opened, count, 256)
            macs = torch.cat([x.macs.reshape(self.mac_keys, -1) for x in shares], dim=1).contiguous()
            check_opened_device(self, opened, macs, context=context)
            result, offset = [], 0
            for share in shares:
                size = share.values.numel()
                result.append(opened[offset:offset + size].reshape(share.values.shape))
                offset += size
            return result



def _check_device_compressed(party, context, local_reduction):
    """Run the common committed coin and compressed-check reveal protocol.

    Call inside ``party.operation()``, after all checked values and residual
    buffers are fixed. ``local_reduction(seed)`` returns canonical int32 checks
    and a CUDA int32 failure flag that is preserved through final aggregation.
    """
    from .gpu_hash import digest, digest_batch
    from .spdz_device import _join, _mac_zero
    from .spdz_metadata import upload_bytes
    import triton
    seed = joint_seed(party, context)
    compressed, failure = local_reduction(seed)
    raw = compressed.view(torch.uint8).reshape(-1)
    salt = _entropy(party, '_commit_salt')
    salt_gpu = upload_bytes(salt, party.device)
    binding = DOMAIN + b'/residual/' + context
    own_binding = upload_bytes(binding + struct.pack('<I', party.party), party.device)
    commitment = digest(torch.cat((own_binding, salt_gpu, raw)), block_tree=True)
    commits = party.broadcast_device(commitment, label=binding + b'/commit')
    reveal = torch.cat((salt_gpu, raw))
    reveals = party.broadcast_device(reveal, label=binding + b'/reveal', canonical_byte_offset=32)
    prefix = upload_bytes(b''.join(binding + struct.pack('<I', p) for p in range(4)),
                          party.device).reshape(4, own_binding.numel())
    joined = torch.empty((4, own_binding.numel() + reveal.numel()), dtype=torch.uint8, device=party.device)
    _join[(triton.cdiv(joined.shape[1], 256), 4)](
        prefix, *(reveals[p] for p in range(4)), joined, own_binding.numel(), reveal.numel(), 256)
    tags = digest_batch(joined, block_tree=True)
    if not torch.equal(tags, torch.stack([commits[p] for p in range(4)])):
        raise ProtocolError('device compressed residual commitment opening failed')
    _mac_zero[(triton.cdiv(party.mac_keys, 256),)](
        *(reveals[p] for p in range(4)), failure, party.mac_keys, 256)
    if failure.item():
        raise AuthenticationError('device compressed field MAC check failed')


def check_opened_device(party, opened, macs, *, context):
    """Check one fixed opening without materializing its full residual matrix."""
    from .field_fusion import field_mac_check
    def local_reduction(seed):
        coefficients, failure = coefficient_rows_gpu(
            seed, context, party.mac_keys, opened.numel(), party.device, return_status=True)
        compressed = field_mac_check(macs, opened, party.alpha_share, coefficients, packed=True)
        return compressed, failure
    _check_device_compressed(party, context, local_reduction)


def check_residual_batches_device(party, residuals, *, context):
    """Check immutable residual segments using one coin and one commitment.

    Ordered segment domains give every key an independent coefficient for every
    queued value. Reducing one segment at a time avoids a concatenated residual
    buffer and limits coefficient storage to the largest pending opening.
    """
    from .field_weighted_accumulate import field_weighted_accumulate
    residuals = tuple(residuals)
    if not residuals:
        raise ValueError('a residual batch must contain at least one segment')
    for residual in residuals:
        if residual.dtype not in (torch.int32, torch.int64) or residual.ndim != 2 or residual.shape[0] != party.mac_keys:
            raise ValueError('residual segments must have shape (keys,values)')
        if residual.device != party.device or not residual.is_contiguous():
            raise ValueError('residual segments must be contiguous on the party device')
    def local_reduction(seed):
        total = torch.zeros(party.mac_keys, dtype=torch.int32, device=party.device)
        failure = torch.zeros(1, dtype=torch.int32, device=party.device)
        for index, residual in enumerate(residuals):
            segment_context = context + b'/segment/' + struct.pack('<Q', index)
            coefficients, rejected = coefficient_rows_gpu(
                seed, segment_context, party.mac_keys, residual.shape[1], party.device, return_status=True)
            field_weighted_accumulate(residual, coefficients, total, failure, rejected)
        return total, failure
    _check_device_compressed(party, context, local_reduction)
