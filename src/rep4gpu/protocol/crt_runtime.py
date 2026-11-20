"""Checked 4-party GPU CRT prime-field runtime with 1-product polynomial gates.

Every arithmetic share lives in product_l F_{p_l} as three uint8 CUDA tensors of
shape (len(primes), *logical_shape). Multiplication, 1D convolution, and fused
square-sum-pooling use the 1-product quadratic polynomial gate with a 2-collector
active-security broadcast:
  - Round 1: P0, P3 send Y(0), Y(3) to collectors P1, P2; P1 <-> P2 exchange Y(1), Y(2).
  - Local (P1, P2): verify third difference Y(3)-3Y(2)+3Y(1)-Y(0) == 0 mod p_l
    BEFORE Round 2; compute leading coefficient C = (Y(2)-2Y(1)+Y(0)) * inv2 mod p_l,
    D_all = BLAKE3(Y(0)||Y(1)||Y(2)||Y(3)), and D_C = BLAKE3(C).
  - Round 2: P1 sends D_all to P0, P2 and C || D_all to P3;
             P2 sends D_all to P0, P1 and D_C || D_all to P3.
  - All parties verify matching D_all (and P3 verifies BLAKE3(C) == D_C) before
    any dependent send or output release.
"""
from __future__ import annotations

import hashlib
import math
import secrets
import struct
import numpy as np
import torch

from ..models.phase_graph import PhaseGraphParty
from .runtime import SCHEDULE, Share
from ..gpu.crypto.gpu_hash import digest, digest_batch
from ..gpu.matmul.crt_conv import (
    DEFAULT_CRT_PRIMES,
    PreparedCrtWeight,
    crt_add_mod,
    crt_conv1d_local,
    crt_eval_linear,
    crt_input_correction,
    crt_input_masks,
    crt_matmul_local,
    crt_poly_decode_check,
    crt_poly_mask,
    crt_square_sum_pool_local,
    prepare_crt_weight,
    validate_primes,
)


def _crt_domain(kind: bytes, primes: tuple[int, ...], excluded: int) -> bytes:
    return kind + bytes((excluded, len(primes), *primes))


class _FastCpuBinaryParty:
    """Minimal NumPy uint64 Rep4 binary party for low-latency small-batch sign trees.

    Preserves the exact Rep4 6-joint schedule, BLAKE2b integrity tags, fresh
    SHAKE-256 correlated masks with monotonic counters, and fail-closed checks
    without PyTorch CPU tensor dispatch or per-call AES wrapper overhead.
    """

    def __init__(self, party: int, transport, seeds: dict[int, bytes]):
        self.party = party
        self.transport = transport
        self.seeds = dict(seeds)
        self.counters = {e: 0 for e in seeds}

    def _mask(self, excluded: int, shape: tuple[int, ...]) -> np.ndarray:
        ctr = self.counters[excluded]
        self.counters[excluded] = ctr + 1
        nbytes = math.prod(shape) * 8
        ctx = b'Rep4GPU-cpu-bin-v1' + struct.pack('<QQ', ctr, nbytes)
        raw = hashlib.shake_256(self.seeds[excluded] + ctx).digest(nbytes)
        return np.frombuffer(raw, dtype='<u8').reshape(shape)

    def public(self, value: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = np.zeros_like(value, dtype=np.uint64)
        parts = [z, z.copy(), z.copy()]
        if self.party != 0:
            parts[(0 - self.party - 1) % 4] = value.astype(np.uint64, copy=True)
        return tuple(parts)

    def xor(self, x, y):
        return (x[0] ^ y[0], x[1] ^ y[1], x[2] ^ y[2])

    def joint_input_batch(self, specs, shape: tuple[int, ...]):
        size = math.prod(shape) * 8
        outputs = [[np.zeros(shape, dtype=np.uint64) for _ in range(3)] for _ in specs]
        payloads, hashes = {}, {}
        for idx, (s, b, r, o, val) in enumerate(specs):
            if self.party != r:
                mask = self._mask(r, shape)
                outputs[idx][(r - self.party - 1) % 4] ^= mask
                if self.party in (s, b):
                    masked = val ^ mask
                    outputs[idx][(o - self.party - 1) % 4] ^= masked
                    data = masked.astype('<u8', copy=False).tobytes()
                    if self.party == s:
                        payloads.setdefault(r, []).append(data)
                    else:
                        hashes.setdefault(r, []).append((s, hashlib.blake2b(data, digest_size=32).digest()))
        outgoing = {
            r: b''.join(payloads.get(r, ())) + b''.join(h for _, h in sorted(hashes.get(r, ())))
            for r in set(payloads) | set(hashes)
        }
        incoming = self.transport.exchange(outgoing)
        offsets = {p: 0 for p in incoming}
        expected = []
        for idx, (s, b, r, o, _) in enumerate(specs):
            if self.party == r:
                start = offsets[s]
                data = incoming[s][start:start + size]
                if len(data) != size:
                    raise ValueError('truncated CPU binary Rep4 joint input')
                offsets[s] += size
                outputs[idx][(o - self.party - 1) % 4] ^= np.frombuffer(data, dtype='<u8').reshape(shape)
                expected.append((b, s, hashlib.blake2b(data, digest_size=32).digest()))
        for b, s, digest_val in sorted(expected):
            start = offsets[b]
            actual = incoming[b][start:start + 32]
            offsets[b] += 32
            if not secrets.compare_digest(actual, digest_val):
                raise ValueError(f'CPU binary Rep4 hash mismatch for sender {s} and backup {b}')
        if any(offsets[p] != len(data) for p, data in incoming.items()):
            raise ValueError('unexpected trailing CPU binary Rep4 bytes')
        return [tuple(parts) for parts in outputs]

    def mul(self, x, y):
        a0, a1, a2 = x
        b0, b1, b2 = y
        shape = a0.shape
        contrib = [None] * 6
        contrib[(self.party - 1) % 4] = ((a0 ^ a1) & b0) ^ (a0 & b1)
        contrib[self.party] = ((a1 ^ a2) & b1) ^ (a1 & b2)
        contrib[4] = (a0 & b2) ^ (a2 & b0)
        contrib[5] = contrib[4]
        specs = [(s, b, r, o, contrib[k] if self.party in (s, b) else None)
                 for k, (s, b, r, o) in enumerate(SCHEDULE)]
        terms = self.joint_input_batch(specs, shape)
        out = [np.zeros(shape, dtype=np.uint64) for _ in range(3)]
        for term in terms:
            out[0] ^= term[0]
            out[1] ^= term[1]
            out[2] ^= term[2]
        return tuple(out)

    def open(self, x) -> np.ndarray:
        payload = x[0].astype('<u8', copy=False).tobytes()
        backup_tag = hashlib.blake2b(x[2].astype('<u8', copy=False).tobytes(), digest_size=32).digest()
        sender, receiver = (self.party - 1) % 4, (self.party + 1) % 4
        # Send share to receiver and independent backup digest to sender in one round.
        incoming = self.transport.exchange({receiver: payload, sender: backup_tag})
        missing = incoming[sender]
        if len(missing) != len(payload):
            raise ValueError('invalid CPU binary open frame length')
        if not secrets.compare_digest(incoming[receiver], hashlib.blake2b(missing, digest_size=32).digest()):
            raise ValueError('CPU binary open BLAKE2b hash mismatch')
        return np.frombuffer(missing, dtype='<u8').reshape(x[0].shape) ^ x[0] ^ x[1] ^ x[2]


class CrtPhaseGraphParty(PhaseGraphParty):
    """PhaseGraphParty extended with byte-lane odd-prime CRT operations."""

    def __init__(self, party, transport, seeds, device='cuda', *,
                 primes=DEFAULT_CRT_PRIMES, binary_bits=None,
                 optimized=True, gpu_open=True):
        self.primes = validate_primes(primes)
        self.crt_modulus = math.prod(self.primes)
        self.crt_nbits = self.crt_modulus.bit_length()
        if binary_bits is None:
            binary_bits = 64 if self.crt_nbits >= 32 else 32
        super().__init__(party, transport, seeds, device=device,
                         bits=binary_bits, binary_bits=binary_bits,
                         optimized=optimized, gpu_open=gpu_open)
        self._crt_true = torch.tensor(True, dtype=torch.bool, device=self.device)
        self._crt_primes_tensor = torch.tensor(self.primes, dtype=torch.int64, device=self.device)
        coeffs = [(self.crt_modulus // q) * pow(self.crt_modulus // q, -1, q) for q in self.primes]
        self._crt_coeffs = torch.tensor(coeffs, dtype=torch.int64, device=self.device)
        self._crt_poly_keys = tuple(
            self._device_keys[(self.party + s + 1) % 4].key(
                _crt_domain(b'crt-poly-v1', self.primes, (self.party + s + 1) % 4))
            for s in range(3)
        )
        self._crt_poly_excluded = tuple((self.party + s + 1) % 4 for s in range(3))
        # Persistent CPU Party with cryptographically domain-separated seeds for
        # tiny final-score sign extraction without GPU context-switch overhead.
        cpu_seeds = {
            e: hashlib.blake2b(seed, digest_size=32, person=b'crt-cpu-sign-v1').digest()
            for e, seed in seeds.items()
        }
        byte_transport = getattr(transport, 'underlying', transport)
        self._cpu_sign_party = _FastCpuBinaryParty(party, byte_transport, cpu_seeds)

    def _crt_nonces(self, excluded, tag):
        if self._phase_builder is not None:
            return self._phase_builder.reserve_nonces_tensor(excluded, tag=tag)
        reserved = self._reserve_nonces(tuple(e for e in excluded if e is not None))
        it = iter(reserved)
        values = [0 if e is None else next(it) for e in excluded]
        return torch.tensor(values, dtype=torch.uint64, device=self.device)

    def _validate_crt_share(self, share: Share):
        if not isinstance(share, Share) or share.binary:
            raise ValueError('expected an arithmetic CRT share')
        if any(t.dtype != torch.uint8 or t.device != self.device or t.shape[0] != len(self.primes)
               for t in share.parts):
            raise ValueError('CRT share parts must be uint8 CUDA tensors with leading prime dimension')

    def crt_input(self, owner: int, value=None, shape=None) -> Share:
        """Privately share an integer tensor into checked CRT shares (2 rounds)."""
        if owner not in range(4):
            raise ValueError('input owner must be in [0, 3]')
        if value is not None:
            if self.party != owner:
                raise ValueError('only the input owner supplies plaintext')
            if isinstance(value, torch.Tensor) and value.dtype == torch.int8:
                value = value.to(device=self.device)
            else:
                value = torch.as_tensor(value, dtype=torch.int64, device=self.device)
            shape = tuple(value.shape)
        if shape is None or (self.party == owner and value is None):
            raise ValueError('owner supplies plaintext and other parties supply shape')
        shape = tuple(shape)
        n_lanes = len(self.primes)
        excl0, excl1 = (owner + 1) % 4, (owner + 3) % 4
        key0 = None if self.party == excl0 else self._device_keys[excl0].key(
            _crt_domain(b'crt-in-v1', self.primes, excl0))
        key1 = None if self.party == excl1 else self._device_keys[excl1].key(
            _crt_domain(b'crt-in-v1', self.primes, excl1))
        nonces = self._crt_nonces((None if key0 is None else excl0, None if key1 is None else excl1),
                                  tag=('crt_input', owner, shape))
        mask0, mask1 = crt_input_masks((key0, key1), nonces, self.primes, shape, self.device)
        parts: list[torch.Tensor | None] = [None, None, None]
        if self.party != excl0:
            parts[(excl0 - self.party - 1) % 4] = mask0
        if self.party != excl1:
            parts[(excl1 - self.party - 1) % 4] = mask1
        if self.party != owner:
            parts[(owner - self.party - 1) % 4] = torch.zeros((n_lanes, *shape), dtype=torch.uint8, device=self.device)
        recipients = (excl0, excl1)
        outgoing = {}
        if self.party == owner:
            parts[1] = crt_input_correction(value, parts[0], parts[2], self.primes)
            wire = parts[1].reshape(-1)
            outgoing = {p: wire for p in recipients}
        wire_bytes = n_lanes * math.prod(shape)
        sizes = {owner: wire_bytes} if self.party in recipients else {}
        incoming = self._exchange_device(outgoing, sizes)
        # Round 2: both recipients exchange BLAKE3(wire) simultaneously to catch owner equivocation.
        tag_out = {}
        tag_sizes = {}
        if self.party in recipients:
            wire = incoming[owner]
            if wire.numel() != wire_bytes or wire.dtype != torch.uint8 or wire.device != self.device:
                raise ValueError('invalid CRT private input frame')
            parts[((owner + 2) % 4 - self.party - 1) % 4] = wire.reshape(n_lanes, *shape)
            my_tag = digest(wire, block_tree=self.hash_block_tree)
            peer = (self.party + 2) % 4
            tag_out = {peer: my_tag}
            tag_sizes = {peer: 32}
        tag_in = self._exchange_device(tag_out, tag_sizes)
        if self.party in recipients:
            peer = (self.party + 2) % 4
            self._verify_tags(tag_in[peer], my_tag, 'CRT private input BLAKE3 hash mismatch')
        return Share(tuple(parts), False)

    def crt_prepare_weight(self, weight: Share) -> PreparedCrtWeight:
        self._validate_crt_share(weight)
        return prepare_crt_weight(weight.parts, self.party, self.primes)

    def crt_add(self, x: Share, y: Share) -> Share:
        self._validate_crt_share(x)
        self._validate_crt_share(y)
        return Share(tuple(crt_add_mod(a, b, self.primes) for a, b in zip(x.parts, y.parts)), False)

    def _alloc_wire_buffer(self, size: int) -> torch.Tensor | None:
        if self._phase_builder is not None and size >= (256 << 10):
            return self._phase_builder.alloc_wire_buffer(size)
        return None

    def _crt_poly_exchange(self, h_eval: torch.Tensor) -> Share:
        """2-collector checked broadcast for one degree-2 CRT polynomial evaluation."""
        shape = tuple(h_eval.shape)
        n_bytes = h_eval.numel()
        nonces = self._crt_nonces(self._crt_poly_excluded, tag=('crt_poly', shape))
        y_wire, out_parts = crt_poly_mask(
            h_eval, self._crt_poly_keys, nonces, self.party, self.primes,
            y_wire_out=self._alloc_wire_buffer(n_bytes),
        )
        # Round 1: collectors P1 and P2 gather all 4 evaluations Y(0..3).
        if self.party == 0:
            r1_out, r1_sizes = {1: y_wire, 2: y_wire}, {}
        elif self.party == 1:
            r1_out, r1_sizes = {2: y_wire}, {0: n_bytes, 2: n_bytes, 3: n_bytes}
        elif self.party == 2:
            r1_out, r1_sizes = {1: y_wire}, {0: n_bytes, 1: n_bytes, 3: n_bytes}
        else:
            r1_out, r1_sizes = {1: y_wire, 2: y_wire}, {}
        r1_in = self._exchange_device(r1_out, r1_sizes)
        # Collectors verify degree-2 third difference BEFORE sending Round 2.
        if self.party in (1, 2):
            ys = (r1_in[0], y_wire if self.party == 1 else r1_in[1],
                  y_wire if self.party == 2 else r1_in[2], r1_in[3])
            slot0 = (0 - self.party - 1) % 4
            tail = 32 if self.party == 1 else 0
            c_out_buf = self._alloc_wire_buffer(n_bytes + tail) if self.party == 1 else None
            c_buf, ok_flag = crt_poly_decode_check(
                ys, out_parts[slot0], self.primes, extra_tail_bytes=tail, c_out_buf=c_out_buf)
            self._verify_tags(ok_flag, self._crt_true, 'CRT polynomial third-difference check failed')
            # Only Y(0) and Y(3) are sent to both collectors (Y(1)/Y(2) go to one peer and are
            # verified uniquely by the third-difference check), so hashing Y(0) and Y(3) suffices
            # to detect any equivocation while cutting collector BLAKE3 work in half.
            leaf_tags = torch.cat((digest(ys[0], block_tree=self.hash_block_tree),
                                   digest(ys[3], block_tree=self.hash_block_tree)))
            d_all = digest(leaf_tags, block_tree=self.hash_block_tree)
            if self.party == 1:
                c_buf[n_bytes:].copy_(d_all)
                r2_out = {0: d_all, 2: d_all, 3: c_buf}
                r2_sizes = {2: 32}
            else:
                d_c = digest(c_buf, block_tree=self.hash_block_tree)
                r2_out = {0: d_all, 1: d_all, 3: torch.cat((d_c, d_all))}
                r2_sizes = {1: 32}
        elif self.party == 0:
            r2_out, r2_sizes = {}, {1: 32, 2: 32}
        else:
            r2_out, r2_sizes = {}, {1: n_bytes + 32, 2: 64}
        r2_in = self._exchange_device(r2_out, r2_sizes)
        if self.party == 0:
            self._verify_tags(r2_in[1], r2_in[2], 'CRT polynomial collector digest mismatch at party 0')
        elif self.party == 1:
            self._verify_tags(r2_in[2], d_all, 'CRT polynomial collector digest mismatch at party 1')
        elif self.party == 2:
            self._verify_tags(r2_in[1], d_all, 'CRT polynomial collector digest mismatch at party 2')
        else:
            c_recv = r2_in[1][:n_bytes]
            d_all_1 = r2_in[1][n_bytes:]
            d_c_2 = r2_in[2][:32]
            d_all_2 = r2_in[2][32:]
            c_digest = digest(c_recv, block_tree=self.hash_block_tree)
            self._verify_tags(torch.stack((d_all_1, c_digest)), torch.stack((d_all_2, d_c_2)),
                              'CRT polynomial correction or collector digest mismatch at party 3')
            slot0 = (0 - self.party - 1) % 4
            crt_add_mod(out_parts[slot0], c_recv.reshape(shape), self.primes, inplace=True)
        return Share(out_parts, False)

    def crt_eval_input(self, x: Share, *, out: torch.Tensor | None = None) -> torch.Tensor:
        """Locally form centered int8 F(party) from secret shares (no communication)."""
        self._validate_crt_share(x)
        return crt_eval_linear(x.parts, self.party, self.primes, out=out)

    def crt_conv1d_from_eval(self, fx: torch.Tensor, weight: PreparedCrtWeight,
                             bias: Share | None = None) -> Share:
        """Private 1D convolution from locally evaluated F_X(party) and prepared weights."""
        h_eval = crt_conv1d_local(fx, weight)
        out = self._crt_poly_exchange(h_eval)
        return self.crt_add(out, bias) if bias is not None else out

    def crt_conv1d(self, x: Share, weight: PreparedCrtWeight, bias: Share | None = None) -> Share:
        """Private 1D convolution with secret-shared inputs and prepared secret weights."""
        fx = self.crt_eval_input(x)
        return self.crt_conv1d_from_eval(fx, weight, bias)

    def crt_square_sum_pool(self, h: Share, bias: Share | None = None) -> Share:
        """Fused private bias-add, square activation, and global sum pooling over sequence."""
        self._validate_crt_share(h)
        if bias is not None:
            self._validate_crt_share(bias)
        s_eval = crt_square_sum_pool_local(h.parts, self.party, self.primes,
                                           bias_parts=None if bias is None else bias.parts)
        return self._crt_poly_exchange(s_eval)

    def crt_matmul(self, x: Share, weight: PreparedCrtWeight, bias: Share | None = None) -> Share:
        """Private matrix multiplication with secret-shared inputs and prepared secret weights."""
        self._validate_crt_share(x)
        fx = crt_eval_linear(x.parts, self.party, self.primes)
        h_eval = crt_matmul_local(fx, weight)
        out = self._crt_poly_exchange(h_eval)
        return self.crt_add(out, bias) if bias is not None else out

    def crt_nonnegative_label(self, score: Share) -> Share:
        """Return binary-shared (score >= 0) directly from CRT shares without opening score."""
        self._validate_crt_share(score)
        m = self.crt_modulus
        nbits = self.crt_nbits
        power2 = 1 << nbits
        logical_shape = tuple(score.shape[1:])
        coeff_view = self._crt_coeffs.view(len(self.primes), *([1] * len(logical_shape)))
        mod_m = {
            (self.party + s + 1) % 4: torch.remainder((part.to(torch.int64) * coeff_view).sum(0), m)
            for s, part in enumerate(score.parts)
        }
        specs = []
        for left, right, sender, backup, recv, outsider in ((0, 1, 2, 3, 0, 1), (2, 3, 0, 1, 2, 3)):
            value = None
            if self.party in (sender, backup):
                value = torch.remainder(2 * (mod_m[left] + mod_m[right]), m)
                if left == 2:
                    value = value + (power2 - m)
            specs.append((sender, backup, recv, outsider, value))
        u, vshift = self.joint_input_batch(specs, shape=logical_shape, binary=True, device=self.device)
        lsb = Share(tuple((a ^ b) & 1 for a, b in zip(u.parts, vshift.parts)), True)
        from ..gpu.bits.carry_tree_swar import carry_bits
        carry = carry_bits(self, u, vshift, [nbits])[0]
        return self.xor(lsb, carry)

    def crt_nonnegative_label_open_cpu(self, score: Share) -> torch.Tensor:
        """Privately compute and open (score >= 0) on CPU using bit-sliced Rep4 carry tree."""
        self._validate_crt_share(score)
        m = self.crt_modulus
        nbits = self.crt_nbits
        power2 = 1 << nbits
        logical_shape = tuple(score.shape[1:])
        count = math.prod(logical_shape)
        if not count:
            return torch.empty(logical_shape, dtype=torch.int64, device=self.device)
        n_words = (count + 63) // 64
        padded = n_words * 64
        coeffs = self._crt_coeffs.cpu().numpy().reshape(len(self.primes), 1)
        stacked_parts = torch.stack(score.parts).cpu().numpy().reshape(3, len(self.primes), count).astype(np.int64)
        mod_m = {
            (self.party + s + 1) % 4: (stacked_parts[s] * coeffs).sum(axis=0) % m
            for s in range(3)
        }

        def to_bitplanes(vec: np.ndarray) -> np.ndarray:
            buf = np.zeros(padded, dtype=np.uint64)
            buf[:count] = vec.astype(np.uint64)
            bits = ((buf[:, None] >> np.arange(nbits, dtype=np.uint64)[None, :]) & 1).astype(np.uint8)
            return np.ascontiguousarray(np.packbits(bits.T, axis=1, bitorder='little')).view(np.uint64)

        specs = []
        for left, right, sender, backup, recv, outsider in ((0, 1, 2, 3, 0, 1), (2, 3, 0, 1, 2, 3)):
            planes = None
            if self.party in (sender, backup):
                vec = (2 * (mod_m[left] + mod_m[right])) % m
                if left == 2:
                    vec = vec + (power2 - m)
                planes = to_bitplanes(vec)
            specs.append((sender, backup, recv, outsider, planes))
        cpu_rt = self._cpu_sign_party
        u, vshift = cpu_rt.joint_input_batch(specs, shape=(nbits, n_words))
        size = 1 << (nbits - 1).bit_length()
        p = cpu_rt.xor(u, vshift)
        lsb = tuple(part[0] for part in p)
        g = cpu_rt.mul(u, vshift)
        if size > nbits:
            pad_g = cpu_rt.public(np.zeros((size - nbits, n_words), dtype=np.uint64))
            pad_p = cpu_rt.public(np.full((size - nbits, n_words), np.uint64(0xFFFFFFFFFFFFFFFF), dtype=np.uint64))
            g = tuple(np.concatenate((a, b), axis=0) for a, b in zip(g, pad_g))
            p = tuple(np.concatenate((a, b), axis=0) for a, b in zip(p, pad_p))
        width = size
        while width > 1:
            half = width // 2
            p_odd = tuple(part[1::2] for part in p)
            g_even = tuple(part[0::2] for part in g)
            g_odd = tuple(part[1::2] for part in g)
            if half == 1:
                g = cpu_rt.xor(g_odd, cpu_rt.mul(p_odd, g_even))
            else:
                p_even = tuple(part[0::2] for part in p)
                left = tuple(np.concatenate((a, a[1:]), axis=0) for a in p_odd)
                right = tuple(np.concatenate((ge, pe[1:]), axis=0) for ge, pe in zip(g_even, p_even))
                prod = cpu_rt.mul(left, right)
                g = cpu_rt.xor(g_odd, tuple(part[:half] for part in prod))
                zero_row = cpu_rt.public(np.zeros((1, n_words), dtype=np.uint64))
                p = tuple(np.concatenate((z, hi[half:]), axis=0) for z, hi in zip(zero_row, prod))
            width = half
        carry_out = tuple(part[0] for part in g)
        packed_labels = cpu_rt.open(cpu_rt.xor(lsb, carry_out))
        unpacked = np.unpackbits(packed_labels.view(np.uint8), bitorder='little')[:count]
        return torch.from_numpy(unpacked.astype(np.int64)).reshape(logical_shape)

    def crt_open(self, x: Share) -> torch.Tensor:
        """Reveal a CRT share as canonical signed int64 in [-M//2, +M//2] (testing/oracle use)."""
        self._validate_crt_share(x)
        n_bytes = x.parts[0].numel()
        sender, receiver = (self.party - 1) % 4, (self.party + 1) % 4
        payload = x.parts[0].contiguous().reshape(-1)
        backup_wire = x.parts[2].contiguous().reshape(-1)
        incoming = self._exchange_device({receiver: payload}, {sender: n_bytes})
        missing = incoming[sender]
        tags = digest_batch(torch.stack((backup_wire, missing)), block_tree=self.hash_block_tree)
        checks = self._exchange_device({sender: tags[0]}, {receiver: 32})
        self._verify_tags(checks[receiver], tags[1], 'CRT opening BLAKE3 hash mismatch')
        p_view = self._crt_primes_tensor.view(len(self.primes), *([1] * (x.parts[0].ndim - 1)))
        total = torch.remainder(
            missing.reshape(x.parts[0].shape).to(torch.int64)
            + x.parts[0].to(torch.int64)
            + x.parts[1].to(torch.int64)
            + x.parts[2].to(torch.int64),
            p_view,
        )
        coeff_view = self._crt_coeffs.view(len(self.primes), *([1] * (x.parts[0].ndim - 1)))
        m = self.crt_modulus
        val = torch.remainder((total * coeff_view).sum(0), m)
        return torch.where(val > m // 2, val - m, val)
