"""Rep4 arithmetic over Z/(2**bits), with three components per party.

A party numbered p stores components excluded from p+1, p+2, and p+3
(modulo four). Online multiplication follows the six joint inputs in
MP-SPDZ Protocols/Rep4.hpp; no operand is opened.
"""
from dataclasses import dataclass
import hashlib
import math
import secrets
import threading
import torch
from .prg import random_bytes


@dataclass(frozen=True)
class Share:
    parts: tuple
    binary: bool = False

    def __post_init__(self):
        if len(self.parts) != 3:
            raise ValueError('a Rep4 party stores exactly three components')
        if any(t.dtype != torch.int64 for t in self.parts):
            raise TypeError('share components must use torch.int64')
        if any(t.shape != self.parts[0].shape or t.device != self.parts[0].device for t in self.parts):
            raise ValueError('share components must have matching shape and device')

    @property
    def shape(self):
        return self.parts[0].shape

    @property
    def device(self):
        return self.parts[0].device

    def reshape(self, *shape):
        return Share(tuple(t.reshape(*shape) for t in self.parts), self.binary)

    def __getitem__(self, item):
        return Share(tuple(t[item] for t in self.parts), self.binary)


@dataclass(frozen=True)
class PreparedMatrix:
    """Three prepacked local weight components, usable only by matmul."""
    parts: tuple
    binary: bool = False

    @property
    def shape(self):
        return self.parts[0].shape

    @property
    def device(self):
        return self.parts[0].device


def prepare_weight_share(weight, bits=64, *, storage='limbs'):
    if weight.binary:
        raise ValueError('only arithmetic weight shares can be prepacked')
    from ..gpu.matmul.kernels import prepare_weight
    return PreparedMatrix(tuple(prepare_weight(t, bits=bits, storage=storage) for t in weight.parts))


SCHEDULE = ((0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0),
            (3, 0, 2, 1), (0, 2, 3, 1), (1, 3, 2, 0))


def tensor_bytes(value, bits=64):
    dtype = '<u4' if bits == 32 else '<i8'
    return value.detach().contiguous().cpu().numpy().astype(dtype, copy=False).tobytes()


def bytes_tensor(data, shape, device, bits=64):
    # bytearray gives torch a writable, owned buffer before the device copy.
    if math.prod(shape) == 0:
        return torch.empty(shape, dtype=torch.int64, device=device)
    dtype = torch.uint32 if bits == 32 else torch.int64
    return torch.frombuffer(bytearray(data), dtype=dtype).to(dtype=torch.int64).clone().reshape(shape).to(device)


def make_seeds():
    """Trusted setup utility: one independent secret per excluded party.

    Provision each party only the three seeds whose index differs from its ID.
    Production provisioning must use an authenticated confidential channel.
    """
    return {i: secrets.token_bytes(32) for i in range(4)}


class Party:
    """One protocol role with a public, sequential operation schedule.

    Run one protocol operation at a time per Party and use the same operation
    order across roles. Atomic mask counters protect nonce uniqueness; they do
    not make concurrent protocol operations a supported execution schedule.
    """
    def __init__(self, party, transport, seeds, device='cpu', bits=64, prg='shake', binary_bits=64):
        if bits not in (32, 64):
            raise ValueError('ring bits must be 32 or 64')
        self.bits = bits
        if binary_bits not in (32, 64) or (binary_bits == 32 and bits != 32):
            raise ValueError('32-bit binary words require the 32-bit ring')
        self.binary_bits = binary_bits
        if prg not in ('shake', 'aes'):
            raise ValueError('PRG backend must be shake or aes')
        self.prg = prg
        if party not in range(4):
            raise ValueError('party must be 0, 1, 2, or 3')
        if set(seeds) != set(range(4)) - {party}:
            raise ValueError('provide only the three seeds available to this party')
        if any(len(seed) < 32 for seed in seeds.values()):
            raise ValueError('PRG seeds must contain at least 256 bits')
        self.party = party
        self.transport = transport
        self.seeds = dict(seeds)
        self.device = torch.device(device)
        self.counters = {p: 0 for p in seeds}
        self._counter_lock = threading.Lock()
        self.multiplications = self.joint_inputs = 0

    def _normalize(self, value, binary=False):
        return value & 0xffffffff if self._wire_bits(binary) == 32 else value

    def _share(self, parts, binary=False):
        return Share(tuple(self._normalize(t, binary) for t in parts), binary)

    def _wire_bits(self, binary):
        return self.binary_bits if binary else self.bits

    def _reserve_nonces(self, excluded_parties):
        """Reserve ordered draw counters atomically, including repeated seeds.

        Reservations are consumed before cryptographic work starts. A failed
        mask generation burns its reservation rather than reusing a nonce.
        """
        with self._counter_lock:
            pending, reserved = {}, []
            for excluded in excluded_parties:
                counter = pending.get(excluded, self.counters[excluded])
                if not 0 <= counter < 1 << 64:
                    raise ValueError('PRG draw counter exhausted')
                reserved.append(counter)
                pending[excluded] = counter+1
            self.counters.update(pending)
            return tuple(reserved)

    def _random_bytes(self, excluded, shape, binary=False):
        count = math.prod(shape)
        counter = self._reserve_nonces((excluded,))[0]
        return random_bytes(self.seeds[excluded], counter, count, ring_bits=self.bits,
                            binary=binary, binary_bits=self.binary_bits, backend=self.prg)

    def _random(self, excluded, shape, device, binary=False):
        data = self._random_bytes(excluded, shape, binary)
        return bytes_tensor(data, shape, device, self._wire_bits(binary))

    def public(self, value, like=None, binary=None):
        if binary is None:
            binary = like.binary if like is not None else False
        device = like.device if like is not None else self.device
        value = torch.as_tensor(value, dtype=torch.int64, device=device)
        if like is not None:
            value = torch.broadcast_to(value, like.shape)
        parts = [torch.zeros_like(value) for _ in range(3)]
        if self.party != 0:
            parts[(0 - self.party - 1) % 4] = value
        return self._share(parts, binary)

    def add(self, x, y):
        if x.binary != y.binary:
            raise ValueError('cannot mix arithmetic and binary shares')
        op = torch.bitwise_xor if x.binary else torch.add
        return self._share((op(a, b) for a, b in zip(x.parts, y.parts)), x.binary)

    def sub(self, x, y):
        if x.binary:
            return self.add(x, y)
        return self._share(a - b for a, b in zip(x.parts, y.parts))

    xor = add

    def mul_public(self, x, value):
        op = torch.bitwise_and if x.binary else torch.mul
        return self._share((op(a, value) for a in x.parts), x.binary)

    def lshift(self, x, bits):
        return self._share((a << bits for a in x.parts), x.binary)

    def rshift(self, x, bits):
        return self._share((a >> bits for a in x.parts), x.binary)

    def joint_input_batch(self, specs, shape=None, binary=False, device=None):
        """Reshare joint values; specs contain (sender, backup, receiver, outsider, value).

        Values need exist only at sender and backup. All parties supply the
        public output shape. Hashes cover each masked tensor's little-endian
        ring bytes and are appended after all payloads, ordered by sender.
        """
        device = self.device if device is None else torch.device(device)
        if shape is None:
            shape = next(v.shape for *_, v in specs if v is not None)
        shape = tuple(shape)
        combine = torch.bitwise_xor if binary else torch.add
        subtract = torch.bitwise_xor if binary else torch.sub
        outputs, payloads, hashes = [], {}, {}
        for sender, backup, receiver, outsider, value in specs:
            parts = [torch.zeros(shape, dtype=torch.int64, device=device) for _ in range(3)]
            if self.party != receiver:
                mask = self._random(receiver, shape, device, binary)
                slot = (receiver - self.party - 1) % 4
                parts[slot] = combine(parts[slot], mask)
                if self.party in (sender, backup):
                    masked = self._normalize(subtract(value, mask), binary)
                    slot = (outsider - self.party - 1) % 4
                    parts[slot] = combine(parts[slot], masked)
                    data = tensor_bytes(masked, self._wire_bits(binary))
                    if self.party == sender:
                        payloads.setdefault(receiver, []).append(data)
                    else:
                        hashes.setdefault(receiver, []).append((sender, hashlib.blake2b(data, digest_size=32).digest()))
            outputs.append(parts)
        outgoing = {}
        for receiver in set(payloads) | set(hashes):
            outgoing[receiver] = b''.join(payloads.get(receiver, [])) + b''.join(
                h for _, h in sorted(hashes.get(receiver, [])))
        incoming = self.transport.exchange(outgoing)
        offsets = {p: 0 for p in incoming}
        expected = []
        size = math.prod(shape) * (self._wire_bits(binary) // 8)
        for index, (sender, backup, receiver, outsider, _) in enumerate(specs):
            if self.party == receiver:
                start = offsets[sender]
                data = incoming[sender][start:start + size]
                if len(data) != size:
                    raise ValueError('truncated Rep4 joint input')
                offsets[sender] += size
                slot = (outsider - self.party - 1) % 4
                outputs[index][slot] = combine(outputs[index][slot], bytes_tensor(data, shape, device, self._wire_bits(binary)))
                expected.append((backup, sender, hashlib.blake2b(data, digest_size=32).digest()))
        for backup, sender, digest in sorted(expected):
            start = offsets[backup]
            actual = incoming[backup][start:start + 32]
            offsets[backup] += 32
            if not secrets.compare_digest(actual, digest):
                raise ValueError(f'joint input hash mismatch for sender {sender} and backup {backup}')
        if any(offsets[p] != len(data) for p, data in incoming.items()):
            raise ValueError('unexpected trailing Rep4 message bytes')
        self.joint_inputs += len(specs)
        return [self._share(parts, binary) for parts in outputs]

    def joint_input(self, sender, backup, receiver, outsider, value, shape=None, binary=False):
        return self.joint_input_batch([(sender, backup, receiver, outsider, value)], shape, binary)[0]

    def _product(self, x, y, product):
        if x.binary != y.binary:
            raise ValueError('cannot mix arithmetic and binary shares')
        add = torch.bitwise_xor if x.binary else torch.add
        a = [None] * 5
        for i in range(2):
            a[(self.party + i - 1) % 4] = add(product(add(x.parts[i], x.parts[i + 1]), y.parts[i]), product(x.parts[i], y.parts[i + 1]))
        a[4] = add(product(x.parts[0], y.parts[2]), product(x.parts[2], y.parts[0]))
        return self._reshare_product(a, x.binary, x.device)

    def _reshare_product(self, a, binary, device):
        specs = [(*entry, a[i if i < 4 else 4]) for i, entry in enumerate(SCHEDULE)]
        results = self.joint_input_batch(specs, a[4].shape, binary, device)
        result = results[0]
        for term in results[1:]:
            result = self.add(result, term)
        self.multiplications += 1
        return result

    def mul(self, x, y):
        return self._product(x, y, torch.bitwise_and if x.binary else torch.mul)

    def matmul(self, x, y):
        if x.binary:
            raise ValueError('binary matrix products are not implemented')
        if x.device.type == 'cpu':
            return self._product(x, y, torch.matmul)
        from ..gpu.matmul.kernels import local_products
        local = local_products(*x.parts, *y.parts, bits=self.bits)
        a = [None] * 5
        a[(self.party - 1) % 4], a[self.party], a[4] = local
        return self._reshare_product(a, False, x.device)

    def prepare_weight_share(self, weight, *, storage='limbs'):
        return prepare_weight_share(weight, bits=self.bits, storage=storage)

    def trunc(self, x, bits):
        from ..arithmetic.conversion import trunc
        return trunc(self, x, bits)

    def trunc_relu(self, x, bits):
        from ..arithmetic.conversion import trunc_relu
        return trunc_relu(self, x, bits)

    def sign_bit(self, x):
        from ..arithmetic.conversion import sign_bit
        return sign_bit(self, x)

    def lt_public(self, x, value):
        from ..arithmetic.conversion import signed_lt
        return signed_lt(self, x, self.public(value, like=x))

    def input(self, owner, value=None, shape=None, binary=False):
        """Share an owner's private tensor with Rep4Input mask placement.

        The two recipients cross-check their received correction. This detects
        an owner sending inconsistent corrections without revealing the input.
        """
        if value is not None:
            value = torch.as_tensor(value, dtype=torch.int64, device=self.device)
            shape = value.shape
        if shape is None:
            raise ValueError('all parties must supply the public input shape')
        shape = tuple(shape)
        parts = [torch.zeros(shape, dtype=torch.int64, device=self.device) for _ in range(3)]
        for excluded in ((owner + 1) % 4, (owner + 3) % 4):
            if self.party != excluded:
                parts[(excluded - self.party - 1) % 4] = self._random(excluded, shape, self.device, binary)
        outgoing = {}
        if self.party == owner:
            if value is None:
                raise ValueError('owner must provide its input')
            correction = value ^ parts[0] ^ parts[2] if binary else value - parts[0] - parts[2]
            correction = self._normalize(correction, binary)
            parts[1] = correction
            data = tensor_bytes(correction, self._wire_bits(binary))
            outgoing = {(owner + 1) % 4: data, (owner + 3) % 4: data}
        incoming = self.transport.exchange(outgoing)
        recipients = ((owner + 1) % 4, (owner + 3) % 4)
        outgoing = {}
        if self.party in recipients:
            data = incoming[owner]
            if len(data) != math.prod(shape) * (self._wire_bits(binary) // 8):
                raise ValueError('truncated private input')
            parts[((owner + 2) % 4 - self.party - 1) % 4] = bytes_tensor(data, shape, self.device, self._wire_bits(binary))
            data_hash = hashlib.blake2b(data, digest_size=32).digest()
            empty_hash = hashlib.blake2b(b'', digest_size=32).digest()
            pair = (data_hash, empty_hash) if self.party == (owner - 1) % 4 else (empty_hash, data_hash)
        received_hashes = []
        for i in range(2):
            outgoing = {(self.party + 2) % 4: pair[i]} if self.party in recipients else {}
            checks = self.transport.exchange(outgoing)
            if self.party in recipients:
                received_hashes.append(checks[(self.party + 2) % 4])
        if self.party in recipients:
            if any(not secrets.compare_digest(pair[i], received_hashes[1 - i]) for i in range(2)):
                raise ValueError('private input hash mismatch')
        return self._share(parts, binary)

    def open(self, x):
        """Explicitly reveal a result with Rep4MC's eager hash check."""
        # Rep4MC sends slot zero clockwise and checks slot two counterclockwise.
        data = tensor_bytes(x.parts[0], self._wire_bits(x.binary))
        incoming = self.transport.exchange({(self.party + 1) % 4: data})
        missing = incoming[(self.party - 1) % 4]
        digest = hashlib.blake2b(tensor_bytes(x.parts[2], self._wire_bits(x.binary)), digest_size=32).digest()
        checks = self.transport.exchange({(self.party - 1) % 4: digest})
        if not secrets.compare_digest(hashlib.blake2b(missing, digest_size=32).digest(), checks[(self.party + 1) % 4]):
            raise ValueError('output component hash mismatch')
        result = bytes_tensor(missing, x.shape, x.device, self._wire_bits(x.binary))
        for part in x.parts:
            result = result ^ part if x.binary else result + part
        if self.bits == 32 and not x.binary:
            result = ((result & 0xffffffff) ^ 0x80000000) - 0x80000000
        return result
