"""GPU Rep4 multiplication with the reference Party's exact wire transcript.

Cryptographic masks and BLAKE2b checks stay on the CPU. A packed host/device
transfer carries each mask batch, each masked output batch, and each receive
batch; Triton assembles the three result components after verification.
"""
import hashlib
import math
import secrets
import torch
import triton
import triton.language as tl
from .runtime import Party, Share, SCHEDULE


@triton.jit
def _masked(X0, X1, X2, Y0, Y1, Y2, MASKS, OUT,
            N: tl.constexpr, BINARY: tl.constexpr, BITS: tl.constexpr,
            POINTWISE: tl.constexpr, JOINTS: tl.constexpr, COLUMNS: tl.constexpr,
            BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x0 = tl.load(X0 + i, i < N, 0).to(tl.uint64)
    x1 = tl.load(X1 + i, i < N, 0).to(tl.uint64)
    x2 = tl.load(X2 + i, i < N, 0).to(tl.uint64)
    if POINTWISE:
        y0 = tl.load(Y0 + i, i < N, 0).to(tl.uint64)
        y1 = tl.load(Y1 + i, i < N, 0).to(tl.uint64)
        y2 = tl.load(Y2 + i, i < N, 0).to(tl.uint64)
        if BINARY:
            a0 = ((x0 ^ x1) & y0) ^ (x0 & y1)
            a1 = ((x1 ^ x2) & y1) ^ (x1 & y2)
            a4 = (x0 & y2) ^ (x2 & y0)
        else:
            a0 = (x0 + x1) * y0 + x0 * y1
            a1 = (x1 + x2) * y1 + x1 * y2
            a4 = x0 * y2 + x2 * y0
    else:
        a0, a1, a4 = x0, x1, x2
    for j in tl.static_range(len(JOINTS)):
        if COLUMNS[j] == 0:
            value = a0
        elif COLUMNS[j] == 1:
            value = a1
        else:
            value = a4
        mask = tl.load(MASKS + JOINTS[j] * N + i, i < N, 0).to(tl.uint64)
        if BINARY:
            value ^= mask
        else:
            value -= mask
        tl.store(OUT + j * N + i, value, i < N)


@triton.jit
def _assemble(MASKS, LOCAL, RECEIVED, OUT, N: tl.constexpr,
              BINARY: tl.constexpr, BITS: tl.constexpr,
              MASK_SLOTS: tl.constexpr, OUTSIDE_SLOTS: tl.constexpr,
              LOCAL_IDS: tl.constexpr, RECEIVE_IDS: tl.constexpr,
              BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    for slot in tl.static_range(3):
        value = tl.full((BLOCK,), 0, tl.uint64)
        for k in tl.static_range(len(MASK_SLOTS)):
            if MASK_SLOTS[k] == slot:
                term = tl.load(MASKS + k * N + i, i < N, 0).to(tl.uint64)
                if BINARY:
                    value ^= term
                else:
                    value += term
            if OUTSIDE_SLOTS[k] == slot:
                if LOCAL_IDS[k] >= 0:
                    term = tl.load(LOCAL + LOCAL_IDS[k] * N + i, i < N, 0).to(tl.uint64)
                    if BINARY:
                        value ^= term
                    else:
                        value += term
                elif RECEIVE_IDS[k] >= 0:
                    term = tl.load(RECEIVED + RECEIVE_IDS[k] * N + i, i < N, 0).to(tl.uint64)
                    if BINARY:
                        value ^= term
                    else:
                        value += term
        if BITS < 64:
            value &= (1 << BITS)-1
        tl.store(OUT + slot * N + i, value.to(tl.int64), i < N)


@triton.jit
def _local_operation(A0, A1, A2, B0, B1, B2, OUT,
                     N: tl.constexpr, OP: tl.constexpr, SCALAR: tl.constexpr,
                     BINARY: tl.constexpr, BITS: tl.constexpr,
                     PARTY: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    for slot in tl.static_range(3):
        if OP == 5:
            if (PARTY + slot + 1) % 4 == 0:
                value = tl.full((BLOCK,), SCALAR, tl.int64)
            else:
                value = tl.full((BLOCK,), 0, tl.int64)
        else:
            if slot == 0:
                value = tl.load(A0 + i, i < N, 0).to(tl.int64)
            elif slot == 1:
                value = tl.load(A1 + i, i < N, 0).to(tl.int64)
            else:
                value = tl.load(A2 + i, i < N, 0).to(tl.int64)
            if OP == 0 or OP == 1:
                if slot == 0:
                    other = tl.load(B0 + i, i < N, 0).to(tl.int64)
                elif slot == 1:
                    other = tl.load(B1 + i, i < N, 0).to(tl.int64)
                else:
                    other = tl.load(B2 + i, i < N, 0).to(tl.int64)
                if BINARY:
                    value ^= other
                elif OP == 0:
                    value += other
                else:
                    value -= other
            elif OP == 2:
                if BINARY:
                    value &= SCALAR
                else:
                    value *= SCALAR
            elif OP == 3:
                value = value.to(tl.uint64) << SCALAR
            else:
                value >>= SCALAR
        if BITS < 64:
            value &= (1 << BITS)-1
        tl.store(OUT + slot * N + i, value.to(tl.int64), i < N)


@triton.jit
def _broadcast_local(A, B, OUT, N: tl.constexpr, SHAPE: tl.constexpr,
                     AS: tl.constexpr, BS: tl.constexpr, SUBTRACT: tl.constexpr,
                     BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    remainder = i
    ai = tl.full((BLOCK,), 0, tl.int64)
    bi = tl.full((BLOCK,), 0, tl.int64)
    for dim in tl.static_range(len(SHAPE)-1, -1, -1):
        coordinate = remainder % SHAPE[dim]
        remainder //= SHAPE[dim]
        ai += coordinate*AS[dim]
        bi += coordinate*BS[dim]
    a = tl.load(A+ai, i < N, 0).to(tl.int32)
    b = tl.load(B+bi, i < N, 0).to(tl.int32)
    tl.store(OUT+i, a-b if SUBTRACT else a+b, i < N)


class FusedParty(Party):
    """Reference-compatible Party with fused CUDA online multiplication."""
    def _local_apply(self, x, op, y=None, scalar=0):
        self._validate_share_storage(x, y)
        count = math.prod(x.shape)
        result = torch.empty((3, *x.shape), dtype=self._storage_dtype(x.binary), device=x.device)
        if count:
            a = tuple(t.contiguous() for t in x.parts)
            b = a if y is None else tuple(t.contiguous() for t in y.parts)
            _local_operation[(triton.cdiv(count, 256),)](*a, *b, result, count, op, scalar,
                                                        x.binary, self._wire_bits(x.binary), self.party, 256)
        return Share(tuple(result.unbind(0)), x.binary)

    def add(self, x, y):
        self._validate_share_storage(x, y)
        if self._storage_dtype(x.binary) in (torch.uint8, torch.uint16) and x.shape != y.shape:
            return self._packed_broadcast(x, y, False)
        if x.device.type != 'cuda' or x.shape != y.shape:
            return super().add(x, y)
        if x.binary != y.binary:
            raise ValueError('cannot mix arithmetic and binary shares')
        return self._local_apply(x, 0, y)

    xor = add

    def sub(self, x, y):
        self._validate_share_storage(x, y)
        if self._storage_dtype(x.binary) in (torch.uint8, torch.uint16) and x.shape != y.shape:
            return self._packed_broadcast(x, y, True)
        if x.device.type != 'cuda' or x.shape != y.shape:
            return super().sub(x, y)
        if x.binary != y.binary:
            raise ValueError('cannot mix arithmetic and binary shares')
        return self._local_apply(x, 1, y)

    def mul_public(self, x, value):
        self._validate_share_storage(x)
        if x.device.type == 'cuda' and isinstance(value, int) and -(1 << 63) <= value < (1 << 63):
            return self._local_apply(x, 2, scalar=value)
        if x.parts[0].dtype in (torch.uint8, torch.uint16):
            x = Share(tuple(t.to(torch.int64) for t in x.parts), x.binary)
        return super().mul_public(x, value)

    def _packed_broadcast(self, x, y, subtract):
        if x.binary != y.binary or x.device != y.device:
            raise ValueError('broadcast operands must have matching sharing and device')
        shape = torch.broadcast_shapes(x.shape, y.shape)
        result = torch.empty((3, *shape), dtype=self._storage_dtype(x.binary), device=x.device)
        count = math.prod(shape)
        if count:
            for a, b, out in zip(x.parts, y.parts, result.unbind(0)):
                a, b = torch.broadcast_to(a, shape), torch.broadcast_to(b, shape)
                _broadcast_local[(triton.cdiv(count, 256),)](
                    a, b, out, count, tuple(shape), a.stride(), b.stride(), subtract, 256)
        return Share(tuple(result.unbind(0)))

    def lshift(self, x, bits):
        self._validate_share_storage(x)
        if x.device.type == 'cuda' and isinstance(bits, int) and 0 <= bits < 64:
            return self._local_apply(x, 3, scalar=bits)
        if x.parts[0].dtype in (torch.uint8, torch.uint16):
            x = Share(tuple(t.to(torch.int64) for t in x.parts), x.binary)
        return super().lshift(x, bits)

    def rshift(self, x, bits):
        self._validate_share_storage(x)
        if x.device.type == 'cuda' and isinstance(bits, int) and 0 <= bits < 64:
            return self._local_apply(x, 4, scalar=bits)
        if x.parts[0].dtype in (torch.uint8, torch.uint16):
            x = Share(tuple(t.to(torch.int64) for t in x.parts), x.binary)
        return super().rshift(x, bits)

    def public(self, value, like=None, binary=None):
        self._validate_share_storage(like)
        device = self.device if like is None else like.device
        if torch.device(device).type != 'cuda' or not isinstance(value, int) or not -(1 << 63) <= value < (1 << 63):
            return super().public(value, like, binary)
        binary = (like.binary if like is not None else False) if binary is None else binary
        shape = () if like is None else like.shape
        count = math.prod(shape)
        result = torch.empty((3, *shape), dtype=self._storage_dtype(binary), device=device)
        if count:
            _local_operation[(triton.cdiv(count, 256),)](result, result, result, result, result, result,
                                                        result, count, 5, value, binary, self._wire_bits(binary), self.party, 256)
        return Share(tuple(result.unbind(0)), binary)

    def _mask_batch(self, shape, binary, device):
        count = math.prod(shape)
        width = self._wire_bits(binary)
        records = []
        for _, _, receiver, _ in SCHEDULE:
            if self.party == receiver:
                records.append(bytes(count * (width // 8)))
                continue
            records.append(self._random_bytes(receiver, shape, binary))
        dtype = torch.uint32 if width == 32 else torch.int64
        host = torch.frombuffer(bytearray(b''.join(records)), dtype=dtype)
        return host.to(device)

    def _fused_product(self, local, binary, device, pointwise=None):
        shape = local[0].shape
        count = math.prod(shape)
        if count == 0:
            raise ValueError('fused multiplication requires a nonempty tensor')
        width = self._wire_bits(binary)
        dtype = torch.uint32 if width == 32 else torch.int64
        local_joints = tuple(k for k, (sender, backup, _, _) in enumerate(SCHEDULE) if self.party in (sender, backup))
        local_columns = tuple(0 if k == (self.party - 1) % 4 else 1 if k == self.party else 2 for k in local_joints)
        receive_joints = tuple(k for k, (_, _, receiver, _) in enumerate(SCHEDULE) if self.party == receiver)
        masks = self._mask_batch(shape, binary, device)
        masked = torch.empty((3, count), dtype=dtype, device=device)
        if pointwise is None:
            args = (*local, *local)
        else:
            args = (*local, *pointwise)
        _masked[(triton.cdiv(count, 256),)](*args, masks, masked, count, binary, width,
                                          pointwise is not None, local_joints, local_columns, 256)
        raw = masked.cpu().numpy().tobytes()
        size = count * (width // 8)
        payloads, hashes = {}, {}
        for j, k in enumerate(local_joints):
            sender, backup, receiver, _ = SCHEDULE[k]
            data = raw[j * size:(j + 1) * size]
            if self.party == sender:
                payloads.setdefault(receiver, []).append(data)
            else:
                hashes.setdefault(receiver, []).append((sender, hashlib.blake2b(data, digest_size=32).digest()))
        outgoing = {receiver: b''.join(payloads.get(receiver, [])) + b''.join(h for _, h in sorted(hashes.get(receiver, [])))
                    for receiver in set(payloads) | set(hashes)}
        incoming = self.transport.exchange(outgoing)
        offsets = {p: 0 for p in incoming}
        received, expected = [], []
        for k in receive_joints:
            sender, backup, _, _ = SCHEDULE[k]
            start = offsets[sender]
            data = incoming[sender][start:start + size]
            if len(data) != size:
                raise ValueError('truncated Rep4 joint input')
            offsets[sender] += size
            received.append(data)
            expected.append((backup, sender, hashlib.blake2b(data, digest_size=32).digest()))
        for backup, sender, digest in sorted(expected):
            start = offsets[backup]
            actual = incoming[backup][start:start + 32]
            offsets[backup] += 32
            if not secrets.compare_digest(actual, digest):
                raise ValueError(f'joint input hash mismatch for sender {sender} and backup {backup}')
        if any(offsets[p] != len(data) for p, data in incoming.items()):
            raise ValueError('unexpected trailing Rep4 message bytes')
        received_gpu = torch.frombuffer(bytearray(b''.join(received)), dtype=dtype).to(device)
        result = torch.empty((3, *shape), dtype=torch.int64, device=device)
        mask_slots = tuple(-1 if self.party == r else (r - self.party - 1) % 4 for _, _, r, _ in SCHEDULE)
        outside_slots = tuple(-1 if self.party == o else (o - self.party - 1) % 4 for _, _, _, o in SCHEDULE)
        local_ids = tuple(local_joints.index(k) if k in local_joints else -1 for k in range(6))
        receive_ids = tuple(receive_joints.index(k) if k in receive_joints else -1 for k in range(6))
        _assemble[(triton.cdiv(count, 256),)](masks, masked, received_gpu, result, count, binary, width,
                                            mask_slots, outside_slots, local_ids, receive_ids, 256)
        self.joint_inputs += 6
        self.multiplications += 1
        return Share(tuple(result.unbind(0)), binary)

    def mul(self, x, y):
        self._validate_share_storage(x, y)
        if x.device.type != 'cuda':
            return super().mul(x, y)
        if x.binary != y.binary:
            raise ValueError('cannot mix arithmetic and binary shares')
        shape = torch.broadcast_shapes(x.shape, y.shape)
        if math.prod(shape) == 0:
            if self._storage_dtype(x.binary) in (torch.uint8, torch.uint16):
                x = Share(tuple(t.to(torch.int64) for t in x.parts), x.binary)
                y = Share(tuple(t.to(torch.int64) for t in y.parts), y.binary)
            return super().mul(x, y)
        xp = tuple(torch.broadcast_to(t, shape).contiguous() for t in x.parts)
        yp = tuple(torch.broadcast_to(t, shape).contiguous() for t in y.parts)
        return self._fused_product(xp, x.binary, x.device, yp)

    def _reshare_product(self, a, binary, device):
        if torch.device(device).type != 'cuda' or a[4].numel() == 0:
            return super()._reshare_product(a, binary, device)
        local = tuple(a[k].contiguous() for k in ((self.party - 1) % 4, self.party, 4))
        return self._fused_product(local, binary, device)
