"""Experimental GPU-resident Rep4 online protocol using ChaCha20 and BLAKE3.

This variant preserves the six joint inputs, recipients, ring encodings, and
32-byte integrity tags. Its BLAKE3 tags and ChaCha20 masks are deliberately not
byte compatible with the reference Party's BLAKE2b/SHAKE transcript. Private
input and explicit output retain the inherited checked BLAKE2b byte protocol.
"""
import math
import torch
import triton
import triton.language as tl
from .runtime import Share, SCHEDULE
from .fused_runtime import FusedParty, _masked, _assemble


# The four nonzero cross-pair products keep their two knowledgeable holders.
# Orient their routes so every party sends and receives one payload and tag.
_SPARSE_SCHEDULE = ((2,1,0,3), (3,0,1,2), (0,2,3,1), (1,3,2,0))


def mask_domain(bits, binary_bits, binary, excluded):
    """Canonical public domain for one excluded party's correlated masks."""
    width = binary_bits if binary else bits
    return bytes((bits, binary_bits, int(binary), width, excluded))


@triton.jit
def _joint_prepare(VALUE, MASK, WIRE, OUT,
                    N: tl.constexpr, WIDTH: tl.constexpr, BINARY: tl.constexpr,
                    MASK_SLOT: tl.constexpr, OUTSIDE_SLOT: tl.constexpr,
                    HAS_VALUE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    mask = tl.load(MASK+i, i < N, 0).to(tl.uint64)
    if HAS_VALUE:
        value = tl.load(VALUE+i, i < N, 0).to(tl.uint64)
        masked = value ^ mask if BINARY else value-mask
        if WIDTH == 32:
            masked &= 0xffffffff
        tl.store(WIRE+i, masked, i < N)
    for slot in tl.static_range(3):
        result = tl.full((BLOCK,), 0, tl.uint64)
        if slot == MASK_SLOT:
            result = mask
        if HAS_VALUE and slot == OUTSIDE_SLOT:
            result = masked
        tl.store(OUT+slot*N+i, result.to(tl.int64), i < N)


@triton.jit
def _joint_received(WIRE, OUT, N: tl.constexpr, SLOT: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    value = tl.load(WIRE+i, i < N, 0).to(tl.int64)
    for slot in tl.static_range(3):
        result = value if slot == SLOT else tl.full((BLOCK,), 0, tl.int64)
        tl.store(OUT+slot*N+i, result, i < N)


class DeviceParty(FusedParty):
    """GPU online masks/tags with a device-aware four-party transport.

    The transport must offer both ``exchange`` for inherited byte operations
    and ``exchange_device`` for online CUDA byte buffers. GPU tag comparison
    synchronizes one Boolean result; payloads and masks stay on the device.
    Each live call must advance host draw counters; replaying a captured online
    call would reuse its nonce and is limited to fixed benchmark fixtures.
    Like Party, this class supports one protocol operation at a time per role;
    atomic reservations prevent nonce reuse but do not reorder protocol rounds.
    """
    def __init__(self, party, transport, seeds, device='cuda', bits=64, binary_bits=64, hash_block_tree=True, prg_batch=False):
        if torch.device(device).type != 'cuda':
            raise ValueError('DeviceParty requires a CUDA device')
        if not hasattr(transport, 'exchange_device'):
            raise ValueError('DeviceParty requires a device-aware transport')
        device = torch.device(device)
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        super().__init__(party, transport, seeds, device=device, bits=bits,
                         binary_bits=binary_bits, prg='shake')
        from ..gpu.crypto.gpu_prg import DomainKeys
        self.prg = 'chacha20'
        self.device_hash = 'blake3'
        self.hash_block_tree = bool(hash_block_tree)
        self.prg_batch = bool(prg_batch)
        self._batch_keys = {}
        self._device_keys = {excluded: DomainKeys(seed, self.device)
                             for excluded, seed in self.seeds.items()}

    def _random(self, excluded, shape, device, binary=False, *, packed=False):
        if torch.device(device) != self.device:
            raise ValueError('DeviceParty masks must stay on its CUDA device')
        from ..gpu.crypto.gpu_prg import generate
        count = math.prod(shape)
        counter = self._reserve_nonces((excluded,))[0]
        domain = mask_domain(self.bits, self.binary_bits, binary, excluded)
        key = self._device_keys[excluded].key(domain)
        if packed and self._wire_bits(binary) == 32:
            from ..gpu.crypto.gpu_prg_batch import generate_rows
            result = generate_rows((key,), (counter,), count, bits=32, packed=True)[0]
        else:
            result = generate(key, counter, count, bits=self._wire_bits(binary))
        return result.reshape(shape)

    def _random_bytes(self, excluded, shape, binary=False):
        raise RuntimeError('DeviceParty generates masks directly on CUDA')

    def _mask_batch(self, shape, binary, device, *, schedule=SCHEDULE):
        count = math.prod(shape)
        if self.prg_batch:
            from ..gpu.crypto.gpu_prg_batch import generate_rows
            excluded = tuple(receiver for _, _, receiver, _ in schedule if self.party != receiver)
            reservations = iter(self._reserve_nonces(excluded))
            nonces = tuple(None if self.party == receiver else next(reservations)
                           for _, _, receiver, _ in schedule)
            cache_key = (self.bits, self.binary_bits, bool(binary), schedule)
            if cache_key not in self._batch_keys:
                self._batch_keys[cache_key] = tuple(
                    None if self.party == receiver else self._device_keys[receiver].key(
                        mask_domain(self.bits, self.binary_bits, binary, receiver))
                    for _, _, receiver, _ in schedule)
            return generate_rows(self._batch_keys[cache_key], nonces, count,
                                 bits=self._wire_bits(binary), packed=True).reshape(-1)
        masks = [torch.zeros(count, dtype=torch.int64, device=device) if self.party == receiver
                 else self._random(receiver, (count,), device, binary)
                 for _, _, receiver, _ in schedule]
        return torch.stack(masks).reshape(-1)

    def _exchange_device(self, outgoing, incoming_sizes):
        return self.transport.exchange_device(outgoing, incoming_sizes)

    def _verify_tags(self, actual, wanted, message='device Rep4 joint input BLAKE3 hash mismatch'):
        if not torch.equal(actual, wanted):
            error = ValueError(message)
            abort = getattr(self.transport, 'abort', None)
            if abort is not None:
                abort(error)
            raise error

    def _exchange_joint_wires(self, entries, shape, binary, *, prepared_outgoing=None):
        """Send masked payloads and backup tags, then verify every received tag.

        entries are (sender, backup, receiver, CUDA uint8 payload or None), in
        the public joint-input order. The returned list has a payload only at
        each joint's receiver. All framing decisions use public metadata.
        A graph backend can supply its already-computed outgoing buffers;
        incoming validation and digest checks still execute here.
        """
        from ..gpu.crypto.gpu_hash import digest_batch

        def hash_payloads(payloads):
            # Every joint in this exchange has the same public wire size.
            rows = payloads[0].unsqueeze(0) if len(payloads) == 1 else torch.stack(payloads)
            return digest_batch(rows, block_tree=self.hash_block_tree)
        count = math.prod(shape)
        size = count*(self._wire_bits(binary)//8)
        payloads, hashes = {}, {}
        backup_payloads = []
        incoming_sizes = {}
        for sender, backup, receiver, data in entries:
            if self.party == sender and prepared_outgoing is None:
                payloads.setdefault(receiver, []).append(data)
            if self.party == backup and prepared_outgoing is None:
                backup_payloads.append((receiver, sender, data))
            if self.party == receiver:
                incoming_sizes[sender] = incoming_sizes.get(sender, 0)+size
                incoming_sizes[backup] = incoming_sizes.get(backup, 0)+32
        if backup_payloads:
            tags = hash_payloads([data for _, _, data in backup_payloads])
            for (receiver, sender, _), tag in zip(backup_payloads, tags.unbind(0)):
                hashes.setdefault(receiver, []).append((sender, tag))
        outgoing = prepared_outgoing
        if outgoing is None:
            outgoing = {}
            for receiver in set(payloads) | set(hashes):
                pieces = payloads.get(receiver, []) + [tag for _, tag in sorted(hashes.get(receiver, []), key=lambda item: item[0])]
                outgoing[receiver] = pieces[0] if len(pieces) == 1 else torch.cat(pieces)
        incoming = self._exchange_device(outgoing, incoming_sizes)
        if set(incoming) != set(incoming_sizes) or any(
                incoming[p].numel() != n or incoming[p].dtype != torch.uint8
                or incoming[p].device != self.device or incoming[p].ndim != 1
                or not incoming[p].is_contiguous() for p, n in incoming_sizes.items()):
            raise ValueError('invalid device Rep4 incoming frame sizes')
        offsets = {p: 0 for p in incoming}
        received, expected = [None]*len(entries), []
        for k, (sender, backup, receiver, _) in enumerate(entries):
            if self.party == receiver:
                start = offsets[sender]
                data = incoming[sender][start:start+size]
                offsets[sender] += size
                received[k] = data
                expected.append((backup, sender, data))
        expected.sort(key=lambda item: item[:2])
        wanted = hash_payloads([data for _, _, data in expected]) if expected else None
        actual = []
        for backup, sender, _ in expected:
            start = offsets[backup]
            actual.append(incoming[backup][start:start+32])
            offsets[backup] += 32
        if actual:
            self._verify_tags(torch.stack(actual), wanted)
        if any(offsets[p] != incoming[p].numel() for p in incoming):
            raise ValueError('unexpected trailing device Rep4 message bytes')
        return received

    def _fused_product(self, local, binary, device, pointwise=None, *, joints=tuple(range(6)), schedule=None):
        shape, count = local[0].shape, local[0].numel()
        if not count:
            raise ValueError('device multiplication requires a nonempty tensor')
        width = self._wire_bits(binary)
        dtype = torch.uint32 if width == 32 else torch.int64
        schedule = tuple(SCHEDULE[k] for k in joints) if schedule is None else schedule
        local_joints = tuple(k for k, (s, b, _, _) in enumerate(schedule) if self.party in (s, b))
        columns = tuple(0 if joints[k] == (self.party-1)%4 else 1 if joints[k] == self.party else 2 for k in local_joints)
        receive_joints = tuple(k for k, (_, _, r, _) in enumerate(schedule) if self.party == r)
        masks = self._mask_batch(shape, binary, device, schedule=schedule)
        masked = torch.empty((len(local_joints), count), dtype=dtype, device=device)
        args = (*local, *(local if pointwise is None else pointwise))
        _masked[(triton.cdiv(count, 256),)](*args, masks, masked, count, binary, width,
                                          pointwise is not None, local_joints, columns, 256)
        wires = {k: masked[j].view(torch.uint8) for j, k in enumerate(local_joints)}
        entries = [(s, b, r, wires.get(k)) for k, (s, b, r, _) in enumerate(schedule)]
        received = self._exchange_joint_wires(entries, shape, binary)
        incoming = (torch.cat([received[k] for k in receive_joints]).view(dtype)
                    if receive_joints else torch.empty(0, dtype=dtype, device=device))
        result = torch.empty((3, *shape), dtype=torch.int64, device=device)
        mask_slots = tuple(-1 if self.party == r else (r-self.party-1)%4 for _, _, r, _ in schedule)
        outside_slots = tuple(-1 if self.party == o else (o-self.party-1)%4 for _, _, _, o in schedule)
        local_ids = tuple(local_joints.index(k) if k in local_joints else -1 for k in range(len(schedule)))
        receive_ids = tuple(receive_joints.index(k) if k in receive_joints else -1 for k in range(len(schedule)))
        _assemble[(triton.cdiv(count, 256),)](masks, masked, incoming, result, count, binary, width,
                                            mask_slots, outside_slots, local_ids, receive_ids, 256)
        self.joint_inputs += len(schedule)
        self.multiplications += 1
        return Share(tuple(result.unbind(0)), binary)

    def _reshare_product_selected(self, contributions, binary, device, joints=(1,3,4,5)):
        """Reshare complementary-support products, omitting public zero terms.

        The caller must establish that global contributions 0 and 2 are zero.
        Each retained term keeps its two knowledgeable holders; balanced routes
        give each party one payload and one integrity tag in each direction.
        """
        if tuple(joints) != (1,3,4,5):
            raise ValueError('selected products require joint indices (1,3,4,5)')
        local = tuple(contributions[k].contiguous() for k in ((self.party-1)%4,self.party,4))
        if not local[0].numel():
            return self.public(0, like=Share(local, binary), binary=binary)
        return self._fused_product(local, binary, device, joints=tuple(joints), schedule=_SPARSE_SCHEDULE)

    def _mul_sparse_pairs(self, x, y):
        """Multiply shares supported on complementary component pairs {0,1}/{2,3}.

        This internal operation relies on support established by joint-input
        construction, not on inspecting secret tensor values.
        """
        if x.binary != y.binary or x.shape != y.shape or x.device != y.device or x.device != self.device:
            raise ValueError('sparse pairs require matching shares on the party device')
        if not x.parts[0].numel():
            return self.public(0, like=x)
        return self._fused_product(tuple(t.contiguous() for t in x.parts), x.binary, x.device,
                                   tuple(t.contiguous() for t in y.parts), joints=(1,3,4,5),
                                   schedule=_SPARSE_SCHEDULE)

    def joint_input_batch(self, specs, shape=None, binary=False, device=None):
        device = self.device if device is None else torch.device(device)
        if device != self.device:
            raise ValueError('DeviceParty joint inputs must stay on its CUDA device')
        if shape is None:
            shape = next(value.shape for *_, value in specs if value is not None)
        shape, width = tuple(shape), self._wire_bits(binary)
        count = math.prod(shape)
        dtype = torch.uint32 if width == 32 else torch.int64
        results, entries = [], []
        for sender, backup, receiver, outsider, value in specs:
            output = torch.empty((3, *shape), dtype=torch.int64, device=device)
            wire = None
            if self.party != receiver:
                mask = self._random(receiver, shape, device, binary, packed=True).reshape(-1)
                has_value = self.party in (sender, backup)
                wire = torch.empty(count if has_value else 0, dtype=dtype, device=device)
                source = value.contiguous().reshape(-1) if has_value else mask
                if count:
                    _joint_prepare[(triton.cdiv(count, 256),)](source, mask, wire, output, count, width, binary,
                                                              (receiver-self.party-1)%4,
                                                              (outsider-self.party-1)%4, has_value, 256)
            results.append(output)
            entries.append((sender, backup, receiver, None if wire is None else wire.view(torch.uint8)))
        received = self._exchange_joint_wires(entries, shape, binary)
        for k, (_, _, receiver, outsider, _) in enumerate(specs):
            if self.party == receiver and count:
                wire = received[k].view(dtype)
                _joint_received[(triton.cdiv(count, 256),)](wire, results[k], count, (outsider-self.party-1)%4, 256)
        self.joint_inputs += len(specs)
        return [Share(tuple(output.unbind(0)), binary) for output in results]
