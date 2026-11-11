"""Checked Rep4 scalar and Ristretto operations for projected commitments.

Scalar vectors live over the Ristretto group order. Ring lifts reconstruct the
canonical unsigned ring64 word, rather than interpreting its sign bit as a
negative integer. This CPU implementation moves only projected ring shares
from a GPU runtime; it does not replace the runtime's tensor arithmetic.
"""
from dataclasses import dataclass
import hashlib
import secrets
import struct
import threading

from .runtime import SCHEDULE
from .ristretto import Q, IDENTITY, scalar_encode, scalar_decode, point_validate, point_add, point_sub, point_mul, base_mul

PEDERSEN_G = bytes.fromhex('e2f2ae0a6abc4e71a884a961c500515f58e30b6aa582dd8db6a65945e08d2d76')
PEDERSEN_H = bytes.fromhex('8c9240b456a9e6dc65c377a1048d745f94a08cdb7f44cbcd7b46f34048871134')


def check_pedersen_generators(generator, blinding_generator):
    """Require the native Bulletproof backend's fixed generator pair."""
    point_validate(generator)
    point_validate(blinding_generator)
    if generator != PEDERSEN_G or blinding_generator != PEDERSEN_H:
        raise ValueError('Pedersen generators do not match the pinned backend')


@dataclass(frozen=True)
class FieldShare:
    parts: tuple

    def __post_init__(self):
        if len(self.parts) != 3 or len({len(part) for part in self.parts}) != 1:
            raise ValueError('field shares require three equally sized components')
        if any(type(value) is not int or not 0 <= value < Q for part in self.parts for value in part):
            raise ValueError('field share components must be canonical scalars')

    @property
    def count(self):
        return len(self.parts[0])


@dataclass(frozen=True)
class GroupShare:
    parts: tuple

    def __post_init__(self):
        if len(self.parts) != 3 or len({len(part) for part in self.parts}) != 1:
            raise ValueError('group shares require three equally sized components')
        for part in self.parts:
            for point in part:
                point_validate(point)

    @property
    def count(self):
        return len(self.parts[0])


class FieldGroupRuntime:
    """Sequential checked group operations using an existing Rep4 transport."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.party = runtime.party
        self.transport = runtime.transport
        # A second wrapper around the same role continues the nonce stream.
        with runtime._counter_lock:
            if not hasattr(runtime, '_field_group_mask_state'):
                runtime._field_group_mask_state = ({}, threading.Lock())
            self._counters, self._lock = runtime._field_group_mask_state

    def _random(self, excluded, count, tag):
        if excluded == self.party:
            raise ValueError('excluded role cannot draw its component mask')
        key = (tag, excluded)
        with self._lock:
            nonce = self._counters.get(key, 0)
            if nonce >= 1 << 64:
                raise ValueError('field/group mask nonce exhausted')
            self._counters[key] = nonce+1
        context = b'Rep4/field-group/mask/v1'+struct.pack('<I', len(tag))+tag+struct.pack('<QQ', nonce, count)
        raw = hashlib.shake_256(self.runtime.seeds[excluded]+context).digest(64*count)
        return tuple(int.from_bytes(raw[i*64:(i+1)*64], 'little') % Q for i in range(count))

    def public_field(self, values):
        values = tuple(int(value) % Q for value in values)
        return FieldShare(tuple(values if (self.party+slot+1)%4 == 0 else (0,)*len(values)
                                for slot in range(3)))

    def public_group(self, values):
        values = tuple(point_validate(value) for value in values)
        return GroupShare(tuple(values if (self.party+slot+1)%4 == 0 else (IDENTITY,)*len(values)
                                for slot in range(3)))

    def secret_field(self, count):
        """Draw fresh PRSS components; each component excludes one role."""
        if type(count) is not int or count <= 0:
            raise ValueError('secret field count must be positive')
        return FieldShare(tuple(self._random((self.party+slot+1)%4, count, b'field-prss')
                                for slot in range(3)))

    def check_field_share(self, value):
        """Check that each externally supplied component agrees at its holders.

        This checks replicated consistency, not the client's claim about the
        reconstructed value. The commitment relation checks that claim.
        """
        known = {(self.party+slot+1)%4:part for slot,part in enumerate(value.parts)}
        digests = {excluded:hashlib.blake2b(
            b'Rep4/field-group/external-component/v1'+struct.pack('<BQ',excluded,value.count)+
            b''.join(scalar_encode(item) for item in part),digest_size=32).digest()
            for excluded,part in known.items()}
        outgoing = {peer:b''.join(digests[excluded] for excluded in sorted(known)
                                  if excluded != peer)
                    for peer in range(4) if peer != self.party}
        incoming = self.transport.exchange(outgoing)
        for peer in range(4):
            if peer == self.party:
                continue
            expected = b''.join(digests[excluded] for excluded in sorted(known)
                                if excluded != peer)
            if not secrets.compare_digest(incoming.get(peer,b''),expected):
                raise ValueError('external field share component mismatch')
        return value

    @staticmethod
    def field_add(left, right):
        if left.count != right.count:
            raise ValueError('field shares must have matching counts')
        return FieldShare(tuple(tuple((a+b)%Q for a,b in zip(xs,ys))
                                for xs,ys in zip(left.parts,right.parts)))

    @staticmethod
    def field_sub(left, right):
        if left.count != right.count:
            raise ValueError('field shares must have matching counts')
        return FieldShare(tuple(tuple((a-b)%Q for a,b in zip(xs,ys))
                                for xs,ys in zip(left.parts,right.parts)))

    @staticmethod
    def group_add(left, right):
        if left.count != right.count:
            raise ValueError('group shares must have matching counts')
        return GroupShare(tuple(tuple(point_add(a,b) for a,b in zip(xs,ys))
                                for xs,ys in zip(left.parts,right.parts)))

    @staticmethod
    def group_sub(left, right):
        if left.count != right.count:
            raise ValueError('group shares must have matching counts')
        return GroupShare(tuple(tuple(point_sub(a,b) for a,b in zip(xs,ys))
                                for xs,ys in zip(left.parts,right.parts)))

    def joint_input_batch(self, specs, count, *, group=False):
        """Share pair-known vectors, checking every payload against its backup."""
        if type(count) is not int or count < 0:
            raise ValueError('joint input count must be nonnegative')
        zero = IDENTITY if group else 0
        add = point_add if group else lambda a,b:(a+b)%Q
        sub = point_sub if group else lambda a,b:(a-b)%Q
        encode = point_validate if group else scalar_encode
        decode = point_validate if group else scalar_decode
        outputs, payloads, hashes = [], {}, {}
        for sender, backup, receiver, outsider, value in specs:
            if sorted((sender,backup,receiver,outsider)) != [0,1,2,3]:
                raise ValueError('joint input roles must be a permutation')
            parts = [[zero]*count for _ in range(3)]
            if self.party != receiver:
                masks = self._random(receiver, count, b'group-joint' if group else b'field-joint')
                if group:
                    masks = tuple(base_mul(mask) for mask in masks)
                slot = (receiver-self.party-1)%4
                parts[slot] = [add(a,b) for a,b in zip(parts[slot],masks)]
                if self.party in (sender,backup):
                    if value is None or len(value) != count:
                        raise ValueError('joint input holders require the complete vector')
                    masked = tuple(sub(a,b) for a,b in zip(value,masks))
                    slot = (outsider-self.party-1)%4
                    parts[slot] = [add(a,b) for a,b in zip(parts[slot],masked)]
                    data = b''.join(encode(item) for item in masked)
                    if self.party == sender:
                        payloads.setdefault(receiver,[]).append(data)
                    else:
                        hashes.setdefault(receiver,[]).append((sender,hashlib.blake2b(data,digest_size=32).digest()))
            outputs.append(parts)
        outgoing = {receiver:b''.join(payloads.get(receiver,[]))+b''.join(
            digest for _,digest in sorted(hashes.get(receiver,[])))
            for receiver in set(payloads)|set(hashes)}
        incoming = self.transport.exchange(outgoing)
        offsets = {peer:0 for peer in incoming}
        expected = []
        for index,(sender,backup,receiver,outsider,_) in enumerate(specs):
            if self.party == receiver:
                start = offsets.get(sender,0)
                data = incoming.get(sender,b'')[start:start+32*count]
                if len(data) != 32*count:
                    raise ValueError('truncated field/group joint input')
                offsets[sender] = start+32*count
                slot = (outsider-self.party-1)%4
                decoded = tuple(decode(data[i*32:(i+1)*32]) for i in range(count))
                outputs[index][slot] = [add(a,b) for a,b in zip(outputs[index][slot],decoded)]
                expected.append((backup,sender,hashlib.blake2b(data,digest_size=32).digest()))
        for backup,sender,digest in sorted(expected):
            start = offsets.get(backup,0)
            actual = incoming.get(backup,b'')[start:start+32]
            offsets[backup] = start+32
            if not secrets.compare_digest(actual,digest):
                raise ValueError('field/group joint input hash mismatch')
        if any(offsets[peer] != len(data) for peer,data in incoming.items()):
            raise ValueError('trailing field/group joint input bytes')
        share_type = GroupShare if group else FieldShare
        return [share_type(tuple(tuple(part) for part in parts)) for parts in outputs]

    def _reshare(self, local, *, group=False):
        contributions = [None]*5
        contributions[(self.party-1)%4], contributions[self.party], contributions[4] = local
        specs = [(*entry,contributions[i if i<4 else 4]) for i,entry in enumerate(SCHEDULE)]
        shares = self.joint_input_batch(specs,len(local[0]),group=group)
        add = self.group_add if group else self.field_add
        result = shares[0]
        for share in shares[1:]:
            result = add(result,share)
        return result

    def field_mul(self, left, right):
        if left.count != right.count:
            raise ValueError('field multiplication requires matching counts')
        x0,x1,x2 = left.parts
        y0,y1,y2 = right.parts
        local = tuple(tuple(value%Q for value in values) for values in (
            ((a+b)*c+a*d for a,b,c,d in zip(x0,x1,y0,y1)),
            ((a+b)*c+a*d for a,b,c,d in zip(x1,x2,y1,y2)),
            (a*d+b*c for a,b,c,d in zip(x0,x2,y0,y2))))
        return self._reshare(local)

    def field_weighted_dot(self, left, right, weights=None):
        """Reduce the bilinear dot locally before its six checked joint inputs."""
        if left.count != right.count:
            raise ValueError('field dot requires matching counts')
        weights = (1,)*left.count if weights is None else tuple(int(x)%Q for x in weights)
        if len(weights) != left.count:
            raise ValueError('field dot weights require one entry per coordinate')
        x0,x1,x2 = left.parts
        y0,y1,y2 = right.parts
        local = tuple((sum(values)%Q,) for values in (
            (w*((a+b)*c+a*d) for w,a,b,c,d in zip(weights,x0,x1,y0,y1)),
            (w*((a+b)*c+a*d) for w,a,b,c,d in zip(weights,x1,x2,y1,y2)),
            (w*(a*d+b*c) for w,a,b,c,d in zip(weights,x0,x2,y0,y2))))
        return self._reshare(local)

    def lift_ring64(self, value):
        """Lift projected ring words to their exact unsigned scalar values."""
        if self.runtime.bits != 64 or value.binary or value.parts[0].numel() == 0:
            raise ValueError('ring lift requires nonempty arithmetic ring64 shares')
        if self.runtime._wire_bits(True) != 64:
            raise ValueError('ring lift requires full 64-bit binary wires')
        if value.parts[0].numel() > 256:
            raise ValueError('ring lift accepts at most 256 projected scalars')
        from .conversion import a2b
        word = a2b(self.runtime,value)
        parts = tuple(part.detach().contiguous().cpu().reshape(-1).tolist() for part in word.parts)
        count = len(parts[0])
        planes = tuple(tuple((item >> bit)&1 for bit in range(64) for item in part) for part in parts)
        known = {(self.party+slot+1)%4:planes[slot] for slot in range(3)}
        p01 = tuple(a*b for a,b in zip(known[0],known[1])) if self.party in (2,3) else None
        p23 = tuple(a*b for a,b in zip(known[2],known[3])) if self.party in (0,1) else None
        q,r = self.joint_input_batch([(2,3,1,0,p01),(0,1,3,2,p23)],64*count)
        left_parts, right_parts = [], []
        for slot in range(3):
            excluded = (self.party+slot+1)%4
            component = planes[slot]
            left_parts.append(tuple(((bit if excluded in (0,1) else 0)-2*prod)%Q
                                    for bit,prod in zip(component,q.parts[slot])))
            right_parts.append(tuple(((bit if excluded in (2,3) else 0)-2*prod)%Q
                                     for bit,prod in zip(component,r.parts[slot])))
        left,right = FieldShare(tuple(left_parts)),FieldShare(tuple(right_parts))
        x0,x1,x2 = left.parts
        y0,y1,y2 = right.parts
        local = [[],[],[]]
        linear = [[],[],[]]
        for item in range(count):
            totals = [0,0,0]
            sums = [0,0,0]
            for bit in range(64):
                i = bit*count+item
                weight = 1 << bit
                totals[0] += weight*((x0[i]+x1[i])*y0[i]+x0[i]*y1[i])
                totals[1] += weight*((x1[i]+x2[i])*y1[i]+x1[i]*y2[i])
                totals[2] += weight*(x0[i]*y2[i]+x2[i]*y0[i])
                for slot in range(3):
                    sums[slot] += weight*(left.parts[slot][i]+right.parts[slot][i])
            for slot in range(3):
                local[slot].append(totals[slot]%Q)
                linear[slot].append(sums[slot]%Q)
        product = self._reshare(tuple(tuple(part) for part in local))
        return FieldShare(tuple(tuple((value-2*prod)%Q for value,prod in zip(xs,ys))
                                for xs,ys in zip(linear,product.parts)))

    def group_linear(self, values, blindings, generator, blinding_generator, commitments):
        """Share G*value+H*blinding-C without opening individual differences."""
        if values.count != blindings.count or len(commitments) != values.count:
            raise ValueError('commitment binding requires matching counts')
        point_validate(generator)
        point_validate(blinding_generator)
        combined = GroupShare(tuple(tuple(point_add(point_mul(generator,a),point_mul(blinding_generator,b))
                                          for a,b in zip(xs,ys))
                                    for xs,ys in zip(values.parts,blindings.parts)))
        return self.group_sub(combined,self.public_group(commitments))

    def group_weighted_dot(self, scalars, points):
        """Compute a secret scalar/group dot with six checked point payloads."""
        if scalars.count != points.count:
            raise ValueError('group dot requires matching counts')
        x0,x1,x2 = scalars.parts
        y0,y1,y2 = points.parts
        local = [IDENTITY]*3
        for a,b,c,p,r,s in zip(x0,x1,x2,y0,y1,y2):
            terms = (point_add(point_mul(p,(a+b)%Q),point_mul(r,a)),
                     point_add(point_mul(r,(b+c)%Q),point_mul(s,b)),
                     point_add(point_mul(s,a),point_mul(p,c)))
            for slot in range(3):
                local[slot] = point_add(local[slot],terms[slot])
        return self._reshare(tuple((point,) for point in local),group=True)

    def _open(self, value, *, group=False):
        encode = point_validate if group else scalar_encode
        decode = point_validate if group else scalar_decode
        data = b''.join(encode(item) for item in value.parts[0])
        incoming = self.transport.exchange({(self.party+1)%4:data})
        missing = incoming.get((self.party-1)%4,b'')
        own_check = b''.join(encode(item) for item in value.parts[2])
        checks = self.transport.exchange({(self.party-1)%4:hashlib.blake2b(own_check,digest_size=32).digest()})
        if len(missing) != value.count*32 or not secrets.compare_digest(
                hashlib.blake2b(missing,digest_size=32).digest(),checks.get((self.party+1)%4,b'')):
            raise ValueError('field/group opening hash mismatch')
        result = [decode(missing[i*32:(i+1)*32]) for i in range(value.count)]
        add = point_add if group else lambda a,b:(a+b)%Q
        for part in value.parts:
            result = [add(a,b) for a,b in zip(result,part)]
        return tuple(result)

    def open_field(self, value):
        return self._open(value)

    def open_group(self, value):
        return self._open(value,group=True)

    def check_group_zero(self, differences):
        """Open only a fresh secret-weighted aggregate commitment difference.

        With ideal uniform hidden weights, a nonzero vector aggregates to a
        uniform group point and passes with probability 1/Q. PRSS derives its
        weights computationally from independent seeds; 64-byte reduction has
        negligible statistical bias. Individual differences remain private.
        """
        if differences.count <= 0:
            raise ValueError('group check requires nonempty differences')
        alpha = self.secret_field(differences.count)
        aggregate = self.group_weighted_dot(alpha,differences)
        return self.open_group(aggregate) == (IDENTITY,)
