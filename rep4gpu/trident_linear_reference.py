"""Ring32 Trident linear algebra with trusted test-only preprocessing.

This experiment measures serialized online messages between four role-local
objects. The fixture knows every input and mask and supplies unchecked mask
products; it does not implement Trident's distributed preprocessing or establish
malicious security. Do not use it with private data. Truncation and ReLU are not
implemented here.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import secrets

import numpy as np

from .transport import IdealNetwork


def ring(value):
    return np.asarray(value).astype(np.uint32)


def _bytes(value):
    return np.asarray(value, dtype='<u4').tobytes()


def _hash(value):
    return hashlib.blake2b(value, digest_size=32).digest()


def _next(p):
    return p % 3 + 1


def _previous(p):
    return (p + 1) % 3 + 1


@dataclass
class Share:
    masked: np.ndarray | None
    masks: dict
    shape: tuple
    token: str


@dataclass
class PreparedProduct:
    gamma: dict
    output_masks: dict
    output_shape: tuple
    input_tokens: tuple
    output_token: str
    used: bool = False


class TrustedFixture:
    """Deterministic test masks; no preprocessing network is simulated."""
    def __init__(self, seed=1):
        self.rng = np.random.default_rng(seed)
        self.counter = 0

    def _token(self):
        self.counter += 1
        return f'test-wire-{self.counter}'

    def _random(self, shape):
        return self.rng.integers(0, 1 << 32, shape, dtype=np.uint32)

    def _masks(self, shape):
        return {i: self._random(shape) for i in (1, 2, 3)}

    @staticmethod
    def _local(parts, party):
        return {i: v.copy() for i, v in parts.items() if i != party}

    def share(self, value):
        value = ring(value)
        masks = self._masks(value.shape)
        masked = value + masks[1] + masks[2] + masks[3]
        token = self._token()
        return tuple(Share(None if p == 0 else masked.copy(),
                           self._local(masks, p), value.shape, token)
                     for p in range(4))

    def prepare(self, xs, ys):
        if len(xs[0].shape) != 2 or len(ys[0].shape) != 2 or xs[0].shape[1] != ys[0].shape[0]:
            raise ValueError('matmul requires compatible matrices')
        lx = sum(xs[0].masks.values(), np.zeros(xs[0].shape, dtype=np.uint32))
        ly = sum(ys[0].masks.values(), np.zeros(ys[0].shape, dtype=np.uint32))
        product = lx @ ly
        shape = product.shape
        gamma = {1: self._random(shape), 2: self._random(shape)}
        gamma[3] = product - gamma[1] - gamma[2]
        masks = self._masks(shape)
        token = self._token()
        return tuple(PreparedProduct(self._local(gamma, p), self._local(masks, p),
                     shape, (xs[p].token, ys[p].token), token) for p in range(4))


class Party:
    """An online role consuming trusted fixture correlations once."""
    def __init__(self, party, transport):
        self.party, self.transport = party, transport

    def _checked(self, incoming, shape):
        data = incoming.get(_next(self.party), b'')
        digest = incoming.get(_previous(self.party), b'')
        if set(incoming) != {_next(self.party), _previous(self.party)}:
            raise ValueError('unexpected checked-message peers')
        if len(data) != int(np.prod(shape)) * 4:
            raise ValueError('invalid ring32 payload length')
        if not secrets.compare_digest(_hash(data), digest):
            raise ValueError('online consistency check failed')
        return np.frombuffer(data, dtype='<u4').reshape(shape).copy()

    def matmul(self, x, y, prepared):
        try:
            if prepared.used:
                raise ValueError('preprocessing is single-use')
            if prepared.input_tokens != (x.token, y.token):
                raise ValueError('preprocessing belongs to different input wires')
            prepared.used = True
            outgoing = {}
            local = {}
            if self.party:
                for i in x.masks:
                    local[i] = (prepared.gamma[i] + prepared.output_masks[i]
                                - x.masked @ y.masks[i] - x.masks[i] @ y.masked)
                outgoing[_previous(self.party)] = _bytes(local[_previous(self.party)])
                outgoing[_next(self.party)] = _hash(_bytes(local[_next(self.party)]))
            incoming = self.transport.exchange(outgoing)
            masked = None
            if self.party:
                missing = self._checked(incoming, prepared.output_shape)
                masked = x.masked @ y.masked + missing
                for value in local.values():
                    masked = masked + value
            # Zero-byte rendezvous propagates a failed check before any role
            # starts a dependent operation in this in-process experiment.
            self.transport.exchange({})
            return Share(masked, prepared.output_masks, prepared.output_shape,
                         prepared.output_token)
        except BaseException as error:
            self.transport.network.abort(error)
            raise

    def open(self, x):
        try:
            outgoing = {}
            if self.party:
                outgoing[_previous(self.party)] = _bytes(x.masks[_previous(self.party)])
                outgoing[_next(self.party)] = _hash(_bytes(x.masks[_next(self.party)]))
                if self.party == 1:
                    outgoing[0] = _bytes(x.masked)
                elif self.party == 2:
                    outgoing[0] = _hash(_bytes(x.masked))
            incoming = self.transport.exchange(outgoing)
            if self.party:
                missing = self._checked(incoming, x.shape)
                value = x.masked - missing
            else:
                data = incoming.get(1, b'')
                if set(incoming) != {1, 2} or len(data) != int(np.prod(x.shape))*4:
                    raise ValueError('invalid output frame')
                if not secrets.compare_digest(_hash(data), incoming[2]):
                    raise ValueError('output consistency check failed')
                value = np.frombuffer(data, dtype='<u4').reshape(x.shape).copy()
            for mask in x.masks.values():
                value = value - mask
            self.transport.exchange({})
            return value
        except BaseException as error:
            self.transport.network.abort(error)
            raise


def run_matmul(x, y, *, seed=1, mutate=None):
    """Return an opened result and measured application payload counts.

    ``mutate(party, round, outgoing)`` can alter serialized messages for fault
    tests. Input sharing and fixture preprocessing use no transport and are
    explicitly excluded from the reported bytes. Wall time is not a distributed
    performance measurement.
    """
    fixture = TrustedFixture(seed)
    xs, ys = fixture.share(x), fixture.share(y)
    prepared = fixture.prepare(xs, ys)
    network = IdealNetwork(timeout=10)
    parties = [Party(p, network.endpoint(p)) for p in range(4)]
    if mutate is not None:
        for party in parties:
            original = party.transport.exchange
            def exchange(outgoing, p=party, original=original):
                return original(mutate(p.party, p.transport.round, dict(outgoing)))
            party.transport.exchange = exchange

    def run(p):
        product = p.matmul(xs[p.party], ys[p.party], prepared[p.party])
        online_bytes = p.transport.bytes_sent
        opened = p.open(product)
        return opened, online_bytes, p.transport.bytes_sent - online_bytes

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run, p) for p in parties]
        results = [future.result() for future in futures]
    if any(not np.array_equal(results[0][0], r[0]) for r in results[1:]):
        raise AssertionError('opened outputs disagree')
    return dict(output=results[0][0], online_payload_bytes=sum(r[1] for r in results),
                opening_payload_bytes=sum(r[2] for r in results),
                online_bytes_per_party=[r[1] for r in results],
                online_exchange_rounds=2, opening_exchange_rounds=2,
                preprocessing='trusted deterministic fixture; no distributed preprocessing executed',
                preprocessing_payload_bytes=None, input_payload_bytes=None,
                test_only=True)
