"""Commit/reveal setup for the four Rep4 correlated PRG seeds.

Each excluded-party seed is the XOR of fresh contributions from its three
holders. The excluded party receives neither contributions nor commitments for
that seed. This setup requires authenticated confidential peer connections.
"""
import hashlib
import secrets


class SetupError(ValueError):
    pass


def _hash(context, tag, excluded, sender, value):
    return hashlib.blake2b(b'Rep4GPU-setup-v1' + len(context).to_bytes(8, 'little')
                           + context + tag + bytes([excluded, sender]) + value,
                           digest_size=32).digest()


def _frame(tag, records):
    return tag + b''.join(bytes([excluded]) + value for excluded, value in sorted(records.items()))


def _parse(payload, tag, expected):
    if not payload.startswith(tag) or len(payload) != len(tag) + 33 * len(expected):
        raise SetupError('malformed correlated-seed setup frame')
    result = {}
    for offset in range(len(tag), len(payload), 33):
        excluded = payload[offset]
        if excluded not in expected or excluded in result:
            raise SetupError('unexpected correlated-seed setup record')
        result[excluded] = payload[offset + 1:offset + 33]
    if set(result) != set(expected):
        raise SetupError('missing correlated-seed setup record')
    return result


def _agreement(transport, party, valid):
    outgoing = {p: b'R4OK' + bytes([bool(valid)]) for p in range(4) if p != party}
    incoming = transport.exchange(outgoing)
    if not valid or set(incoming) != set(outgoing) or any(v != b'R4OK\x01' for v in incoming.values()):
        raise SetupError('correlated-seed setup transcript check failed')


def correlated_seeds(party, transport, *, context=b'', _allow_insecure_test_transport=False):
    """Return the three seeds held by ``party`` after checked joint setup.

    ``context`` is a public session identifier shared by all four parties.
    Every call uses fresh OS entropy. SocketTransport must use mutually
    authenticated TLS with distinct party certificate names in deployment.
    The private test override exists for deterministic protocol fault tests.
    """
    if party not in range(4):
        raise ValueError('party must be in [0, 3]')
    if not isinstance(context, bytes):
        raise TypeError('context must be bytes')
    if not getattr(transport, 'authenticated', False) and not _allow_insecure_test_transport:
        raise ValueError('correlated seed setup requires authenticated confidential transport')
    holders = set(range(4)) - {party}
    contributions = {excluded: secrets.token_bytes(32) for excluded in holders}
    commits = {excluded: {party: _hash(context, b'commit', excluded, party, value)}
               for excluded, value in contributions.items()}

    def exchange_records(tag, own):
        outgoing = {peer: _frame(tag, {excluded: own[excluded] for excluded in holders if excluded != peer})
                    for peer in range(4) if peer != party}
        incoming = transport.exchange(outgoing)
        valid = set(incoming) == set(outgoing)
        received = {}
        for peer in outgoing:
            try:
                received[peer] = _parse(incoming.get(peer, b''), tag, holders - {peer})
            except SetupError:
                valid = False
        return received, valid

    received, valid = exchange_records(b'R4C1', {e: c[party] for e, c in commits.items()})
    for peer, records in received.items():
        for excluded, commitment in records.items():
            commits[excluded][peer] = commitment
    digests = {}
    for excluded in holders:
        transcript = b''.join(bytes([p]) + commits[excluded].get(p, bytes(32))
                              for p in range(4) if p != excluded)
        digests[excluded] = _hash(context, b'commit-transcript', excluded, 0, transcript)
    checks, check_valid = exchange_records(b'R4C2', digests)
    valid &= check_valid and all(value == digests[e] for records in checks.values() for e, value in records.items())
    _agreement(transport, party, valid)

    revealed, valid = exchange_records(b'R4R1', contributions)
    values = {excluded: {party: value} for excluded, value in contributions.items()}
    for peer, records in revealed.items():
        for excluded, value in records.items():
            values[excluded][peer] = value
            valid &= secrets.compare_digest(_hash(context, b'commit', excluded, peer, value), commits[excluded][peer])
    digests = {}
    for excluded in holders:
        transcript = b''.join(bytes([p]) + values[excluded].get(p, bytes(32))
                              for p in range(4) if p != excluded)
        digests[excluded] = _hash(context, b'reveal-transcript', excluded, 0, transcript)
    checks, check_valid = exchange_records(b'R4R2', digests)
    valid &= check_valid and all(value == digests[e] for records in checks.values() for e, value in records.items())
    _agreement(transport, party, valid)
    seeds = {}
    for excluded in holders:
        seed = bytearray(32)
        for value in values[excluded].values():
            for i, byte in enumerate(value):
                seed[i] ^= byte
        seeds[excluded] = bytes(seed)
    return seeds
