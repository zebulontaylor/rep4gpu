"""SHA-256 Merkle commitments to ordered byte strings.

The wire format uses separate leaf, node, and root domains. Leaf indices and
leaf counts are unsigned 64-bit big-endian integers. Odd levels duplicate the
last hash, and proofs include that duplicate explicitly.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import hashlib
import hmac


_MAX_COUNT = (1 << 64) - 1
_LEAF_DOMAIN = b'rep4gpu.merkle.v1.leaf\x00'
_NODE_DOMAIN = b'rep4gpu.merkle.v1.node\x00'
_ROOT_DOMAIN = b'rep4gpu.merkle.v1.root\x00'


def _leaf_hash(index: int, leaf: bytes) -> bytes:
    digest = hashlib.sha256(_LEAF_DOMAIN)
    digest.update(index.to_bytes(8, 'big'))
    digest.update(leaf)
    return digest.digest()


def hash_leaf(index: int, payload: bytes) -> bytes:
    """Hash a payload at its committed index for ``from_leaf_hashes``.

    Indices must fit the tree's unsigned 64-bit leaf-count limit.
    """
    if type(index) is not int or not 0 <= index < _MAX_COUNT:
        raise ValueError('leaf index is outside the supported range')
    if not isinstance(payload, bytes):
        raise TypeError('leaf payload must be bytes')
    return _leaf_hash(index, payload)


def _node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(_NODE_DOMAIN + left + right).digest()


def _root_hash(count: int, top: bytes) -> bytes:
    return hashlib.sha256(_ROOT_DOMAIN + count.to_bytes(8, 'big') + top).digest()


class MerkleTree:
    """Commit to nonempty leaves, retaining hashes rather than leaf payloads.

    Construction consumes ``leaves`` once and hashes each payload immediately.
    ``proof`` returns sibling hashes from the leaf level toward the root.
    """

    def __init__(self, leaves: Iterable[bytes]):
        hashes = []
        for index, leaf in enumerate(leaves):
            if index >= _MAX_COUNT:
                raise ValueError('leaf count exceeds the unsigned 64-bit limit')
            if not isinstance(leaf, bytes):
                raise TypeError('leaves must be bytes')
            hashes.append(_leaf_hash(index, leaf))
        self._build(hashes)

    @classmethod
    def from_leaf_hashes(cls, hashes: Iterable[bytes]) -> 'MerkleTree':
        """Build from ordered digests produced by ``hash_leaf``.

        Callers must hash each payload with its position in this iterable.
        Digest lengths are checked; payloads and their indices cannot be
        recovered or checked from a digest alone.
        """
        digests = []
        for index, digest in enumerate(hashes):
            if index >= _MAX_COUNT:
                raise ValueError('leaf count exceeds the unsigned 64-bit limit')
            if not isinstance(digest, bytes):
                raise TypeError('leaf hashes must be bytes')
            if len(digest) != 32:
                raise ValueError('leaf hashes must contain exactly 32 bytes')
            digests.append(digest)
        tree = cls.__new__(cls)
        tree._build(digests)
        return tree

    def _build(self, hashes: list[bytes]) -> None:
        if not hashes:
            raise ValueError('a Merkle tree requires at least one leaf')
        self._levels = [tuple(hashes)]
        while len(hashes) > 1:
            hashes = [
                _node_hash(hashes[i], hashes[min(i + 1, len(hashes) - 1)])
                for i in range(0, len(hashes), 2)
            ]
            self._levels.append(tuple(hashes))
        self._root = _root_hash(self.leaf_count, hashes[0])

    @property
    def root(self) -> bytes:
        """The 32-byte root binding the leaf count and ordered payloads."""
        return self._root

    @property
    def leaf_count(self) -> int:
        return len(self._levels[0])

    def proof(self, index: int) -> tuple[bytes, ...]:
        """Return a membership proof; raise IndexError for an invalid index."""
        if type(index) is not int or not 0 <= index < self.leaf_count:
            raise IndexError('leaf index is out of range')
        siblings = []
        for level in self._levels[:-1]:
            siblings.append(level[min(index ^ 1, len(level) - 1)])
            index //= 2
        return tuple(siblings)


def verify_merkle(
    root: bytes,
    leaf: bytes,
    index: int,
    leaf_count: int,
    siblings: Iterable[bytes],
) -> bool:
    """Verify membership, returning False for malformed or inconsistent proofs.

    The proof must contain exactly one 32-byte sibling per tree level. At an
    odd level's last position, that sibling must equal the running hash.
    """
    if (not isinstance(root, bytes) or len(root) != 32
            or not isinstance(leaf, bytes)
            or type(leaf_count) is not int or not 1 <= leaf_count <= _MAX_COUNT
            or type(index) is not int or not 0 <= index < leaf_count):
        return False
    try:
        proof = iter(siblings)
    except TypeError:
        return False
    current = _leaf_hash(index, leaf)
    width = leaf_count
    while width > 1:
        try:
            sibling = next(proof)
        except (StopIteration, TypeError):
            return False
        if not isinstance(sibling, bytes) or len(sibling) != 32:
            return False
        if index % 2:
            current = _node_hash(sibling, current)
        else:
            if index + 1 == width and not hmac.compare_digest(sibling, current):
                return False
            current = _node_hash(current, sibling)
        index //= 2
        width = (width + 1) // 2
    try:
        next(proof)
    except StopIteration:
        return hmac.compare_digest(root, _root_hash(leaf_count, current))
    except TypeError:
        return False
    return False


@dataclass(frozen=True)
class PackedMerkleLayout:
    """Prepare byte offsets for retained bottom-up levels and a bound root."""
    leaf_count: int
    levels: tuple[tuple[int, int], ...]
    byte_count: int

    def __init__(self, leaf_count: int):
        if type(leaf_count) is not int or not 1 <= leaf_count <= _MAX_COUNT:
            raise ValueError('leaf count is outside the supported range')
        object.__setattr__(self, 'leaf_count', leaf_count)
        levels, offset, width = [], 0, leaf_count
        while True:
            levels.append((offset, width))
            offset += width * 32
            if width == 1:
                break
            width = (width + 1) // 2
        object.__setattr__(self, 'levels', tuple(levels))
        object.__setattr__(self, 'byte_count', offset + 32)

    def bind(self, packed: bytes) -> 'PackedMerkleTree':
        """Retain completed hashes without slicing levels or recomputing hashes."""
        if not isinstance(packed, bytes) or len(packed) != self.byte_count:
            raise ValueError('packed tree must contain exactly the layout number of bytes')
        tree = PackedMerkleTree.__new__(PackedMerkleTree)
        tree._packed = packed
        tree._layout = self
        return tree


class PackedMerkleTree:
    """Retain immutable native hashes and slice membership paths on demand.

    The buffer contains every level from leaves toward the top followed by the
    count-bound root. Binding checks its shape; the producer is responsible for
    computing the supplied hashes correctly.
    """
    @classmethod
    def from_packed(cls, packed: bytes, leaf_count: int) -> 'PackedMerkleTree':
        return PackedMerkleLayout(leaf_count).bind(packed)

    @property
    def root(self) -> bytes:
        return self._packed[-32:]

    @property
    def leaf_count(self) -> int:
        return self._layout.leaf_count

    @property
    def leaf_hashes(self):
        """A lazy sequence of indexed leaf hashes retained by this tree."""
        return _PackedLeafHashes(self)

    def leaf_hash(self, index: int) -> bytes:
        if type(index) is not int or not 0 <= index < self.leaf_count:
            raise IndexError('leaf index is out of range')
        start = index * 32
        return self._packed[start:start + 32]

    def proof(self, index: int) -> tuple[bytes, ...]:
        if type(index) is not int or not 0 <= index < self.leaf_count:
            raise IndexError('leaf index is out of range')
        siblings = []
        for offset, width in self._layout.levels[:-1]:
            start = offset + min(index ^ 1, width - 1) * 32
            siblings.append(self._packed[start:start + 32])
            index //= 2
        return tuple(siblings)


class _PackedLeafHashes(Sequence):
    def __init__(self, tree):
        self._tree = tree

    def __len__(self):
        return self._tree.leaf_count

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self._tree.leaf_hash(i) for i in range(*index.indices(len(self))))
        if type(index) is int and index < 0:
            index += len(self)
        return self._tree.leaf_hash(index)
