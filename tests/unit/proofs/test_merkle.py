"""Ordered Merkle commitments reject altered data and malformed paths."""

import hashlib

import pytest

from rep4gpu.proofs.merkle import MerkleTree, verify_merkle


@pytest.mark.parametrize('count', [1, 2, 3, 5, 7, 8, 9, 31, 32, 33])
def test_every_leaf_roundtrips(count):
    leaves = [str(i).encode() for i in range(count)]
    tree = MerkleTree(iter(leaves))
    assert tree.leaf_count == count
    assert len(tree.root) == 32
    for index, leaf in enumerate(leaves):
        proof = tree.proof(index)
        assert len(proof) == (count - 1).bit_length()
        assert verify_merkle(tree.root, leaf, index, count, proof)
        assert not verify_merkle(tree.root, leaf + b'x', index, count, proof)
        assert not verify_merkle(tree.root, leaf, index, count, proof + (bytes(32),))
        if proof:
            assert not verify_merkle(tree.root, leaf, index, count, proof[:-1])
            for position in range(len(proof)):
                changed = list(proof)
                changed[position] = bytes(x ^ 1 for x in changed[position])
                assert not verify_merkle(tree.root, leaf, index, count, changed)


def test_hash_format_and_count_binding():
    leaf = hashlib.sha256(b'rep4gpu.merkle.v1.leaf\x00' + bytes(8) + b'').digest()
    expected = hashlib.sha256(b'rep4gpu.merkle.v1.root\x00' + (1).to_bytes(8, 'big') + leaf).digest()
    assert MerkleTree([b'']).root == expected
    tree = MerkleTree([b'a', b'b', b'c'])
    assert tree.root != MerkleTree([b'a', b'b', b'c', b'c']).root
    assert not verify_merkle(tree.root, b'c', 2, 4, tree.proof(2))


def test_indices_are_bound_for_identical_payloads():
    tree = MerkleTree([b'a'] * 4)
    for index in range(4):
        for other in range(4):
            assert verify_merkle(tree.root, b'a', other, 4, tree.proof(index)) == (index == other)


def test_odd_sibling_must_be_duplicate_even_with_matching_forged_root():
    tree = MerkleTree([b'a', b'b', b'c'])
    proof = list(tree.proof(2))
    proof[0] = bytes(32)
    current = hashlib.sha256(b'rep4gpu.merkle.v1.leaf\x00' + (2).to_bytes(8, 'big') + b'c').digest()
    current = hashlib.sha256(b'rep4gpu.merkle.v1.node\x00' + current + proof[0]).digest()
    current = hashlib.sha256(b'rep4gpu.merkle.v1.node\x00' + proof[1] + current).digest()
    root = hashlib.sha256(b'rep4gpu.merkle.v1.root\x00' + (3).to_bytes(8, 'big') + current).digest()
    assert not verify_merkle(root, b'c', 2, 3, proof)


@pytest.mark.parametrize('index', [-1, 3, 1.0, True, None])
def test_invalid_indices(index):
    tree = MerkleTree([b'a'] * 3)
    with pytest.raises(IndexError):
        tree.proof(index)
    assert not verify_merkle(tree.root, b'a', index, 3, ())


@pytest.mark.parametrize('count', [0, -1, 2**64, 1.0, True, None])
def test_invalid_counts(count):
    assert not verify_merkle(bytes(32), b'a', 0, count, ())


@pytest.mark.parametrize('root,leaf,siblings', [
    (b'', b'a', ()), (bytes(32), 'a', ()),
    (bytes(32), b'a', None), (bytes(32), b'a', (b'',)),
    (bytes(32), b'a', ('x' * 32,)),
])
def test_malformed_proof(root, leaf, siblings):
    assert not verify_merkle(root, leaf, 0, 2, siblings)


def test_constructor_rejects_empty_and_nonbytes():
    with pytest.raises(ValueError, match='at least one'):
        MerkleTree(iter(()))
    with pytest.raises(TypeError, match='bytes'):
        MerkleTree([bytearray(b'a')])


def test_streams_once_and_retains_only_hashes():
    class Leaves:
        def __init__(self):
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            assert self.iterations == 1
            yield b'payload' * 100
            yield b''

    leaves = Leaves()
    tree = MerkleTree(leaves)
    assert leaves.iterations == 1
    assert all(len(value) == 32 for level in tree._levels for value in level)
    assert verify_merkle(tree.root, b'payload' * 100, 0, 2, iter(tree.proof(0)))


@pytest.mark.parametrize('count', [1, 2, 3, 7, 16, 33])
def test_prehashed_construction_matches_payload_construction(count):
    from rep4gpu.proofs.merkle import hash_leaf

    leaves = [str(index).encode() for index in range(count)]
    direct = MerkleTree(leaves)
    prehashed = MerkleTree.from_leaf_hashes(
        hash_leaf(index, payload) for index, payload in enumerate(leaves)
    )
    assert prehashed.root == direct.root
    assert prehashed.leaf_count == count
    for index, leaf in enumerate(leaves):
        assert prehashed.proof(index) == direct.proof(index)
        assert verify_merkle(prehashed.root, leaf, index, count, prehashed.proof(index))


@pytest.mark.parametrize('digest', [b'', bytes(31), bytes(33)])
def test_prehashed_construction_rejects_invalid_lengths(digest):
    with pytest.raises(ValueError, match='32 bytes'):
        MerkleTree.from_leaf_hashes([digest])


def test_prehashed_construction_rejects_empty_and_nonbytes():
    with pytest.raises(ValueError, match='at least one'):
        MerkleTree.from_leaf_hashes(iter(()))
    with pytest.raises(TypeError, match='bytes'):
        MerkleTree.from_leaf_hashes([bytearray(32)])


@pytest.mark.parametrize('index', [-1, 2**64 - 1, 2**64, True, 1.0, None])
def test_hash_leaf_rejects_invalid_indices(index):
    from rep4gpu.proofs.merkle import hash_leaf

    with pytest.raises(ValueError, match='index'):
        hash_leaf(index, b'a')


def test_hash_leaf_rejects_nonbytes_payload():
    from rep4gpu.proofs.merkle import hash_leaf

    with pytest.raises(TypeError, match='bytes'):
        hash_leaf(0, bytearray(b'a'))
