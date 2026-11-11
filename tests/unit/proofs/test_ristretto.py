"""Ristretto wrappers preserve canonical encodings and group operations."""
import hashlib

import pytest

from rep4gpu.proofs.ristretto import (Q, IDENTITY, base_mul, hash_to_point, point_add,
                               point_mul, point_sub, point_validate,
                               random_scalar, scalar_decode, scalar_encode,
                               scalar_reduce)


@pytest.mark.parametrize('value', [0, 1, 2**64-1, Q-1])
def test_scalar_roundtrip(value):
    assert scalar_decode(scalar_encode(value)) == value


@pytest.mark.parametrize('value', [-1, Q, Q+1])
def test_scalar_encode_rejects_noncanonical(value):
    with pytest.raises(ValueError, match='canonical'):
        scalar_encode(value)


@pytest.mark.parametrize('value', [Q, Q+1, 2**256-1])
def test_scalar_decode_rejects_noncanonical(value):
    with pytest.raises(ValueError, match='canonical'):
        scalar_decode(value.to_bytes(32, 'little'))


def test_wide_reduction():
    for raw in (bytes(64), bytes([255])*64, hashlib.sha512(b'scalar').digest()):
        assert scalar_decode(scalar_reduce(raw)) == int.from_bytes(raw, 'little') % Q
    assert 0 <= random_scalar() < Q


def test_group_laws():
    generator = base_mul(1)
    assert point_validate(generator) == generator
    assert point_validate(IDENTITY) == IDENTITY
    assert point_add(base_mul(17), base_mul(23)) == base_mul(40)
    assert point_sub(base_mul(17), base_mul(23)) == base_mul(Q-6)
    assert point_mul(base_mul(17), 23) == base_mul(17*23)
    assert point_mul(generator, scalar_encode(Q-1)) == point_sub(IDENTITY, generator)
    assert point_add(generator, point_sub(IDENTITY, generator)) == IDENTITY
    assert base_mul(0) == IDENTITY
    assert point_mul(generator, 0) == IDENTITY
    assert point_mul(IDENTITY, 123) == IDENTITY


def test_hash_mapping():
    raw = hashlib.sha512(b'independent-generator').digest()
    point = hash_to_point(raw)
    assert point_validate(point) == point
    assert point != IDENTITY
    assert hash_to_point(raw) == point
    assert point_mul(point, 2) == point_add(point, point)


@pytest.mark.parametrize('point', [bytes([255])*32, bytes([1])+bytes(31)])
def test_invalid_point_rejected(point):
    for operation in (point_validate, lambda value: point_add(value, IDENTITY),
                      lambda value: point_sub(IDENTITY, value),
                      lambda value: point_mul(value, 0)):
        with pytest.raises(ValueError, match='encoding'):
            operation(point)


@pytest.mark.parametrize('value', [b'', bytes(31), bytes(33)])
def test_incorrect_lengths(value):
    with pytest.raises(ValueError, match='32 bytes'):
        scalar_decode(value)
    with pytest.raises(ValueError, match='32 bytes'):
        point_validate(value)
    with pytest.raises(ValueError, match='64 bytes'):
        scalar_reduce(value)
    with pytest.raises(ValueError, match='64 bytes'):
        hash_to_point(value)
