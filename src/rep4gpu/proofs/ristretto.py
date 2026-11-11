"""Canonical Ristretto255 points and scalars through libsodium."""
from __future__ import annotations

import ctypes
import ctypes.util
import secrets

Q = 2**252 + 27742317777372353535851937790883648493
SCALAR_BYTES = POINT_BYTES = 32
IDENTITY = bytes(POINT_BYTES)


def _load_sodium():
    candidates = (ctypes.util.find_library('sodium'), 'libsodium.so.26')
    failures = []
    for name in candidates:
        if name is None:
            continue
        try:
            return ctypes.CDLL(name)
        except OSError as error:
            failures.append(str(error))
    raise ImportError('Ristretto255 requires libsodium: '+ '; '.join(failures))


_sodium = _load_sodium()
_sodium.sodium_init.argtypes = []
_sodium.sodium_init.restype = ctypes.c_int
if _sodium.sodium_init() < 0:
    raise RuntimeError('libsodium initialization failed')
for _name, _arity, _return in (
    ('crypto_core_ristretto255_is_valid_point', 1, ctypes.c_int),
    ('crypto_core_ristretto255_scalar_reduce', 2, None),
    ('crypto_core_ristretto255_add', 3, ctypes.c_int),
    ('crypto_core_ristretto255_sub', 3, ctypes.c_int),
    ('crypto_core_ristretto255_from_hash', 2, ctypes.c_int),
    ('crypto_scalarmult_ristretto255', 3, ctypes.c_int),
    ('crypto_scalarmult_ristretto255_base', 2, ctypes.c_int),
):
    _function = getattr(_sodium, _name)
    _function.argtypes = [ctypes.c_void_p]*_arity
    _function.restype = _return


def _bytes(value, length, label):
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f'{label} must be bytes')
    value = bytes(value)
    if len(value) != length:
        raise ValueError(f'{label} must contain {length} bytes')
    return value


def _buffer(value):
    return (ctypes.c_ubyte*len(value)).from_buffer_copy(value)


def scalar_encode(value):
    """Encode an integer in [0,Q); callers reduce field arithmetic modulo Q."""
    if type(value) is not int:
        raise TypeError('scalar must be an integer')
    if not 0 <= value < Q:
        raise ValueError('scalar is outside the canonical range')
    return value.to_bytes(SCALAR_BYTES, 'little')


def scalar_decode(value):
    """Decode a canonical little-endian scalar, rejecting values at least Q."""
    value = int.from_bytes(_bytes(value, SCALAR_BYTES, 'scalar'), 'little')
    if value >= Q:
        raise ValueError('scalar is outside the canonical range')
    return value


def scalar_reduce(value):
    """Reduce a 64-byte little-endian integer and return canonical bytes."""
    source = _buffer(_bytes(value, 64, 'wide scalar'))
    result = (ctypes.c_ubyte*SCALAR_BYTES)()
    _sodium.crypto_core_ristretto255_scalar_reduce(result, source)
    return bytes(result)


def random_scalar():
    """Return a scalar from a fresh 64-byte random integer reduced modulo Q."""
    return scalar_decode(scalar_reduce(secrets.token_bytes(64)))


def point_validate(value):
    """Return canonical point bytes or reject an invalid Ristretto encoding."""
    value = _bytes(value, POINT_BYTES, 'point')
    if _sodium.crypto_core_ristretto255_is_valid_point(_buffer(value)) != 1:
        raise ValueError('invalid Ristretto255 point encoding')
    return value


def _scalar(value):
    if type(value) is int:
        return scalar_encode(value)
    value = _bytes(value, SCALAR_BYTES, 'scalar')
    scalar_decode(value)
    return value


def _point_binary(name, left, right):
    left = _buffer(point_validate(left))
    right = _buffer(point_validate(right))
    result = (ctypes.c_ubyte*POINT_BYTES)()
    if getattr(_sodium, name)(result, left, right) != 0:
        raise ValueError('Ristretto255 point operation failed')
    return bytes(result)


def point_add(left, right):
    """Add two canonical point encodings."""
    return _point_binary('crypto_core_ristretto255_add', left, right)


def point_sub(left, right):
    """Subtract two canonical point encodings."""
    return _point_binary('crypto_core_ristretto255_sub', left, right)


def point_mul(point, scalar):
    """Multiply a canonical point by an integer or encoded canonical scalar."""
    point, scalar = point_validate(point), _scalar(scalar)
    if point == IDENTITY or scalar == bytes(SCALAR_BYTES):
        return IDENTITY
    result = (ctypes.c_ubyte*POINT_BYTES)()
    if _sodium.crypto_scalarmult_ristretto255(result, _buffer(scalar), _buffer(point)) != 0:
        raise ValueError('Ristretto255 scalar multiplication failed')
    return bytes(result)


def base_mul(scalar):
    """Multiply the standard Ristretto255 generator by a canonical scalar."""
    scalar = _scalar(scalar)
    if scalar == bytes(SCALAR_BYTES):
        return IDENTITY
    result = (ctypes.c_ubyte*POINT_BYTES)()
    if _sodium.crypto_scalarmult_ristretto255_base(result, _buffer(scalar)) != 0:
        raise ValueError('Ristretto255 base multiplication failed')
    return bytes(result)


def hash_to_point(value):
    """Map exactly 64 hash bytes to a canonical Ristretto255 point."""
    source = _buffer(_bytes(value, 64, 'point hash'))
    result = (ctypes.c_ubyte*POINT_BYTES)()
    if _sodium.crypto_core_ristretto255_from_hash(result, source) != 0:
        raise ValueError('Ristretto255 hash mapping failed')
    return bytes(result)
