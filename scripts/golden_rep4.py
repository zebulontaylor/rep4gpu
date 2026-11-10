#!/usr/bin/env python3
"""Deterministic Rep4 arithmetic/transcript oracle, independent of GPU code."""
import argparse
import hashlib
import json
from pathlib import Path

JOINT = [(0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0),
         (3, 0, 2, 1), (0, 2, 3, 1), (1, 3, 2, 0)]


def digest(data):
    return hashlib.blake2b(data, digest_size=32).digest()


def shares(parts):
    """Map excluded-party components to MP-SPDZ's three local slots."""
    return [[parts[(p + j + 1) % 4] for j in range(3)] for p in range(4)]


def multiply(xparts, yparts, binary=False, width=64):
    mask = (1 << width) - 1
    add = (lambda a, b: a ^ b) if binary else (lambda a, b: (a + b) & mask)
    sub = (lambda a, b: a ^ b) if binary else (lambda a, b: (a - b) & mask)
    mul = (lambda a, b: a & b) if binary else (lambda a, b: (a * b) & mask)
    xs, ys = shares(xparts), shares(yparts)
    local = [[0] * 5 for _ in range(4)]
    for p, (x, y) in enumerate(zip(xs, ys)):
        for i in range(2):
            local[p][(p + i - 1) % 4] = add(mul(add(x[i], x[i + 1]), y[i]), mul(x[i], y[i + 1]))
        local[p][4] = add(mul(x[0], y[2]), mul(x[2], y[0]))
    result = [[0] * 3 for _ in range(4)]
    data = [[bytearray() for _ in range(4)] for _ in range(4)]
    hashes = {}
    joint_values = []
    for k, (sender, backup, receiver, outsider) in enumerate(JOINT):
        column = k if k < 4 else 4
        assert local[sender][column] == local[backup][column]
        # Domain-separated deterministic masks replace correlated PRNG output.
        r = int.from_bytes(digest(f'mask/{binary}/{k}'.encode())[:8], 'little') & mask
        value = sub(local[sender][column], r)
        joint_values.append(value)
        encoded = value.to_bytes((width + 7) // 8, 'little')
        data[sender][receiver].extend(encoded)
        hashes.setdefault((backup, receiver, sender), bytearray()).extend(encoded)
        for p in range(4):
            if p != receiver:
                slot = (receiver - p) % 4 - 1
                result[p][slot] = add(result[p][slot], r)
            if p in (sender, backup, receiver):
                slot = (outsider - p) % 4 - 1
                result[p][slot] = add(result[p][slot], value)
    # append_hashes traverses receiver offset first, then sender index.
    for (backup, receiver, sender), payload in sorted(hashes.items(), key=lambda kv: (kv[0][0], (kv[0][1] - kv[0][0]) % 4, kv[0][2])):
        data[backup][receiver].extend(digest(payload))
    outparts = [result[(excluded + 1) % 4][2] for excluded in range(4)]
    assert shares(outparts) == result
    def total(values):
        acc = 0
        for value in values:
            acc = add(acc, value)
        return acc
    expected = mul(total(xparts), total(yparts))
    assert total(outparts) == expected
    return dict(domain='binary' if binary else 'ring', width=width,
                x_components=xparts, y_components=yparts, local_products=local,
                joint_masked_values=joint_values, output_shares=result,
                opened=expected, edges=[dict(sender=p, receiver=q, bytes=len(data[p][q]), hex=data[p][q].hex())
                    for p in range(4) for q in range(4) if data[p][q]],
                total_payload_bytes=sum(len(v) for row in data for v in row))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    cases = [multiply([1, 2, 3, 4], [5, 6, 7, 8]),
             multiply([2**64 - 1, 7, 0, 3], [11, 0, 2**63, 9]),
             multiply([0x55, 0xaa, 0x81, 0x18], [0xf0, 0xcc, 0x33, 0x5a], True),
             multiply([2**32 - 1, 7, 0, 3], [11, 0, 2**31, 9], width=32)]
    rendered = json.dumps(dict(schema=1, mask_source='deterministic oracle masks; not MP-SPDZ PRNG seeds', cases=cases), indent=2) + '\n'
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end='')


if __name__ == '__main__':
    main()
