"""Compare actual runtime bytes and share slots against independent fixtures."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import unittest

import torch

from rep4gpu.protocol.runtime import Party, Share, SCHEDULE
from rep4gpu.protocol.transport import IdealNetwork

ROOT = Path(__file__).resolve().parents[2]
MASK = (1 << 64) - 1


def signed(value):
    value &= MASK
    return value - (1 << 64) if value >> 63 else value


class RecordingTransport:
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.records = []

    def exchange(self, outgoing):
        self.records.append(dict(outgoing))
        return self.endpoint.exchange(outgoing)


class RuntimeGoldenTests(unittest.TestCase):
    def test_serialized_multiplication_transcripts(self):
        fixture = json.loads((ROOT / 'tests/fixtures/mpspdz/rep4_transcripts.json').read_text())
        for case in fixture['cases']:
            with self.subTest(domain=case['domain'], x=case['x_components']):
                network = IdealNetwork()
                transports = [RecordingTransport(network.endpoint(p)) for p in range(4)]
                binary = case['domain'] == 'binary'
                def run(p):
                    party = Party(p, transports[p], {e: bytes([e]) * 32 for e in range(4) if e != p}, bits=case['width'])
                    draws = iter(k for k, entry in enumerate(SCHEDULE) if entry[2] != p)
                    def mask(excluded, shape, device, binary=False):
                        k = next(draws)
                        self.assertEqual(excluded, SCHEDULE[k][2])
                        value = int.from_bytes(hashlib.blake2b(f'mask/{case["domain"] == "binary"}/{k}'.encode(), digest_size=32).digest()[:8], 'little')
                        value &= (1 << case['width']) - 1
                        return torch.full(shape, signed(value), dtype=torch.int64, device=device)
                    party._random = mask
                    x = Share(tuple(torch.tensor([signed(case['x_components'][(p+j+1)%4])]) for j in range(3)), binary)
                    y = Share(tuple(torch.tensor([signed(case['y_components'][(p+j+1)%4])]) for j in range(3)), binary)
                    return party.mul(x, y)
                with ThreadPoolExecutor(max_workers=4) as pool:
                    result = list(pool.map(run, range(4)))
                expected = {(edge['sender'], edge['receiver']): bytes.fromhex(edge['hex']) for edge in case['edges']}
                actual = {(p, q): payload for p in range(4) for q, payload in transports[p].records[0].items()}
                self.assertEqual(actual, expected)
                for p in range(4):
                    self.assertEqual([int(t.item()) & MASK for t in result[p].parts], case['output_shares'][p])

    def test_input_and_open_payload_routes(self):
        for owner in range(4):
            with self.subTest(owner=owner):
                self.check_input_and_open(owner)

    def check_input_and_open(self, owner):
        # Deterministic triple masks are independent of private input contents.
        network = IdealNetwork()
        transports = [RecordingTransport(network.endpoint(p)) for p in range(4)]
        value = torch.tensor([-7, (1 << 63)-1], dtype=torch.int64)
        def run(p):
            party = Party(p, transports[p], {e: bytes([e]) * 32 for e in range(4) if e != p})
            party._random = lambda excluded, shape, device, binary=False: torch.full(shape, 100+excluded, dtype=torch.int64, device=device)
            share = party.input(owner, value if p == owner else None, shape=value.shape)
            return share, party.open(share)
        with ThreadPoolExecutor(max_workers=4) as pool:
            result = list(pool.map(run, range(4)))
        correction = (value - (100+(owner+1)%4) - (100+(owner-1)%4)).numpy().astype('<i8').tobytes()
        self.assertEqual(transports[owner].records[0], {(owner+1)%4: correction, (owner-1)%4: correction})
        for p in range(4):
            if p != owner:
                self.assertEqual(transports[p].records[0], {})
        empty_hash = hashlib.blake2b(b'', digest_size=32).digest()
        correction_hash = hashlib.blake2b(correction, digest_size=32).digest()
        # MP-SPDZ sends the two input checks in separate pass-around rounds.
        for p, expected in (((owner+1)%4, empty_hash + correction_hash), ((owner-1)%4, correction_hash + empty_hash)):
            self.assertEqual(len(transports[p].records), 5)
            actual = b''.join(record.get((p+2)%4, b'') for record in transports[p].records[1:3])
            self.assertEqual(actual, expected)
        for p, (share, revealed) in enumerate(result):
            self.assertTrue(torch.equal(revealed, value))
            data_outgoing, check_outgoing = transports[p].records[-2:]
            data = share.parts[0].numpy().astype('<i8').tobytes()
            check = hashlib.blake2b(share.parts[2].numpy().astype('<i8').tobytes(), digest_size=32).digest()
            self.assertEqual(data_outgoing, {(p+1)%4: data})
            self.assertEqual(check_outgoing, {(p-1)%4: check})


if __name__ == '__main__':
    unittest.main()
