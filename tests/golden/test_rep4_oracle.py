"""Algebra and fixture checks for the independent MP-SPDZ schedule oracle."""
import importlib.util
import json
from pathlib import Path
import random
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('golden_rep4', ROOT / 'scripts/golden_rep4.py')
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


class Rep4OracleTests(unittest.TestCase):
    def test_full_ring_and_binary_replication(self):
        rng = random.Random(4096)
        for binary in (False, True):
            for _ in range(100):
                case = ORACLE.multiply([rng.getrandbits(64) for _ in range(4)],
                                       [rng.getrandbits(64) for _ in range(4)], binary)
                self.assertEqual(case['total_payload_bytes'], 240)
                self.assertEqual(len(case['edges']), 10)
                for party in range(4):
                    sent = sum(edge['bytes'] for edge in case['edges'] if edge['sender'] == party)
                    self.assertEqual(sent, 48 if party < 2 else 72)

    def test_saved_transcripts(self):
        fixtures = json.loads((ROOT / 'tests/golden/rep4_transcripts.json').read_text())
        for saved in fixtures['cases']:
            observed = ORACLE.multiply(saved['x_components'], saved['y_components'],
                                       saved['domain'] == 'binary', saved['width'])
            self.assertEqual(saved, observed)


if __name__ == '__main__':
    unittest.main()
