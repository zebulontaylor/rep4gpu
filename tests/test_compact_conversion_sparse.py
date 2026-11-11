"""Sparse parity conversion against an independent CPU joint-input oracle."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import math
import struct

import numpy as np
import pytest
import torch

from rep4gpu.compact_conversion import weighted_b2a
from rep4gpu.compact_conversion_gpu import weighted_b2a_gpu
from rep4gpu.runtime import Party, Share, bytes_tensor, make_seeds
from rep4gpu.transport import IdealNetwork, RecordingTransport


_SPARSE_TERMS = (1, 3, 4, 5)
_SPARSE_ROUTES = ((2,1,0,3), (3,0,1,2), (0,2,3,1), (1,3,2,0))


class _SparseOracle(Party):
    def _random(self, excluded, shape, device, binary=False):
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
        counter = self.counters[excluded]
        self.counters[excluded] += 1
        width = self.binary_bits if binary else self.bits
        domain = bytes((self.bits, self.binary_bits, int(binary), width, excluded))
        key = hashlib.sha256(self.seeds[excluded]+b'Rep4GPU-DevicePRGv1\0'+struct.pack('<Q', len(domain))+domain).digest()
        stream = Cipher(algorithms.ChaCha20(key, struct.pack('<QQ', 0, counter)), mode=None).encryptor()
        data = stream.update(bytes(math.prod(shape)*width//8))+stream.finalize()
        return bytes_tensor(data, shape, device, width)

    def _reshare_product(self, contributions, binary, device):
        for index in (0, 2):
            value = contributions[index]
            if value is not None:
                assert torch.count_nonzero(value) == 0
        specs = [(*route, contributions[index if index<4 else 4])
                 for index,route in zip(_SPARSE_TERMS, _SPARSE_ROUTES)]
        shares = self.joint_input_batch(specs, contributions[4].shape, binary, device)
        result = shares[0]
        for share in shares[1:]:
            result = self.add(result, share)
        self.multiplications += 1
        return result


def _run(components, weights, bits, seeds, gpu):
    network = IdealNetwork(timeout=120)
    def work(party):
        try:
            transport = network.endpoint(party)
            if gpu:
                from rep4gpu.device_inference import DeviceInferenceParty
                from rep4gpu.device_transport import DeviceByteBridge, DeviceRecorder
                transport = DeviceRecorder(DeviceByteBridge(transport, 'cuda'))
                rt = DeviceInferenceParty(party, transport,
                    {e:seed for e,seed in seeds.items() if e!=party}, bits=bits,
                    sparse_products=True)
            else:
                transport = RecordingTransport(transport)
                rt = _SparseOracle(party, transport,
                    {e:seed for e,seed in seeds.items() if e!=party}, bits=bits, binary_bits=bits)
            planes = Share(tuple(components[(party+j+1)%4].to(rt.device) for j in range(3)), True)
            outputs = []
            for _ in range(2):
                result = (weighted_b2a_gpu if gpu else weighted_b2a)(rt, planes, weights)
                outputs.append(tuple(part.cpu() for part in result.parts))
            return outputs, dict(rt.counters), rt.joint_inputs, rt.multiplications, [frame.outgoing if gpu else frame[0] for frame in transport.transcript]
        except BaseException as error:
            network.abort(error)
            raise
    with ThreadPoolExecutor(4) as pool:
        return list(pool.map(work, range(4)))


@pytest.mark.parametrize('bits', [32, 64])
@pytest.mark.parametrize('shape', [(1, 17), (3, 5), (7, 2, 13)])
def test_sparse_weighted_b2a_matches_cpu_oracle(bits, shape):
    rng = np.random.default_rng(589)
    components = [torch.from_numpy(rng.bit_generator.random_raw(math.prod(shape)).view(np.int64).reshape(shape).copy()) for _ in range(4)]
    weights = torch.arange(2*shape[0], dtype=torch.int64).reshape(2, shape[0])-shape[0]
    seeds = make_seeds()
    expected = _run(components, weights, bits, seeds, gpu=False)
    parity = components[0]&1
    for component in components[1:]:
        parity ^= component&1
    wanted = (parity.unsqueeze(1)*weights.T.reshape((*weights.T.shape,)+(1,)*(parity.ndim-1))).sum(0)
    if bits == 32:
        wanted &= 0xffffffff
    for repeat in range(2):
        actual = sum(expected[0][0][repeat])+expected[1][0][repeat][2]
        if bits == 32:
            actual &= 0xffffffff
        assert torch.equal(actual, wanted)
    assert all(row[2:4] == (12, 2) for row in expected)
    for party,row in enumerate(expected):
        assert row[1] == {e:2*(1+int(e in (1,3))) for e in range(4) if e!=party}
        for stage in (1, 3):
            assert len(row[4][stage]) == 2
            assert sum(map(len, row[4][stage].values())) == wanted.numel()*(bits//8)+32
    assert any(not torch.equal(a,b) for a,b in zip(expected[0][0][0], expected[0][0][1]))
    if not torch.cuda.is_available():
        return
    actual = _run(components, weights, bits, seeds, gpu=True)
    for got, want in zip(actual, expected):
        assert got[1:4] == want[1:4]
        for got_parts, want_parts in zip(got[0], want[0]):
            assert all(torch.equal(a,b) for a,b in zip(got_parts, want_parts))

    # Translate only the CPU oracle's integrity algorithm; payload bytes stay
    # untouched, so this compares the four-joint ordering and exact mask draws.
    blake3 = pytest.importorskip('blake3').blake3
    schedules = (((2,3,1,0), (0,1,3,2)), _SPARSE_ROUTES)
    for stage in range(4):
        size = (math.prod(shape) if stage%2 == 0 else wanted.numel())*(bits//8)
        payloads, tags, offsets = [{} for _ in range(4)], [{} for _ in range(4)], {}
        for sender, backup, receiver, _ in schedules[stage%2]:
            offset = offsets.get((sender,receiver), 0)
            payload = expected[sender][4][stage][receiver][offset:offset+size]
            assert len(payload) == size
            offsets[sender,receiver] = offset+size
            payloads[sender].setdefault(receiver, []).append(payload)
            tags[backup].setdefault(receiver, []).append((sender, blake3(payload).digest()))
        for party in range(4):
            wires = {peer: b''.join(payloads[party].get(peer, []))+b''.join(tag for _,tag in sorted(tags[party].get(peer, [])))
                for peer in set(payloads[party]) | set(tags[party])}
            assert actual[party][4][stage] == wires
