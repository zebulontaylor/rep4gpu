"""Compare eager and prepared product graphs using fixed message fixtures.

Run from the repository root with ``python -m experiments.benchmarks.kernels.graph_product``.
Each timed call includes host counter reservation, nonce upload, input copies,
ChaCha20, BLAKE3 checks, assembly, and final synchronization. Counters reset to
zero to match the recorded messages: these timings deliberately reuse masks
and must never be used for live requests. Fresh live transcript equivalence is
checked separately by ``tests/unit/inference/test_graph_runtime.py``. Graph preparation and
fixture recording are outside timing. The device fixture excludes the network.
"""
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from rep4gpu.models.device_inference import DeviceInferenceParty
from rep4gpu.protocol.device_transport import DeviceIdealEndpoint, DeviceRecorder, DeviceReplay
from rep4gpu.models.graph_runtime import GraphDeviceParty
from rep4gpu.protocol.runtime import make_seeds
from rep4gpu.protocol.transport import IdealNetwork


def benchmark(count):
    seeds = make_seeds()
    network = IdealNetwork(timeout=90)
    records = [DeviceRecorder(DeviceIdealEndpoint(network.endpoint(p))) for p in range(4)]
    value = torch.full((count,), 7, dtype=torch.int64, device='cuda')

    def operation(role):
        a = role.public(value, binary=False)
        return role.mul(a, a)

    def record(party):
        try:
            operation(DeviceInferenceParty(party, records[party],
                                           {e:s for e,s in seeds.items() if e != party}))
        except BaseException as error:
            network.abort(error)
            raise

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(record, range(4)))
    result = {}
    for kind in (DeviceInferenceParty, GraphDeviceParty):
        medians = []
        for party in range(4):
            replay = DeviceReplay(records[party].transcript, check_outgoing=False)
            role = kind(party, replay, {e:s for e,s in seeds.items() if e != party})
            if kind is GraphDeviceParty:
                role.prepare_product((count,), False, True)
            samples = []
            for repeat in range(24):
                # Reset only this recorded timing fixture, never live state.
                role.counters = {e:0 for e in role.counters}
                replay.round = 0
                torch.cuda.synchronize()
                start = time.perf_counter()
                operation(role)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter()-start)*1000
                replay.assert_consumed()
                if repeat >= 4:
                    samples.append(elapsed)
            medians.append(statistics.median(samples))
        result[kind.__name__] = medians
    print(json.dumps(dict(count=count, party_median_ms=result,
                          max_party_ms={k:max(v) for k,v in result.items()})), flush=True)


if __name__ == '__main__':
    for count in (8192, 1048576):
        benchmark(count)
