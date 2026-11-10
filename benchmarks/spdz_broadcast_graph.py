"""Compare eager and prepared local RBC phases on a fixed replay fixture.

Run ``python -m benchmarks.spdz_broadcast_graph`` from the repository root.
The fixture deliberately resets the public round counter and replays incoming
messages. It is not a live session. Graph preparation and fixture recording are
excluded; fresh input-pointer uploads, dynamic headers, hashing, canonical and
consistency checks, host decisions and final synchronization remain timed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import time

import torch

from rep4gpu.device_transport import DeviceIdealEndpoint,DeviceRecorder,DeviceReplay
from rep4gpu.spdz_device import DeviceDistributedParty
from rep4gpu.spdz_device_graph import GraphDeviceDistributedParty
from rep4gpu.transport import IdealNetwork


def run_case(size,repeats=30):
    network = IdealNetwork(timeout=30)
    records = [DeviceRecorder(DeviceIdealEndpoint(network.endpoint(p))) for p in range(4)]
    alpha = torch.arange(1,11,dtype=torch.int64,device='cuda')
    payload = torch.zeros(size,dtype=torch.uint8,device='cuda')
    session,label = b'graph-rbc-timing-fixture',b'fixed-fixture'
    def make(kind,party,transport):
        return kind(party,transport,alpha,session=session,_allow_insecure_test_transport=True)
    def operation(role):
        return role.broadcast_device(payload,label=label,canonical_byte_offset=0)
    def record(party):
        try:
            operation(make(DeviceDistributedParty,party,records[party]))
        except BaseException as error:
            network.abort(error)
            raise
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(record,range(4)))
    results = {}
    for kind in (DeviceDistributedParty,GraphDeviceDistributedParty):
        party_samples = []
        for party in range(4):
            role = make(kind,party,DeviceReplay(records[party].transcript))
            if kind is GraphDeviceDistributedParty:
                role.prepare_broadcast(size,0)
            samples = []
            for repeat in range(repeats+5):
                role.transport = DeviceReplay(records[party].transcript,check_outgoing=repeat==0)
                role.round = 0
                torch.cuda.synchronize()
                start = time.perf_counter()
                operation(role)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter()-start)*1000
                role.transport.assert_consumed()
                if repeat >= 5:
                    samples.append(elapsed)
            party_samples.append(samples)
        medians = [statistics.median(samples) for samples in party_samples]
        results[kind.__name__] = dict(party_median_ms=medians,max_party_median_ms=max(medians),
                                      samples_ms=party_samples)
    return dict(payload_bytes=size,repeats=repeats,results=results,
                methodology='fixed replay fixture; header generation and graph pointer uploads included; network excluded')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes',type=int,nargs='+',default=[32,512,4096,65536])
    parser.add_argument('--repeats',type=int,default=30)
    parser.add_argument('--output',type=Path,default=Path('artifacts/spdz_broadcast_graph.json'))
    args = parser.parse_args()
    rows = []
    for size in args.sizes:
        if size < 0 or size % 4:
            parser.error('sizes must be nonnegative multiples of four')
        row = run_case(size,args.repeats)
        rows.append(row)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps(dict(payload_bytes=size,max_party_median_ms={
            key:value['max_party_median_ms'] for key,value in row['results'].items()})),flush=True)


if __name__ == '__main__':
    main()
