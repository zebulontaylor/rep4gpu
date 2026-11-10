"""Compare scheduled ChaCha mask batching with separate draws and stacking.

Repeated nonces are used only for this deterministic throughput measurement.
The production caller must reserve a fresh nonce for every active row.
"""
import argparse
import json
from pathlib import Path

import torch

from benchmarks.timing import measure
from rep4gpu.gpu_prg import DomainKeys, generate
from rep4gpu.gpu_prg_batch import generate_rows
from rep4gpu.runtime import SCHEDULE


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts",type=int,nargs="+",default=[8192,262144])
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    rows=[]
    for party in (0,1):
        caches={r:DomainKeys(bytes([r])*32) for r in range(4) if r!=party}
        for bits in (32,64):
            keys=[]
            nonces=[]
            counters={r:0 for r in caches}
            for _,_,receiver,_ in SCHEDULE:
                keys.append(None if receiver==party else caches[receiver].key(bytes((bits,receiver))))
                nonces.append(None if receiver==party else counters[receiver])
                if receiver!=party:
                    counters[receiver]+=1
            for count in args.counts:
                separate=lambda:torch.stack([torch.zeros(count,device="cuda",dtype=torch.int64) if key is None
                    else generate(key,nonce,count,bits) for key,nonce in zip(keys,nonces)])
                batched=lambda:generate_rows(keys,nonces,count,bits)
                torch.testing.assert_close(separate(),batched(),rtol=0,atol=0)
                for backend,fn in (("separate",separate),("batched",batched)):
                    row=dict(party=party,bits=bits,count=count,backend=backend,**measure(fn,rep=30))
                    print(json.dumps(row),flush=True)
                    rows.append(row)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(rows,indent=2)+"\n")


if __name__=="__main__":
    main()
