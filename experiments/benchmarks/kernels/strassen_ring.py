"""Measure exact matrix-level Strassen, including all online sums and joins."""
import argparse
import json
from pathlib import Path

import torch
import triton

from rep4gpu.gpu.matmul.kernels import prepare_weight
from rep4gpu.gpu.matmul.strassen_kernels import prepare_strassen_weight, ring_matmul_strassen
from rep4gpu.gpu.matmul.tuned_kernels import ring_matmul_tuned


def graph_ms(function):
    function()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output=function()
    return triton.testing.do_bench(graph.replay),output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('experiments/results/kernels/rep4_strassen_compare.json'))
    args=parser.parse_args()
    rows=[]
    for m,k,n in [(128,2048,8192),(128,8192,2048)]:
        a=torch.randint(-2**63,2**63-1,(m,k),device='cuda',dtype=torch.int64)
        b=torch.randint(-2**63,2**63-1,(k,n),device='cuda',dtype=torch.int64)
        weight=prepare_weight(b,32,storage='words32')
        del b
        baseline,reference=graph_ms(lambda:ring_matmul_tuned(a,weight,32))
        row=dict(shape=[m,k,n],baseline_single_ring_product_ms=baseline,
                 baseline_static_weight_bytes=weight.words.numel()*4,variants=[])
        for levels in (1,2):
            prepared=prepare_strassen_weight(weight,levels)
            actual=ring_matmul_strassen(a,prepared)
            torch.testing.assert_close(actual,reference,rtol=0,atol=0)
            ms,_=graph_ms(lambda:ring_matmul_strassen(a,prepared))
            result=dict(levels=levels,graph_ms=ms,retained_static_weight_bytes=prepared.storage_bytes,
                        exact_full_output_match=True,baseline_ratio=ms/baseline)
            row['variants'].append(result)
            print(dict(shape=[m,k,n],**result),flush=True)
            del prepared,actual
        rows.append(row)
        print(dict(shape=[m,k,n],baseline_single_ring_product_ms=baseline),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(kind='single ring matrix product component benchmark',
        online_boundary='all dynamic input sums, exact leaf products, output joins, allocations and final graph completion; static weight preparation excluded',
        gpu=torch.cuda.get_device_name(),results=rows),indent=2)+'\n')


if __name__=='__main__':
    main()
