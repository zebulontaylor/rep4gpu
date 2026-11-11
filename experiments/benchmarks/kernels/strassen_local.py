"""Measure full Rep4 local contributions with exact matrix Strassen."""
import json
from pathlib import Path
import torch
from experiments.benchmarks.kernels.strassen_ring import graph_ms
from rep4gpu.gpu.matmul.kernels import prepare_weight
from rep4gpu.gpu.matmul.strassen_kernels import prepare_strassen_weight,local_products_strassen
from rep4gpu.gpu.matmul.tuned_kernels import local_products_tuned


def main():
    results=[]
    for m,k,n in [(128,2048,8192),(128,8192,2048)]:
        xs=[torch.randint(-2**63,2**63-1,(m,k),device='cuda',dtype=torch.int64) for _ in range(3)]
        ys=[prepare_weight(torch.randint(-2**63,2**63-1,(k,n),device='cuda',dtype=torch.int64),32,storage='words32') for _ in range(3)]
        baseline,expected=graph_ms(lambda:local_products_tuned(*xs,*ys,32))
        row=dict(shape=[m,k,n],baseline_local_products_ms=baseline,variants=[])
        for levels in (1,2):
            weights=[prepare_strassen_weight(y,levels) for y in ys]
            actual=local_products_strassen(xs,weights)
            for got,want in zip(actual,expected):
                torch.testing.assert_close(got,want,rtol=0,atol=0)
            ms,_=graph_ms(lambda:local_products_strassen(xs,weights))
            variant=dict(levels=levels,graph_ms=ms,baseline_ratio=ms/baseline,
                         retained_static_bytes=sum(w.storage_bytes for w in weights),exact_output_match=True)
            row['variants'].append(variant)
            print(dict(shape=[m,k,n],**variant),flush=True)
            del weights,actual
        print(dict(shape=[m,k,n],baseline_local_products_ms=baseline),flush=True)
        results.append(row)
    Path('experiments/results/kernels/rep4_strassen_local_compare.json').write_text(json.dumps(dict(
        kind='three Rep4 local contributions; all online input sums, leaf products and output joins included',
        gpu=torch.cuda.get_device_name(),results=results),indent=2)+'\n')


if __name__=='__main__':
    main()
