"""Pair maintained, opaque, Strassen and combined exact local products."""
import json
from pathlib import Path
import torch
import triton
from rep4gpu.kernels import prepare_weight
from rep4gpu.ring32_opaque import local_products_opaque
from rep4gpu.strassen_kernels import prepare_strassen_weight,local_products_strassen
from rep4gpu.tuned_kernels import local_products_tuned


def main():
    rows=[]
    columns=['maintained_local_products_ms','opaque32_local_products_ms',
             'strassen1_legacy_local_products_ms','strassen1_opaque32_local_products_ms']
    for m,k,n in [(128,2048,8192),(128,8192,2048)]:
        xs=[torch.randint(-2**63,2**63-1,(m,k),device='cuda',dtype=torch.int64) for _ in range(3)]
        ys=[prepare_weight(torch.randint(-2**63,2**63-1,(k,n),device='cuda',dtype=torch.int64),32,storage='words32') for _ in range(3)]
        weights=[prepare_strassen_weight(y,1) for y in ys]
        functions=[lambda:local_products_tuned(*xs,*ys,32),
                   lambda:local_products_opaque(*xs,*ys,tile=(32,128,32)),
                   lambda:local_products_strassen(xs,weights),
                   lambda:local_products_strassen(xs,weights,opaque=True)]
        expected=functions[0]()
        graphs=[]
        for f in functions:
            for got,want in zip(f(),expected):
                torch.testing.assert_close(got,want,rtol=0,atol=0)
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                output=f()
            graphs.append(g)
        pairs=[[triton.testing.do_bench(g.replay) for g in graphs] for _ in range(3)]
        row=dict(shape=[m,k,n],columns=columns,pairs=pairs,
                 retained_strassen_static_bytes=sum(w.storage_bytes for w in weights),
                 all_full_outputs_exact=True)
        rows.append(row)
        print(row,flush=True)
    Path('artifacts/rep4_strassen_opaque_paired.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':
    main()
