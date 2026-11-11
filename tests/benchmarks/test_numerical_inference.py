"""Complete numerical FFNs retain correctness when graph inputs change."""
from concurrent.futures import ThreadPoolExecutor
import pytest
import torch
from experiments.benchmarks.inference.numerical_inference_common import fixture, oracle, infer


@pytest.mark.parametrize('kind',['silu','gelu','swiglu'])
def test_floor_forward_graph_refresh(kind):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from rep4gpu.models.phase_graph import PhaseGraphParty
    from rep4gpu.protocol.device_transport import DeviceIdealEndpoint
    from rep4gpu.protocol.runtime import make_seeds
    from rep4gpu.protocol.transport import IdealNetwork
    torch.set_num_threads(1)
    _,values=fixture(kind,1,8,16,8)
    network,seeds=IdealNetwork(timeout=180),make_seeds()
    parties=[PhaseGraphParty(i,DeviceIdealEndpoint(network.endpoint(i),torch.device('cuda:0')),
              {j:s for j,s in seeds.items() if j!=i},device='cuda:0',bits=64,binary_bits=64,
              optimized=True,gpu_open=True,rounding='floor',sparse_products=True) for i in range(4)]
    with ThreadPoolExecutor(4) as pool:
        def parallel(fn):
            def guarded(p):
                try:return fn(p)
                except BaseException as error:network.abort(error);raise
            return list(pool.map(guarded,parties))
        shares=parallel(lambda p:[p.input(0,v if p.party==0 else None,v.shape) for v in values])
        plans=[p.prepare_inference(lambda x,p=p:infer(p,[x,*shares[p.party][1:]],kind=kind),shares[p.party][0],check_at_send=True) for p in parties]
        for refresh in range(2):
            if refresh:
                values[0]=values[0]//2
                updated=parallel(lambda p:p.input(0,values[0] if p.party==0 else None,values[0].shape))
                for i in range(4):shares[i][0]=updated[i]
            wanted,_=oracle(kind,values,probabilistic=False)
            outputs=parallel(lambda p:plans[p.party].run(shares[p.party][0]))
            assert torch.equal(wanted.lo,wanted.hi)
            assert all(torch.equal(o.cpu(),wanted.lo) for o in outputs)
