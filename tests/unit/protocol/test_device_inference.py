from concurrent.futures import ThreadPoolExecutor
import pytest
import torch
from rep4gpu.arithmetic.activations import FixedPoint
from rep4gpu.protocol.ring_inference import DeviceInferenceParty
from rep4gpu.transports.device import DeviceIdealEndpoint,DeviceRecorder,DeviceReplay
from rep4gpu.protocol.runtime import make_seeds
from rep4gpu.transports.byte import IdealNetwork


def _ffn_parity(bits,shift,reference_optimized,actual_optimized,reference_storage=None,actual_storage=None,batch=5,dimensions=(3,7,5)):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    device = torch.device('cuda',torch.cuda.current_device())
    generator = torch.Generator().manual_seed(481)
    width,hidden,out = dimensions
    scale = 1 << shift
    x = torch.randint(-3,4,(batch,width),generator=generator)*max(1,scale//8)
    w1 = torch.randint(-4,5,(width,hidden),generator=generator)*max(1,scale//16)
    b1 = torch.randint(-3,4,(hidden,),generator=generator)*max(1,scale//64)
    w2 = torch.randint(-3,4,(hidden,out),generator=generator)*max(1,scale//16)
    accumulated = x@w1+b1*scale
    expected_hidden = torch.clamp(accumulated>>shift,min=0)
    expected = (expected_hidden@w2)>>shift
    assert bool(torch.all((accumulated >= -(1 << 31)) & (accumulated < 1 << 31)))
    assert bool(torch.all(((expected_hidden@w2) >= -(1 << 31)) & ((expected_hidden@w2) < 1 << 31)))
    inputs = tuple(t.to(device) for t in (x,w1,b1,w2))
    seeds = make_seeds()
    network = IdealNetwork(timeout=90)
    recordings = [DeviceRecorder(DeviceIdealEndpoint(network.endpoint(p),device=device)) for p in range(4)]

    def forward(runtime,storage=None):
        fp = FixedPoint(runtime,fractional_bits=shift)
        shared = [runtime.input(owner,value if runtime.party == owner else None,value.shape)
                  for owner,value in zip((0,1,1,1),inputs)]
        first,second = shared[1],shared[3]
        if storage is not None:
            first = runtime.prepare_weight_share(first,storage=storage)
            second = runtime.prepare_weight_share(second,storage=storage)
        hidden = fp.matmul_relu(shared[0],first,shared[2])
        result = fp.matmul(hidden,second)
        opened = runtime.open(result)
        return (tuple(t.cpu() for t in hidden.parts),tuple(t.cpu() for t in result.parts),
                opened.cpu(),dict(runtime.counters))

    def reference(party):
        try:
            runtime = DeviceInferenceParty(party,recordings[party],
                                          {e:s for e,s in seeds.items() if e != party},
                                          device=device,bits=bits,optimized=reference_optimized)
            return forward(runtime,reference_storage)
        except BaseException as error:
            network.abort(error)
            raise
    with ThreadPoolExecutor(4) as pool:
        references = list(pool.map(reference,range(4)))
    assert all(torch.equal(result[2],expected) for result in references)

    # Fixed messages and seeds are reused only as this deterministic test
    # fixture. Live requests must use fresh counters and session seeds.
    for party in range(4):
        replay = DeviceReplay(recordings[party].transcript,device=device,check_outgoing=True)
        runtime = DeviceInferenceParty(party,replay,{e:s for e,s in seeds.items() if e != party},
                                      device=device,bits=bits,optimized=actual_optimized)
        actual = forward(runtime,actual_storage)
        for group,reference in zip(actual[:2],references[party][:2]):
            assert all(torch.equal(a,b) for a,b in zip(group,reference))
        assert torch.equal(actual[2],expected)
        assert actual[3] == references[party][3]
        replay.assert_consumed()


@pytest.mark.parametrize('bits', [32, 64])
@pytest.mark.parametrize('shift', [0, 14])
def test_optimized_ffn_exact_integer_output(bits, shift):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    device = torch.device('cuda', torch.cuda.current_device())
    generator = torch.Generator().manual_seed(481)
    width, hidden, out = 3, 7, 5
    scale = 1 << shift
    x = torch.randint(-3, 4, (5, width), generator=generator) * max(1, scale // 8)
    w1 = torch.randint(-4, 5, (width, hidden), generator=generator) * max(1, scale // 16)
    b1 = torch.randint(-3, 4, (hidden,), generator=generator) * max(1, scale // 64)
    w2 = torch.randint(-3, 4, (hidden, out), generator=generator) * max(1, scale // 16)
    expected = (torch.clamp((x @ w1 + b1 * scale) >> shift, min=0) @ w2) >> shift
    inputs = tuple(t.to(device) for t in (x, w1, b1, w2))
    for optimized in (False, True):
        seeds, network = make_seeds(), IdealNetwork(timeout=90)
        def role(party):
            try:
                rt = DeviceInferenceParty(
                    party, DeviceIdealEndpoint(network.endpoint(party), device=device),
                    {e: s for e, s in seeds.items() if e != party},
                    device=device, bits=bits, optimized=optimized,
                )
                fp = FixedPoint(rt, fractional_bits=shift)
                shared = [rt.input(owner, val if party == owner else None, val.shape)
                          for owner, val in zip((0, 1, 1, 1), inputs)]
                h = fp.matmul_relu(shared[0], shared[1], shared[2])
                return rt.open(fp.matmul(h, shared[3])).cpu()
            except BaseException as err:
                network.abort(err)
                raise
        with ThreadPoolExecutor(4) as pool:
            outputs = list(pool.map(role, range(4)))
        assert all(torch.equal(o, expected) for o in outputs)


def test_tuned_prefill_ffn_matches_limbs_transcript_and_local_shares():
    _ffn_parity(32, 14, True, True, reference_storage='limbs', actual_storage='words32', batch=17)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('bits', [32, 64])
@pytest.mark.parametrize('optimized', [False, True])
@pytest.mark.parametrize('binary', [False, True])
def test_sign_including_signed_boundaries(bits, optimized, binary):
    values = torch.tensor([0, 1, -1, -(1 << (bits-1)), (1 << (bits-1))-1])
    expected = (values >> (bits-1)) & 1
    network, seeds = IdealNetwork(timeout=30), make_seeds()
    def role(p):
        try:
            rt = DeviceInferenceParty(p, DeviceIdealEndpoint(network.endpoint(p)),
                                      {i:s for i,s in seeds.items() if i != p},
                                      bits=bits, optimized=optimized)
            share = rt.input(0, values if p == 0 else None, values.shape, binary=binary)
            return rt.open(rt.sign_bit(share)).cpu()
        except BaseException as error:
            network.abort(error)
            raise
    with ThreadPoolExecutor(4) as pool:
        assert all(torch.equal(value, expected) for value in pool.map(role, range(4)))


@pytest.mark.parametrize('batch',[2,32])
def test_strassen32_ffn_matches_words32_transcript_and_local_shares(batch):
    _ffn_parity(32,14,True,True,reference_storage='words32',actual_storage='strassen32',
               batch=batch,dimensions=(4,8,6))
