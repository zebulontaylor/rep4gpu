"""Inference capture keeps live protocol boundaries and reservations."""
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch


@pytest.mark.parametrize('bits',[32,64])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_inference_graph_matches_eager_and_preserves_outputs(bits):
    from rep4gpu.models.device_inference import DeviceInferenceParty
    from rep4gpu.models.phase_graph import PhaseGraphParty
    from rep4gpu.protocol.device_transport import DeviceIdealEndpoint
    from rep4gpu.protocol.runtime import make_seeds
    from rep4gpu.protocol.transport import IdealNetwork

    seeds = make_seeds()
    def execute(kind):
        network = IdealNetwork(timeout=90)
        roles = [kind(p, DeviceIdealEndpoint(network.endpoint(p)),
                      {e:s for e,s in seeds.items() if e != p}, bits=bits, gpu_open=True)
                 for p in range(4)]
        inputs = [r.public(torch.full((17,), 3, dtype=torch.int64, device=r.device)) for r in roles]
        functions = [lambda x, r=r: r.open(r.mul(x,x)) for r in roles]
        plans = [r.prepare_inference(fn,x) for r,fn,x in zip(roles,functions,inputs)] if kind is PhaseGraphParty else None
        assert all(not any(r.counters.values()) for r in roles)
        def worker(p):
            r, x = roles[p], inputs[p]
            try:
                fn = plans[p].run if plans else functions[p]
                first = fn(x)
                saved = first.clone()
                second = fn(r.public(torch.full((17,), 5, dtype=torch.int64, device=r.device)))
                torch.testing.assert_close(first, saved)
                return first.cpu(), second.cpu(), dict(r.counters)
            except BaseException as error:
                network.abort(error)
                raise
        with ThreadPoolExecutor(4) as pool:
            return list(pool.map(worker,range(4)))
    eager, captured = execute(DeviceInferenceParty), execute(PhaseGraphParty)
    for (a,b,c),(x,y,z) in zip(eager,captured):
        torch.testing.assert_close(a,x)
        torch.testing.assert_close(b,y)
        assert c == z
        assert torch.all(x == 9) and torch.all(y == 25)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_failed_check_stops_dependent_segment_and_poisons_plan():
    from rep4gpu.models.phase_graph import PhaseGraphParty
    from rep4gpu.protocol.device_transport import DeviceIdealEndpoint
    from rep4gpu.protocol.runtime import make_seeds
    from rep4gpu.protocol.transport import IdealNetwork

    network = IdealNetwork(timeout=5)
    seeds = make_seeds()
    role = PhaseGraphParty(0, DeviceIdealEndpoint(network.endpoint(0)),
                           {e:s for e,s in seeds.items() if e != 0}, bits=32)
    value = torch.zeros(1, dtype=torch.int64, device=role.device)
    def function(x):
        actual = torch.zeros(32, dtype=torch.uint8, device=role.device)
        wanted = torch.ones_like(actual)
        role._verify_tags(actual, wanted)
        role._exchange_device({1: actual}, {1: 32})
        return x.add_(1)
    plan = role.prepare_inference(function,value)
    with pytest.raises(ValueError, match='hash mismatch'):
        plan.run(value)
    assert plan.inputs[0].item() == 0
    with pytest.raises(RuntimeError, match='poisoned'):
        plan.run(value)


@pytest.mark.parametrize('check_at_send', [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_reused_plan_checks_current_input_before_send(check_at_send):
    from rep4gpu.models.phase_graph import PhaseGraphParty
    from rep4gpu.protocol.device_transport import DeviceIdealEndpoint
    from rep4gpu.protocol.runtime import make_seeds
    from rep4gpu.protocol.transport import IdealNetwork

    seeds = make_seeds()
    network = IdealNetwork(timeout=1)
    transport = DeviceIdealEndpoint(network.endpoint(0))
    role = PhaseGraphParty(0, transport, {e:s for e,s in seeds.items() if e != 0})
    value = torch.zeros(32, dtype=torch.uint8, device=role.device)
    def function(x):
        role._verify_tags(x, torch.zeros_like(x))
        role._exchange_device({1:x}, {})
        return x
    plan = role.prepare_inference(function, value, check_at_send=check_at_send)
    sends = []
    def exchange(outgoing, sizes, buffers):
        sends.append(outgoing[1].clone())
        return buffers
    transport.exchange_device_into = exchange
    plan.run(value)
    assert len(sends) == 1
    with pytest.raises(ValueError, match='hash mismatch'):
        plan.run(torch.ones_like(value))
    assert len(sends) == 1
    with pytest.raises(RuntimeError, match='poisoned'):
        plan.run(value)


@pytest.mark.parametrize('invalid', ['peers', 'size', 'dtype'])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_invalid_frame_poisons_plan(invalid):
    from rep4gpu.models.phase_graph import PhaseGraphParty
    from rep4gpu.protocol.device_transport import DeviceIdealEndpoint
    from rep4gpu.protocol.runtime import make_seeds
    from rep4gpu.protocol.transport import IdealNetwork

    network = IdealNetwork(timeout=5)
    seeds = make_seeds()
    transport = DeviceIdealEndpoint(network.endpoint(0))
    role = PhaseGraphParty(0,transport,{e:s for e,s in seeds.items() if e != 0},bits=32)
    value = torch.zeros(32,dtype=torch.uint8,device=role.device)
    def function(x):
        return role._exchange_device({1:x},{1:32})[1]+1
    plan = role.prepare_inference(function,value)
    def exchange(outgoing,sizes):
        if invalid == 'peers':
            return {}
        return {1:torch.zeros(31 if invalid == 'size' else 32,
                             dtype=torch.int64 if invalid == 'dtype' else torch.uint8,
                             device=role.device)}
    transport.exchange_device = exchange
    with pytest.raises((ValueError,TypeError),match='invalid inference incoming|public schedule|expected|flat uint8'):
        plan.run(value)
    with pytest.raises(RuntimeError,match='poisoned'):
        plan.run(value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_build_rejects_live_byte_exchange_without_changing_counters():
    from rep4gpu.models.phase_graph import PhaseGraphParty
    from rep4gpu.protocol.device_transport import DeviceIdealEndpoint
    from rep4gpu.protocol.runtime import make_seeds
    from rep4gpu.protocol.transport import IdealNetwork
    network = IdealNetwork(timeout=1)
    transport = DeviceIdealEndpoint(network.endpoint(0))
    seeds = make_seeds()
    role = PhaseGraphParty(0,transport,{e:s for e,s in seeds.items() if e != 0})
    value = torch.zeros(1,dtype=torch.int64,device=role.device)
    def function(x):
        role._random(1,x.shape,x.device)
        role.transport.exchange({}, {})
        return x
    before = dict(role.counters)
    with pytest.raises(RuntimeError,match='live transport'):
        role.prepare_inference(function,value)
    assert role.counters == before
    assert role.transport is transport and role._phase_builder is None


@pytest.mark.parametrize('check_at_send',[False,True])
@pytest.mark.parametrize('sparse_products',[False,True])
@pytest.mark.parametrize('rounding,bounded',[('floor',False),('probabilistic',False),('probabilistic',True)])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_tiny_private_ffn_transcripts_match_eager(rounding,bounded,check_at_send,sparse_products,storage='words32'):
    from experiments.benchmarks.inference.inference import infer
    from rep4gpu.models.device_inference import DeviceInferenceParty
    from rep4gpu.models.phase_graph import PhaseGraphParty
    from rep4gpu.protocol.device_transport import DeviceIdealEndpoint, DeviceRecorder
    from rep4gpu.protocol.runtime import make_seeds
    from rep4gpu.protocol.transport import IdealNetwork
    seeds = make_seeds()
    values = [torch.tensor([[3,-5,2]]), torch.arange(12).reshape(3,4)-5,
              torch.tensor([2,-1,3,0]), torch.arange(8).reshape(4,2)-3, torch.tensor([1,-2])]
    if storage=='strassen32':
        values=[torch.tensor([[3,-5,2,1],[-2,1,4,-3]]),torch.arange(16).reshape(4,4)-5,
                torch.tensor([2,-1,3,0]),torch.arange(8).reshape(4,2)-3,torch.tensor([1,-2])]
    def execute(kind):
        network = IdealNetwork(timeout=90)
        records = [DeviceRecorder(DeviceIdealEndpoint(network.endpoint(p))) for p in range(4)]
        roles = [kind(p,records[p],{e:s for e,s in seeds.items() if e != p},bits=32,
                      gpu_open=True,rounding=rounding,bounded_sign=bounded,
                      sparse_products=sparse_products) for p in range(4)]
        def guarded(fn):
            def work(p):
                try:
                    return fn(p)
                except BaseException as error:
                    network.abort(error)
                    raise
            with ThreadPoolExecutor(4) as pool:
                return list(pool.map(work,range(4)))
        def distribute(p):
            r = roles[p]
            result = []
            for owner,value in zip((0,1,1,1,1),values):
                result.append(r.input(owner,value.to(r.device) if p == owner else None,value.shape))
            result[1] = r.prepare_weight_share(result[1],storage=storage)
            result[3] = r.prepare_weight_share(result[3],storage=storage)
            return result
        shares = guarded(distribute)
        changed = guarded(lambda p:roles[p].input(0,(values[0]+1).to(roles[p].device) if p == 0 else None,values[0].shape))
        for record in records:
            record.transcript.clear()
        counters = [dict(r.counters) for r in roles]
        plans = [r.prepare_inference(lambda v,r=r:infer(r,v,2),v,check_at_send=check_at_send) for r,v in zip(roles,shares)] if kind is PhaseGraphParty else None
        assert [r.counters for r in roles] == counters
        outputs = []
        for iteration in range(2):
            if iteration:
                for p in range(4):
                    shares[p][0] = changed[p]
            outputs.append(guarded(lambda p:plans[p].run(shares[p]) if plans else infer(roles[p],shares[p],2)))
        torch.cuda.synchronize()
        return outputs,[dict(r.counters) for r in roles],[r.transcript for r in records]
    expected,ec,et = execute(DeviceInferenceParty)
    actual,ac,at = execute(PhaseGraphParty)
    assert ec == ac
    for a,b in zip(actual,expected):
        for x,y in zip(a,b):
            torch.testing.assert_close(x,y)
    assert at == et


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_copy_failure_after_upload_poisons_plan(monkeypatch):
    from rep4gpu.models.phase_graph import PhaseGraphParty
    from rep4gpu.protocol.device_transport import DeviceIdealEndpoint
    from rep4gpu.protocol.runtime import make_seeds
    from rep4gpu.protocol.transport import IdealNetwork
    network = IdealNetwork(timeout=1)
    seeds = make_seeds()
    role = PhaseGraphParty(0,DeviceIdealEndpoint(network.endpoint(0)),
                           {e:s for e,s in seeds.items() if e != 0})
    value = torch.zeros(1,dtype=torch.int64,device=role.device)
    plan = role.prepare_inference(lambda x:x+role._random(1,x.shape,x.device),value)
    original = torch.Tensor.copy_
    def copy(destination,source,*args,**kwargs):
        if destination is plan.inputs[0]:
            raise RuntimeError('injected input copy failure')
        return original(destination,source,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'copy_',copy)
    with pytest.raises(RuntimeError,match='input copy failure'):
        plan.run(value)
    assert role.counters[1] == 1
    plan.ready.synchronize()
    with pytest.raises(RuntimeError,match='poisoned'):
        plan.run(value)


@pytest.mark.parametrize('exchange',[False,True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_deferred_checks_keep_first_failure_and_prevent_send(exchange):
    from rep4gpu.models.phase_graph import PhaseGraphParty
    from rep4gpu.protocol.device_transport import DeviceIdealEndpoint
    from rep4gpu.protocol.runtime import make_seeds
    from rep4gpu.protocol.transport import IdealNetwork
    seeds = make_seeds()
    network = IdealNetwork(timeout=1)
    transport = DeviceIdealEndpoint(network.endpoint(0))
    role = PhaseGraphParty(0,transport,{e:s for e,s in seeds.items() if e != 0})
    value = torch.zeros(32,dtype=torch.uint8,device=role.device)
    def function(x):
        role._verify_tags(x,x+1,'first invalid check')
        role._verify_tags(x,x,'second valid check')
        role._verify_tags(x+1,x+1,'third valid check')
        output = x+3
        if exchange:
            role._exchange_device({}, {})
        return output
    plan = role.prepare_inference(function,value,check_at_send=True)
    batches = [value for kind,value in plan.actions if kind == 'verify_batch']
    assert len(batches) == 1
    assert batches[0][0].shape == (3,)
    assert batches[0][1] == ('first invalid check', 'second valid check', 'third valid check')
    calls = []
    def unexpected(*args,**kwargs):
        calls.append(args)
        raise AssertionError('failed verification allowed a dependent exchange')
    transport.exchange_device = unexpected
    with pytest.raises(ValueError,match='first invalid check'):
        plan.run(value)
    assert not calls
    with pytest.raises(RuntimeError,match='poisoned'):
        plan.run(value)


@pytest.mark.parametrize('staging',['device','host'])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_bound_receive_replay_checks_tampered_payload(staging):
    from rep4gpu.models.phase_graph import PhaseGraphParty
    from rep4gpu.protocol.device_transport import DeviceIdealEndpoint,DeviceReplay,DeviceFrame
    from rep4gpu.protocol.runtime import make_seeds
    from rep4gpu.protocol.transport import IdealNetwork
    seeds = make_seeds()
    network = IdealNetwork(timeout=1)
    role = PhaseGraphParty(0,DeviceIdealEndpoint(network.endpoint(0)),
                           {e:s for e,s in seeds.items() if e != 0})
    value = torch.zeros(32,dtype=torch.uint8,device=role.device)
    def function(x):
        received = role._exchange_device({1:x},{3:32})[3]
        role._verify_tags(received,x,'tampered bound frame')
        return received+1
    plan = role.prepare_inference(function,value,check_at_send=True)
    frame = DeviceFrame('device',{1:bytes(32)},{3:bytes([1])*32})
    role.transport = DeviceReplay([frame],device=role.device,staging=staging,
                                  incoming_buffers=plan.receive_buffers)
    with pytest.raises(ValueError,match='tampered bound frame'):
        plan.run(value)
    assert plan.failed


@pytest.mark.parametrize('rounding',['floor','probabilistic'])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_strassen_private_ffn_graph_matches_eager(rounding):
    test_tiny_private_ffn_transcripts_match_eager(rounding,True,True,True,storage='strassen32')
