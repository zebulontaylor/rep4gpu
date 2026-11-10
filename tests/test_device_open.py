from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import math
import pytest
import torch
from rep4gpu.runtime import Share,tensor_bytes
from rep4gpu import device_open


def _fixture(bits,binary_bits,binary,shape,noncontiguous=False):
    from blake3 import blake3
    generator = torch.Generator().manual_seed(981)
    components = []
    for component in range(4):
        if noncontiguous:
            value = torch.randint(-(1 << 62),1 << 62,tuple(reversed(shape)),generator=generator).T
        else:
            value = torch.randint(-(1 << 62),1 << 62,shape,generator=generator)
        if value.numel():
            value[(0,)*len(shape)] = (-(1 << 63),(1 << 63)-1,-1,1)[component]
        components.append(value)
    width = binary_bits if binary else bits
    targets = ([-(1 << (width-1)),(1 << (width-1))-1,-1,0,1]
               if width == 64 or not binary else [0x80000000,0xffffffff,0,1,0x55555555])
    for index,target in enumerate(targets[:min(5,math.prod(shape))]):
        remainder,coordinates = index,[]
        for dimension in reversed(shape):
            coordinates.append(remainder % dimension)
            remainder //= dimension
        coordinates = tuple(reversed(coordinates))
        value = torch.tensor(target,dtype=torch.int64)
        for part in components[:3]:
            value = value ^ part[coordinates] if binary else value-part[coordinates]
        components[3][coordinates] = value
    shares = [Share(tuple(components[(party+j+1)%4] for j in range(3)),binary) for party in range(4)]
    outgoing = [[{(party+1)%4:tensor_bytes(shares[party].parts[0],width)},
                 {(party-1)%4:blake3(tensor_bytes(shares[party].parts[2],width)).digest()}]
                for party in range(4)]
    incoming = [[{(party-1)%4:outgoing[(party-1)%4][0][party]},
                 {(party+1)%4:outgoing[(party+1)%4][1][party]}]
                for party in range(4)]
    expected = components[0]
    for part in components[1:]:
        expected = expected ^ part if binary else expected+part
    if width == 32:
        expected &= 0xffffffff
        if not binary:
            expected = (expected ^ 0x80000000)-0x80000000
    return shares,outgoing,incoming,expected


def _cuda_bytes(data):
    if not data:
        return torch.empty(0,dtype=torch.uint8,device='cuda')
    return torch.frombuffer(bytearray(data),dtype=torch.uint8).to('cuda')


class _Replay:
    def __init__(self,outgoing,incoming,corrupt_round=None,unaligned=False):
        self.outgoing = [{p:_cuda_bytes(data) for p,data in frames.items()} for frames in outgoing]
        self.incoming = [{p:_cuda_bytes(data) for p,data in frames.items()} for frames in incoming]
        if unaligned:
            for frame in self.incoming:
                for peer,value in list(frame.items()):
                    buffer = torch.empty(value.numel()+1,dtype=torch.uint8,device=value.device)
                    buffer[1:] = value
                    frame[peer] = buffer[1:]
        if corrupt_round is not None:
            first = next(iter(self.incoming[corrupt_round].values()))
            first[0] ^= 1
        self.round = 0

    def exchange_device(self,outgoing,incoming_sizes):
        expected = self.outgoing[self.round]
        assert set(outgoing) == set(expected)
        assert all(torch.equal(value,expected[p]) for p,value in outgoing.items())
        incoming = self.incoming[self.round]
        assert {p:t.numel() for p,t in incoming.items()} == incoming_sizes
        self.round += 1
        return incoming


def _runtime(party,transport,bits,binary_bits):
    return SimpleNamespace(party=party,transport=transport,hash_block_tree=True,
                           _wire_bits=lambda binary:binary_bits if binary else bits)


@pytest.mark.parametrize('bits,binary_bits,binary',[(64,64,False),(64,64,True),(32,64,False),(32,64,True),(32,32,True)])
@pytest.mark.parametrize('shape',[(),(0,),(33,),(3,11)])
def test_device_open_independent_cpu_blake3_transcript(bits,binary_bits,binary,shape):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    pytest.importorskip('blake3')
    shares,outgoing,incoming,expected = _fixture(bits,binary_bits,binary,shape,noncontiguous=len(shape)==2)
    for party in range(4):
        replay = _Replay(outgoing[party],incoming[party],unaligned=True)
        runtime = _runtime(party,replay,bits,binary_bits)
        share = Share(tuple(part.to('cuda') for part in shares[party].parts),binary)
        result = device_open.open_device(runtime,share)
        assert result.shape == expected.shape and result.dtype == torch.int64
        assert torch.equal(result.cpu(),expected)
        assert replay.round == 2
    width = binary_bits if binary else bits
    assert sum(len(data) for frames in outgoing for frame in frames for data in frame.values()) == 4*math.prod(shape)*(width//8)+4*32


@pytest.mark.parametrize('corrupt_round,shape',[(0,(33,)),(1,(33,)),(1,(0,))])
def test_device_open_corruption_rejected_before_reconstruction(monkeypatch,corrupt_round,shape):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    pytest.importorskip('blake3')
    shares,outgoing,incoming,_ = _fixture(32,32,False,shape)
    replay = _Replay(outgoing[0],incoming[0],corrupt_round=corrupt_round)
    runtime = _runtime(0,replay,32,32)
    def forbidden(*args,**kwargs):
        raise AssertionError('reconstruction must not run after a failed check')
    monkeypatch.setattr(device_open,'_finish',forbidden)
    share = Share(tuple(part.to('cuda') for part in shares[0].parts))
    with pytest.raises(ValueError,match='BLAKE3 component hash mismatch'):
        device_open.open_device(runtime,share)
    assert replay.round == 2


@pytest.mark.parametrize('bridge',[False,True])
@pytest.mark.parametrize('shape',[(0,),(65,)])
def test_device_open_four_role_adapters(bridge,shape):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    pytest.importorskip('blake3')
    from rep4gpu.transport import IdealNetwork
    from rep4gpu.device_transport import DeviceIdealEndpoint,DeviceByteBridge
    shares,_,_,expected = _fixture(32,32,False,shape)
    device = torch.device('cuda',torch.cuda.current_device())
    network = IdealNetwork(timeout=60)
    def work(party):
        try:
            transport = (DeviceByteBridge if bridge else DeviceIdealEndpoint)(network.endpoint(party),device=device)
            runtime = _runtime(party,transport,32,32)
            share = Share(tuple(part.to(device) for part in shares[party].parts))
            return device_open.open_device(runtime,share).cpu()
        except BaseException as error:
            network.abort(error)
            raise
    with ThreadPoolExecutor(4) as pool:
        actual = list(pool.map(work,range(4)))
    assert all(torch.equal(value,expected) for value in actual)


def test_device_open_rejects_cpu():
    share = Share(tuple(torch.zeros(1,dtype=torch.int64) for _ in range(3)))
    with pytest.raises(ValueError,match='CUDA'):
        device_open.open_device(None,share)


def test_device_open_cpu_oracle_boundary_outputs():
    pytest.importorskip('blake3')
    for bits,binary_bits,binary in [(64,64,False),(64,64,True),(32,64,False),(32,64,True),(32,32,True)]:
        _,outgoing,_,expected = _fixture(bits,binary_bits,binary,(33,))
        width = binary_bits if binary else bits
        targets = ([-(1 << (width-1)),(1 << (width-1))-1,-1,0,1]
                   if width == 64 or not binary else [0x80000000,0xffffffff,0,1,0x55555555])
        assert expected[:5].tolist() == targets
        for party in range(4):
            assert set(outgoing[party][0]) == {(party+1)%4}
            assert set(outgoing[party][1]) == {(party-1)%4}
