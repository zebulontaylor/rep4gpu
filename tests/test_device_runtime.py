import hashlib
import math
import struct
import pytest
import torch
from rep4gpu.runtime import Share, SCHEDULE, make_seeds, tensor_bytes, bytes_tensor
from rep4gpu.device_runtime import DeviceParty, mask_domain


def _cpu_mask(seed, excluded, counter, shape, bits, binary_bits, binary):
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
    width = binary_bits if binary else bits
    domain = bytes([bits,binary_bits,1 if binary else 0,width,excluded])
    assert mask_domain(bits,binary_bits,binary,excluded) == domain
    key = hashlib.sha256(seed+b'Rep4GPU-DevicePRGv1\0'+struct.pack('<Q', len(domain))+domain).digest()
    width = binary_bits if binary else bits
    encryptor = Cipher(algorithms.ChaCha20(key, struct.pack('<QQ', 0, counter)), mode=None).encryptor()
    data = encryptor.update(bytes(math.prod(shape)*width//8))+encryptor.finalize()
    return bytes_tensor(data, shape, 'cpu', width)


def _oracle(seeds, values, schedule, bits, binary_bits, binary):
    """Independent scalar/tensor joint-input oracle using CPU crypto packages."""
    from blake3 import blake3
    width = binary_bits if binary else bits
    shape = values[0].shape
    counters = {e:0 for e in range(4)}
    outputs = [[[] for _ in schedule] for _ in range(4)]
    payloads = [{ } for _ in range(4)]
    hashes = [{ } for _ in range(4)]
    add = torch.bitwise_xor if binary else torch.add
    sub = torch.bitwise_xor if binary else torch.sub
    normalize = (lambda t:t & 0xffffffff) if width == 32 else (lambda t:t)
    for k, ((sender, backup, receiver, outsider), value) in enumerate(zip(schedule, values)):
        r = _cpu_mask(seeds[receiver], receiver, counters[receiver], shape, bits, binary_bits, binary)
        counters[receiver] += 1
        masked = normalize(sub(value, r))
        data = tensor_bytes(masked, width)
        payloads[sender].setdefault(receiver, []).append(data)
        hashes[backup].setdefault(receiver, []).append((sender, blake3(data).digest()))
        for party in range(4):
            parts = [torch.zeros(shape, dtype=torch.int64) for _ in range(3)]
            if party != receiver:
                slot = (receiver-party-1)%4
                parts[slot] = add(parts[slot], r)
            if party in (sender, backup, receiver):
                slot = (outsider-party-1)%4
                parts[slot] = add(parts[slot], masked)
            outputs[party][k] = Share(tuple(normalize(t) for t in parts), binary)
    outgoing = []
    for party in range(4):
        outgoing.append({receiver:b''.join(payloads[party].get(receiver, []))+b''.join(
            tag for _, tag in sorted(hashes[party].get(receiver, []), key=lambda item:item[0]))
            for receiver in set(payloads[party]) | set(hashes[party])})
    incoming = [{sender:outgoing[sender][receiver] for sender in range(4) if receiver in outgoing[sender]}
                for receiver in range(4)]
    return outputs, outgoing, incoming, counters


class _Replay:
    def __init__(self, outgoing, incoming, corrupt=False):
        self.expected = {p:torch.frombuffer(bytearray(data), dtype=torch.uint8).to('cuda') for p,data in outgoing.items()}
        self.incoming = {p:torch.frombuffer(bytearray(data), dtype=torch.uint8).to('cuda') for p,data in incoming.items()}
        if corrupt:
            peer = next(iter(self.incoming))
            self.incoming[peer][0] ^= 1
        self.used = False

    def exchange_device(self, outgoing, incoming_sizes):
        assert not self.used
        assert set(outgoing) == set(self.expected)
        assert all(torch.equal(data, self.expected[p]) for p,data in outgoing.items())
        assert {p:t.numel() for p,t in self.incoming.items()} == incoming_sizes
        self.used = True
        return self.incoming


def _product_fixture(bits, binary_bits, binary, matrix):
    gen = torch.Generator().manual_seed(830)
    xs = [torch.randint(-(1 << 62), 1 << 62, (3, 5), generator=gen) for _ in range(4)]
    ys = [torch.randint(-(1 << 62), 1 << 62, (5, 4) if matrix else (3, 5), generator=gen) for _ in range(4)]
    xs[0][0,0] = -(1 << 63)
    ys[0][0,0] = -1
    local = []
    product = torch.matmul if matrix else torch.bitwise_and if binary else torch.mul
    add = torch.bitwise_xor if binary else torch.add
    for party in range(4):
        x = [xs[(party+j+1)%4] for j in range(3)]
        y = [ys[(party+j+1)%4] for j in range(3)]
        a = [None]*5
        for j in range(2):
            a[(party+j-1)%4] = add(product(add(x[j],x[j+1]),y[j]),product(x[j],y[j+1]))
        a[4] = add(product(x[0],y[2]),product(x[2],y[0]))
        local.append(a)
    values = [local[sender][k if k < 4 else 4] for k,(sender,_,_,_) in enumerate(SCHEDULE)]
    seeds = make_seeds()
    outputs, outgoing, incoming, counters = _oracle(seeds, values, SCHEDULE, bits, binary_bits, binary)
    def shares(party):
        return (Share(tuple(xs[(party+j+1)%4].to('cuda') for j in range(3)), binary),
                Share(tuple(ys[(party+j+1)%4].to('cuda') for j in range(3)), binary))
    expected = []
    for party in range(4):
        parts = [torch.zeros_like(outputs[party][0].parts[0]) for _ in range(3)]
        for result in outputs[party]:
            parts = [add(a,b) for a,b in zip(parts,result.parts)]
        if (binary_bits if binary else bits) == 32:
            parts = [t & 0xffffffff for t in parts]
        expected.append(parts)
    clear_x, clear_y = xs[0], ys[0]
    for x, y in zip(xs[1:], ys[1:]):
        clear_x, clear_y = add(clear_x,x), add(clear_y,y)
    clear_product = product(clear_x, clear_y)
    reconstructed = expected[1][2]
    for part in expected[0]:
        reconstructed = add(reconstructed,part)
    width = binary_bits if binary else bits
    if width == 32:
        clear_product, reconstructed = clear_product & 0xffffffff, reconstructed & 0xffffffff
    assert torch.equal(clear_product,reconstructed)
    assert sum(len(data) for row in outgoing for data in row.values()) == 6*values[0].numel()*(width//8)+6*32
    return seeds, shares, outgoing, incoming, counters, expected


@pytest.mark.parametrize('prg_batch',[False,True])
@pytest.mark.parametrize('hash_block_tree',[False,True])
@pytest.mark.parametrize('bits,binary_bits,binary', [(64,64,False),(64,64,True),(32,64,False),(32,64,True),(32,32,True)])
def test_device_product_cpu_crypto_transcript(bits, binary_bits, binary, hash_block_tree, prg_batch):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    pytest.importorskip('cryptography')
    pytest.importorskip('blake3')
    seeds, shares, outgoing, incoming, counters, expected = _product_fixture(bits,binary_bits,binary,False)
    for party in range(4):
        replay = _Replay(outgoing[party],incoming[party])
        runtime = DeviceParty(party,replay,{e:s for e,s in seeds.items() if e != party},bits=bits,binary_bits=binary_bits,hash_block_tree=hash_block_tree,prg_batch=prg_batch)
        result = runtime.mul(*shares(party))
        assert all(torch.equal(a.cpu(),b) for a,b in zip(result.parts,expected[party]))
        assert runtime.counters == {e:counters[e] for e in range(4) if e != party}
        assert replay.used and runtime.joint_inputs == 6 and runtime.multiplications == 1


@pytest.mark.parametrize('bits', [32,64])
def test_device_matmul_cpu_crypto_transcript(bits):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    pytest.importorskip('cryptography')
    pytest.importorskip('blake3')
    seeds, shares, outgoing, incoming, counters, expected = _product_fixture(bits,64,False,True)
    for party in range(4):
        runtime = DeviceParty(party,_Replay(outgoing[party],incoming[party]),
                              {e:s for e,s in seeds.items() if e != party},bits=bits)
        result = runtime.matmul(*shares(party))
        assert all(torch.equal(a.cpu(),b) for a,b in zip(result.parts,expected[party]))


@pytest.mark.parametrize('bits,binary_bits,binary', [(64,64,False),(32,32,False),(32,32,True)])
def test_device_two_joint_inputs(bits,binary_bits,binary):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    pytest.importorskip('cryptography')
    pytest.importorskip('blake3')
    schedule = ((0,1,3,2),(2,3,1,0))
    values = [torch.tensor([-(1 << 63),-1,17],dtype=torch.int64),torch.tensor([7,91,0],dtype=torch.int64)]
    seeds = make_seeds()
    outputs,outgoing,incoming,counters = _oracle(seeds,values,schedule,bits,binary_bits,binary)
    for party in range(4):
        runtime = DeviceParty(party,_Replay(outgoing[party],incoming[party]),
                              {e:s for e,s in seeds.items() if e != party},bits=bits,binary_bits=binary_bits)
        specs = [(*entry,value.to('cuda') if party in entry[:2] else None) for entry,value in zip(schedule,values)]
        actual = runtime.joint_input_batch(specs,values[0].shape,binary)
        for a,b in zip(actual,outputs[party]):
            assert all(torch.equal(x.cpu(),y) for x,y in zip(a.parts,b.parts))


def test_device_hash_tamper_rejected():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    pytest.importorskip('cryptography')
    pytest.importorskip('blake3')
    seeds,shares,outgoing,incoming,_,_ = _product_fixture(32,32,False,False)
    runtime = DeviceParty(3,_Replay(outgoing[3],incoming[3],corrupt=True),
                          {e:s for e,s in seeds.items() if e != 3},bits=32,binary_bits=32)
    with pytest.raises(ValueError,match='BLAKE3 hash mismatch'):
        runtime.mul(*shares(3))


def test_device_rejects_cpu_and_plain_transport():
    seeds = {e:bytes([e])*32 for e in (1,2,3)}
    with pytest.raises(ValueError,match='CUDA'):
        DeviceParty(0,None,seeds,device='cpu')
    with pytest.raises(ValueError,match='device-aware'):
        DeviceParty(0,None,seeds)


@pytest.mark.parametrize('bits,binary_bits', [(64,64),(32,32)])
def test_device_four_role_private_input_product_output(bits,binary_bits):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from concurrent.futures import ThreadPoolExecutor
    from rep4gpu.transport import IdealNetwork
    from rep4gpu.device_transport import DeviceIdealEndpoint
    device = torch.device('cuda',torch.cuda.current_device())
    network = IdealNetwork(timeout=60)
    seeds = make_seeds()
    x = torch.tensor([-(1 << 63),-1,0,17,2**31-1],dtype=torch.int64,device=device)
    y = torch.tensor([-1,71,7,-91,17],dtype=torch.int64,device=device)
    def work(party):
        transport = DeviceIdealEndpoint(network.endpoint(party),device=device)
        runtime = DeviceParty(party,transport,{e:s for e,s in seeds.items() if e != party},
                              device=device,bits=bits,binary_bits=binary_bits)
        a = runtime.input(0,x if party == 0 else None,x.shape)
        b = runtime.input(1,y if party == 1 else None,y.shape)
        before = transport.bytes_sent
        product = runtime.mul(a,b)
        sent = transport.bytes_sent-before
        output = runtime.open(product)
        return output,sent
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(work,range(4)))
    expected = x*y
    if bits == 32:
        expected = ((expected & 0xffffffff) ^ 0x80000000)-0x80000000
    assert all(torch.equal(value,expected) for value,_ in results)
    assert sum(sent for _,sent in results) == 6*x.numel()*(bits//8)+6*32


class _CompactDeviceParty(DeviceParty):
    def trunc(self,x,shift):
        from rep4gpu.exact_truncation import trunc
        return trunc(self,x,shift)

    def trunc_relu(self,x,shift):
        from rep4gpu.exact_truncation import trunc_relu
        return trunc_relu(self,x,shift)


@pytest.mark.parametrize('prg_batch',[False,True])
@pytest.mark.parametrize('bits', [32,64])
@pytest.mark.parametrize('bridge', [False,True])
def test_device_live_ffn_mixed_paths(bits,bridge,prg_batch):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from concurrent.futures import ThreadPoolExecutor
    from rep4gpu.activations import FixedPoint
    from rep4gpu.transport import IdealNetwork
    from rep4gpu.device_transport import DeviceIdealEndpoint,DeviceByteBridge
    device = torch.device('cuda',torch.cuda.current_device())
    network = IdealNetwork(timeout=90)
    seeds = make_seeds()
    x = torch.tensor([[16,-24,8],[-16,32,17]],dtype=torch.int64,device=device)
    w1 = torch.tensor([[8,-16,17,31],[16,8,-8,0],[-24,16,1,-17]],dtype=torch.int64,device=device)
    b1 = torch.tensor([1,-2,3,4],dtype=torch.int64,device=device)
    w2 = torch.tensor([[16,-8],[-17,31],[8,8],[1,-16]],dtype=torch.int64,device=device)
    expected_hidden = torch.clamp((x.cpu()@w1.cpu()+b1.cpu()*16)>>4,min=0)
    expected = (expected_hidden@w2.cpu())>>4
    def work(party):
        try:
            endpoint = network.endpoint(party)
            transport = (DeviceByteBridge if bridge else DeviceIdealEndpoint)(endpoint,device=device)
            runtime = _CompactDeviceParty(party,transport,{e:s for e,s in seeds.items() if e != party},
                                         device=device,bits=bits,binary_bits=bits,prg_batch=prg_batch)
            fp = FixedPoint(runtime,fractional_bits=4)
            a = runtime.input(0,x if party == 0 else None,x.shape)
            first = runtime.input(1,w1 if party == 1 else None,w1.shape)
            bias = runtime.input(1,b1 if party == 1 else None,b1.shape)
            second = runtime.input(1,w2 if party == 1 else None,w2.shape)
            hidden = fp.matmul_relu(a,first,bias)
            result = fp.matmul(hidden,second)
            return runtime.open(result).cpu()
        except BaseException as error:
            network.abort(error)
            raise
    with ThreadPoolExecutor(4) as pool:
        actual = list(pool.map(work,range(4)))
    assert all(torch.equal(value,expected) for value in actual)
