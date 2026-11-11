"""Packed carry trees against unsigned arithmetic and protocol byte counts."""
from concurrent.futures import ThreadPoolExecutor
import random

import pytest
import torch

from rep4gpu.gpu.bits.carry_tree import carry_bits, reduction_word_counts, truncation_bits
from rep4gpu.arithmetic.conversion import b2a
from rep4gpu.protocol.runtime import Party, make_seeds
from rep4gpu.protocol.transport import IdealNetwork


def run_four(function, bits, word_bits=64):
    seeds = make_seeds()
    network = IdealNetwork()
    parties = [Party(p, network.endpoint(p), {e: seed for e, seed in seeds.items() if e != p}, bits=bits, binary_bits=word_bits) for p in range(4)]
    with ThreadPoolExecutor(4) as pool:
        return list(pool.map(function, parties))


def fixture(bits, count):
    rng = random.Random(bits+count)
    mask = (1 << bits)-1
    values = [0, 1, (1 << (bits-1))-1, 1 << (bits-1), mask, mask-1]
    x = [values[i] if i < len(values) else rng.getrandbits(bits) for i in range(count)]
    y = [values[-i-1] if i < len(values) else rng.getrandbits(bits) for i in range(count)]
    def tensor(v):
        return torch.tensor([n if n < (1 << 63) else n-(1 << 64) for n in v], dtype=torch.int64)
    return x, y, tensor(x), tensor(y)


@pytest.mark.parametrize('bits', [32, 64])
@pytest.mark.parametrize('shift', [1, 12, 31])
@pytest.mark.parametrize('count', [1, 19, 127])
def test_selected_carries_sign_and_payload(bits, shift, count):
    xs, ys, x, y = fixture(bits, count)
    def work(rt):
        a = rt.input(0, x if rt.party == 0 else None, x.shape, binary=True)
        b = rt.input(1, y if rt.party == 1 else None, y.shape, binary=True)
        before = rt.transport.bytes_sent
        low, high, sign = truncation_bits(rt, a, b, shift)
        sent = rt.transport.bytes_sent-before
        opened = tuple(rt.open(value) for value in (low, high, sign))
        return opened, sent
    mask = (1 << bits)-1
    expected = [torch.tensor([((a & ((1 << shift)-1))+(b & ((1 << shift)-1))) >> shift for a,b in zip(xs,ys)]),
                torch.tensor([(a+b) >> bits for a,b in zip(xs,ys)]),
                torch.tensor([((a+b)&mask) >> (bits-1) for a,b in zip(xs,ys)])]
    result = run_four(work, bits)
    for opened, _ in result:
        assert all(torch.equal(actual, target) for actual,target in zip(opened,expected))
    rounds = reduction_word_counts(count,bits,(shift,bits,bits-1))
    assert sum(sent for _,sent in result) == sum(48*words+192 for words in rounds)
    prefix_words = count*(1+2*(bits.bit_length()-1))
    assert sum(rounds) < prefix_words


@pytest.mark.parametrize('bits', [32,64])
def test_all_carry_positions_and_duplicates(bits):
    xs,ys,x,y=fixture(bits,17)
    positions=list(range(bits+1))+[12,0,bits]
    def work(rt):
        a=rt.input(2,x if rt.party==2 else None,x.shape,binary=True)
        b=rt.input(3,y if rt.party==3 else None,y.shape,binary=True)
        return [rt.open(carry) for carry in carry_bits(rt,a,b,positions)]
    for actual in run_four(work,bits):
        for position,carry in zip(positions,actual):
            mask=(1 << position)-1
            expected=torch.tensor([((a&mask)+(b&mask)) >> position for a,b in zip(xs,ys)])
            assert torch.equal(carry,expected)


@pytest.mark.parametrize('bits', [32,64])
def test_exact_truncation_correction(bits):
    xs,ys,x,y=fixture(bits,127)
    shift=12
    def work(rt):
        a=rt.input(0,x if rt.party==0 else None,x.shape,binary=True)
        b=rt.input(1,y if rt.party==1 else None,y.shape,binary=True)
        selected=truncation_bits(rt,a,b,shift)
        converted=[b2a(rt,bit,bits=1,signed=False) for bit in selected]
        upper_a=torch.tensor([value >> shift for value in xs])
        upper_b=torch.tensor([value >> shift for value in ys])
        sa=rt.input(0,upper_a if rt.party==0 else None,x.shape)
        sb=rt.input(1,upper_b if rt.party==1 else None,y.shape)
        low,high,sign=converted
        corrected=rt.sub(rt.add(rt.add(sa,sb),low),rt.lshift(rt.add(high,sign),bits-shift))
        return rt.open(corrected)
    expected=[]
    for a,b in zip(xs,ys):
        value=(a+b)&((1<<bits)-1)
        value=value-(1<<bits) if value>>(bits-1) else value
        expected.append(value>>shift)
    for actual in run_four(work,bits):
        assert torch.equal(actual,torch.tensor(expected))


def test_empty_and_zero_carries_do_not_communicate():
    class NoNetwork:
        def exchange(self,outgoing):
            raise AssertionError('no gates are required')
    seeds=make_seeds()
    party=Party(0,NoNetwork(),{e:seed for e,seed in seeds.items() if e!=0})
    x=party.public(torch.empty(0,dtype=torch.int64),binary=True)
    assert carry_bits(party,x,x,[0,12,64])[0].shape==x.shape
    y=party.public(torch.tensor([1]),binary=True)
    assert len(carry_bits(party,y,y,[0,0]))==2
    assert reduction_word_counts(0,64,[12,64,63])==[]


@pytest.mark.parametrize('count', [1,19,127])
@pytest.mark.parametrize('shift', [1,12,31])
def test_32_bit_packed_word_carries_and_payload(count,shift):
    xs,ys,x,y=fixture(32,count)
    def work(rt):
        a=rt.input(0,x if rt.party==0 else None,x.shape,binary=True)
        b=rt.input(1,y if rt.party==1 else None,y.shape,binary=True)
        before=rt.transport.bytes_sent
        result=truncation_bits(rt,a,b,shift)
        sent=rt.transport.bytes_sent-before
        return [rt.open(carry) for carry in result],sent
    expected=[torch.tensor([((a&((1<<shift)-1))+(b&((1<<shift)-1)))>>shift for a,b in zip(xs,ys)]),
              torch.tensor([(a+b)>>32 for a,b in zip(xs,ys)]),
              torch.tensor([((a+b)&0xffffffff)>>31 for a,b in zip(xs,ys)])]
    results=run_four(work,32,word_bits=32)
    for actual,_ in results:
        assert all(torch.equal(value,target) for value,target in zip(actual,expected))
    words=reduction_word_counts(count,32,(shift,32,31),word_bits=32)
    assert sum(sent for _,sent in results)==sum(24*word_count+192 for word_count in words)
    prefix=24*count*11+192*6
    assert sum(sent for _,sent in results)<prefix
