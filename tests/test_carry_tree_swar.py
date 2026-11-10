"""Whole-word packing must preserve every reference node and gate bit."""
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch


def test_cpu_interpreter_matches_all_packing_stages():
    # The Triton interpreter operates on CPU tensors in a separate process so
    # this check does not initialize CUDA or change other tests' JIT settings.
    script = r'''
import torch
import triton
from rep4gpu import carry_tree_gpu as reference, carry_tree_swar as swar
for width,lengths,storage in [(32,(14,31,32),torch.int64), (64,(14,63,64),torch.int64), (32,(1,3,17,64),torch.int64), (64,(1,3,17,32),torch.int64), (32,(14,31,32),torch.uint32)]:
 for count in (1,3,33):
  generator=torch.Generator().manual_seed(width+count)
  values=[torch.randint(-(1<<62),1<<62,(count,),generator=generator) for _ in range(6)]
  sizes=tuple(1<<(p-1).bit_length() for p in lengths)
  words,offsets,total=reference._layout(count,sizes,width)
  state=torch.empty((2,3,total),dtype=torch.int64)
  actual=torch.empty(state.shape,dtype=storage)
  for length,size,n,offset in zip(lengths,sizes,words,offsets):
   reference._initialize[(triton.cdiv(n,32),3)](*values,state,count,length,size,total,offset,n,width,2,32)
   swar._initialize[(triton.cdiv(n,128),3)](*values,actual,count,length,size,total,offset,n,width,2,128)
  assert torch.equal(state,actual.to(torch.int64)),('initialize',width,count,lengths)
  while any(size>1 for size in sizes):
   gatesizes=[]; gatecount=0
   for size in sizes:
    gatesizes.append(gatecount)
    if size>1:gatecount+=count*size
   gatewords=(gatecount+width-1)//width
   expected_gates=torch.empty((2,3,gatewords),dtype=torch.int64)
   actual_gates=torch.empty_like(expected_gates)
   reference._pack_gates[(triton.cdiv(gatewords,32),3)](state,expected_gates,count,total,width,gatewords,gatecount,sizes,offsets,tuple(gatesizes),32)
   swar._pack_gates[(triton.cdiv(gatewords,128),3)](actual,actual_gates,count,total,width,gatewords,gatecount,sizes,offsets,tuple(gatesizes),128)
   assert torch.equal(expected_gates,actual_gates),('pack',width,count,sizes)
   products=torch.randint(-(1<<62),1<<62,(3,gatewords),generator=generator)
   nextsizes=tuple(size//2 if size>1 else 1 for size in sizes)
   nextwords,nextoffsets,nexttotal=reference._layout(count,nextsizes,width)
   expected_next=torch.empty((2,3,nexttotal),dtype=torch.int64)
   actual_next=torch.empty(expected_next.shape,dtype=storage)
   for oldsize,size,oldoffset,offset,n,goffset in zip(sizes,nextsizes,offsets,nextoffsets,nextwords,gatesizes):
    reference._next_tree[(triton.cdiv(n,32),3)](state,*products,expected_next,count,oldsize,size,total,oldoffset,nexttotal,offset,n,width,goffset,32)
    swar._next_tree[(triton.cdiv(n,128),3)](actual,*products,actual_next,count,oldsize,size,total,oldoffset,nexttotal,offset,n,width,goffset,gatewords,128)
   assert torch.equal(expected_next,actual_next.to(torch.int64)),('next',width,count,sizes)
   state,actual,sizes,words,offsets,total=expected_next,actual_next,nextsizes,nextwords,nextoffsets,nexttotal
print('all CPU-interpreted node and gate buffers match')
'''
    environment = dict(os.environ, TRITON_INTERPRET='1')
    result = subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).resolve().parents[1],
                            env=environment, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('bits,binary_bits,shift', [(32,32,14), (64,64,14), (32,64,12)])
def test_gpu_swar_matches_cpu_protocol_transcript(bits, binary_bits, shift):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    pytest.importorskip('cryptography')
    from test_carry_tree_gpu import _reference
    from rep4gpu import carry_tree_swar
    from rep4gpu.fused_runtime import FusedParty
    from rep4gpu.transport import ReplayTransport
    seeds, transports, shares, expected = _reference(bits, binary_bits, (), shape=(33,), shift=shift)
    for party in range(4):
        replay = ReplayTransport(transports[party].transcript)
        runtime = FusedParty(party, replay, {e:s for e,s in seeds.items() if e != party},
                             device='cuda', bits=bits, binary_bits=binary_bits, prg='aes')
        actual = carry_tree_swar.truncation_bits(runtime, *shares(party, 'cuda'), shift)
        for result, reference in zip(actual, expected[party][0]):
            assert all(torch.equal(a.cpu(), b) for a,b in zip(result.parts,reference.parts))
        assert runtime.counters == expected[party][1]
        replay.assert_consumed()
