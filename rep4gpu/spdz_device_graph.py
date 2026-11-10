"""Optional prepared graphs for authenticated device broadcast local phases.

Prepare public payload layouts outside concurrent CUDA work. Live calls upload
fresh device pointers; framing, salts, coins, exchanges and host check decisions
remain in the underlying protocol. No graph owns preprocessing correlations.
"""
import gc

import torch
import triton
import triton.language as tl

from .gpu_hash import digest_batch
from .spdz_device import DeviceDistributedParty, _frame_inputs, _broadcast_checks
from .spdz_compression import DeviceCompressedParty


@triton.jit
def _hash_inputs(P, OUT, L: tl.constexpr, BLOCK: tl.constexpr):
    p0 = tl.load(P+0).to(tl.pointer_type(tl.uint8))
    p1 = tl.load(P+1).to(tl.pointer_type(tl.uint8))
    p2 = tl.load(P+2).to(tl.pointer_type(tl.uint8))
    p3 = tl.load(P+3).to(tl.pointer_type(tl.uint8))
    _frame_inputs(p0,p1,p2,p3,OUT,L,BLOCK)


@triton.jit
def _check_inputs(P, FRAMES, VIEW, FLAG, L: tl.constexpr, OFFSET: tl.constexpr,
                  WORDS: tl.constexpr, BLOCK: tl.constexpr):
    vh = tl.load(P+0).to(tl.pointer_type(tl.uint8))
    p0 = tl.load(P+1).to(tl.pointer_type(tl.uint8))
    p1 = tl.load(P+2).to(tl.pointer_type(tl.uint8))
    p2 = tl.load(P+3).to(tl.pointer_type(tl.uint8))
    p3 = tl.load(P+4).to(tl.pointer_type(tl.uint8))
    eh = tl.load(P+5).to(tl.pointer_type(tl.uint8))
    _broadcast_checks(FRAMES,vh,p0,p1,p2,p3,eh,VIEW,FLAG,L,OFFSET,WORDS,BLOCK)


class _PointerTable:
    def __init__(self, tensors, device):
        self.host = torch.empty(len(tensors),dtype=torch.uint64,pin_memory=True)
        self.device = torch.empty(len(tensors),dtype=torch.uint64,device=device)
        self.upload_done = None
        self.update(tensors)

    def update(self, tensors):
        # A failed exchange can skip the later synchronizing flag read.
        if self.upload_done is not None:
            self.upload_done.synchronize()
        self.host.numpy()[:] = [value.data_ptr() for value in tensors]
        stream = torch.cuda.current_stream(self.device.device)
        for value in tensors:
            value.record_stream(stream)
        self.device.copy_(self.host,non_blocking=True)
        if self.upload_done is None:
            self.upload_done = torch.cuda.Event()
        self.upload_done.record(stream)


def _capture(run, device):
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        run()
    stream.synchronize()
    # Automatic collection can destroy an older CUDA graph and invoke a
    # forbidden driver operation during capture. The PyTorch context already
    # collects before capture; keep automatic collection disabled until it ends.
    collect_enabled = gc.isenabled()
    gc.disable()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph,stream=stream):
            output = run()
    finally:
        if collect_enabled:
            gc.enable()
    ready = torch.cuda.Event()
    ready.record(stream)
    return graph,output,ready


class _HashPlan:
    def __init__(self, size, device):
        self.size = size
        self.dummy = tuple(torch.zeros(93+size,dtype=torch.uint8,device=device) for _ in range(4))
        self.pointers = _PointerTable(self.dummy,device)
        self.hashed = torch.empty((4,97+size),dtype=torch.uint8,device=device)
        def run():
            _hash_inputs[(triton.cdiv(97+size,256),4)](self.pointers.device,self.hashed,93+size,256)
            return digest_batch(self.hashed,block_tree=True).reshape(-1)
        self.graph,self.view,self.ready = _capture(run,device)

    def replay(self, frames):
        stream = torch.cuda.current_stream(self.hashed.device)
        stream.wait_event(self.ready)
        self.pointers.update(tuple(frames[p] for p in range(4)))
        self.graph.replay()
        self.ready.record(stream)
        return self.hashed,self.view


def _new_hash_plan(size,device):
    if size <= 927:
        from .spdz_hash_frames import SmallFrameHashPlan
        return SmallFrameHashPlan(size,device)
    return _HashPlan(size,device)


class _CheckPlan:
    def __init__(self, hash_plan, offset, words):
        self.hash_plan = hash_plan
        self.offset,self.words = offset,words
        device = hash_plan.hashed.device
        self.dummy_headers = tuple(torch.zeros((4,93),dtype=torch.uint8,device=device) for _ in range(2))
        self.dummy_echoes = tuple(torch.zeros(221,dtype=torch.uint8,device=device) for _ in range(4))
        self.pointers = _PointerTable((self.dummy_headers[0],*self.dummy_echoes,self.dummy_headers[1]),device)
        self.flag = torch.empty(1,dtype=torch.int32,device=device)
        def run():
            self.flag.zero_()
            _check_inputs[(triton.cdiv(max(884,4*words),256),)](
                self.pointers.device,hash_plan.hashed,hash_plan.view,self.flag,
                93+hash_plan.size,offset,words,256)
        self.graph,_,self.ready = _capture(run,device)

    def replay(self, vheaders, echoes, eheaders):
        stream = torch.cuda.current_stream(self.flag.device)
        stream.wait_event(self.ready)
        stream.wait_event(self.hash_plan.ready)
        self.pointers.update((vheaders,*(echoes[p] for p in range(4)),eheaders))
        self.graph.replay()
        self.ready.record(stream)
        # The base protocol reads this flag before returning any payload.
        return self.flag


class BroadcastGraphMixin:
    """Graph local hash/check work for explicitly prepared broadcast layouts.

    Call ``prepare_observed_broadcasts`` after an eager warmup has collected the
    public layouts, with no protocol threads or other CUDA work running. Each
    live phase uploads its current input addresses, including fresh headers.
    Unprepared layouts retain the underlying eager implementation.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._broadcast_hash_graphs = {}
        self._broadcast_check_graphs = {}
        self._broadcast_last_views = {}
        self.observed_broadcast_layouts = set()

    def prepare_broadcast(self,size,canonical_byte_offset=None):
        if not isinstance(size,int) or size < 0:
            raise ValueError('broadcast payload size must be nonnegative')
        offset = 0 if canonical_byte_offset is None else canonical_byte_offset
        if canonical_byte_offset is not None and (offset < 0 or offset > size or (size-offset)%4):
            raise ValueError('invalid canonical field offset')
        words = 0 if canonical_byte_offset is None else (size-offset)//4
        if size not in self._broadcast_hash_graphs:
            self._broadcast_hash_graphs[size] = _new_hash_plan(size,self.device)
        key = (size,offset,words)
        if key not in self._broadcast_check_graphs:
            self._broadcast_check_graphs[key] = _CheckPlan(self._broadcast_hash_graphs[size],offset,words)

    def prepare_observed_broadcasts(self):
        for size,offset,words in sorted(self.observed_broadcast_layouts):
            if size not in self._broadcast_hash_graphs:
                self._broadcast_hash_graphs[size] = _new_hash_plan(size,self.device)
            key = (size,offset,words)
            if key not in self._broadcast_check_graphs:
                self._broadcast_check_graphs[key] = _CheckPlan(self._broadcast_hash_graphs[size],offset,words)

    def _broadcast_hash_frames(self,frames,size,*,digest_output=None):
        plan = self._broadcast_hash_graphs.get(size)
        if plan is None:
            if digest_output is None:
                return super()._broadcast_hash_frames(frames,size)
            return super()._broadcast_hash_frames(frames,size,digest_output=digest_output)
        hashed,view = plan.replay(frames)
        if digest_output is not None:
            digest_output.copy_(view)
            view = digest_output
        self._broadcast_last_views[size] = view
        return hashed,view

    def _broadcast_check_frames(self,hashed,vheaders,echoes,eheaders,view,size,offset,words):
        key = (size,offset,words)
        self.observed_broadcast_layouts.add(key)
        plan = self._broadcast_check_graphs.get(key)
        if plan is None:
            return super()._broadcast_check_frames(hashed,vheaders,echoes,eheaders,view,size,offset,words)
        if hashed is not plan.hash_plan.hashed or view is not self._broadcast_last_views.get(size):
            raise RuntimeError('broadcast graph hash/check plans do not match')
        return plan.replay(vheaders,echoes,eheaders)


class GraphDeviceDistributedParty(BroadcastGraphMixin,DeviceDistributedParty):
    """Full-vector device MAC checks with optional prepared broadcast graphs."""


class GraphDeviceCompressedParty(BroadcastGraphMixin,DeviceCompressedParty):
    """Compressed device MAC checks with optional prepared broadcast graphs."""
