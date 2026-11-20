"""Prepared CUDA graphs for checked Rep4 product preparation.

Prepare each public shape before starting role threads. Graphs stop before
communication; the inherited receiver checks finish before result assembly.
"""
import math

import torch
import triton

from .device_inference import DeviceInferenceParty
from ..protocol.device_runtime import mask_domain
from ..protocol.fused_runtime import _masked, _assemble
from ..gpu.crypto.gpu_hash import digest
from ..gpu.crypto.gpu_prg_graph import generate_rows_into
from ..protocol.runtime import SCHEDULE, Share


class _ProductPlan:
    def __init__(self, rt, shape, binary, pointwise):
        self.shape, self.binary, self.pointwise = shape, binary, pointwise
        count = math.prod(shape)
        width = rt._wire_bits(binary)
        dtype = {8: torch.uint8, 16: torch.uint16, 32: torch.uint32, 64: torch.int64}[width]
        self.local = tuple(torch.zeros(shape, dtype=rt._storage_dtype(binary), device=rt.device) for _ in range(3))
        self.other = tuple(torch.zeros_like(t) for t in self.local) if pointwise else self.local
        self.host_nonces = torch.zeros(6, dtype=torch.uint64, pin_memory=True)
        self.upload_done = None
        self.nonces = torch.zeros(6, dtype=torch.uint64, device=rt.device)
        self.masks = torch.empty((6, count), dtype=torch.int64, device=rt.device)
        self.masked = torch.empty((3, count), dtype=dtype, device=rt.device)
        self.local_joints = tuple(k for k, (s, b, _, _) in enumerate(SCHEDULE) if rt.party in (s, b))
        columns = tuple(0 if k == (rt.party-1)%4 else 1 if k == rt.party else 2 for k in self.local_joints)
        keys = tuple(None if rt.party == r else rt._device_keys[r].key(
            mask_domain(rt.bits, rt.binary_bits, binary, r)) for _, _, r, _ in SCHEDULE)
        self.wires = {k: self.masked[j].view(torch.uint8) for j, k in enumerate(self.local_joints)}
        self.entries = [(s, b, r, self.wires.get(k)) for k, (s, b, r, _) in enumerate(SCHEDULE)]

        def run():
            generate_rows_into(keys, self.nonces, self.masks, width)
            _masked[(triton.cdiv(count, 256),)](*self.local, *self.other, self.masks, self.masked,
                                              count, binary, width, pointwise, self.local_joints, columns, 256)
            payloads, hashes = {}, {}
            for sender, backup, receiver, data in self.entries:
                if rt.party == sender:
                    payloads.setdefault(receiver, []).append(data)
                if rt.party == backup:
                    hashes.setdefault(receiver, []).append((sender, digest(data, block_tree=rt.hash_block_tree)))
            outgoing = {}
            for receiver in sorted(set(payloads) | set(hashes)):
                pieces = payloads.get(receiver, []) + [tag for _, tag in sorted(hashes.get(receiver, []))]
                outgoing[receiver] = pieces[0] if len(pieces) == 1 else torch.cat(pieces)
            return outgoing

        # Dummy inputs warm kernels without reserving draws or sending messages.
        stream = torch.cuda.Stream(device=rt.device)
        stream.wait_stream(torch.cuda.current_stream(rt.device))
        with torch.cuda.stream(stream):
            run()
        stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.outgoing = run()
        self.ready = torch.cuda.Event()
        self.ready.record(stream)

    def replay(self, rt, local, pointwise):
        stream = torch.cuda.current_stream(rt.device)
        stream.wait_event(self.ready)
        # Keep pinned source storage alive even if the previous exchange failed
        # frame validation before reaching the synchronizing hash check.
        if self.upload_done is not None:
            self.upload_done.synchronize()
        excluded = tuple(r for _, _, r, _ in SCHEDULE if rt.party != r)
        values = iter(rt._reserve_nonces(excluded))
        nonces = [0 if rt.party == r else next(values) for _, _, r, _ in SCHEDULE]
        self.host_nonces.copy_(torch.tensor(nonces, dtype=torch.uint64))
        self.nonces.copy_(self.host_nonces, non_blocking=True)
        if self.upload_done is None:
            self.upload_done = torch.cuda.Event()
        self.upload_done.record(stream)
        for dst, src in zip(self.local, local):
            dst.copy_(src)
        if pointwise is not None:
            for dst, src in zip(self.other, pointwise):
                dst.copy_(src)
        self.graph.replay()


class GraphDeviceParty(DeviceInferenceParty):
    """Use prepared product graphs while keeping fresh masks and eager checks.

    ``prepare_product`` must run outside concurrent CUDA/protocol activity.
    Unprepared public shapes execute the inherited eager implementation.
    Transports must finish consuming outgoing buffers before exchange returns,
    as DeviceIdealEndpoint and DeviceByteBridge do.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._product_graphs = {}

    def prepare_product(self, shape, binary=False, pointwise=False):
        shape = tuple(shape)
        if math.prod(shape) <= 0:
            raise ValueError('product graphs require a nonempty shape')
        key = (shape, bool(binary), bool(pointwise), self.bits, self.binary_bits, self.packed_arithmetic)
        if key not in self._product_graphs:
            self._product_graphs[key] = _ProductPlan(self, *key[:3])

    def _fused_product(self, local, binary, device, pointwise=None):
        shape = tuple(local[0].shape)
        key = (shape, bool(binary), pointwise is not None, self.bits, self.binary_bits, self.packed_arithmetic)
        if key not in self._product_graphs:
            return super()._fused_product(local, binary, device, pointwise)
        plan = self._product_graphs[key]
        plan.replay(self, local, pointwise)
        received = self._exchange_joint_wires(plan.entries, shape, binary,
                                              prepared_outgoing=plan.outgoing)
        # Checks above synchronize and reject all received data before use.
        receive_joints = tuple(k for k, (_, _, r, _) in enumerate(SCHEDULE) if self.party == r)
        count, width = math.prod(shape), self._wire_bits(binary)
        dtype = {8: torch.uint8, 16: torch.uint16, 32: torch.uint32, 64: torch.int64}[width]
        incoming = torch.cat([received[k] for k in receive_joints]).view(dtype)
        result = torch.empty((3, *shape), dtype=self._storage_dtype(binary), device=device)
        mask_slots = tuple(-1 if self.party == r else (r-self.party-1)%4 for _, _, r, _ in SCHEDULE)
        outside_slots = tuple(-1 if self.party == o else (o-self.party-1)%4 for _, _, _, o in SCHEDULE)
        local_ids = tuple(plan.local_joints.index(k) if k in plan.local_joints else -1 for k in range(6))
        receive_ids = tuple(receive_joints.index(k) if k in receive_joints else -1 for k in range(6))
        _assemble[(triton.cdiv(count, 256),)](plan.masks, plan.masked, incoming, result, count, binary,
                                            width, mask_slots, outside_slots, local_ids, receive_ids, 256)
        plan.ready.record(torch.cuda.current_stream(self.device))
        self.joint_inputs += 6
        self.multiplications += 1
        return Share(tuple(result.unbind(0)), binary)
