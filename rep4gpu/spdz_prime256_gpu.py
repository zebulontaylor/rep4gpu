"""Exact GPU products inside the checked CPU q256 prepared protocol.

Only the post-check D@W value/MAC products run on the GPU. Inputs, salted MAC
checks, square operations, framing, and network transport retain the CPU bigint
implementation. Packing, transfers, and synchronization are part of each call.
"""
from dataclasses import dataclass
import time
import weakref

import torch

from .spdz_distributed import ProtocolError
from .spdz_prime256 import PRIME,QShare,Prime256Party


def pack_limbs(values,shape):
    """Encode canonical Python integers into owned CPU radix65536 tensors."""
    if len(values)!=shape[0]*shape[1] or any(type(v)is not int or not 0<=v<PRIME for v in values):
        raise ValueError('canonical q256 matrix values required')
    limbs=[(value>>(16*i))&65535 for value in values for i in range(16)]
    return torch.tensor(limbs,dtype=torch.int32).reshape(*shape,16)


def unpack_limbs(source):
    """Copy GPU output to CPU and require the exact canonical limb encoding."""
    if source.dtype!=torch.int32 or source.ndim!=3 or source.shape[-1]!=16:
        raise ProtocolError('invalid q256 GPU product representation')
    rows=source.cpu().reshape(-1,16).tolist()
    if any(not 0<=limb<65536 for row in rows for limb in row):raise ProtocolError('invalid q256 GPU limb')
    values=[sum(limb<<(16*i) for i,limb in enumerate(row)) for row in rows]
    if any(value>=PRIME for value in values):raise ProtocolError('noncanonical q256 GPU product')
    return values


@dataclass
class _Weight:
    source: object
    prepared: object
    digits_version: int
    sums_version: int


class GPUPrime256Party(Prime256Party):
    """A checked CPU/TLS party with explicit cached GPU model products.

    ``prepare_handles`` charges weight packing, host-to-device transfer, and
    snapshot preparation to setup. Every query charges fresh D packing and
    transfers in both directions. Four local parties sharing one CUDA device
    remain four contending contexts, not a physical four-GPU measurement.
    """
    def __init__(self,*args,device='cuda',**kwargs):
        super().__init__(*args,**kwargs)
        with self.operation():
            if self.prime!=PRIME:raise ValueError('GPU q256 products require q=2**256-189')
            self.gpu_device=torch.device(device)
            if self.gpu_device.type!='cuda' or not torch.cuda.is_available():raise ValueError('CUDA device required')
            if self.gpu_device.index is None:self.gpu_device=torch.device('cuda',torch.cuda.current_device())
            self._gpu_weights={};self._gpu_warmed_shapes=set()
            self.gpu_setup_ms=0.0;self.gpu_query_product_ms=0.0;self.gpu_product_calls=0
            self.gpu_query_device_interval_ms=0.0
            self.gpu_product_samples=[]
            with torch.cuda.device(self.gpu_device):
                self._gpu_start=torch.cuda.Event(enable_timing=True);self._gpu_stop=torch.cuda.Event(enable_timing=True)

    def _prepare_weight(self,weights):
        from .field_q256_bytes import prepare_q256_bytes
        key=id(weights);cached=self._gpu_weights.get(key)
        if cached is not None:
            if cached.source() is not weights:raise ProtocolError('GPU prepared model identity mismatch')
            if (cached.prepared.digits._version!=cached.digits_version or cached.prepared.sums._version!=cached.sums_version):
                raise ProtocolError('GPU prepared model snapshot was mutated')
            return cached.prepared
        started=time.perf_counter();k,n=weights.shape
        # Values and their MAC coordinate share D, so one product computes both.
        combined=[]
        for row in range(k):
            combined.extend(weights.values[row*n:(row+1)*n]);combined.extend(weights.macs[row*n:(row+1)*n])
        with torch.cuda.device(self.gpu_device):
            raw=pack_limbs(combined,(k,2*n)).to(self.gpu_device)
            prepared=prepare_q256_bytes(raw)
            torch.cuda.current_stream(self.gpu_device).synchronize()
        self._gpu_weights[key]=_Weight(weakref.ref(weights),prepared,prepared.digits._version,prepared.sums._version)
        self.gpu_setup_ms+=(time.perf_counter()-started)*1000
        return prepared

    def prepare_handles(self,handles):
        """Prepack unused owned capabilities and warm kernels with public zeros.

        Warmup never opens or consumes A/F. Its compilation and execution cost
        is included in GPU setup, before fresh query timing begins.
        """
        from .spdz_prime256_prepared import PreparedQRows,_states
        with self.operation():
            for handle in handles:
                if type(handle)is not PreparedQRows or handle not in _states:raise ValueError('unknown prepared q256 rows')
                state=_states[handle]
                if (state.used or state.owner()is not self or state.session!=self.session
                        or state.alpha!=self.alpha_share or state.role!=self.party):
                    raise ProtocolError('GPU setup requires an unused party-bound prepared handle')
                prepared=self._prepare_weight(state.weights)
                shape=(id(state.weights),state.a.shape[0])
                if shape not in self._gpu_warmed_shapes:
                    from .field_q256_bytes import field_matmul_q256_bytes
                    started=time.perf_counter()
                    with torch.cuda.device(self.gpu_device):
                        public_zero=torch.zeros((*state.a.shape,16),dtype=torch.int32,device=self.gpu_device)
                        field_matmul_q256_bytes(public_zero,prepared)
                        torch.cuda.current_stream(self.gpu_device).synchronize()
                    self.gpu_setup_ms+=(time.perf_counter()-started)*1000
                    self._gpu_warmed_shapes.add(shape)

    def _prepared_right_product(self,d,weights,factor,shape):
        from .field_q256_bytes import field_matmul_q256_bytes
        # The caller already burned this query capability and authenticated D.
        # Requiring explicit setup keeps lazy model packing out of query totals.
        cached=self._gpu_weights.get(id(weights))
        if cached is None:raise ProtocolError('call prepare_handles before the first GPU query')
        prepared=self._prepare_weight(weights)
        started=time.perf_counter();m,k=shape;n=weights.shape[1]
        with torch.cuda.device(self.gpu_device):
            packed=pack_limbs(d,shape)
            self._gpu_start.record()
            inputs=packed.to(self.gpu_device)
            output=field_matmul_q256_bytes(inputs,prepared)
            self._gpu_stop.record()
            decoded=unpack_limbs(output)
            self._gpu_stop.synchronize()
            interval=self._gpu_start.elapsed_time(self._gpu_stop)
        product=[];mac_product=[]
        for row in range(m):
            product.extend(decoded[row*2*n:row*2*n+n]);mac_product.extend(decoded[row*2*n+n:(row+1)*2*n])
        result=QShare([(a+b)%PRIME for a,b in zip(factor.values,product)],
                      [(a+b)%PRIME for a,b in zip(factor.macs,mac_product)],(m,n))
        wall=(time.perf_counter()-started)*1000
        self.gpu_query_product_ms+=wall;self.gpu_query_device_interval_ms+=interval;self.gpu_product_calls+=1
        self.gpu_product_samples.append(dict(wall_ms=wall,device_stream_interval_ms=interval,shape=list(shape),columns=n))
        return result
