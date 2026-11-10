"""Batch exact MAC checks for capability-masked Goldilocks differences.

Only the prepared-row and square-row helpers authorize tentative openings.
Their candidates stay inside a fixed arithmetic circuit. Ordinary open/open_many
checks every pending residual before transmitting any unmasked output share.
There is no random projection: every residual coordinate is checked separately.
"""
from dataclasses import dataclass
import json
import secrets
import struct

from .spdz_distributed import ProtocolError,_hash
from .spdz_prime256 import Prime256Party,GOLDILOCKS_PRIME


@dataclass(frozen=True)
class _Pending:
    residual: tuple
    schema: bytes


class DeferredGoldilocksParty(Prime256Party):
    """Opt-in fixed-circuit batching with a mandatory checked-output boundary.

    Fresh opaque row capabilities are required for every tentative difference.
    The public queue cap is bound into protocol contexts; exceeding it checks
    the existing queue first. An oversized single difference is checked
    immediately. A failure retires this session and, when supplied, the complete
    native key epoch containing ``key_epoch_role_path``.
    """
    def __init__(self,*args,max_pending_values=2**20,key_epoch_role_path=None,**kwargs):
        self._pending=[];self._pending_values=0;self._authorized=set();self._flush_index=0
        self._key_epoch_role_path=key_epoch_role_path
        self._retirement_error=None
        kwargs.setdefault('field_prime',GOLDILOCKS_PRIME)
        super().__init__(*args,**kwargs)
        with self.operation():
            if self.prime!=GOLDILOCKS_PRIME:raise ValueError('deferred Goldilocks checks require the Goldilocks field')
            if type(max_pending_values)is not int or not 1<=max_pending_values<2**31:
                raise ValueError('max_pending_values must be in [1,2**31)')
            self.max_pending_values=max_pending_values

    @property
    def pending_values(self):return self._pending_values

    def abort(self,error):
        # No tentative state remains available after an authentication failure.
        self._pending.clear();self._pending_values=0
        if self._key_epoch_role_path is not None:
            try:
                from .lowgear_key_cache import retire_epoch
                retire_epoch(self._key_epoch_role_path,'failed deferred Goldilocks session')
            except BaseException as retirement_error:self._retirement_error=retirement_error
        super().abort(error)

    def _open_goldilocks_difference(self,share,handle,*,kind,label):
        from .spdz_goldilocks_stream import _validate_masked_capability
        with self.operation():
            state=_validate_masked_capability(self,handle,kind,label)
            authorization=(kind,state.identifier)
            if authorization in self._authorized:raise ProtocolError('masked capability already authorized an opening')
            self._authorized.add(authorization)
            self._validate(share)
            if share.shape!=state.a.shape:raise ValueError('masked capability dimensions mismatch')
            count=len(share.values)
            if count>self.max_pending_values:
                self.flush_pending()
                return super().open_many((share,),label=b'Goldilocks-immediate-oversize\0'+struct.pack('<Q',self.max_pending_values)+label)[0]
            if self._pending_values+count>self.max_pending_values:self.flush_pending()
            schema=json.dumps(dict(field=self.prime,kind=kind,shape=share.shape,
                mask_shape=state.a.shape,factor_shape=state.factor.shape,identifier=state.identifier.hex(),
                label=label.hex(),cap=self.max_pending_values),sort_keys=True,separators=(',',':')).encode()
            context=(b'Goldilocks-deferred-value-v1\0'+self._session_hash+struct.pack('<Q',self.round)+_hash(schema))
            # Snapshot both local coordinates before publishing a value share.
            values=tuple(share.values);macs=tuple(share.macs)
            received=self.broadcast(self._pack(values),label=context+b'/values')
            decoded=[self._unpack(received[p],count) for p in range(4)]
            opened=[sum(row[i] for row in decoded)%self.prime for i in range(count)]
            residual=tuple((mac-self.alpha_share*value)%self.prime for mac,value in zip(macs,opened))
            self._pending.append(_Pending(residual,context))
            self._pending_values+=count
            return opened

    def flush_pending(self):
        """Authenticate the frozen queue without broadcasting any output value."""
        with self.operation():
            if not self._pending:return
            if not 0<=self._flush_index<2**64:raise ProtocolError('deferred check counter exhausted')
            schema=json.dumps(dict(field=self.prime,cap=self.max_pending_values,
                entries=[dict(context=entry.schema.hex(),count=len(entry.residual)) for entry in self._pending]),
                sort_keys=True,separators=(',',':')).encode()
            context=(b'Goldilocks-deferred-MAC-v1\0'+self._session_hash+
                     struct.pack('<QQ',self.round,self._flush_index)+_hash(schema))
            residual=tuple(value for entry in self._pending for value in entry.residual)
            count=len(residual);salt=self._commit_salt()
            if not isinstance(salt,bytes) or len(salt)!=32:raise ProtocolError('invalid commitment salt')
            reveal=salt+self._pack(residual)
            commits=self.broadcast(_hash(context+struct.pack('<I',self.party)+reveal),label=context+b'/commit')
            if any(len(value)!=32 for value in commits.values()):raise ProtocolError('invalid deferred commitment length')
            reveals=self.broadcast(reveal,label=context+b'/reveal')
            checked=[]
            for p in range(4):
                expected=_hash(context+struct.pack('<I',p)+reveals[p])
                if len(reveals[p])!=32+self.field_bytes*count or not secrets.compare_digest(expected,commits[p]):
                    raise ProtocolError('deferred residual commitment mismatch')
                checked.append(self._unpack(reveals[p][32:],count))
            if any(sum(row[i] for row in checked)%self.prime for i in range(count)):
                raise ProtocolError('deferred Goldilocks MAC check failed')
            self._pending.clear();self._pending_values=0;self._flush_index+=1

    def _before_ordinary_open_many(self):
        # Base-method aliases must respect the same output barrier.
        self.flush_pending()

    def open_many(self,shares,*,label=b''):
        with self.operation():
            self.flush_pending()
            return super().open_many(shares,label=label)

    def input(self,mask,value=None):
        with self.operation():
            self.flush_pending()
            return super().input(mask,value)
