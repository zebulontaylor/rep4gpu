"""GPU-resident authenticated Beaver online protocol over F_(2**31-1).

This protocol variant uses BLAKE3 payload commitments and consistency echoes.
Its 93-byte GPU frames have a distinct magic; they are not byte compatible with
DistributedParty's BLAKE2b frames. Input/setup byte broadcasts remain inherited.
Preprocessing must supply verified, single-use authenticated correlations.
"""
import json
import struct
import weakref
import torch
import triton
import triton.language as tl
from .spdz import PRIME,AuthenticationError,field_matmul
from .spdz_distributed import DistributedParty,AuthShare,MatrixTriple,ElementwiseTriple,ProtocolError,_hash


@triton.jit
def _reduce(x):
    x=x.to(tl.int64)
    x=(x&2147483647)+(x>>31)
    x=(x&2147483647)+(x>>31)
    x=tl.where(x<0,x+2147483647,x)
    return tl.where(x>=2147483647,x-2147483647,x)


@triton.jit
def _word(P,index,OFFSET:tl.constexpr,valid):
    address=OFFSET+4*index
    a=tl.load(P+address,valid,0).to(tl.uint32)
    b=tl.load(P+address+1,valid,0).to(tl.uint32)
    c=tl.load(P+address+2,valid,0).to(tl.uint32)
    d=tl.load(P+address+3,valid,0).to(tl.uint32)
    return (a|(b<<8)|(c<<16)|(d<<24)).to(tl.int64)


@triton.jit
def _canonical(V,M,FLAG,N:tl.constexpr,H:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    v=tl.load(V+i,i<N,0).to(tl.int64)
    m=tl.load(M+i,i<N*H,0).to(tl.int64)
    bad=((i<N)&((v<0)|(v>=2147483647)))|((i<N*H)&((m<0)|(m>=2147483647)))
    if tl.sum(bad.to(tl.int32),0)>0:
        tl.atomic_or(FLAG,1)


@triton.jit
def _arithmetic(XV,XM,YV,YM,OV,OM,N:tl.constexpr,H:tl.constexpr,SUB:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    a=tl.load(XV+i,i<N,0).to(tl.int64)
    b=tl.load(YV+i,i<N,0).to(tl.int64)
    am=tl.load(XM+i,i<N*H,0).to(tl.int64)
    bm=tl.load(YM+i,i<N*H,0).to(tl.int64)
    value=a-b if SUB else a+b
    mac=am-bm if SUB else am+bm
    tl.store(OV+i,_reduce(value).to(tl.int32),i<N)
    tl.store(OM+i,_reduce(mac).to(tl.int32),i<N*H)


@triton.jit
def _public(XV,XM,P,A,OV,OM,N:tl.constexpr,H:tl.constexpr,OWNER:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    value=tl.load(XV+i,i<N,0).to(tl.int64)
    pub=tl.load(P+i,i<N,0).to(tl.int64)
    if OWNER:
        value+=pub
    tl.store(OV+i,_reduce(value).to(tl.int32),i<N)
    mac=tl.load(XM+i,i<N*H,0).to(tl.int64)
    key=tl.load(A+i//N,i<N*H,0).to(tl.int64)
    p=tl.load(P+i%N,i<N*H,0).to(tl.int64)
    tl.store(OM+i,_reduce(mac+key*p).to(tl.int32),i<N*H)


@triton.jit
def _frame_inputs(F0,F1,F2,F3,OUT,L:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=tl.program_id(1)
    if row==0:p=F0
    elif row==1:p=F1
    elif row==2:p=F2
    else:p=F3
    data=tl.load(p+i-4,(i>=4)&(i<L+4),0)
    value=tl.where(i<4,(row>>(8*i))&255,data)
    tl.store(OUT+row*(L+4)+i,value.to(tl.uint8),i<L+4)


@triton.jit
def _broadcast_checks(FRAMES,VHEADERS,E0,E1,E2,E3,EHEADERS,VIEW,FLAG,
                       L:tl.constexpr,CANONICAL_OFFSET:tl.constexpr,FIELD_WORDS:tl.constexpr,
                       BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=i//93;column=i%93
    actual=tl.load(FRAMES+row*(L+4)+4+column,i<4*93,0)
    expected=tl.load(VHEADERS+i,i<4*93,0)
    bad=(i<4*93)&(actual!=expected)
    row=i//221;column=i%221
    p=tl.where(row==0,E0,tl.where(row==1,E1,tl.where(row==2,E2,E3)))
    actual_e=tl.load(p+column,i<4*221,0)
    header=tl.load(EHEADERS+row*93+column,(i<4*221)&(column<93),0)
    view=tl.load(VIEW+column-93,(i<4*221)&(column>=93),0)
    expected_e=tl.where(column<93,header,view)
    bad|=(i<4*221)&(actual_e!=expected_e)
    if FIELD_WORDS>0:
        row=i//FIELD_WORDS;word=i%FIELD_WORDS
        base=row*(L+4)+4+93+CANONICAL_OFFSET+word*4
        a=tl.load(FRAMES+base,i<4*FIELD_WORDS,0).to(tl.uint32)
        b=tl.load(FRAMES+base+1,i<4*FIELD_WORDS,0).to(tl.uint32)
        c=tl.load(FRAMES+base+2,i<4*FIELD_WORDS,0).to(tl.uint32)
        d=tl.load(FRAMES+base+3,i<4*FIELD_WORDS,0).to(tl.uint32)
        field=a|(b<<8)|(c<<16)|(d<<24)
        bad|=(i<4*FIELD_WORDS)&(field>=2147483647)
    if tl.sum(bad.to(tl.int32),0)>0:
        tl.atomic_or(FLAG,1)


@triton.jit
def _sum_values(P0,P1,P2,P3,OUT,N:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    value=_word(P0,i,0,i<N)+_word(P1,i,0,i<N)+_word(P2,i,0,i<N)+_word(P3,i,0,i<N)
    tl.store(OUT+i,_reduce(value),i<N)


@triton.jit
def _residual(MACS,OPENED,ALPHA,OUT,N:tl.constexpr,H:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    m=tl.load(MACS+i,i<N*H,0).to(tl.int64)
    x=tl.load(OPENED+i%N,i<N*H,0).to(tl.int64)
    a=tl.load(ALPHA+i//N,i<N*H,0).to(tl.int64)
    tl.store(OUT+i,_reduce(m-a*x).to(tl.int32),i<N*H)


@triton.jit
def _residual_pair(M0,M1,OPENED,ALPHA,OUT,N0:tl.constexpr,N1:tl.constexpr,H:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    n:tl.constexpr=N0+N1
    key=i//n;position=i%n
    a=tl.load(M0+key*N0+position,(i<n*H)&(position<N0),0).to(tl.int64)
    b=tl.load(M1+key*N1+position-N0,(i<n*H)&(position>=N0),0).to(tl.int64)
    opened=tl.load(OPENED+position,i<n*H,0).to(tl.int64)
    alpha=tl.load(ALPHA+key,i<n*H,0).to(tl.int64)
    tl.store(OUT+i,_reduce(a+b-alpha*opened).to(tl.int32),i<n*H)


@triton.jit
def _join(PREFIX,P0,P1,P2,P3,OUT,P:tl.constexpr,N:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=tl.program_id(1)
    if row==0:source=P0
    elif row==1:source=P1
    elif row==2:source=P2
    else:source=P3
    prefix=tl.load(PREFIX+row*P+i,i<P,0)
    body=tl.load(source+i-P,(i>=P)&(i<P+N),0)
    tl.store(OUT+row*(P+N)+i,tl.where(i<P,prefix,body),i<P+N)


@triton.jit
def _mac_zero(P0,P1,P2,P3,FLAG,N:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    total=_word(P0,i,32,i<N)+_word(P1,i,32,i<N)+_word(P2,i,32,i<N)+_word(P3,i,32,i<N)
    if tl.sum(((i<N)&(_reduce(total)!=0)).to(tl.int32),0)>0:
        tl.atomic_or(FLAG,1)


@triton.jit
def _square_local(AV,AM,CV,CM,D,A,OV,OM,N:tl.constexpr,H:tl.constexpr,OWNER:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    d=tl.load(D+i,i<N,0).to(tl.int64)
    a=tl.load(AV+i,i<N,0).to(tl.int64)
    c=tl.load(CV+i,i<N,0).to(tl.int64)
    value=_reduce(c+2*d*a)
    if OWNER:value=_reduce(value+_reduce(d*d))
    tl.store(OV+i,value.to(tl.int32),i<N)
    dm=tl.load(D+i%N,i<N*H,0).to(tl.int64)
    am=tl.load(AM+i,i<N*H,0).to(tl.int64)
    cm=tl.load(CM+i,i<N*H,0).to(tl.int64)
    alpha=tl.load(A+i//N,i<N*H,0).to(tl.int64)
    mac=_reduce(_reduce(cm+2*dm*am)+alpha*_reduce(dm*dm))
    tl.store(OM+i,mac.to(tl.int32),i<N*H)


@triton.jit
def _elementwise_local(AV,AM,BV,BM,CV,CM,D,E,A,OV,OM,N:tl.constexpr,H:tl.constexpr,OWNER:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    d=tl.load(D+i,i<N,0).to(tl.int64);e=tl.load(E+i,i<N,0).to(tl.int64)
    a=tl.load(AV+i,i<N,0).to(tl.int64);b=tl.load(BV+i,i<N,0).to(tl.int64)
    c=tl.load(CV+i,i<N,0).to(tl.int64)
    value=_reduce(c+d*b+a*e)
    if OWNER:value=_reduce(value+_reduce(d*e))
    tl.store(OV+i,value.to(tl.int32),i<N)
    dm=tl.load(D+i%N,i<N*H,0).to(tl.int64);em=tl.load(E+i%N,i<N*H,0).to(tl.int64)
    am=tl.load(AM+i,i<N*H,0).to(tl.int64);bm=tl.load(BM+i,i<N*H,0).to(tl.int64)
    cm=tl.load(CM+i,i<N*H,0).to(tl.int64);alpha=tl.load(A+i//N,i<N*H,0).to(tl.int64)
    mac=_reduce(_reduce(cm+dm*bm+am*em)+alpha*_reduce(dm*em))
    tl.store(OM+i,mac.to(tl.int32),i<N*H)


class DeviceDistributedParty(DistributedParty):
    """Four-role CUDA online protocol with checked GPU-resident messages.

    Canonical bounds are checked once for each immutable tensor version, not
    repeatedly by local arithmetic. Received field encodings and consistency
    echoes are checked before reconstruction; commitments are checked before
    MAC aggregation, and MACs before opened values can enter Beaver products.
    """
    def __init__(self,party,transport,alpha_share,*,parties=4,session,
                 _allow_insecure_test_transport=False,stack_macs=True):
        if alpha_share.device.type!='cuda' or parties!=4 or not hasattr(transport,'exchange_device'):
            raise ValueError('device authenticated protocol requires four CUDA roles and device transport')
        self._canonical_versions=[]
        super().__init__(party,transport,alpha_share.to(torch.int64),parties=parties,session=session,
                         _allow_insecure_test_transport=_allow_insecure_test_transport,stack_macs=stack_macs)
        self.crypto_variant='SPDZ31G1/BLAKE3'
        self.local_backend='torch-field'
        from .spdz_metadata import upload_bytes
        self._device_session_hash=upload_bytes(self._session_hash,self.device)

    def _validate(self,x):
        if not isinstance(x,AuthShare) or x.values.dtype not in (torch.int32,torch.int64) or x.values.ndim!=2:
            raise ValueError('expected a local authenticated int32/int64 matrix')
        if x.macs.dtype not in (torch.int32,torch.int64) or x.macs.shape!=(self.mac_keys,*x.values.shape):
            raise ValueError('authenticated matrix MAC dimensions mismatch')
        if x.values.device!=self.device or x.macs.device!=self.device:
            raise ValueError('authenticated matrices must use the party device')

    @staticmethod
    def _version(tensor):
        try:return tensor._version
        except RuntimeError:return None

    def _known(self,x):
        self._canonical_versions=[entry for entry in self._canonical_versions if entry[0]() is not None and entry[2]() is not None]
        vv_now,mv_now=self._version(x.values),self._version(x.macs)
        if vv_now is None or mv_now is None:return False
        return any(v() is x.values and m() is x.macs and vv==vv_now and mv==mv_now
                   for v,vv,m,mv in self._canonical_versions)

    def _mark(self,x):
        vv,mv=self._version(x.values),self._version(x.macs)
        if vv is not None and mv is not None:
            self._canonical_versions.append((weakref.ref(x.values),vv,weakref.ref(x.macs),mv))
        return x

    def _ensure_canonical(self,shares):
        unknown=[]
        for x in shares:
            self._validate(x)
            if not self._known(x):unknown.append(x)
        if not unknown:return
        flag=torch.zeros(1,dtype=torch.int32,device=self.device)
        for x in unknown:
            n=x.values.numel()
            if n:
                _canonical[(triton.cdiv(n*self.mac_keys,256),)](x.values.contiguous(),x.macs.contiguous(),flag,n,self.mac_keys,256)
        if flag.item():raise ValueError('shares must be canonical field elements')
        for x in unknown:self._mark(x)

    def validate_share(self,x):
        """Check preprocessing representation once; this does not verify its MACs."""
        with self.operation():
            self._ensure_canonical((x,))
            return x

    def _headers(self,stage,round_id,label,size):
        from .spdz_metadata import headers
        return headers(self._device_session_hash,_hash(label),round_id,size,stage)

    def _broadcast_frame(self,payload,stage,round_id,label):
        from .spdz_metadata import header_frame
        return header_frame(self._device_session_hash,_hash(label),round_id,stage,self.party,payload)

    def _broadcast_hash_frames(self,frames,size,*,digest_output=None):
        if size<=927:
            from .spdz_hash_frames import hash_frames
            return hash_frames(frames,size,digest_output=digest_output)
        from .gpu_hash import digest_batch
        hashed=torch.empty((4,97+size),dtype=torch.uint8,device=self.device)
        _frame_inputs[(triton.cdiv(97+size,256),4)](*(frames[p] for p in range(4)),hashed,93+size,256)
        view=digest_batch(hashed,block_tree=True).reshape(-1)
        if digest_output is not None:
            digest_output.copy_(view)
            view=digest_output
        return hashed,view

    def _broadcast_check_frames(self,hashed,vheaders,echoes,eheaders,view,size,offset,words):
        flag=torch.zeros(1,dtype=torch.int32,device=self.device)
        _broadcast_checks[(triton.cdiv(max(884,4*words),256),)](hashed,vheaders,*(echoes[p] for p in range(4)),
                                                             eheaders,view,flag,93+size,offset,words,256)
        return flag

    def broadcast_device(self,payload,*,label,canonical_byte_offset=None):
        from .gpu_hash import digest_batch
        with self.operation():
            if payload.dtype!=torch.uint8 or payload.ndim!=1 or payload.device!=self.device or not payload.is_contiguous():
                raise ValueError('device broadcast payload must be contiguous CUDA bytes')
            if not isinstance(label,bytes) or not 0<=self.round<1<<64:
                raise ProtocolError('invalid broadcast label or exhausted round counter')
            size=payload.numel();round_id=self.round;self.round+=1
            if size<=927:
                from .spdz_broadcast_metadata import prepare_broadcast
                vheaders,own,eheaders,echo=prepare_broadcast(self._device_session_hash,_hash(label),round_id,self.party,payload)
            else:
                vheaders,own=self._broadcast_frame(payload,b'V',round_id,label)
            peers=set(range(4))-{self.party}
            incoming=self.transport.exchange_device({p:own for p in peers},{p:93+size for p in peers})
            if set(incoming)!=peers:raise ProtocolError('device broadcast peer set mismatch')
            frames={self.party:own,**incoming}
            if set(frames)!=set(range(4)) or any(t.dtype!=torch.uint8 or t.device!=self.device or t.ndim!=1 or not t.is_contiguous() or t.numel()!=93+size for t in frames.values()):
                raise ProtocolError('device broadcast frame layout mismatch')
            if size<=927:
                hashed,view=self._broadcast_hash_frames(frames,size,digest_output=echo[93:])
            else:
                hashed,view=self._broadcast_hash_frames(frames,size)
                eheaders,echo=self._broadcast_frame(view,b'E',round_id,label)
            checks=self.transport.exchange_device({p:echo for p in peers},{p:221 for p in peers})
            if set(checks)!=peers:raise ProtocolError('device echo peer set mismatch')
            echoes={self.party:echo,**checks}
            if set(echoes)!=set(range(4)) or any(t.dtype!=torch.uint8 or t.device!=self.device or t.ndim!=1 or not t.is_contiguous() or t.numel()!=221 for t in echoes.values()):
                raise ProtocolError('device echo frame layout mismatch')
            offset=0 if canonical_byte_offset is None else canonical_byte_offset
            if canonical_byte_offset is not None and (offset<0 or offset>size or (size-offset)%4):
                raise ValueError('invalid canonical field offset')
            words=0 if canonical_byte_offset is None else (size-offset)//4
            flag=self._broadcast_check_frames(hashed,vheaders,echoes,eheaders,view,size,offset,words)
            if flag.item():raise ProtocolError('device broadcast header, canonical encoding, or echo check failed')
            return {p:frames[p][93:] for p in range(4)}

    # Alias retained for protocol helpers that spell the verb first.
    device_broadcast=broadcast_device

    def _packed_values(self,shares):
        """Borrow canonical int32 bytes; framing copies before publication."""
        parts=[x.values.reshape(-1) for x in shares]
        values=parts[0] if len(parts)==1 else torch.cat(parts)
        words=values.view(torch.uint32) if values.dtype==torch.int32 else values.to(torch.uint32)
        return words.view(torch.uint8)

    def _residual(self,shares,opened):
        """Return a fresh, owned residual snapshot, independent of all inputs."""
        count=opened.numel()
        result=torch.empty((self.mac_keys,count),dtype=torch.int32,device=self.device)
        if count:
            if len(shares)==1:
                macs=shares[0].macs.contiguous().reshape(self.mac_keys,-1)
                _residual[(triton.cdiv(result.numel(),256),)](macs,opened,self.alpha_share,result,count,self.mac_keys,256)
            elif len(shares)==2:
                n0,n1=shares[0].values.numel(),shares[1].values.numel()
                _residual_pair[(triton.cdiv(result.numel(),256),)](shares[0].macs.contiguous(),shares[1].macs.contiguous(),
                    opened,self.alpha_share,result,n0,n1,self.mac_keys,256)
            else:
                macs=torch.cat([x.macs.reshape(self.mac_keys,-1) for x in shares],dim=1).contiguous()
                _residual[(triton.cdiv(result.numel(),256),)](macs,opened,self.alpha_share,result,count,self.mac_keys,256)
        return result

    def open_many(self,shares,*,label=b''):
        from .gpu_hash import digest,digest_batch
        from .spdz_metadata import upload_bytes
        with self.operation():
            shares=tuple(shares)
            if not shares:raise ValueError('open_many requires at least one matrix')
            self._ensure_canonical(shares)
            layout=json.dumps({'field':PRIME,'dtype':'uint32-le','keys':self.mac_keys,
                               'shapes':[list(x.values.shape) for x in shares]},sort_keys=True,separators=(',',':')).encode()
            context=b'SPDZ31G-open-v1'+self._session_hash+struct.pack('<Q',self.round)+_hash(layout)+_hash(label)
            count=sum(x.values.numel() for x in shares)
            values=self.broadcast_device(self._packed_values(shares),label=context+b'values',canonical_byte_offset=0)
            opened=torch.empty(count,dtype=torch.int64,device=self.device)
            if count:
                _sum_values[(triton.cdiv(count,256),)](*(values[p] for p in range(4)),opened,count,256)
            residual=self._residual(shares,opened)
            raw=residual.view(torch.uint8).reshape(-1)
            salt=self._commit_salt()
            if not isinstance(salt,bytes) or len(salt)!=32:raise ValueError('commitment salts must contain 32 bytes')
            salt_gpu=upload_bytes(salt,self.device)
            binding=context+struct.pack('<I',self.party)
            binding_gpu=upload_bytes(binding,self.device)
            commitment=digest(torch.cat((binding_gpu,salt_gpu,raw)),block_tree=True)
            commits=self.broadcast_device(commitment,label=context+b'commit')
            reveal=torch.cat((salt_gpu,raw))
            reveals=self.broadcast_device(reveal,label=context+b'reveal',canonical_byte_offset=32)
            prefix=upload_bytes(b''.join(context+struct.pack('<I',p) for p in range(4)),self.device).reshape(4,len(binding))
            joined=torch.empty((4,len(binding)+reveal.numel()),dtype=torch.uint8,device=self.device)
            _join[(triton.cdiv(joined.shape[1],256),4)](prefix,*(reveals[p] for p in range(4)),joined,len(binding),reveal.numel(),256)
            tags=digest_batch(joined,block_tree=True)
            if not torch.equal(tags,torch.stack([commits[p] for p in range(4)])):
                raise ProtocolError('device residual commitment opening failed')
            flag=torch.zeros(1,dtype=torch.int32,device=self.device)
            if count:
                _mac_zero[(triton.cdiv(count*self.mac_keys,256),)](*(reveals[p] for p in range(4)),flag,count*self.mac_keys,256)
            if flag.item():raise AuthenticationError('distributed device field MAC check failed')
            result=[];offset=0
            for x in shares:
                n=x.values.numel();result.append(opened[offset:offset+n].reshape(x.values.shape));offset+=n
            return result

    def _arithmetic(self,x,y,subtract):
        with self.operation():
            self._ensure_canonical((x,y))
            if x.values.shape!=y.values.shape:raise ValueError('authenticated addition dimensions mismatch')
            values=torch.empty(x.values.shape,dtype=torch.int32,device=self.device)
            macs=torch.empty(x.macs.shape,dtype=torch.int32,device=self.device)
            n=values.numel()
            if n:
                _arithmetic[(triton.cdiv(n*self.mac_keys,256),)](x.values.contiguous(),x.macs.contiguous(),y.values.contiguous(),y.macs.contiguous(),values,macs,n,self.mac_keys,subtract,256)
            return self._mark(AuthShare(values,macs))

    def add(self,x,y):return self._arithmetic(x,y,False)
    def subtract(self,x,y):return self._arithmetic(x,y,True)

    def add_public(self,x,value):
        with self.operation():
            self._ensure_canonical((x,))
            public=torch.as_tensor(value,dtype=torch.int64,device=self.device).remainder(PRIME)
            public=torch.broadcast_to(public,x.values.shape).contiguous()
            values=torch.empty(x.values.shape,dtype=torch.int32,device=self.device);macs=torch.empty(x.macs.shape,dtype=torch.int32,device=self.device)
            n=values.numel()
            if n:
                _public[(triton.cdiv(n*self.mac_keys,256),)](x.values.contiguous(),x.macs.contiguous(),public,self.alpha_share,values,macs,n,self.mac_keys,self.party==0,256)
            return self._mark(AuthShare(values,macs))

    def _masked_open_many(self,shares,*,label=b''):
        """Internal masked openings; the default checks them immediately."""
        return self.open_many(shares,label=label)

    def matmul(self,x,y,triple):
        with self.operation():
            identifier=self._burn(triple)
            if type(triple) is not MatrixTriple:raise ValueError('matmul requires a matrix triple')
            self._ensure_canonical((x,y,triple.a,triple.b,triple.c))
            if x.values.shape!=triple.a.values.shape or y.values.shape!=triple.b.values.shape or x.values.shape[1]!=y.values.shape[0] or triple.c.values.shape!=(x.values.shape[0],y.values.shape[1]):
                raise ValueError('matrix triple dimensions mismatch')
            d,e=self._masked_open_many((self.subtract(x,triple.a),self.subtract(y,triple.b)),label=b'matmul'+identifier)
            return self._matmul_local(triple,d,e)

    def _matmul_local(self,triple,d,e):
        try:
            from .field_fusion import authenticated_local
        except ImportError:
            authenticated_local=None
        if authenticated_local is not None:
            self.local_backend='fused-field'
            if d.shape[0]<=4:
                from .field_decode_auth import authenticated_local_decode
                authenticated_local=authenticated_local_decode
                self.local_backend='fused-field-decode'
            values,macs=authenticated_local(d,e,triple.a.values,triple.a.macs,triple.b.values,triple.b.macs,
                                            triple.c.values,triple.c.macs,self.alpha_share,self.party,packed=True)
            return self._mark(AuthShare(values,macs))
        self.local_backend='torch-field'
        a,b,c=triple.a,triple.b,triple.c
        public=field_matmul(d.long(),e.long())
        m,n=public.shape
        right=field_matmul(torch.cat((a.values.long(),*a.macs.long().unbind(0)),dim=0),e.long()).reshape(self.mac_keys+1,m,n)
        left=field_matmul(d.long(),torch.cat((b.values.long(),*b.macs.long().unbind(0)),dim=1))
        values=(c.values.long()+left[:,:n]+right[0]).remainder(PRIME)
        macs=(c.macs.long()+left[:,n:].reshape(m,self.mac_keys,n).permute(1,0,2)+right[1:]).remainder(PRIME)
        result=AuthShare(values.int(),macs.int())
        self._mark(result)
        return self.add_public(result,public)


    def mul(self,x,y,triple):
        with self.operation():
            identifier=self._burn(triple)
            if type(triple) is not ElementwiseTriple:raise ValueError('mul requires an elementwise triple')
            self._ensure_canonical((x,y,triple.a,triple.b,triple.c))
            if any(s.values.shape!=x.values.shape for s in (y,triple.a,triple.b,triple.c)):
                raise ValueError('elementwise triple dimensions mismatch')
            d,e=self._masked_open_many((self.subtract(x,triple.a),self.subtract(y,triple.b)),label=b'elementwise'+identifier)
            values=torch.empty(x.values.shape,dtype=torch.int32,device=self.device);macs=torch.empty(x.macs.shape,dtype=torch.int32,device=self.device)
            n=values.numel()
            if n:
                _elementwise_local[(triton.cdiv(n*self.mac_keys,256),)](triple.a.values.contiguous(),triple.a.macs.contiguous(),
                    triple.b.values.contiguous(),triple.b.macs.contiguous(),triple.c.values.contiguous(),triple.c.macs.contiguous(),
                    d.contiguous(),e.contiguous(),self.alpha_share,values,macs,n,self.mac_keys,self.party==0,256)
            return self._mark(AuthShare(values,macs))

    def _square_device(self,x,mask):
        with self.operation():
            spent=getattr(self,'_spent_square_masks',None)
            if spent is None:spent=self._spent_square_masks=set()
            old=mask.used;mask.used=True
            if old or mask.identifier in spent:raise ValueError('square masks are single-use')
            spent.add(mask.identifier)
            if not isinstance(mask.identifier,bytes) or len(mask.identifier)!=32:
                raise ValueError('invalid square mask identifier')
            self._ensure_canonical((x,mask.a,mask.a_squared))
            if x.values.shape!=mask.a.values.shape or x.values.shape!=mask.a_squared.values.shape:
                raise ValueError('square mask shape does not match input')
            d=self._masked_open_many((self.subtract(x,mask.a),),label=b'square'+mask.identifier)[0]
            values=torch.empty(x.values.shape,dtype=torch.int32,device=self.device);macs=torch.empty(x.macs.shape,dtype=torch.int32,device=self.device)
            n=values.numel()
            if n:
                _square_local[(triton.cdiv(n*self.mac_keys,256),)](mask.a.values.contiguous(),mask.a.macs.contiguous(),
                    mask.a_squared.values.contiguous(),mask.a_squared.macs.contiguous(),d.contiguous(),self.alpha_share,
                    values,macs,n,self.mac_keys,self.party==0,256)
            return self._mark(AuthShare(values,macs))
