"""Explicit client channels for externally proved activation witnesses.

These byte channels connect an external witness client to four MPC roles.
They are separate from the four-role MPC transport. Every send counts the
immutable frame actually delivered, including its length prefix.
"""
from __future__ import annotations

import queue
import struct
import threading

_FRAME=struct.Struct('<4sQ')
_MAGIC=b'SZK1'
CLIENT_ROLE=4
MAX_VALUES=(1 << 35)-1
_ABORT=object()


class ProofClientAborted(RuntimeError):
    """A client or verifier failed before completing the client exchange."""



class ProofClientNetwork:
    """In-process byte mailboxes with exact external-client payload accounting."""
    def __init__(self,timeout=120):
        self.timeout=timeout
        self.failure=None
        self._abort_lock=threading.Lock()
        self._queues={(source,target):queue.Queue()
                      for source in range(5) for target in range(5)
                      if (source == CLIENT_ROLE) != (target == CLIENT_ROLE)}
        self._endpoints={role:ProofClientEndpoint(self,role) for role in range(5)}

    def abort(self,error):
        """Wake every pending client-channel receive without counting payload."""
        with self._abort_lock:
            if self.failure is None:
                self.failure=error
                for mailbox in self._queues.values(): mailbox.put(_ABORT)

    def endpoint(self,role):
        if role not in self._endpoints:
            raise ValueError('client channel role must be in [0,4]')
        return self._endpoints[role]

    @property
    def bytes_sent(self):
        return sum(endpoint.bytes_sent for endpoint in self._endpoints.values())


class ProofClientEndpoint:
    """Send and receive framed immutable bytes on one client-party link."""
    def __init__(self,network,role):
        self.network,self.role=network,role
        self.bytes_sent=0
        self.messages_sent=0
        self._lock=threading.Lock()

    def send(self,target,payload):
        if self.network.failure is not None:
            raise ProofClientAborted('proof client exchange aborted') from self.network.failure
        if not isinstance(payload,bytes):
            raise TypeError('client messages must be immutable bytes')
        key=(self.role,target)
        if key not in self.network._queues:
            raise ValueError('client channels connect a client and one MPC role')
        frame=_FRAME.pack(_MAGIC,len(payload))+payload
        with self._lock:
            self.bytes_sent+=len(frame)
            self.messages_sent+=1
        self.network._queues[key].put(frame)

    def receive(self,source):
        key=(source,self.role)
        if key not in self.network._queues:
            raise ValueError('client channels connect a client and one MPC role')
        if self.network.failure is not None:
            raise ProofClientAborted('proof client exchange aborted') from self.network.failure
        frame=self.network._queues[key].get(timeout=self.network.timeout)
        if frame is _ABORT:
            raise ProofClientAborted('proof client exchange aborted') from self.network.failure
        if len(frame)<_FRAME.size:
            raise ValueError('truncated client message frame')
        magic,length=_FRAME.unpack_from(frame)
        if magic != _MAGIC or length != len(frame)-_FRAME.size:
            raise ValueError('invalid client message frame')
        return frame[_FRAME.size:]


def encode_public_bundle(public):
    """Encode commitments and proofs once as raw bytes rather than hex text."""
    version=public['version'].encode('utf8')
    count=public['count']
    if not 0<len(version)<256 or not 1<=count<=MAX_VALUES:
        raise ValueError('invalid public proof metadata')
    out=bytearray(struct.pack('<B',len(version))+version)
    out.extend(struct.pack('<QQQ',count,public['chunk_size'],len(public['chunks'])))
    for name in ('commitments_x','commitments_y'):
        if len(public[name])!=count: raise ValueError('commitment count mismatch')
        for item in public[name]:
            point=bytes.fromhex(item)
            if len(point)!=32: raise ValueError('commitment requires 32 bytes')
            out.extend(point)
    for chunk in public['chunks']:
        proof=bytes.fromhex(chunk['proof'])
        out.extend(struct.pack('<QQQ',chunk['start'],chunk['count'],len(proof)))
        out.extend(proof)
    return bytes(out)


def decode_public_bundle(payload):
    """Decode a bounded frame, rejecting truncation and trailing bytes."""
    position=0
    def take(count):
        nonlocal position
        if count<0 or count>len(payload)-position: raise ValueError('truncated public proof bundle')
        result=payload[position:position+count];position+=count
        return result
    version=take(struct.unpack('<B',take(1))[0]).decode('utf8')
    count,chunk_size,chunks=struct.unpack('<QQQ',take(24))
    if not 1<=count<=MAX_VALUES or chunk_size<=0 or count>(len(payload)-position)//64:
        raise ValueError('invalid public proof count')
    result={'version':version,'count':count,'chunk_size':chunk_size}
    for name in ('commitments_x','commitments_y'):
        result[name]=[take(32).hex() for _ in range(count)]
    result['chunks']=[]
    for _ in range(chunks):
        start,size,length=struct.unpack('<QQQ',take(24))
        result['chunks'].append({'start':start,'count':size,'proof':take(length).hex()})
    if position!=len(payload): raise ValueError('trailing public proof bytes')
    return result


def projection_weights(seed,count,public_digest):
    """Generate contiguous row-major LSB-first Bernoulli challenge bits."""
    import hashlib
    if len(seed)!=32 or len(public_digest)!=32 or not 1<=count<=MAX_VALUES:
        raise ValueError('invalid projection challenge metadata')
    context=b'Rep4/silu-proof/projection/v1'+public_digest+seed+struct.pack('<Q',count)
    return hashlib.shake_256(context).digest(16*count)


def _serve_proof_client(endpoint,public,private,*,owner=0):
    """Deliver public proofs, owner output, and fresh masked blinding shares.

    This client function receives plaintext prover artifacts only. The four
    incoming messages contain the agreed public seed, never verifier shares.
    """
    import hashlib
    import secrets
    from .ristretto import Q, scalar_decode, scalar_encode
    if endpoint.role!=CLIENT_ROLE or owner not in range(4):
        raise ValueError('invalid client endpoint or output owner')
    bundle=encode_public_bundle(public)
    count=public['count']
    if len(private['output_y'])!=count: raise ValueError('output count mismatch')
    for role in range(4): endpoint.send(role,bundle)
    endpoint.send(owner,struct.pack('<'+'q'*count,*private['output_y']))
    seeds=[endpoint.receive(role) for role in range(4)]
    if any(seed!=seeds[0] or len(seed)!=32 for seed in seeds):
        raise ValueError('MPC roles disagreed on public challenge')
    packed=projection_weights(seeds[0],count,hashlib.sha256(bundle).digest())
    import numpy as np
    weights=np.unpackbits(np.frombuffer(packed,dtype=np.uint8),bitorder='little').reshape(128,count)
    projected=[]
    for name in ('blindings_x','blindings_y'):
        if len(private[name])!=count: raise ValueError('blinding count mismatch')
        blindings=[scalar_decode(bytes.fromhex(value)) for value in private[name]]
        for row in weights:
            projected.append(sum(value for bit,value in zip(row,blindings) if bit)%Q)
    components=[[secrets.randbelow(Q) for _ in projected] for _ in range(3)]
    components.append([(value-sum(parts))%Q for value,parts in zip(projected,zip(*components))])
    for role in range(4):
        packet=b''.join(scalar_encode(value) for slot in range(3)
                        for value in components[(role+slot+1)%4])
        endpoint.send(role,packet)


def _verify_proof_activation(runtime,x,endpoint,*,binary,workdir,owner=0,metrics=None):
    """Release shared output after proof verification and private input binding.

    The native proof process sees public commitments and proofs only.
    The count is below 2**35: each proved shifted value is at most 2**29,
    so every selected candidate sum is strictly below 2**64. Actual
    ring projections and masked blinding shares stay in this MPC role and its
    checked field/group runtime. The sole group opening is its masked aggregate.
    """
    import hashlib
    import json
    from pathlib import Path
    import subprocess
    import time
    import numpy as np
    import torch
    from .runtime import Share
    from .product_witness import ProductWitnessBatch
    from .field_group_runtime import FieldGroupRuntime, FieldShare, check_pedersen_generators
    from .ristretto import scalar_decode
    if runtime.bits!=64 or runtime.binary_bits!=64:
        raise ValueError('proof activation requires ring64 and 64-bit binary words')
    if endpoint.role!=runtime.party or owner not in range(4):
        raise ValueError('client endpoint does not match MPC role')
    bundle=endpoint.receive(CLIENT_ROLE)
    public=decode_public_bundle(bundle)
    count=public['count']
    if x.binary or x.parts[0].numel()!=count:
        raise ValueError('proof activation input shape mismatch')
    output_payload=endpoint.receive(CLIENT_ROLE) if runtime.party==owner else None
    if output_payload is not None and len(output_payload)!=8*count:
        raise ValueError('invalid client output length')
    output=(torch.frombuffer(bytearray(output_payload),dtype=torch.int64).clone().reshape(x.shape)
            if output_payload is not None else None)
    y=runtime.input(owner,output,x.shape)
    digest=hashlib.sha256(bundle).digest()
    outgoing={peer:digest for peer in range(4) if peer!=runtime.party}
    incoming=runtime.transport.exchange(outgoing)
    if set(incoming)!=set(outgoing) or any(value!=digest for value in incoming.values()):
        raise ValueError('MPC roles received inconsistent proof bundles')
    directory=Path(workdir);directory.mkdir(parents=True,exist_ok=True)
    public_path=directory/'public.json'
    public_path.write_text(json.dumps(public))
    command=str(Path(binary).resolve())
    measured={} if metrics is None else metrics
    started=time.perf_counter()
    subprocess.run([command,'verify',str(public_path)],check=True,capture_output=True)
    measured['native_proof_verification_seconds']=time.perf_counter()-started
    metadata=json.loads(subprocess.run([command,'metadata'],check=True,capture_output=True,text=True).stdout)
    check_pedersen_generators(bytes.fromhex(metadata['G']),bytes.fromhex(metadata['H']))
    if metadata['version']!=public['version']:
        raise ValueError('proof generator metadata mismatch')
    seed=ProductWitnessBatch(runtime)._public_seed()
    endpoint.send(CLIENT_ROLE,seed)
    packed=projection_weights(seed,count,digest)
    weights=np.unpackbits(np.frombuffer(packed,dtype=np.uint8),bitorder='little').reshape(128,count).astype(np.uint64)
    # NumPy uint64 sums deliberately implement the original ring, including
    # wrapped projections of invalid input words. Only 256 scalars are lifted.
    projected=[]
    for slot in range(3):
        values=[]
        for source,shift in ((x,16*(1 << 24)),(y,1 << 24)):
            part=source.parts[slot].detach().cpu().reshape(-1).numpy().view(np.uint64)
            # Public constants occupy the component excluded from party zero.
            if (runtime.party+slot+1)%4==0:
                part=part+np.uint64(shift)
            values.append((weights@part).view(np.int64))
        projected.append(torch.from_numpy(np.concatenate(values).copy()).to(runtime.device))
    ring=Share(tuple(projected))
    weights_path=directory/'weights.bin';weights_path.write_bytes(packed)
    projection_path=directory/'projection.json'
    started=time.perf_counter()
    subprocess.run([command,'project',str(public_path),str(weights_path),str(projection_path)],
                   check=True,capture_output=True)
    measured['native_public_projection_seconds']=time.perf_counter()-started
    public_projection=json.loads(projection_path.read_text())
    commitments=tuple(bytes.fromhex(value) for name in ('commitments_x','commitments_y')
                      for value in public_projection[name])
    payload=endpoint.receive(CLIENT_ROLE)
    if len(payload)!=3*256*32: raise ValueError('invalid projected blinding share length')
    scalars=tuple(scalar_decode(payload[start:start+32]) for start in range(0,len(payload),32))
    blindings=FieldShare(tuple(scalars[slot*256:(slot+1)*256] for slot in range(3)))
    started=time.perf_counter()
    field=FieldGroupRuntime(runtime)
    field.check_field_share(blindings)
    actual=field.lift_ring64(ring)
    difference=field.group_linear(actual,blindings,bytes.fromhex(metadata['G']),
                                  bytes.fromhex(metadata['H']),commitments)
    if not field.check_group_zero(difference):
        raise ValueError('proof activation input or output binding failed')
    measured['private_binding_seconds']=time.perf_counter()-started
    return y


def serve_proof_client(endpoint,public,private,*,owner=0):
    """Serve prover artifacts, waking pending roles if the client fails."""
    try:
        return _serve_proof_client(endpoint,public,private,owner=owner)
    except BaseException as error:
        endpoint.network.abort(error)
        raise


def verify_proof_activation(runtime,x,endpoint,*,binary,workdir,owner=0,metrics=None):
    """Verify an activation and wake the external client on failure.

    As with the other runtime protocols, the caller aborts its MPC transport
    when a role fails. Client-channel aborts separately wake the prover thread.
    """
    try:
        return _verify_proof_activation(runtime,x,endpoint,binary=binary,workdir=workdir,owner=owner,metrics=metrics)
    except BaseException as error:
        endpoint.network.abort(error)
        raise
