"""Framed role-local control and strict public replies for persistent native OLE."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import select
import struct
import time

from .direct_ole_io import LocalOleBatch

MAGIC=b'SIVSVC01'
SCHEMAS={
    'ready':{'status','sequence','ole_role','self_role','peer_role','tls_session','context_setup_seconds','context_sent_bytes','context_received_bytes','entropy_substreams','security_reviewed','transport_authenticated'},
    'complete':{'status','sequence','request_id','length','calls','request_seconds','fresh_rot_choices','total_rot_choices','sent_bytes','received_bytes','entropy_substreams'},
    'closed':{'status','sequence','total_calls','total_rot_choices','poisoned'},
    'aborted':{'status','sequence','poisoned'},
}


def frame(*,sequence,request_id=None,length=0,calls=0,input_path=None,output_path=None,close=False):
    if type(sequence)is not int or not 0<=sequence<(1<<64):raise ValueError('invalid service sequence')
    if close:
        if request_id is not None or length or calls or input_path is not None or output_path is not None:raise ValueError('close requires an empty canonical frame')
        return MAGIC+struct.pack('<QQ',2,sequence)+bytes(32)+struct.pack('<QQQQ',0,0,0,0)
    if not isinstance(request_id,bytes) or len(request_id)!=32 or not any(request_id):raise ValueError('nonzero 32-byte request ID required')
    if type(length)is not int or type(calls)is not int or not 1<=length<=8192 or not 1<=calls<=4096 or length*calls>(1<<24):raise ValueError('unsupported service dimensions')
    paths=[str(Path(path).resolve()).encode('utf8') for path in (input_path,output_path)]
    if any(not path or len(path)>4096 or b'\0' in path for path in paths) or paths[0]==paths[1]:raise ValueError('invalid service request paths')
    return MAGIC+struct.pack('<QQ',1,sequence)+request_id+struct.pack('<QQQQ',length,calls,len(paths[0]),len(paths[1]))+paths[0]+paths[1]


def send_frame(pipe,**arguments):
    payload=frame(**arguments)
    pipe.write(payload);pipe.flush()


class ReplyReader:
    """Read bounded newline-delimited public replies with a total deadline."""
    def __init__(self,pipe):self.pipe=pipe;self.buffer=bytearray()
    def read(self,timeout):
        deadline=time.monotonic()+timeout
        while b'\n' not in self.buffer:
            remaining=deadline-time.monotonic()
            if remaining<=0 or not select.select([self.pipe.fileno()],[],[],max(0,remaining))[0]:raise TimeoutError('native OLE service reply timeout')
            block=os.read(self.pipe.fileno(),65536)
            if not block:raise RuntimeError('native OLE service closed before its reply')
            self.buffer.extend(block)
            if len(self.buffer)>65536:raise ValueError('native OLE public reply exceeds limit')
        line,_,tail=self.buffer.partition(b'\n');self.buffer=bytearray(tail)
        reply=json.loads(line)
        if type(reply)is not dict or reply.get('status') not in SCHEMAS or set(reply)!=SCHEMAS[reply['status']]:raise ValueError('unexpected native OLE public reply fields')
        if type(reply.get('sequence'))is not int or reply['sequence']<0:raise ValueError('invalid native OLE reply sequence')
        return reply


def check_ready(reply,*,ole_role,self_role,peer_role,session):
    if (reply['status']!='ready' or reply['sequence']!=0 or reply['ole_role']!=ole_role or reply['self_role']!=self_role or reply['peer_role']!=peer_role or reply['tls_session']!=session or reply['security_reviewed']is not False or reply['transport_authenticated']is not True):raise ValueError('native OLE service READY binding mismatch')
    return reply


def check_complete(reply,*,sequence,request_id,length,calls,start_rot_choices):
    if (reply['status']!='complete' or reply['sequence']!=sequence or reply['request_id']!=request_id.hex() or reply['length']!=length or reply['calls']!=calls or reply['fresh_rot_choices']!=508*calls or reply['total_rot_choices']!=start_rot_choices+508*calls):raise ValueError('native OLE service COMPLETE binding mismatch')
    return reply


@dataclass(frozen=True)
class LocalServiceOutput:
    batch:LocalOleBatch
    start_rot_choices:int
    final_rot_choices:int

    @classmethod
    def read(cls,path,*,ole_role,self_role,peer_role,session,sequence,request_id,length,calls,start_rot_choices):
        batch=LocalOleBatch.read(path,role=ole_role,length=length,calls=calls,authenticated=True,self_role=self_role,peer_role=peer_role,session=session)
        raw=json.loads(Path(path).read_text())
        final=start_rot_choices+508*calls
        if (raw.get('service_sequence')!=sequence or raw.get('request_id')!=request_id.hex() or raw.get('start_rot_choices')!=start_rot_choices or raw.get('final_rot_choices')!=final):raise ValueError('native OLE private result sequence/identifier/stream mismatch')
        for index,call in enumerate(raw['sessions']):
            if call.get('total_rot_choices')!=start_rot_choices+508*(index+1):raise ValueError('native OLE stream did not advance for every call')
        counters=[call['entropy_substreams'] for call in raw['sessions']]
        if any(type(value)is not int or value<0 for value in counters) or any(counters[i]>=counters[i+1] for i in range(calls-1)):raise ValueError('native OLE entropy stream did not advance')
        return cls(batch,start_rot_choices,final)
