"""Optional single-launch packing and BLAKE3 hashing for small RBC frames.

The message is exactly little-endian uint32(sender) followed by that sender's
entire frame. Both the packed messages and four 32-byte digests are returned;
callers retain all existing header, canonical-value, and echo checks.
"""
from __future__ import annotations

import ctypes as ct
import ctypes.util
from functools import lru_cache

import torch

from . import gpu_hash

# Reuse the independently verified compression rounds and ROOT-output encoding.
# The extension supplies only input packing and the single-chunk block loop.
_COMPRESSION=gpu_hash._SOURCE.split('extern "C" __global__ void chunks(',1)[0]
_SOURCE=_COMPRESSION+r'''
__device__ __forceinline__ void frame_hash_impl(const U8* frame,int sender,
    U8* packed,U8* digests,int frame_length) {
    int length=frame_length+4;
    U8* message=packed+sender*length;
    for(int i=threadIdx.x;i<length;i+=blockDim.x)
        message[i]=i<4 ? (U8)((U32)sender>>(8*i)) : frame[i-4];
    __syncthreads();
    if(threadIdx.x!=0) return;
    U32 cv[8];
    #pragma unroll
    for(int i=0;i<8;i++) cv[i]=IV[i];
    int blocks=(length+63)/64;
    for(int block=0;block<blocks;block++) {
        int remaining=length-block*64;
        U32 len=remaining<64 ? remaining : 64;
        U32 m[16];
        #pragma unroll
        for(int word=0;word<16;word++) {
            U32 value=0;
            #pragma unroll
            for(int byte=0;byte<4;byte++)
                if(word*4+byte<len)
                    value|=(U32)message[block*64+word*4+byte]<<(8*byte);
            m[word]=value;
        }
        U32 flags=(block==0 ? 1 : 0)|(block==blocks-1 ? 2|8 : 0);
        compress(cv,m,0,len,flags);
    }
    write_digest(digests+sender*32,cv);
}
extern "C" __global__ void frame_hash(const U8* F0,const U8* F1,const U8* F2,const U8* F3,
    U8* packed,U8* digests,int frame_length) {
    int sender=blockIdx.x;
    const U8* frame=sender==0 ? F0 : sender==1 ? F1 : sender==2 ? F2 : F3;
    frame_hash_impl(frame,sender,packed,digests,frame_length);
}
extern "C" __global__ void frame_hash_addresses(const U64* addresses,U8* packed,U8* digests,int frame_length) {
    int sender=blockIdx.x;
    frame_hash_impl((const U8*)addresses[sender],sender,packed,digests,frame_length);
}
'''


@lru_cache(None)
def _compile(major,minor):
    nvrtc=ct.CDLL(ct.util.find_library('nvrtc') or '/usr/local/cuda/lib64/libnvrtc.so')
    program=ct.c_void_p()
    gpu_hash._check(nvrtc.nvrtcCreateProgram(ct.byref(program),_SOURCE.encode(),b'frame_hash.cu',0,None,None),'nvrtcCreateProgram')
    options=(ct.c_char_p*2)(f'--gpu-architecture=compute_{major}{minor}'.encode(),b'--std=c++17')
    status=nvrtc.nvrtcCompileProgram(program,2,options)
    size=ct.c_size_t()
    nvrtc.nvrtcGetProgramLogSize(program,ct.byref(size))
    log=ct.create_string_buffer(size.value)
    nvrtc.nvrtcGetProgramLog(program,log)
    if status:
        nvrtc.nvrtcDestroyProgram(ct.byref(program))
        raise RuntimeError(f'frame BLAKE3 compilation failed: {log.value.decode()}')
    gpu_hash._check(nvrtc.nvrtcGetPTXSize(program,ct.byref(size)),'nvrtcGetPTXSize')
    ptx=ct.create_string_buffer(size.value)
    gpu_hash._check(nvrtc.nvrtcGetPTX(program,ptx),'nvrtcGetPTX')
    nvrtc.nvrtcDestroyProgram(ct.byref(program))
    return ptx.raw


@lru_cache(None)
def _module(device):
    with torch.cuda.device(device):
        driver=ct.CDLL(ct.util.find_library('cuda') or 'libcuda.so.1')
        context=ct.c_void_p()
        gpu_hash._check(driver.cuDevicePrimaryCtxRetain(ct.byref(context),ct.c_int(device)),'cuDevicePrimaryCtxRetain')
        with gpu_hash._current_context(driver,context):
            ptx=ct.create_string_buffer(_compile(*torch.cuda.get_device_capability(device)))
            module=ct.c_void_p()
            gpu_hash._check(driver.cuModuleLoadData(ct.byref(module),ptx),'cuModuleLoadData')
            functions={}
            for name in ('frame_hash','frame_hash_addresses'):
                function=ct.c_void_p()
                gpu_hash._check(driver.cuModuleGetFunction(ct.byref(function),module,name.encode()),'cuModuleGetFunction')
                functions[name]=function
        return driver,module,functions,context


def _frames(frames):
    if isinstance(frames,dict):
        if set(frames)!=set(range(4)):raise ValueError('frame senders must be exactly 0,1,2,3')
        frames=tuple(frames[p] for p in range(4))
    else:
        frames=tuple(frames)
        if len(frames)!=4:raise ValueError('exactly four frames are required')
    first=frames[0]
    if not isinstance(first,torch.Tensor):raise TypeError('frames must be CUDA byte tensors')
    for frame in frames:
        if not isinstance(frame,torch.Tensor) or frame.dtype!=torch.uint8:
            raise TypeError('frames must use torch.uint8')
        if not frame.is_cuda or frame.ndim!=1 or not frame.is_contiguous():
            raise ValueError('frames must be flat contiguous CUDA buffers')
        if frame.device!=first.device or frame.numel()!=first.numel():
            raise ValueError('frames must have equal lengths and devices')
    return frames


def _pack_and_hash_validated(frames,digest_output=None):
    """Return packed [4,L+4] bytes and flat 128 digest bytes for four frames.

    Messages of at most 1024 bytes use one CUDA launch. Longer messages retain
    the existing packing and verified BLAKE3 tree implementation. Warm this
    function before CUDA graph capture to compile/load its CUDA module.
    """
    length=frames[0].numel()
    device=frames[0].device
    if digest_output is not None:
        if (not isinstance(digest_output,torch.Tensor) or digest_output.dtype!=torch.uint8
                or digest_output.shape!=(128,) or digest_output.device!=device or not digest_output.is_contiguous()):
            raise ValueError('digest output must be contiguous CUDA uint8[128] on the frame device')
        start=digest_output.data_ptr();end=start+128
        if any(max(start,frame.data_ptr())<min(end,frame.data_ptr()+length) for frame in frames):
            raise ValueError('digest output must not overlap a frame input')
    packed=torch.empty((4,length+4),dtype=torch.uint8,device=device)
    if length+4>1024:
        import triton
        from .spdz_device import _frame_inputs
        _frame_inputs[(triton.cdiv(length+4,256),4)](*frames,packed,length,256)
        result=gpu_hash.digest_batch(packed,block_tree=True).reshape(-1)
        if digest_output is not None:
            digest_output.copy_(result)
            result=digest_output
        return packed,result
    digests=digest_output if digest_output is not None else torch.empty(128,dtype=torch.uint8,device=device)
    with torch.cuda.device(device):
        driver,_,functions,context=_module(device.index)
        args=[ct.c_void_p(frame.data_ptr()) for frame in frames]
        args.extend((ct.c_void_p(packed.data_ptr()),ct.c_void_p(digests.data_ptr()),ct.c_int(length)))
        parameters=(ct.c_void_p*len(args))(*[ct.cast(ct.byref(arg),ct.c_void_p) for arg in args])
        with gpu_hash._current_context(driver,context):
            stream=ct.c_void_p(torch.cuda.current_stream(device).cuda_stream)
            gpu_hash._check(driver.cuLaunchKernel(functions['frame_hash'],4,1,1,128,1,1,0,stream,parameters,None),'frame BLAKE3 launch')
    return packed,digests


def pack_and_hash_frames(frames,*,digest_output=None):
    """Return packed bytes and four exact digests, optionally into supplied storage.

    ``digest_output`` can be a contiguous 128-byte slice of an echo frame,
    including an unaligned offset. Its byte range must not overlap any input.
    The small path writes it directly; the large fallback copies its digest.
    """
    return _pack_and_hash_validated(_frames(frames),digest_output)


def hash_frames(frames,size,*,digest_output=None):
    """Drop-in local arithmetic for DeviceDistributedParty's frame hash step."""
    if type(size) is not int or size<0:raise ValueError('payload size must be a nonnegative integer')
    frames=_frames(frames)
    if frames[0].numel()!=93+size:raise ValueError('RBC frame length must be 93+payload size')
    return _pack_and_hash_validated(frames,digest_output)


def hash_frame_addresses(addresses,frame_length,packed,digests):
    """Hash validated live pointers into preallocated outputs for graph plans.

    The caller must retain four valid frame allocations and order their writes
    before this stream. This low-level entry reads the current CUDA uint64[4]
    pointer table; it never treats received protocol bytes as device pointers.
    Only single-chunk prefixed messages are supported by this entry.
    """
    if type(frame_length) is not int or not 0<=frame_length<=1020:
        raise ValueError('pointer-table frames must fit one prefixed BLAKE3 chunk')
    if addresses.dtype!=torch.uint64 or addresses.shape!=(4,) or not addresses.is_cuda or not addresses.is_contiguous():
        raise ValueError('pointer table must be contiguous CUDA uint64[4]')
    device=addresses.device
    for output,shape in ((packed,(4,frame_length+4)),(digests,(128,))):
        if output.dtype!=torch.uint8 or output.shape!=shape or output.device!=device or not output.is_contiguous():
            raise ValueError('preallocated frame-hash output has invalid metadata')
    with torch.cuda.device(device):
        driver,_,functions,context=_module(device.index)
        args=[ct.c_void_p(addresses.data_ptr()),ct.c_void_p(packed.data_ptr()),ct.c_void_p(digests.data_ptr()),ct.c_int(frame_length)]
        parameters=(ct.c_void_p*len(args))(*[ct.cast(ct.byref(arg),ct.c_void_p) for arg in args])
        with gpu_hash._current_context(driver,context):
            stream=ct.c_void_p(torch.cuda.current_stream(device).cuda_stream)
            gpu_hash._check(driver.cuLaunchKernel(functions['frame_hash_addresses'],4,1,1,128,1,1,0,stream,parameters,None),'frame BLAKE3 address launch')
    return packed,digests


class SmallFrameHashPlan:
    """Prepared one-kernel hash plan with live frame-pointer uploads.

    This preserves the existing hash/check plan buffer identities and event
    ordering. A caller selects the existing large-frame plan above this class's
    single-chunk limit; no protocol validation is performed by graph replay.
    """
    def __init__(self,size,device):
        if type(size) is not int or not 0<=size<=927:
            raise ValueError('small RBC hash plans require payload size in [0,927]')
        from .spdz_device_graph import _PointerTable,_capture
        self.size=size
        self.dummy=tuple(torch.zeros(93+size,dtype=torch.uint8,device=device) for _ in range(4))
        self.pointers=_PointerTable(self.dummy,device)
        self.hashed=torch.empty((4,97+size),dtype=torch.uint8,device=device)
        self.view=torch.empty(128,dtype=torch.uint8,device=device)
        def run():
            return hash_frame_addresses(self.pointers.device,93+size,self.hashed,self.view)
        self.graph,_,self.ready=_capture(run,device)

    def replay(self,frames):
        # The underlying RBC validates peers, sizes, types, and devices before
        # its local hash step, as with the existing prepared hash plan.
        stream=torch.cuda.current_stream(self.hashed.device)
        stream.wait_event(self.ready)
        self.pointers.update(tuple(frames[p] for p in range(4)))
        self.graph.replay()
        self.ready.record(stream)
        return self.hashed,self.view
