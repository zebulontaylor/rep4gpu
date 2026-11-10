"""Experimental unkeyed BLAKE3-256 over contiguous CUDA byte buffers.

The compression function and flags follow the official BLAKE3 reference:
https://github.com/BLAKE3-team/BLAKE3/blob/master/reference_impl/reference_impl.rs
Chunks run in parallel; subsequent kernels reduce the binary tree. A final
compression sets ROOT, rather than returning an ordinary chaining value.
This module does not change the validated BLAKE2b wire protocol.
"""
from __future__ import annotations

import ctypes as ct
import ctypes.util
from functools import lru_cache
from contextlib import contextmanager

import torch

_SOURCE = r'''
typedef unsigned char U8;
typedef unsigned int U32;
typedef unsigned long long U64;
__device__ __constant__ U32 IV[8] = {0x6A09E667,0xBB67AE85,0x3C6EF372,0xA54FF53A,
    0x510E527F,0x9B05688C,0x1F83D9AB,0x5BE0CD19};
__device__ __forceinline__ U32 rotr(U32 x, int n) { return (x>>n)|(x<<(32-n)); }
__device__ __forceinline__ void g(U32& a,U32& b,U32& c,U32& d,U32 x,U32 y) {
    a=a+b+x; d=rotr(d^a,16); c=c+d; b=rotr(b^c,12);
    a=a+b+y; d=rotr(d^a,8); c=c+d; b=rotr(b^c,7);
}
__device__ __forceinline__ void compress(U32 cv[8],const U32 m[16],U64 counter,U32 len,U32 flags) {
    U32 s[16];
    #pragma unroll
    for(int i=0;i<8;i++) s[i]=cv[i];
    #pragma unroll
    for(int i=0;i<4;i++) s[8+i]=IV[i];
    s[12]=(U32)counter; s[13]=(U32)(counter>>32); s[14]=len; s[15]=flags;
'''
_permutation = (2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8)
_schedule = list(range(16))
for _round in range(7):
    for _i, (_a, _b, _c, _d) in enumerate(((0,4,8,12),(1,5,9,13),(2,6,10,14),(3,7,11,15),
                                          (0,5,10,15),(1,6,11,12),(2,7,8,13),(3,4,9,14))):
        _SOURCE += f'    g(s[{_a}],s[{_b}],s[{_c}],s[{_d}],m[{_schedule[2*_i]}],m[{_schedule[2*_i+1]}]);\n'
    _schedule = [_schedule[i] for i in _permutation]
_SOURCE += r'''
    #pragma unroll
    for(int i=0;i<8;i++) cv[i]=s[i]^s[i+8];
}
__device__ __forceinline__ void write_digest(U8* out,const U32 cv[8]) {
    #pragma unroll
    for(int i=0;i<8;i++) {
        #pragma unroll
        for(int j=0;j<4;j++) out[4*i+j]=(U8)(cv[i]>>(8*j));
    }
}
extern "C" __global__ void chunks(const U8* data,U32* cvs,U8* result,U64 length,U64 count,int batch,int aligned,int tree) {
    __shared__ U32 shared_cv[8][128];
    U64 id=(U64)blockIdx.x*blockDim.x+threadIdx.x;
    U64 groups=(count+127)/128;
    U64 row=tree ? blockIdx.x/groups : id/count;
    U64 chunk=tree ? (blockIdx.x%groups)*128+threadIdx.x : id%count;
    bool active=row<batch && chunk<count;
    if(!tree && !active) return;
    U32 cv[8];
    #pragma unroll
    for(int i=0;i<8;i++) cv[i]=IV[i];
    if(active) {
        U64 begin=chunk*1024;
        U64 available=length>begin ? length-begin : 0;
        int chunk_len=available<1024 ? (int)available : 1024;
        int blocks=chunk_len ? (chunk_len+63)/64 : 1;
        for(int block=0;block<blocks;block++) {
            int remaining=chunk_len-block*64;
            U32 len=remaining<64 ? remaining : 64;
            U32 m[16];
            #pragma unroll
            for(int word=0;word<16;word++) {
                int byte=word*4;
                U64 offset=row*length+begin+block*64+byte;
                U32 value=0;
                if(aligned && ((row*length)&3)==0 && byte+4<=len) {
                    value=((const U32*)data)[offset/4];
                } else {
                    #pragma unroll
                    for(int j=0;j<4;j++) if(byte+j<len) value|=(U32)data[offset+j]<<(8*j);
                }
                m[word]=value;
            }
            U32 flags=(block==0 ? 1 : 0) | (block==blocks-1 ? 2 : 0);
            if(count==1 && block==blocks-1) flags|=8;
            compress(cv,m,chunk,len,flags);
        }
    }
    if(!tree) {
        #pragma unroll
        for(int i=0;i<8;i++) cvs[id*8+i]=cv[i];
        if(count==1) write_digest(result+row*32,cv);
        return;
    }
    if(active) {
        #pragma unroll
        for(int i=0;i<8;i++) shared_cv[i][threadIdx.x]=cv[i];
    }
    __syncthreads();
    U64 available=count-(blockIdx.x%groups)*128;
    int current=available<128 ? (int)available : 128;
    while(current>1) {
        int next=(current+1)/2;
        if(threadIdx.x<next) {
            int left=threadIdx.x*2;
            U32 m[16];
            #pragma unroll
            for(int i=0;i<8;i++) {
                cv[i]=shared_cv[i][left];
                m[i]=cv[i];
                if(left+1<current) m[i+8]=shared_cv[i][left+1];
            }
            if(left+1<current) {
                #pragma unroll
                for(int i=0;i<8;i++) cv[i]=IV[i];
                compress(cv,m,0,64,4|((groups==1 && current==2) ? 8 : 0));
            }
        }
        // Every child must be read before another thread overwrites its slot.
        __syncthreads();
        if(threadIdx.x<next) {
            #pragma unroll
            for(int i=0;i<8;i++) shared_cv[i][threadIdx.x]=cv[i];
        }
        __syncthreads();
        current=next;
    }
    if(threadIdx.x==0) {
        #pragma unroll
        for(int i=0;i<8;i++) {cv[i]=shared_cv[i][0];cvs[((U64)blockIdx.x)*8+i]=cv[i];}
        if(groups==1) write_digest(result+row*32,cv);
    }
}
extern "C" __global__ void parents(const U32* input,U32* output,U8* result,U64 count,int batch) {
    U64 next=(count+1)/2;
    U64 id=(U64)blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=next*batch) return;
    U64 row=id/next, node=id%next;
    U64 left=row*count+node*2;
    if(node*2+1==count) {
        #pragma unroll
        for(int i=0;i<8;i++) output[id*8+i]=input[left*8+i];
        return;
    }
    U32 m[16],cv[8];
    #pragma unroll
    for(int i=0;i<8;i++) {m[i]=input[left*8+i];m[i+8]=input[(left+1)*8+i];cv[i]=IV[i];}
    compress(cv,m,0,64,4|(count==2 ? 8 : 0));
    #pragma unroll
    for(int i=0;i<8;i++) output[id*8+i]=cv[i];
    if(count==2) write_digest(result+row*32,cv);
}
'''


def _check(status, operation):
    if status:
        raise RuntimeError(f'{operation} failed with CUDA status {status}')


@lru_cache(None)
def _compile(major, minor):
    nvrtc = ct.CDLL(ct.util.find_library('nvrtc') or '/usr/local/cuda/lib64/libnvrtc.so')
    program = ct.c_void_p()
    _check(nvrtc.nvrtcCreateProgram(ct.byref(program), _SOURCE.encode(), b'blake3.cu', 0, None, None), 'nvrtcCreateProgram')
    options = (ct.c_char_p*2)(f'--gpu-architecture=compute_{major}{minor}'.encode(), b'--std=c++17')
    status = nvrtc.nvrtcCompileProgram(program, 2, options)
    size = ct.c_size_t()
    nvrtc.nvrtcGetProgramLogSize(program, ct.byref(size))
    log = ct.create_string_buffer(size.value)
    nvrtc.nvrtcGetProgramLog(program, log)
    if status:
        nvrtc.nvrtcDestroyProgram(ct.byref(program))
        raise RuntimeError(f'BLAKE3 NVRTC compilation failed: {log.value.decode()}')
    _check(nvrtc.nvrtcGetPTXSize(program, ct.byref(size)), 'nvrtcGetPTXSize')
    ptx = ct.create_string_buffer(size.value)
    _check(nvrtc.nvrtcGetPTX(program, ptx), 'nvrtcGetPTX')
    nvrtc.nvrtcDestroyProgram(ct.byref(program))
    return ptx.raw


@contextmanager
def _current_context(driver, context):
    previous = ct.c_void_p()
    _check(driver.cuCtxGetCurrent(ct.byref(previous)), 'cuCtxGetCurrent')
    changed = previous.value != context.value
    if changed:
        _check(driver.cuCtxPushCurrent_v2(context), 'cuCtxPushCurrent')
    try:
        yield
    finally:
        if changed:
            popped = ct.c_void_p()
            _check(driver.cuCtxPopCurrent_v2(ct.byref(popped)), 'cuCtxPopCurrent')


@lru_cache(None)
def _module(device):
    with torch.cuda.device(device):
        driver = ct.CDLL(ct.util.find_library('cuda') or 'libcuda.so.1')
        context = ct.c_void_p()
        _check(driver.cuDevicePrimaryCtxRetain(ct.byref(context),ct.c_int(device)), 'cuDevicePrimaryCtxRetain')
        with _current_context(driver,context):
            torch.cuda.current_stream()
            ptx = ct.create_string_buffer(_compile(*torch.cuda.get_device_capability(device)))
            module = ct.c_void_p()
            _check(driver.cuModuleLoadData(ct.byref(module), ptx), 'cuModuleLoadData')
            functions = {}
            for name in ('chunks','parents'):
                function = ct.c_void_p()
                _check(driver.cuModuleGetFunction(ct.byref(function), module, name.encode()), 'cuModuleGetFunction')
                functions[name] = function
        return driver, module, functions, context


def _launch(driver, function, blocks, parameters, device, context):
    params = (ct.c_void_p*len(parameters))(*[ct.cast(ct.byref(p),ct.c_void_p) for p in parameters])
    # Empty tensor operations need not install a CUDA context on a new worker.
    # Restore the caller's context after using the cached module's owner.
    with _current_context(driver,context):
        stream = ct.c_void_p(torch.cuda.current_stream(device).cuda_stream)
        _check(driver.cuLaunchKernel(function, blocks,1,1,128,1,1,0,stream,params,None), 'BLAKE3 cuLaunchKernel')


def digest_batch(data, *, block_tree=False):
    """Hash each row of a contiguous CUDA uint8 matrix to one 32-byte digest.

    Inputs and intermediate chaining values stay on the current CUDA device.
    Warm this function before CUDA graph capture to compile/load its kernels.
    Empty rows are supported; a zero-row batch returns shape [0,32].
    ``block_tree=True`` reduces aligned groups of up to 128 chunks within each
    block before the global tree, reducing launches for short messages.
    """
    if data.dtype != torch.uint8:
        raise TypeError('BLAKE3 input must use torch.uint8')
    if data.ndim != 2 or not data.is_cuda or not data.is_contiguous():
        raise ValueError('BLAKE3 batch input must be a contiguous CUDA matrix')
    batch, length = data.shape
    count = max(1,(length+1023)//1024)
    if batch >= 2**31:
        raise ValueError('BLAKE3 batch size must fit int32')
    result = torch.empty((batch,32),dtype=torch.uint8,device=data.device)
    if not batch:
        return result
    with torch.cuda.device(data.device):
        driver, _, functions, context = _module(data.device.index)
        reduced_count=(count+127)//128 if block_tree else count
        cvs = torch.empty((batch,reduced_count,8),dtype=torch.uint32,device=data.device)
        blocks=batch*reduced_count if block_tree else (batch*count+127)//128
        _launch(driver,functions['chunks'],blocks,
                [ct.c_void_p(data.data_ptr()),ct.c_void_p(cvs.data_ptr()),ct.c_void_p(result.data_ptr()),
                 ct.c_uint64(length),ct.c_uint64(count),ct.c_int(batch),ct.c_int(data.data_ptr()%4==0),
                 ct.c_int(bool(block_tree))],data.device,context)
        count=reduced_count
        while count>1:
            next_count=(count+1)//2
            parents = torch.empty((batch,next_count,8),dtype=torch.uint32,device=data.device)
            _launch(driver,functions['parents'],(batch*next_count+127)//128,
                    [ct.c_void_p(cvs.data_ptr()),ct.c_void_p(parents.data_ptr()),ct.c_void_p(result.data_ptr()),
                     ct.c_uint64(count),ct.c_int(batch)],data.device,context)
            cvs, count = parents, next_count
    return result


def digest(data, *, block_tree=False):
    """Return official unkeyed BLAKE3-256 for a contiguous CUDA uint8 vector."""
    if data.ndim != 1:
        raise ValueError('BLAKE3 input must be a one-dimensional byte buffer')
    return digest_batch(data.unsqueeze(0),block_tree=block_tree)[0]
