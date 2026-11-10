"""Exact shared-memory CUDA matrix products with register-tiled SIMT arithmetic.

NVRTC compiles the embedded source on first use. The CUDA driver launches on
PyTorch's current stream, including during CUDA graph capture after warmup.
"""
from __future__ import annotations

import ctypes as ct
import ctypes.util
from functools import lru_cache

import torch

_SOURCE = r'''
typedef unsigned long long U64;
template<int BITS> struct Word;
template<> struct Word<64> { typedef U64 T; };
template<> struct Word<32> { typedef unsigned int T; };

template<int BITS, bool LOCAL>
__device__ __forceinline__ void tile(const U64* X0, const U64* X1, const U64* X2,
    const U64* Y0, const U64* Y1, const U64* Y2, U64* C, int M, int N, int K) {
    typedef typename Word<BITS>::T T;
    const int BM=32, BN=64, BK=16;
    __shared__ T sa[BM*BK];
    __shared__ T sb[BK*BN];
    __shared__ T sc[LOCAL ? BM*BK : 1];
    __shared__ T sd[LOCAL ? BK*BN : 1];
    const int tid=threadIdx.x, tx=tid%16, ty=tid/16;
    const int row=blockIdx.y*BM, col=blockIdx.x*BN;
    const U64 *a=X0, *a_add=0, *b=Y0, *c=0, *d=0;
    if (LOCAL) {
        if (blockIdx.z==0) { a=X0; a_add=X1; b=Y0; c=X0; d=Y1; }
        else if (blockIdx.z==1) { a=X1; a_add=X2; b=Y1; c=X1; d=Y2; }
        else { a=X0; b=Y2; c=X2; d=Y0; }
    }
    T accum[4][4] = {};
    for (int start=0; start<K; start+=BK) {
        for (int t=tid; t<BM*BK; t+=128) {
            int r=row+t/BK, k=start+t%BK;
            T v=0, w=0;
            if (r<M && k<K) {
                v=(T)a[(U64)r*K+k];
                if (LOCAL) {
                    if (a_add) v+=(T)a_add[(U64)r*K+k];
                    w=(T)c[(U64)r*K+k];
                }
            }
            sa[t]=v;
            if (LOCAL) sc[t]=w;
        }
        for (int t=tid; t<BK*BN; t+=128) {
            int k=start+t/BN, n=col+t%BN;
            T v=0, w=0;
            if (k<K && n<N) {
                v=(T)b[(U64)k*N+n];
                if (LOCAL) w=(T)d[(U64)k*N+n];
            }
            sb[t]=v;
            if (LOCAL) sd[t]=w;
        }
        __syncthreads();
        #pragma unroll
        for (int k=0; k<BK; ++k) {
            T av[4], bv[4], cv[4], dv[4];
            #pragma unroll
            for (int i=0; i<4; ++i) {
                av[i]=sa[(ty+i*8)*BK+k];
                bv[i]=sb[k*BN+tx+i*16];
                if (LOCAL) {
                    cv[i]=sc[(ty+i*8)*BK+k];
                    dv[i]=sd[k*BN+tx+i*16];
                }
            }
            #pragma unroll
            for (int i=0; i<4; ++i)
                #pragma unroll
                for (int j=0; j<4; ++j) {
                    accum[i][j]+=av[i]*bv[j];
                    if (LOCAL) accum[i][j]+=cv[i]*dv[j];
                }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i=0; i<4; ++i)
        #pragma unroll
        for (int j=0; j<4; ++j) {
            int r=row+ty+i*8, n=col+tx+j*16;
            if (r<M && n<N) C[(U64)blockIdx.z*M*N+(U64)r*N+n]=(U64)accum[i][j];
        }
}
'''
_SIGNATURE = 'const U64* X0,const U64* X1,const U64* X2,const U64* Y0,const U64* Y1,const U64* Y2,U64* C,int M,int N,int K'
for _bits in (32, 64):
    for _local in (False, True):
        _SOURCE += ('extern "C" __global__ void mm%d_%d(%s) { tile<%d,%s>'
                    '(X0,X1,X2,Y0,Y1,Y2,C,M,N,K); }\n' %
                    (_bits, int(_local), _SIGNATURE, _bits, str(_local).lower()))


def _check(code, operation):
    if code:
        raise RuntimeError(f"{operation} failed with CUDA status {code}")


@lru_cache(None)
def _compile(major=8, minor=9):
    library = ct.util.find_library("nvrtc")
    if library is None:
        library = "/usr/local/cuda/lib64/libnvrtc.so"
    nvrtc = ct.CDLL(library)
    program = ct.c_void_p()
    _check(nvrtc.nvrtcCreateProgram(ct.byref(program), _SOURCE.encode(), b"simt.cu", 0, None, None), "nvrtcCreateProgram")
    options = [f"--gpu-architecture=compute_{major}{minor}".encode(), b"--std=c++17", b"--device-as-default-execution-space"]
    array = (ct.c_char_p * len(options))(*options)
    code = nvrtc.nvrtcCompileProgram(program, len(options), array)
    size = ct.c_size_t()
    nvrtc.nvrtcGetProgramLogSize(program, ct.byref(size))
    log = ct.create_string_buffer(size.value)
    nvrtc.nvrtcGetProgramLog(program, log)
    if code:
        nvrtc.nvrtcDestroyProgram(ct.byref(program))
        raise RuntimeError(f"NVRTC compilation failed: {log.value.decode()}")
    _check(nvrtc.nvrtcGetPTXSize(program, ct.byref(size)), "nvrtcGetPTXSize")
    ptx = ct.create_string_buffer(size.value)
    _check(nvrtc.nvrtcGetPTX(program, ptx), "nvrtcGetPTX")
    nvrtc.nvrtcDestroyProgram(ct.byref(program))
    return ptx.raw


@lru_cache(None)
def _module(device):
    # PyTorch initializes the primary context before loading the CUDA module.
    with torch.cuda.device(device):
        torch.cuda.current_stream()
        capability = torch.cuda.get_device_capability(device)
        driver = ct.CDLL(ct.util.find_library("cuda") or "libcuda.so.1")
        module = ct.c_void_p()
        ptx = ct.create_string_buffer(_compile(*capability))
        _check(driver.cuModuleLoadData(ct.byref(module), ptx), "cuModuleLoadData")
        functions = {}
        for bits in (32, 64):
            for local in (False, True):
                function = ct.c_void_p()
                _check(driver.cuModuleGetFunction(ct.byref(function), module, f"mm{bits}_{int(local)}".encode()), "cuModuleGetFunction")
                functions[bits, local] = function
        return driver, module, functions


def _validate(a, b, bits):
    if bits not in (32, 64):
        raise ValueError("bits must be 32 or 64")
    if a.dtype != torch.int64 or b.dtype != torch.int64:
        raise TypeError("ring_matmul requires int64 inputs")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected compatible two-dimensional matrices")
    if not a.is_cuda or not b.is_cuda or a.device != b.device:
        raise ValueError("inputs must be on the same CUDA device")


def _launch(xs, ys, output, bits, local):
    a, b = xs[0], ys[0]
    m, k = a.shape
    n = b.shape[1]
    driver, _, functions = _module(a.device.index)
    pointers = [ct.c_void_p(x.data_ptr()) for x in (*xs, *ys, output)]
    integers = [ct.c_int(x) for x in (m, n, k)]
    params = (ct.c_void_p * 10)(*[ct.cast(ct.byref(x), ct.c_void_p) for x in (*pointers, *integers)])
    stream = ct.c_void_p(torch.cuda.current_stream(a.device).cuda_stream)
    _check(driver.cuLaunchKernel(functions[bits, local],
        (n+63)//64, (m+31)//32, 3 if local else 1, 128, 1, 1, 0, stream, params, None), "cuLaunchKernel")


def ring_matmul(a, b, bits=64):
    """Multiply int64 CUDA matrices exactly modulo 2**bits."""
    _validate(a, b, bits)
    a, b = a.contiguous(), b.contiguous()
    output = torch.empty((a.shape[0], b.shape[1]), dtype=torch.int64, device=a.device)
    if not output.numel():
        return output
    if a.shape[1] == 0:
        return output.zero_()
    with torch.cuda.device(a.device):
        _launch((a, a, a), (b, b, b), output, bits, False)
    return output


def local_products(x0, x1, x2, y0, y1, y2, bits=64):
    """Return the three Rep4 local products, fusing their six matrix products."""
    xs, ys = (x0, x1, x2), (y0, y1, y2)
    for x, y in zip(xs, ys):
        _validate(x, y, bits)
        if x.shape != x0.shape or y.shape != y0.shape or x.device != x0.device:
            raise ValueError("local product inputs must have matching shapes and devices")
    xs, ys = tuple(x.contiguous() for x in xs), tuple(y.contiguous() for y in ys)
    output = torch.empty((3, x0.shape[0], y0.shape[1]), dtype=torch.int64, device=x0.device)
    if output.numel():
        if x0.shape[1] == 0:
            output.zero_()
        else:
            with torch.cuda.device(x0.device):
                _launch(xs, ys, output, bits, True)
    return tuple(output.unbind(0))
