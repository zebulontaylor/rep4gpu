"""Compare large-message crypto layouts without changing protocol modules.

Run from the repository root with ``python -m benchmarks.large_crypto_experiment``.
All inputs, keys, and repeated nonces are public benchmark fixtures. The
experiments preserve BLAKE3's 1024-byte chunks and binary tree, but do not
replace the maintained implementation.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import ctypes.util
import json
from pathlib import Path

import torch
import triton.testing

from rep4gpu import gpu_hash
from rep4gpu.gpu_prg_batch import generate_rows


_PARENT = r'''
extern "C" __global__ void grouped_parents(const U32* input,U32* output,U8* result,U64 count) {
    __shared__ U32 shared[8][128];
    U64 groups=(count+127)/128, group=blockIdx.x%groups, row=blockIdx.x/groups;
    U64 node=group*128+threadIdx.x;
    U32 cv[8];
    if(node<count) {
        #pragma unroll
        for(int i=0;i<8;i++)shared[i][threadIdx.x]=input[(row*count+node)*8+i];
    }
    __syncthreads();
    int current=(count-group*128)<128 ? count-group*128:128;
    while(current>1) {
        int next=(current+1)/2;
        if(threadIdx.x<next) {
            int left=threadIdx.x*2;
            U32 m[16];
            #pragma unroll
            for(int i=0;i<8;i++) {
                cv[i]=shared[i][left]; m[i]=cv[i];
                if(left+1<current)m[i+8]=shared[i][left+1];
            }
            if(left+1<current) {
                #pragma unroll
                for(int i=0;i<8;i++)cv[i]=IV[i];
                compress(cv,m,0,64,4|((groups==1 && current==2)?8:0));
            }
        }
        __syncthreads();
        if(threadIdx.x<next) {
            #pragma unroll
            for(int i=0;i<8;i++)shared[i][threadIdx.x]=cv[i];
        }
        __syncthreads();
        current=next;
    }
    if(threadIdx.x==0) {
        #pragma unroll
        for(int i=0;i<8;i++){cv[i]=shared[i][0];output[(U64)blockIdx.x*8+i]=cv[i];}
        if(groups==1)write_digest(result+row*32,cv);
    }
}
'''

_SHARED_LEAF = r'''
extern "C" __global__ void shared_chunks(const U8* data,U32* output,U8* result,U64 length,U64 count) {
    __shared__ U32 words[256][33];
    U64 groups=(count+31)/32, group=blockIdx.x%groups, row=blockIdx.x/groups;
    U64 chunk=group*32+threadIdx.x;
    int current=(count-group*32)<32 ? count-group*32:32;
    for(int index=threadIdx.x;index<current*256;index+=128)
        words[index%256][index/256]=((const U32*)(data+row*length+group*32768))[index];
    __syncthreads();
    U32 cv[8];
    #pragma unroll
    for(int i=0;i<8;i++)cv[i]=IV[i];
    if(threadIdx.x<current) {
        for(int block=0;block<16;block++) {
            U32 m[16];
            #pragma unroll
            for(int i=0;i<16;i++)m[i]=words[block*16+i][threadIdx.x];
            U32 flags=(block==0?1:0)|(block==15?2:0);
            if(count==1 && block==15)flags|=8;
            compress(cv,m,chunk,64,flags);
        }
    }
    if(threadIdx.x<32) {
        for(int stride=1;stride<current;stride*=2) {
            U32 m[16];
            #pragma unroll
            for(int i=0;i<8;i++){m[i]=cv[i];m[i+8]=__shfl_down_sync(0xffffffff,cv[i],stride);}
            if(threadIdx.x%(2*stride)==0 && threadIdx.x+stride<current) {
                #pragma unroll
                for(int i=0;i<8;i++)cv[i]=IV[i];
                compress(cv,m,0,64,4|((groups==1 && stride*2>=current)?8:0));
            }
        }
        if(threadIdx.x==0) {
            #pragma unroll
            for(int i=0;i<8;i++)output[(U64)blockIdx.x*8+i]=cv[i];
            if(groups==1)write_digest(result+row*32,cv);
        }
    }
}
'''


def _compile(source, capability):
    nvrtc = ct.CDLL(ct.util.find_library('nvrtc') or '/usr/local/cuda/lib64/libnvrtc.so')
    program = ct.c_void_p()
    gpu_hash._check(nvrtc.nvrtcCreateProgram(ct.byref(program), source.encode(), b'large_crypto.cu', 0, None, None), 'nvrtcCreateProgram')
    options = (ct.c_char_p*2)(f'--gpu-architecture=compute_{capability[0]}{capability[1]}'.encode(), b'--std=c++17')
    status = nvrtc.nvrtcCompileProgram(program, 2, options)
    size = ct.c_size_t()
    nvrtc.nvrtcGetProgramLogSize(program, ct.byref(size))
    log = ct.create_string_buffer(size.value)
    nvrtc.nvrtcGetProgramLog(program, log)
    if status:
        nvrtc.nvrtcDestroyProgram(ct.byref(program))
        raise RuntimeError(log.value.decode())
    gpu_hash._check(nvrtc.nvrtcGetPTXSize(program, ct.byref(size)), 'nvrtcGetPTXSize')
    ptx = ct.create_string_buffer(size.value)
    gpu_hash._check(nvrtc.nvrtcGetPTX(program, ptx), 'nvrtcGetPTX')
    nvrtc.nvrtcDestroyProgram(ct.byref(program))
    return ptx


class LargeHashExperiment:
    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.driver, _, self.base_functions, self.context = gpu_hash._module(self.device.index)
        prefix = gpu_hash._SOURCE[:gpu_hash._SOURCE.index('extern "C" __global__ void chunks')]
        ptx = _compile(prefix+_PARENT+_SHARED_LEAF, torch.cuda.get_device_capability(self.device))
        self.module = ct.c_void_p()
        self.functions = {}
        with gpu_hash._current_context(self.driver, self.context):
            gpu_hash._check(self.driver.cuModuleLoadData(ct.byref(self.module), ptx), 'cuModuleLoadData')
            for name in ('grouped_parents','shared_chunks'):
                function = ct.c_void_p()
                gpu_hash._check(self.driver.cuModuleGetFunction(ct.byref(function), self.module, name.encode()), 'cuModuleGetFunction')
                self.functions[name] = function

    def digest(self, data, shared=False):
        if data.ndim!=2 or data.dtype!=torch.uint8 or not data.is_contiguous() or data.device!=self.device:
            raise ValueError('expected contiguous CUDA byte rows on the experiment device')
        batch,length = data.shape
        if not batch or not length or length%1024 or data.data_ptr()%16:
            raise ValueError('experiment requires nonempty aligned full-chunk rows')
        count = length//1024
        group_size = 32 if shared else 128
        groups = (count+group_size-1)//group_size
        result = torch.empty((batch,32),dtype=torch.uint8,device=self.device)
        cvs = torch.empty((batch,groups,8),dtype=torch.uint32,device=self.device)
        args = [ct.c_void_p(data.data_ptr()),ct.c_void_p(cvs.data_ptr()),ct.c_void_p(result.data_ptr()),ct.c_uint64(length),ct.c_uint64(count)]
        if shared:
            function = self.functions['shared_chunks']
        else:
            function = self.base_functions['chunks']
            args += [ct.c_int(batch),ct.c_int(1),ct.c_int(1)]
        gpu_hash._launch(self.driver,function,batch*groups,args,self.device,self.context)
        count = groups
        while count>1:
            groups = (count+127)//128
            output = torch.empty((batch,groups,8),dtype=torch.uint32,device=self.device)
            gpu_hash._launch(self.driver,self.functions['grouped_parents'],batch*groups,
                [ct.c_void_p(cvs.data_ptr()),ct.c_void_p(output.data_ptr()),ct.c_void_p(result.data_ptr()),ct.c_uint64(count)],self.device,self.context)
            cvs,count = output,groups
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes-mib', type=int, nargs='+', default=[1,2,4,8,16])
    parser.add_argument('--batches', type=int, nargs='+', default=[1,2])
    parser.add_argument('--repetitions', type=int, default=2)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--skip-prg', action='store_true')
    args = parser.parse_args()
    experiment = LargeHashExperiment()
    results = []
    for mib in args.sizes_mib:
        length = mib*2**20
        for batch in args.batches:
            data = torch.randint(256,(batch,length),device=experiment.device,dtype=torch.uint8)
            inputs = tuple(data[i].clone() for i in range(batch))
            expected = gpu_hash.digest_batch(data,block_tree=True)
            for shared in (False,True):
                assert torch.equal(expected,experiment.digest(data,shared=shared))
            # Check the tree against the independent CPU package outside timing.
            from blake3 import blake3
            for row,digest in zip(data.cpu().numpy(),expected.cpu().numpy()):
                assert blake3(row.tobytes()).digest()==digest.tobytes()
            methods = {'base':lambda:gpu_hash.digest_batch(data,block_tree=True),
                       'grouped_parents':lambda:experiment.digest(data),
                       'shared32_grouped_parents':lambda:experiment.digest(data,shared=True),
                       'stack_copy':lambda:torch.stack(inputs),
                       'stack_plus_base':lambda:gpu_hash.digest_batch(torch.stack(inputs),block_tree=True)}
            for repetition in range(args.repetitions):
                for name in (list(methods) if repetition%2==0 else list(reversed(methods))):
                    ms = triton.testing.do_bench_cudagraph(methods[name],rep=100)
                    result = dict(operation='hash',length=length,batch=batch,repetition=repetition,method=name,ms=ms,
                                  useful_gbps=batch*length/(ms*1e6))
                    results.append(result)
                    print(json.dumps(result),flush=True)
        if not args.skip_prg:
            count = length//4
            key = torch.arange(8,device=experiment.device,dtype=torch.int64).to(torch.uint32)
            for party,receivers in ((0,(3,0,1,2,3,2)),(2,(3,0,1,2,3,2))):
                keys = tuple(None if receiver==party else key for receiver in receivers)
                nonces = tuple(None if receiver==party else i for i,receiver in enumerate(receivers))
                function = lambda:generate_rows(keys,nonces,count,bits=32,packed=True)
                ms = triton.testing.do_bench_cudagraph(function,rep=100)
                result = dict(operation='prg',length=length,party=party,active_rows=sum(k is not None for k in keys),output_rows=6,ms=ms,
                              stored_gbps=6*length/(ms*1e6))
                results.append(result)
                print(json.dumps(result),flush=True)
    if args.output:
        args.output.write_text(json.dumps({'device':torch.cuda.get_device_name(),
            'scope':'public fixed fixtures; captured GPU kernels only; no network or protocol replacement',
            'correctness':'all hash variants compared with maintained GPU and independent CPU BLAKE3',
            'results':results},indent=2)+'\n')


if __name__=='__main__':
    main()
