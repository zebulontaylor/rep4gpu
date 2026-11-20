"""Optional OpenSSL finalization of prepared state and Merkle fingerprints."""
from __future__ import annotations

import ctypes as ct
from functools import lru_cache
import os
from pathlib import Path
import subprocess
import tempfile
import threading
from collections.abc import Callable, Sequence

from .merkle import MerkleTree

_SOURCE = r'''
#include <stdint.h>
#include <stddef.h>
#include <openssl/sha.h>
static void integer(SHA256_CTX* sha, uint64_t n) {
    unsigned char bytes[8];
    for (int i=0;i<8;i++) bytes[7-i]=(unsigned char)(n>>(i*8));
    SHA256_Update(sha,bytes,8);
}
void compute(const unsigned char* metadata,const uint64_t* offsets,
             const uint64_t* starts,const unsigned char* digests,
             const unsigned char* context,unsigned char* out,uint64_t count) {
    SHA256_CTX sha;
    unsigned char fingerprint[32];
    const char state[]="rep4gpu.state-digest.v1";
    const char leaf[]="rep4gpu.merkle.v1.leaf";
    const char node[]="rep4gpu.merkle.v1.node";
    const char root[]="rep4gpu.merkle.v1.root";
    for(uint64_t i=0;i<count;i++) {
        SHA256_Init(&sha);SHA256_Update(&sha,state,sizeof(state));
        integer(&sha,offsets[i+1]-offsets[i]);
        SHA256_Update(&sha,metadata+offsets[i],offsets[i+1]-offsets[i]);
        SHA256_Update(&sha,digests+starts[i],starts[i+1]-starts[i]);
        SHA256_Final(fingerprint,&sha);
        SHA256_Init(&sha);SHA256_Update(&sha,leaf,sizeof(leaf));integer(&sha,i);
        SHA256_Update(&sha,context,32);SHA256_Update(&sha,fingerprint,32);
        SHA256_Final(out+i*32,&sha);
    }
    uint64_t offset=0,width=count;
    while(width>1) {
        uint64_t next=(width+1)/2;
        for(uint64_t i=0;i<next;i++) {
            uint64_t right=i*2+1<width ? i*2+1 : width-1;
            SHA256_Init(&sha);SHA256_Update(&sha,node,sizeof(node));
            SHA256_Update(&sha,out+(offset+i*2)*32,32);
            SHA256_Update(&sha,out+(offset+right)*32,32);
            SHA256_Final(out+(offset+width+i)*32,&sha);
        }
        offset+=width;width=next;
    }
    SHA256_Init(&sha);SHA256_Update(&sha,root,sizeof(root));integer(&sha,count);
    SHA256_Update(&sha,out+offset*32,32);SHA256_Final(out+(offset+1)*32,&sha);
}
'''


_CUDA_SOURCE = r'''
#include <dlfcn.h>
#include <pthread.h>
typedef int (*retain_fn)(void**,int);
typedef int (*current_fn)(void**);
typedef int (*push_fn)(void*);
typedef int (*pop_fn)(void**);
typedef int (*copy_fn)(uint64_t,uint64_t,size_t,void*);
typedef int (*launch_fn)(void*,void*);
typedef int (*record_fn)(void*,void*);
typedef int (*sync_fn)(void*);
static pthread_once_t cuda_once=PTHREAD_ONCE_INIT;
static retain_fn retain_context;
static current_fn current_context;
static push_fn push_context;
static pop_fn pop_context;
static copy_fn copy_device;
static launch_fn launch_graph;
static record_fn record_event;
static sync_fn sync_event;
static sync_fn sync_stream;
static void load_cuda(void) {
    void* lib=dlopen("libcuda.so.1",RTLD_NOW|RTLD_LOCAL);
    if(!lib)return;
    retain_context=(retain_fn)dlsym(lib,"cuDevicePrimaryCtxRetain");
    current_context=(current_fn)dlsym(lib,"cuCtxGetCurrent");
    push_context=(push_fn)dlsym(lib,"cuCtxPushCurrent_v2");
    pop_context=(pop_fn)dlsym(lib,"cuCtxPopCurrent_v2");
    copy_device=(copy_fn)dlsym(lib,"cuMemcpyDtoDAsync_v2");
    launch_graph=(launch_fn)dlsym(lib,"cuGraphLaunch");
    record_event=(record_fn)dlsym(lib,"cuEventRecord");
    sync_event=(sync_fn)dlsym(lib,"cuEventSynchronize");
    sync_stream=(sync_fn)dlsym(lib,"cuStreamSynchronize");
}
int prepare_cuda(int device,void** context) {
    pthread_once(&cuda_once,load_cuda);
    if(!retain_context||!current_context||!push_context||!pop_context||!copy_device||!launch_graph||!record_event||!sync_event||!sync_stream)return -1;
    return retain_context(context,device);
}
typedef struct {
    void *owner,*graph,*ready,*stream;
    const uint64_t *destinations,*sizes,*sources;
    uint64_t inputs,checkpoint_source,checkpoint_destination,checkpoint_bytes;
    const unsigned char *digest_host,*metadata;
    const uint64_t *offsets,*starts;
    const unsigned char *context;
    unsigned char *out;
    uint64_t count;
} Execution;
static int enter_context(Execution* args,int* changed) {
    void* previous=0;int status=current_context(&previous);
    *changed=0;if(status)return status;
    if(previous!=args->owner){status=push_context(args->owner);if(status)return status;*changed=1;}
    return 0;
}
static int leave_context(int changed,int status) {
    if(changed){void* popped=0;int restored=pop_context(&popped);if(!status)status=restored;}
    return status;
}
int launch_cuda(Execution* args) {
    int changed=0,status=enter_context(args,&changed);if(status)return status;
    void* stream=args->stream;
    for(uint64_t i=0;i<args->inputs;i++){
        status=copy_device(args->destinations[i],args->sources[i],args->sizes[i],stream);if(status)goto done;
    }
    status=launch_graph(args->graph,stream);if(status)goto done;
    if(args->checkpoint_bytes){status=copy_device(args->checkpoint_destination,args->checkpoint_source,args->checkpoint_bytes,stream);if(status)goto done;}
    status=record_event(args->ready,stream);
 done:
    // Raw driver copies do not register their pointer lifetimes with PyTorch.
    // Drain earlier queued work before an error lets the caller release them.
    if(status)sync_stream(stream);
    return leave_context(changed,status);
}
int drain_cuda(Execution* args) {
    int changed=0,status=enter_context(args,&changed);if(status)return status;
    status=sync_event(args->ready);
    return leave_context(changed,status);
}
int finish_cuda(Execution* args) {
    int status=drain_cuda(args);if(status)return status;
    compute(args->metadata,args->offsets,args->starts,args->digest_host,args->context,args->out,args->count);
    return 0;
}
int execute_cuda(Execution* args) {
    int status=launch_cuda(args);if(status)return status;
    return finish_cuda(args);
}

'''


@lru_cache(maxsize=1)
def _library():
    """Compile on first use, retaining the private temporary directory."""
    directory = tempfile.TemporaryDirectory(prefix='rep4gpu-merkle-')
    source = Path(directory.name) / 'merkle.c'
    library = Path(directory.name) / 'merkle.so'
    source.write_text(_SOURCE + _CUDA_SOURCE)
    try:
        result = subprocess.run([os.environ.get('CC', 'cc'), '-shared', '-O3', '-fPIC',
                                 '-Wno-deprecated-declarations', str(source), '-o', str(library), '-lcrypto', '-ldl', '-pthread'],
                                capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f'native Merkle compilation failed:\n{result.stderr}')
        loaded = ct.CDLL(str(library))
    except OSError as error:
        directory.cleanup()
        raise RuntimeError(f'native Merkle compiler or OpenSSL library unavailable: {error}') from error
    except Exception:
        directory.cleanup()
        raise
    loaded.compute.argtypes = [ct.c_char_p, ct.POINTER(ct.c_uint64), ct.POINTER(ct.c_uint64),
                               ct.c_char_p, ct.c_char_p, ct.c_void_p, ct.c_uint64]
    loaded.compute.restype = None
    loaded.prepare_cuda.argtypes = [ct.c_int, ct.POINTER(ct.c_void_p)]
    loaded.prepare_cuda.restype = ct.c_int
    loaded.execute_cuda.argtypes = [ct.POINTER(_ExecutionArgs)]
    loaded.execute_cuda.restype = ct.c_int
    for name in ('launch_cuda', 'finish_cuda', 'drain_cuda'):
        function = getattr(loaded, name)
        function.argtypes = [ct.POINTER(_ExecutionArgs)]
        function.restype = ct.c_int
    return loaded, directory


class PreparedNativeMerkle:
    """Prepare immutable metadata for a single native finalization call.

    Construction requires a C compiler and OpenSSL headers/library. Importing
    this module does not compile or load native code. Each ``compute`` owns its
    output buffer, so prepared metadata can be shared between callers.
    """
    def __init__(self, metadata: Sequence[bytes], counts: Sequence[int]):
        if not metadata or len(metadata) != len(counts):
            raise ValueError('expected nonempty states and one count per state')
        if any(not isinstance(value, bytes) for value in metadata):
            raise TypeError('metadata entries must be bytes')
        if any(type(count) is not int or count < 0 for count in counts):
            raise ValueError('tensor counts must be nonnegative integers')
        self.count = len(metadata)
        offsets, starts = [0], [0]
        for value, count in zip(metadata, counts):
            offsets.append(offsets[-1] + len(value))
            starts.append(starts[-1] + count * 32)
        if max(offsets[-1], starts[-1], self.count) >= 1 << 64:
            raise ValueError('metadata, tensor counts and state count must fit uint64')
        self.digest_bytes = starts[-1]
        self.metadata = b''.join(metadata)
        self.offsets = (ct.c_uint64 * len(offsets))(*offsets)
        self.starts = (ct.c_uint64 * len(starts))(*starts)
        self.widths = [self.count]
        while self.widths[-1] > 1:
            self.widths.append((self.widths[-1] + 1) // 2)
        self.output_bytes = (sum(self.widths) + 1) * 32
        self._compute = _library()[0].compute

    def prepare_cuda(self, device: int, graph_exec: int, ready_event: int,
                     digest_host_ptr: int, checkpoint_src: int, checkpoint_bytes: int,
                     input_destinations: Sequence[tuple[int, int]]):
        """Prepare an internal graph executor; callers retain all pointer owners.

        The graph must copy its completed tensor digests into ``digest_host_ptr``.
        Inputs, graph execution, checkpoint copy, and completion event use the
        supplied execution stream. Each helper serializes its calls; helpers sharing
        an event require caller serialization.
        """
        return _PreparedCudaExecution(self, device, graph_exec, ready_event,
                                      digest_host_ptr, checkpoint_src, checkpoint_bytes,
                                      input_destinations)

    def compute(self, packed: bytes, context: bytes) -> tuple[MerkleTree, tuple[bytes, ...]]:
        """Finalize and decode every level into the ordinary Python tree."""
        return self.decode(self.compute_packed(packed, context))

    def compute_packed(self, packed: bytes, context: bytes) -> bytes:
        """Return immutable completed levels and root without parsing hashes."""
        if not isinstance(packed, bytes) or len(packed) != self.digest_bytes:
            raise ValueError('packed tensor digests must contain exactly the prepared number of bytes')
        if not isinstance(context, bytes) or len(context) != 32:
            raise ValueError('context must contain exactly 32 bytes')
        output = ct.create_string_buffer(self.output_bytes)
        self._compute(self.metadata, self.offsets, self.starts, packed, context, output, self.count)
        return output.raw

    def decode(self, raw: bytes) -> tuple[MerkleTree, tuple[bytes, ...]]:
        """Decode completed native output into ordinary retained tree levels."""
        if not isinstance(raw, bytes) or len(raw) != self.output_bytes:
            raise ValueError('packed tree must contain exactly the prepared number of bytes')
        levels, offset = [], 0
        for width in self.widths:
            levels.append(tuple(raw[index:index + 32] for index in range(offset, offset + width * 32, 32)))
            offset += width * 32
        tree = MerkleTree.__new__(MerkleTree)
        tree._levels = levels
        tree._root = raw[offset:offset + 32]
        return tree, levels[0]


@lru_cache(None)
def _cuda_context(device):
    context = ct.c_void_p()
    status = _library()[0].prepare_cuda(device, ct.byref(context))
    if status:
        raise RuntimeError(f'native CUDA preparation failed with driver status {status}')
    return context


class _ExecutionArgs(ct.Structure):
    _fields_ = [(name, ct.c_void_p) for name in ('owner', 'graph', 'ready', 'stream')] + [
        (name, ct.POINTER(ct.c_uint64)) for name in ('destinations', 'sizes', 'sources')] + [
        (name, ct.c_uint64) for name in ('inputs', 'checkpoint_source', 'checkpoint_destination', 'checkpoint_bytes')] + [
        ('digest_host', ct.c_void_p), ('metadata', ct.c_char_p),
        ('offsets', ct.POINTER(ct.c_uint64)), ('starts', ct.POINTER(ct.c_uint64)),
        ('context', ct.c_char_p), ('out', ct.c_void_p), ('count', ct.c_uint64)]


class _PreparedCudaExecution:
    def __init__(self, backend, device, graph, ready, digest_host, checkpoint_source,
                 checkpoint_bytes, destinations):
        self.backend = backend
        self.count = len(destinations)
        self.destinations = (ct.c_uint64 * self.count)(*[pointer for pointer, _ in destinations])
        self.sizes = (ct.c_uint64 * self.count)(*[size for _, size in destinations])
        self.sources = (ct.c_uint64 * self.count)()
        self.output = ct.create_string_buffer(backend.output_bytes)
        self.args = _ExecutionArgs(
            _cuda_context(device), graph, ready, None,
            self.destinations, self.sizes, self.sources, self.count,
            checkpoint_source, 0, checkpoint_bytes, digest_host, backend.metadata,
            backend.offsets, backend.starts, bytes(32), ct.addressof(self.output), backend.count)
        self._args_pointer = ct.pointer(self.args)
        library = _library()[0]
        self._execute = library.execute_cuda
        self._launch = library.launch_cuda
        self._finish = library.finish_cuda
        self._drain = library.drain_cuda
        self._lock = threading.Lock()

    def execute(self, input_src_ptrs: Sequence[int], dest_checkpoint_ptr: int,
                stream_ptr: int, context: bytes) -> bytes:
        """Execute trusted CUDA pointers and return completed immutable hashes."""
        if len(input_src_ptrs) != self.count:
            raise ValueError('source pointer count must match prepared inputs')
        if not isinstance(context, bytes) or len(context) != 32:
            raise ValueError('context must contain exactly 32 bytes')
        with self._lock:
            for index, pointer in enumerate(input_src_ptrs):
                self.sources[index] = pointer
            self.args.checkpoint_destination = dest_checkpoint_ptr
            self.args.stream = stream_ptr
            self.args.context = context
            status = self._execute(self._args_pointer)
            if status:
                raise RuntimeError(f'native CUDA execution failed with driver status {status}')
            return self.output.raw


    def execute_with_prepare(self, input_src_ptrs: Sequence[int], dest_checkpoint_ptr: int,
                             stream_ptr: int, prepare: Callable[[], bytes]) -> bytes:
        """Launch, run ``prepare`` during CUDA work, then finish using its context.

        ``prepare`` returns 32 context bytes and must not recursively execute this
        helper. If it raises or returns invalid bytes, completion is drained before
        the error escapes so pointer owners and the shared graph remain safe.
        """
        if len(input_src_ptrs) != self.count:
            raise ValueError('source pointer count must match prepared inputs')
        with self._lock:
            for index, pointer in enumerate(input_src_ptrs):
                self.sources[index] = pointer
            self.args.checkpoint_destination = dest_checkpoint_ptr
            self.args.stream = stream_ptr
            status = self._launch(self._args_pointer)
            if status:
                raise RuntimeError(f'native CUDA launch failed with driver status {status}')
            try:
                context = prepare()
                if not isinstance(context, bytes) or len(context) != 32:
                    raise ValueError('context must contain exactly 32 bytes')
                self.args.context = context
            except BaseException as error:
                status = self._drain(self._args_pointer)
                if status:
                    raise RuntimeError(f'native CUDA drain failed with driver status {status}') from error
                raise
            status = self._finish(self._args_pointer)
            if status:
                raise RuntimeError(f'native CUDA finish failed with driver status {status}')
            return self.output.raw
