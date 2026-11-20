"""Compare dense and sparse native inference for the bounded integer classifier.

Sparse indices are public only to this native baseline. A private MPC runtime
must keep the learned indices and coefficients secret. All timings use resident
features and weights; packing and compilation are outside inference timing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import tempfile
import time

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_scores(X, INDEX, WEIGHT, BIAS, HEAD, OUT, B2: tl.constexpr,
                   FEATURES: tl.constexpr, HIDDEN: tl.constexpr, NONZEROS: tl.constexpr,
                   H: tl.constexpr, K: tl.constexpr, LABELS: tl.constexpr):
    row = tl.program_id(0)
    h = tl.arange(0, H)
    k = tl.arange(0, K)
    valid = (h[:, None] < HIDDEN) & (k[None, :] < NONZEROS)
    index = tl.load(INDEX + h[:, None] * NONZEROS + k[None, :], valid, 0)
    weight = tl.load(WEIGHT + h[:, None] * NONZEROS + k[None, :], valid, 0)
    value = tl.load(X + row * FEATURES + index, valid, 0).to(tl.int32)
    first = tl.sum(value * weight, 1) + tl.load(BIAS + h, h < HIDDEN, 0)
    score = tl.sum(first * first * tl.load(HEAD + h, h < HIDDEN, 0), 0) + B2
    tl.store(OUT + row, (score >= 0).to(tl.int32) if LABELS else score)


def pack_sparse(w1, b1, w2, b2):
    """Create padded per-hidden sparse columns and certify int32 evaluation."""
    w1, b1, w2, b2 = [np.asarray(v, dtype=np.int64) for v in (w1, b1, w2, b2)]
    if w1.ndim != 2 or not all(w1.shape) or b1.shape != (w1.shape[1],) or w2.shape != (w1.shape[1], 1) or b2.shape != (1,):
        raise ValueError('Expected w1[F,H], b1[H], w2[H,1], b2[1]')
    nonzeros = max(1, int(np.count_nonzero(w1, axis=0).max()))
    index = np.zeros((w1.shape[1], nonzeros), dtype=np.int32)
    weight = np.zeros_like(index)
    for column in range(w1.shape[1]):
        ids = np.flatnonzero(w1[:, column])
        index[column, :len(ids)] = ids
        weight[column, :len(ids)] = w1[ids, column]
    hidden_bound = np.abs(w1).sum(axis=0) + np.abs(b1)
    bound = sum(int(a)**2 * abs(int(b)) for a, b in zip(hidden_bound, w2.reshape(-1))) + abs(int(b2[0]))
    if max(int(hidden_bound.max())**2, bound) >= 2**31:
        raise ValueError('Sparse int32 evaluation can overflow for binary inputs')
    return dict(index=index, weight=weight, bias=b1.astype(np.int32),
                head=w2.reshape(-1).astype(np.int32), b2=int(b2[0]), bound=bound)


def sparse_cuda(x, packed, out=None, *, labels=False):
    """Evaluate exact integer scores; binary x and packed int32 arrays are resident."""
    if x.dtype != torch.uint8 or x.ndim != 2 or not x.is_cuda or not x.is_contiguous():
        raise ValueError('x must be a contiguous uint8 CUDA matrix')
    hidden, nonzeros = packed['index'].shape
    if out is None:
        out = torch.empty(x.shape[0], dtype=torch.int32, device=x.device)
    _sparse_scores[(x.shape[0],)](x, packed['index'], packed['weight'], packed['bias'], packed['head'], out,
                                packed['b2'], x.shape[1], hidden, nonzeros,
                                triton.next_power_of_2(hidden), triton.next_power_of_2(nonzeros), labels, num_warps=4)
    return out


_CPP = r'''
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>
#include <omp.h>
__attribute__((noinline)) void infer(const uint8_t* x, const int32_t* index, const int32_t* weight,
 const int32_t* bias, const int32_t* head, int32_t b2, int32_t* out,
 int rows, int features, int hidden, int nonzeros, int threads, bool labels) {
 #pragma omp parallel for num_threads(threads) if(threads>1)
 for(int row=0;row<rows;++row) {
  int32_t score=b2;
  for(int h=0;h<hidden;++h) {
   int32_t first=bias[h];
   for(int k=0;k<nonzeros;++k) first+=x[row*features+index[h*nonzeros+k]]*weight[h*nonzeros+k];
   score+=first*first*head[h];
  }
  out[row]=labels ? score>=0 : score;
 }
}
template<class T> std::vector<T> read(std::ifstream& f,size_t count) {
 std::vector<T> v(count);f.read(reinterpret_cast<char*>(v.data()),count*sizeof(T));
 if(!f) throw std::runtime_error("truncated fixture");return v;
}
int main(int argc,char** argv) {
 if(argc!=5)return 2;
 std::ifstream f(argv[1],std::ios::binary);auto shape=read<int32_t>(f,5);
 int rows=shape[0],features=shape[1],hidden=shape[2],nonzeros=shape[3],b2=shape[4];
 auto x=read<uint8_t>(f,size_t(rows)*features);auto index=read<int32_t>(f,hidden*nonzeros);
 auto weight=read<int32_t>(f,hidden*nonzeros);auto bias=read<int32_t>(f,hidden);auto head=read<int32_t>(f,hidden);
 std::vector<int32_t> out(rows);int threads=std::stoi(argv[3]),repeats=std::stoi(argv[4]);
 std::vector<double> samples;
 for(int r=-3;r<repeats;++r) {
  auto start=std::chrono::steady_clock::now();
  infer(x.data(),index.data(),weight.data(),bias.data(),head.data(),b2,out.data(),rows,features,hidden,nonzeros,threads,true);
  auto end=std::chrono::steady_clock::now();
  if(r>=0)samples.push_back(std::chrono::duration<double,std::milli>(end-start).count());
 }
 infer(x.data(),index.data(),weight.data(),bias.data(),head.data(),b2,out.data(),rows,features,hidden,nonzeros,threads,false);
 std::ofstream result(argv[2],std::ios::binary);result.write(reinterpret_cast<char*>(out.data()),rows*sizeof(int32_t));
 std::sort(samples.begin(),samples.end());std::cout<<samples[samples.size()/2]<<"\n";
}
'''


def cpp_baseline(x, packed, repeats, threads):
    """Compile a native sparse loop and measure each requested CPU thread count."""
    rows, features = x.shape
    hidden, nonzeros = packed['index'].shape
    with tempfile.TemporaryDirectory(prefix='native-sparse-') as folder:
        folder = Path(folder)
        source, binary, fixture, result = [folder / name for name in ('native.cpp', 'native', 'fixture.bin', 'scores.bin')]
        source.write_text(_CPP)
        subprocess.run(['g++', '-O3', '-march=native', '-fopenmp', '-std=c++17', str(source), '-o', str(binary)], check=True, capture_output=True)
        with fixture.open('wb') as output:
            np.array([rows, features, hidden, nonzeros, packed['b2']], dtype='<i4').tofile(output)
            x.astype(np.uint8).tofile(output)
            for key in ('index', 'weight', 'bias', 'head'):
                packed[key].astype('<i4').tofile(output)
        times, scores = {}, None
        for count in threads:
            measured = subprocess.run([str(binary), str(fixture), str(result), str(count), str(repeats)], check=True, capture_output=True, text=True)
            times[f'cpp_sparse_int32_threads{count}'] = float(measured.stdout.strip())
            current = np.fromfile(result, dtype='<i4')
            if scores is not None and not np.array_equal(scores, current):
                raise AssertionError('CPU thread counts disagree')
            scores = current
    return times, scores


def timed_cuda(fn, repeats, graph_calls):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_calls):
            fn()
    samples = []
    for _ in range(7):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            graph.replay()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000 / (repeats * graph_calls))
    return statistics.median(samples)


def native_sparse_baselines(x, weights, device='cuda', *, repeats=20, graph_calls=16, threads=(1, 8)):
    """Return milliseconds for exact sparse native label inference.

    Inputs are plaintext NumPy features and integer parameter arrays. The
    function validates full scores against a dense integer oracle before timing
    the fused label kernel. Packing, model loading, and compilation are untimed.
    """
    x = np.asarray(x)
    arrays = {key: np.asarray(weights[key], dtype=np.int64) for key in ('w1', 'b1', 'w2', 'b2')}
    if x.ndim != 2 or not np.isin(x, [0, 1]).all():
        raise ValueError('Sparse baseline requires a binary feature matrix')
    packed = pack_sparse(**arrays)
    oracle = (((x.astype(np.int64) @ arrays['w1'] + arrays['b1'])**2) @ arrays['w2'] + arrays['b2']).reshape(-1)
    if torch.device(device).type == 'cpu':
        times, scores = cpp_baseline(x, packed, repeats, threads)
        if not np.array_equal(scores, oracle):
            raise AssertionError('Sparse CPU scores differ from integer oracle')
        return times
    if torch.device(device).type != 'cuda':
        raise ValueError('Sparse baseline supports cpu or cuda')
    gpu = {key: torch.as_tensor(value, device=device) if isinstance(value, np.ndarray) else value
           for key, value in packed.items()}
    gx = torch.as_tensor(x, dtype=torch.uint8, device=device)
    out = torch.empty(len(x), dtype=torch.int32, device=device)
    if not np.array_equal(sparse_cuda(gx, gpu, out).cpu().numpy(), oracle):
        raise AssertionError('Sparse CUDA scores differ from integer oracle')
    if not np.array_equal(sparse_cuda(gx, gpu, out, labels=True).cpu().numpy(), oracle >= 0):
        raise AssertionError('Sparse CUDA labels differ from integer oracle')
    return {'triton_sparse_int32_graph': timed_cuda(lambda: sparse_cuda(gx, gpu, out, labels=True), repeats, graph_calls)}


def run(checkpoint, batch=8192, repeats=20, graph_calls=16, threads=(1, 8), cuda=True):
    with np.load(checkpoint, allow_pickle=False) as model:
        arrays = {name: model[name].astype(np.int64) for name in ('w1', 'b1', 'w2', 'b2')}
        original = model['x_test']
    x = original[np.arange(batch) % len(original)]
    if not np.isin(x, [0, 1]).all():
        raise ValueError('This baseline requires binary token-presence features')
    packed = pack_sparse(**arrays)
    oracle = (((x.astype(np.int64) @ arrays['w1'] + arrays['b1'])**2) @ arrays['w2'] + arrays['b2']).reshape(-1)
    cpu_times, cpu_scores = cpp_baseline(x, packed, repeats, threads)
    if not np.array_equal(cpu_scores, oracle):
        raise AssertionError('Sparse native CPU scores differ from integer oracle')
    # Dense CPU comparison uses the same model and thread counts as sparse CPU.
    for count in threads:
        torch.set_num_threads(count)
        a = torch.from_numpy(x.astype(np.float32))
        w = {k: torch.from_numpy(v.astype(np.float32)) for k, v in arrays.items()}
        def dense_cpu():
            return (a @ w['w1'] + w['b1']).square() @ w['w2'] + w['b2']
        for _ in range(5):
            dense_cpu()
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            out = dense_cpu() >= 0
            samples.append((time.perf_counter() - start) * 1000)
        if not np.array_equal(dense_cpu().numpy().reshape(-1), oracle):
            raise AssertionError('Dense CPU scores differ from integer oracle')
        cpu_times[f'torch_dense_float32_threads{count}'] = statistics.median(samples)
    cuda_times, agreements = {}, {}
    if cuda:
        device_packed = {k: torch.as_tensor(v, device='cuda') if isinstance(v, np.ndarray) else v for k, v in packed.items()}
        gx = torch.as_tensor(x, dtype=torch.uint8, device='cuda')
        out = torch.empty(batch, dtype=torch.int32, device='cuda')
        actual = sparse_cuda(gx, device_packed, out).cpu().numpy()
        if not np.array_equal(actual, oracle):
            raise AssertionError('Sparse CUDA scores differ from integer oracle')
        cuda_times['triton_sparse_int32_graph'] = timed_cuda(lambda: sparse_cuda(gx, device_packed, out, labels=True), repeats, graph_calls)
        for dtype in (torch.float32, torch.float16):
            a = gx.to(dtype)
            w = {k: torch.as_tensor(v, dtype=dtype, device='cuda') for k, v in arrays.items()}
            def dense_gpu():
                return (a @ w['w1'] + w['b1']).square() @ w['w2'] + w['b2']
            score = dense_gpu().cpu().numpy().reshape(-1)
            name = str(dtype).split('.')[-1]
            agreements[name] = dict(exact_scores=bool(np.array_equal(score, oracle)),
                                    exact_labels=bool(np.array_equal(score >= 0, oracle >= 0)))
            cuda_times[f'torch_dense_{name}_graph'] = timed_cuda(lambda: dense_gpu() >= 0, repeats, graph_calls)
    report = dict(batch=batch, features=x.shape[1], hidden=arrays['w1'].shape[1],
                  nonzeros_per_hidden=int(packed['index'].shape[1]),
                  checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  cpp_source_sha256=hashlib.sha256(_CPP.encode()).hexdigest(),
                  torch_version=torch.__version__, cuda_device=torch.cuda.get_device_name() if cuda else None,
                  repeats=repeats, graph_calls=graph_calls, cpu_ms=cpu_times, cuda_ms=cuda_times,
                  dense_cuda_agreement=agreements, sparse_cpu_exact=True, sparse_cuda_exact=True if cuda else None,
                  fastest_cpu_ms=min(cpu_times.values()), fastest_cuda_ms=min(
                      value for name, value in cuda_times.items() if name == 'triton_sparse_int32_graph'
                      or agreements[name.removeprefix('torch_dense_').removesuffix('_graph')]['exact_labels']) if cuda_times else None,
                  scope='Native plaintext only. Learned sparse indices are available to this baseline; MPC must keep them private. Packing, compilation, and model loading excluded. CUDA graphs contain multiple calls to measure saturated resident throughput.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('experiments/results/examples/ring16_text_classifier/model_pruned.npz'))
    parser.add_argument('--batch', type=int, default=8192)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--graph-calls', type=int, default=16)
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 8])
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('experiments/results/inference/sparse_classifier_native.json'))
    args = parser.parse_args()
    if min(args.batch, args.repeats, args.graph_calls, *args.threads) < 1:
        parser.error('Batch, repeats, graph calls, and thread counts must be positive')
    result = run(args.checkpoint, args.batch, args.repeats, args.graph_calls, args.threads, not args.cpu_only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
