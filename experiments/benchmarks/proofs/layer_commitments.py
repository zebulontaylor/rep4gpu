"""Measure completed layer commitments around exact integer matrix inference.

Run from the repository root with ``PYTHONPATH=src .venv/bin/python -m
experiments.benchmarks.proofs.layer_commitments --device cuda --batch 1``.
Use ``--batch 128`` for a prefill-shaped workload. Results include host staging,
checkpoint snapshots, tensor fingerprinting, and final Merkle tree construction.
Weights are fixed public parameters captured by each layer callback.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch

from rep4gpu.proofs.layer_commitments import CommitmentRuntime, Layer, verify_layer


def benchmark(args):
    torch.manual_seed(17)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        from rep4gpu.gpu.matmul.kernels import ring_matmul
        multiply = ring_matmul
    else:
        multiply = torch.matmul
    weights = [torch.randint(-8, 8, (args.width, args.width), dtype=torch.int64, device=device) for _ in range(args.layers)]
    inputs = {"x": torch.randint(-8, 8, (args.batch, args.width), dtype=torch.int64, device=device)}

    def make_layer(weight):
        return lambda state: {"x": multiply(state["x"], weight) + state["x"]}

    layers = [Layer(f"residual_{index}", make_layer(weight), preserves_inputs=True) for index, weight in enumerate(weights)]
    runtime = CommitmentRuntime(layers, checkpoint_interval=args.checkpoint_interval, hash_workers=args.hash_workers, max_pending=args.max_pending)

    def plain():
        state = inputs
        for layer in layers:
            state = layer.forward(state)
        return state

    active_runtime = runtime
    if args.prepared:
        if device.type != "cuda":
            raise ValueError("--prepared requires CUDA")
        active_runtime = runtime.prepare(inputs)
        static_inputs = {name: value.clone() for name, value in inputs.items()}
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                state = static_inputs
                for layer in layers:
                    state = layer.forward(state)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            state = static_inputs
            for layer in layers:
                state = layer.forward(state)
            graph_output = state

        def plain():
            for name, value in inputs.items():
                static_inputs[name].copy_(value)
            graph.replay()
            return graph_output

    request = 0

    def committed():
        nonlocal request
        request += 1
        return active_runtime.commit(inputs, model_id="integer-residual-matmul", run_id=f"benchmark-{request}")

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def measure(function, *, completed=False):
        synchronize()
        start = time.perf_counter()
        result = function()
        if not completed:
            synchronize()
        return time.perf_counter() - start, result

    for _ in range(args.warmups):
        measure(plain)
        measure(committed, completed=True)
    timings = {"plain": [], "committed": []}
    execution = None
    for repeat in range(args.repeats):
        order = ("plain", "committed") if repeat % 2 == 0 else ("committed", "plain")
        for name in order:
            elapsed, result = measure(plain if name == "plain" else committed, completed=name == "committed")
            timings[name].append(elapsed)
            if name == "committed":
                execution = result
    expected = plain()["x"].cpu()
    actual = verify_layer(execution.commitment, execution.open(args.layers), expected_boundary=args.layers)["x"]
    if not torch.equal(actual, expected):
        raise ValueError("committed output differs from plain inference")
    plain_median = statistics.median(timings["plain"])
    committed_median = statistics.median(timings["committed"])
    overhead = (committed_median / plain_median - 1) * 100
    paired = [(committed / plain - 1) * 100 for plain, committed in zip(timings["plain"], timings["committed"])]
    return {
        "workload": "exact int64 residual matrix multiplication modulo 2**64",
        "device": str(device), "prepared": args.prepared,
        "preserves_inputs": True,
        "native_execution": bool(getattr(active_runtime, "_native_executor", None)),
        "native_finalizer": bool(getattr(active_runtime, "_native_finalizer", None)),
        "batched_hashes": bool(getattr(active_runtime, "_batch_hashes", False)),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "run_id_policy": "unique per committed inference",
        "state_digest": "BLAKE3 tensor fingerprints; CUDA hashes on device",
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_capability": list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None,
        "source_sha256": {
            str(path.relative_to(Path(__file__).resolve().parents[3])): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__).resolve(),
                Path(__file__).resolve().parents[3] / "src/rep4gpu/proofs/layer_commitments.py",
                Path(__file__).resolve().parents[3] / "src/rep4gpu/proofs/state_digest.py",
                Path(__file__).resolve().parents[3] / "src/rep4gpu/proofs/merkle.py",
                Path(__file__).resolve().parents[3] / "src/rep4gpu/proofs/native_merkle.py",
                Path(__file__).resolve().parents[3] / "src/rep4gpu/gpu/crypto/gpu_hash.py",
            )
        },
        "batch": args.batch, "width": args.width, "layers": args.layers,
        "threads": args.threads, "checkpoint_interval": args.checkpoint_interval,
        "hash_workers": args.hash_workers, "max_pending": args.max_pending,
        "warmups": args.warmups, "repeats": args.repeats,
        "plain_seconds": timings["plain"], "committed_seconds": timings["committed"],
        "plain_median_seconds": plain_median, "committed_median_seconds": committed_median,
        "overhead_percent": overhead, "under_two_percent": overhead < 2,
        "under_three_percent": overhead < 3,
        "paired_overhead_percent": paired,
        "paired_median_overhead_percent": statistics.median(paired),
        "paired_overhead_quartiles_percent": statistics.quantiles(paired, n=4) if len(paired)>1 else None,
        "output_matches": True, "completed_root_hex": execution.commitment.root.hex(),
        "timing_scope": "inference plus checkpoints, tensor hashing, digest transfer, and completed Merkle root; excludes preparation and replay verification",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prepared", action="store_true")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=8)
    parser.add_argument("--hash-workers", type=int, default=2)
    parser.add_argument("--max-pending", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("experiments/results/proofs/layer_commitments.json"))
    args = parser.parse_args()
    for name in ("batch", "width", "layers", "repeats", "threads", "checkpoint_interval", "hash_workers", "max_pending"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if args.warmups < 0:
        parser.error("warmups must be nonnegative")
    result = benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
