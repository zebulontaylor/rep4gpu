"""Measure the trusted-dealer simulator, including separate preprocessing costs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from rep4gpu.spdz import AuthenticatedOnline, PRIME, TrustedDealer, field_matmul


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", nargs=3, type=int, default=[1, 256, 256], metavar=("M", "K", "N"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--parties", type=int, default=4)
    parser.add_argument("--mac-keys", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(*args.shape, args.repeats) < 1:
        parser.error("shape dimensions and repeats must be positive")
    device = torch.device(args.device)
    m, k, n = args.shape
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    def timed(fn):
        sync()
        start = time.perf_counter()
        result = fn()
        sync()
        return result, (time.perf_counter() - start) * 1000
    a = torch.arange(m * k, dtype=torch.int64, device=device).reshape(m, k) % 127
    b = torch.arange(k * n, dtype=torch.int64, device=device).reshape(k, n) % 127
    af, bf = a.float(), b.float()
    expected = field_matmul(a, b)
    dealer, setup_ms = timed(lambda: TrustedDealer(args.parties, args.mac_keys, device))
    online_times, prep_times, clear_times, total_times = [], [], [], []
    # A full discarded run warms kernels and consumes its triple.
    for iteration in range(args.repeats + 1):
        prepared, prep_ms = timed(lambda: (dealer.share(a), dealer.share(b), dealer.triple(m, k, n)))
        x, y, triple = prepared
        online = AuthenticatedOnline(dealer)
        output, online_ms = timed(lambda: online.open(online.matmul(x, y, triple)))
        if not torch.equal(output, expected):
            raise RuntimeError("authenticated output differs from exact field product")
        _, clear_ms = timed(lambda: af @ bf)
        if iteration:
            prep_times.append(prep_ms)
            online_times.append(online_ms)
            clear_times.append(clear_ms)
            total_times.append(prep_ms + online_ms)
    report = {
        "implementation": "centralized trusted-dealer prime-field online simulator",
        "full_malicious_majority_protocol": False,
        "device": str(device), "shape": args.shape, "parties": args.parties,
        "prime": PRIME, "mac_keys": args.mac_keys, "repeats": args.repeats,
        "setup_ms": setup_ms,
        "preprocessing_ms": statistics.median(prep_times),
        "online_with_output_check_ms": statistics.median(online_times),
        "preprocessing_plus_online_ms": statistics.median(total_times),
        "clear_fp32_matmul_ms": statistics.median(clear_times),
        "online_over_clear_fp32": statistics.median(online_times) / statistics.median(clear_times),
        "opened_elements": online.opened_elements,
        "mac_residual_contributions": online.mac_residual_elements,
        "logical_field_contribution_bytes": 4 * (args.parties * online.opened_elements + online.mac_residual_elements),
        "network_ms": None,
        "correct": True,
    }
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
