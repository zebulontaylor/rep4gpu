#!/usr/bin/env python3
"""Build and run the CPU SPAKE2 checks, retaining logs and source hashes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "native/matrix_vope"
OUTPUT = ROOT / "work/matrix_vope/spake2_eq"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    files = [SOURCE / name for name in (
        "spake2_eq.h", "spake2_eq.cpp", "spake2_rfc9382_vectors.h",
        "test_spake2_eq.cpp", "spake2_eq_production_smoke.cpp", "run_spake2_eq_checks.py")]
    report = {"source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, "checks": []}
    env = os.environ.copy()
    env.pop("EMP_TEST_MODE", None)
    env.pop("SIV_SPAKE2_EQ_TEST_MODE", None)
    status = 0

    def run(name, command, *, expected=0, extra_env=None):
        nonlocal status
        current_env = env.copy()
        current_env.update(extra_env or {})
        log = OUTPUT / (name + ".log")
        with log.open("w") as output:
            try:
                result = subprocess.run(command, cwd=ROOT, env=current_env, stdout=output, stderr=subprocess.STDOUT, timeout=120)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 124
        report["checks"].append({"name": name, "command": command, "exit_code": code, "expected_exit_code": expected, "log": str(log.relative_to(ROOT))})
        print(f"{name}: {'PASS' if code == expected else 'FAIL'} ({code})", flush=True)
        if code != expected and status == 0:
            status = code if code > 0 else 1
        return code == expected

    try:
        common = ["taskset", "-c", "6,7,14,15", "clang++", "-std=c++20", "-Wall", "-Wextra", "-Werror", "-pthread"]
        for variant, flags in [("normal", ["-O2"]), ("ubsan", ["-O1", "-g", "-fsanitize=undefined", "-fno-sanitize-recover=all"])]:
            binary = OUTPUT / ("test_spake2_eq_" + variant)
            command = common + flags + ["-DSIV_SPAKE2_EQ_TESTING", str(SOURCE / "spake2_eq.cpp"), str(SOURCE / "test_spake2_eq.cpp"), "-lcrypto", "-o", str(binary)]
            if run("build_" + variant, command):
                run("test_" + variant, ["taskset", "-c", "6,7,14,15", str(binary), "--test-mode"])
                run("refuse_implicit_" + variant, [str(binary)], expected=2)
        binary = OUTPUT / "spake2_eq_production_smoke"
        if run("build_production", common + ["-O2", str(SOURCE / "spake2_eq.cpp"), str(SOURCE / "spake2_eq_production_smoke.cpp"), "-lcrypto", "-o", str(binary)]):
            for value in ["1", "0", ""]:
                run("production_env_" + (value or "empty"), ["taskset", "-c", "6,7,14,15", str(binary)], extra_env={"EMP_TEST_MODE": value})
    finally:
        report["exit_code"] = status
        (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return status


if __name__ == "__main__":
    sys.exit(main())
