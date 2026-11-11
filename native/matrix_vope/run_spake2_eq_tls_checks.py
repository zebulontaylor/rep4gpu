#!/usr/bin/env python3
"""Exercise the production SPAKE2 adapter across two mutually authenticated processes."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "native/matrix_vope"
BACKEND = ROOT / "work/vole-labs"
OUT = ROOT / "work/matrix_vope/spake2_eq_tls"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = ("spake2_eq.h", "spake2_eq.cpp", "spake2_rfc9382_vectors.h", "spake2_eq_tls.h", "spake2_eq_tls.cpp", "tls_io.h", "tls_io.cpp", "test_spake2_eq_tls.cpp", "run_spake2_eq_tls_checks.py")
    report = {"source_sha256": {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in names}, "cases": []}
    lock = json.loads((HERE / "source-lock.json").read_text())
    for dependency in lock["dependencies"]:
        path = BACKEND / dependency["path"]
        if subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip() != dependency["commit"]:
            raise RuntimeError("EMP dependency commit mismatch")
        if subprocess.check_output(["git", "-C", str(path), "diff", "HEAD", "--name-only"], text=True).strip():
            raise RuntimeError("EMP dependency tracked sources changed")
    for name, expected in lock["source_files_sha256"].items():
        if hashlib.sha256((BACKEND / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("EMP pinned source differs: " + name)
    report["dependencies"] = lock["dependencies"]
    binary = OUT / "test_spake2_eq_tls"
    command = ["taskset", "-c", "6,7,14,15", "c++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Wno-ignored-attributes", "-march=native", "-maes", "-pthread", "-I" + str(BACKEND / "thirdparty/emp-tool"), "-I" + str(BACKEND / "build/emp-tool")]
    command += [str(HERE / n) for n in ("spake2_eq.cpp", "tls_io.cpp", "spake2_eq_tls.cpp", "test_spake2_eq_tls.cpp")]
    command += [str(BACKEND / "build/emp-tool/libemp-tool.a"), "-lssl", "-lcrypto", "-o", str(binary)]
    report["command"] = command
    with (OUT / "build.log").open("w") as log:
        build = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    report["build_exit_code"] = build.returncode
    status = build.returncode
    if status:
        (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return status
    report["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    env = os.environ.copy()
    env.pop("EMP_TEST_MODE", None)
    env.pop("SIV_SPAKE2_EQ_TEST_MODE", None)
    with tempfile.TemporaryDirectory(prefix="siv-spake2-tls-") as temp:
        certdir = Path(temp)
        with (OUT / "certificate_setup.log").open("w") as log:
            def openssl(*args):
                subprocess.run(["openssl", *args], check=True, cwd=certdir, stdout=log, stderr=subprocess.STDOUT)
            openssl("req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes", "-days", "1", "-subj", "/CN=SPAKE2 regression CA", "-keyout", "ca.key", "-out", "ca.crt")
            for party in range(2):
                stem = "party" + str(party)
                openssl("req", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes", "-subj", "/CN=" + stem, "-keyout", stem + ".key", "-out", stem + ".csr")
                (certdir / (stem + ".ext")).write_text("subjectAltName=DNS:" + stem + "\nextendedKeyUsage=serverAuth,clientAuth\n")
                openssl("x509", "-req", "-in", stem + ".csr", "-CA", "ca.crt", "-CAkey", "ca.key", "-CAcreateserial", "-days", "1", "-extfile", stem + ".ext", "-out", stem + ".crt")
            report["certificate_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in certdir.glob("*.crt")}
        for mode in ("equal", "equal_twice", "unequal", "correct_guess", "wrong_guess", "sid", "configuration", "identities", "withhold", "barrier_binding", "barrier_role", "barrier_status", "emp_test_flag"):
            success = mode in ("equal", "equal_twice", "correct_guess")
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            entries = []
            processes = []
            logs = []
            try:
                for party in range(2):
                    raw = party == 0 and mode in ("withhold", "barrier_binding", "barrier_role", "barrier_status")
                    logfile = OUT / f"{mode}_party{party}.log"
                    log = logfile.open("w")
                    logs.append(log)
                    cmd = ["taskset", "-c", "6,7,14,15", str(binary), str(party), str(port), mode, temp, "success" if success else "failure", "raw" if raw else "adapter"]
                    processes.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env))
                    if party == 0:
                        time.sleep(0.1)
                for party, process in enumerate(processes):
                    try:
                        code = process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                        code = 124
                    logs[party].close()
                    logfile = OUT / f"{mode}_party{party}.log"
                    measurements = None
                    for line in logfile.read_text().splitlines():
                        try:
                            measurements = json.loads(line)
                        except json.JSONDecodeError:
                            pass
                    entries.append({"party": party, "exit_code": code, "log": str(logfile.relative_to(ROOT)), "measurements": measurements})
                    if code and not status:
                        status = code if code > 0 else 1
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
                for log in logs:
                    log.close()
            report["cases"].append({"mode": mode, "expected_success": success, "roles": entries})
            print(mode + ": " + ("PASS" if all(e["exit_code"] == 0 for e in entries) else "FAIL"), flush=True)
    report["exit_code"] = status
    report["temporary_private_keys_removed"] = True
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
