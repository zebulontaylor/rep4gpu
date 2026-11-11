"""Compare FP127 arithmetic with GMP and compile its VOLE template callers."""
from pathlib import Path
import json
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BACKEND = REPO / "work/vole-labs"
OUT = REPO / "work/matrix_vope/fp127"
OUT.mkdir(parents=True, exist_ok=True)
flags = ["-std=c++20", "-maes", "-msse4.1", "-mpclmul", "-mrdseed",
         "-I" + str(HERE), "-I" + str(BACKEND),
         "-I" + str(BACKEND / "thirdparty/emp-tool"),
         "-I" + str(BACKEND / "thirdparty/emp-ot"),
         "-I" + str(BACKEND / "build/emp-tool")]
link = ["-L" + str(BACKEND / "build/emp-tool"), "-lemp-tool", "-lcrypto",
        "-lgmpxx", "-lgmp", "-pthread"]
results = {}
for name, command in (
    ("arithmetic", ["c++", *flags, "-O2", str(HERE / "test_fp127.cpp"), *link,
                    "-o", str(OUT / "arithmetic")]),
    ("ubsan", ["clang++", *flags, "-O1", "-g", "-fsanitize=undefined",
               "-fno-sanitize-recover=all", str(HERE / "test_fp127.cpp"), *link,
               "-o", str(OUT / "ubsan")]),
    ("compat", ["c++", *flags, "-O0", "-c", str(HERE / "test_fp127_compat.cpp"),
                "-o", str(OUT / "compat.o")]),
):
    with (OUT / (name + "-build.log")).open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    if name != "compat":
        output = subprocess.check_output([str(OUT / name)], text=True)
        (OUT / (name + ".log")).write_text(output)
        results[name] = output.strip()
    else:
        results[name] = "compiled primal, committed, and chosen-input VOLE callers"
(OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
