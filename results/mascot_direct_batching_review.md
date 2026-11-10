# Direct MASCOT batching and preprocessing cost

The 1,024-row micro model requires 32,768 scalar matrix-product triples,
2,048 squares, 10,272 secret random field elements, and 8,234 input tuples.
The compiler already combines the two `Matrix.dot` operations into two VM
rounds. It still requests scalar triples; the matrix syntax does not select
packed malicious matrix preprocessing.

## Measured batching experiment

| Requested batch size | Fresh generation | Global logical traffic | Party 0 rounds |
| --- | ---: | ---: | ---: |
| 1,000 | 36.392 s | 35.758 GB | about 2,408 |
| 8,192 | 48.750 s | 37.380 GB | about 1,294 |

The second timing is provisional. Four physical cores were available to the
native workers while an isolated two-job LowGear build used two other cores.
The earlier run used six physical cores. The second run also enabled verbose
batch diagnostics. This comparison establishes the batch sizes, traffic, and
round counts; it does not establish a timing regression from increasing `-b`.
Both runs retain statistical security 128, amplification four, checks, encrypted
channels, and fresh one-use material.

The [experiment record](mascot_direct_capacity1024_batch8192.json) and
[party 0 log](../logs/mascot_direct_capacity1024_batch8192_party0.log) show that
requested batch size 8,192 becomes **1,929** for q256 triples. The generator
limits batches to `floor(1e7 / sizeof(T)^2)`. Its q256 authenticated C++ share
object occupies 72 bytes; this is an object-size cap, not the wire encoding
width. At amplification four, each batch uses
`1929 * 256 * 4 = 1,975,296` OTs per peer before the additional authentication
work.

The cap also interacts with square generation. The square fallback consumes
one scalar triple per square. This run generated 19 batches of 1,929 triples:
36,651 total, versus 34,816 needed for the matrix products and squares. The log
reports **1,835 unused triples**. This overgeneration explains the additional
traffic despite fewer rounds. A quiet comparison of `-b1000` against
`-b1821` would avoid most of that waste: the latter predicts 18 matrix batches
plus two 1,024-triple square refills, or 34,826 triples. This is a rounding
experiment, not a proposed change to security parameters.

## What the profile establishes

Party 0 reports 48.285 seconds in preprocessing and 0.056 seconds in the native
online phase. Preprocessing accounts for 9,485.97 MB of its traffic, versus
6.29 MB online. Its communication timers accumulate 32.512 seconds, including
30.402 seconds in one-to-one exchange. On localhost, that includes waiting for
peers' cryptographic work; it is not a measurement of physical network delay.
Increasing the batch size reduces rounds but leaves the dominant scalar OT
work and bytes.

The printed phase CPU totals cannot yield an ideal-network compute estimate.
`TimerWithComm::operator-` subtracts wall time and communication statistics but
leaves `cpu_timer` unchanged; `operator+=` also omits that timer. The resulting
online CPU total includes preprocessing. Subtracting communication elapsed
time from preprocessing wall time is also invalid because waiting and worker
computation overlap.

For a quiet follow-up, sample all four native PIDs with
`perf record -F 99 -e cycles:u -p PID0,PID1,PID2,PID3 -- sleep 45`, then inspect
`perf report --stdio --sort comm,dso,symbol`. This samples instruction addresses
without collecting user-stack contents. If performance counters are unavailable,
add thread CPU and wall counters around OT extension, output hashing,
correlation arithmetic, amplification, MAC authentication, and sacrifice.
Keep socket waiting and serialization separate. Sum CPU work per role and
report the live critical path separately; neither sum alone proves an ideal
pipeline time. Actual GPU cryptography would need its own measured execution.

## Baseline and amortization check

The [served-batch report](spdz_prime256_served_capacity_costs.json) correctly
divides both the complete measured batch stages and the matched clear-model
batch by the same row count. Each row was actually served from a fresh batch,
with exact integer outputs at all four TLS roles. These are throughput figures,
not single-request latency. Compilation and process/certificate setup are
separate, and the sum of stage maxima is not one end-to-end stopwatch.
The measured preprocessing-inclusive target remains unmet. These results do
not establish secure-token throughput or a full GPU authenticated runtime.
