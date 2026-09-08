# Task 09 Report

## 1. Problem and system contract

- Target GPU and software versions:
- Dataset N/D/Q, dtype, metric, and provenance:
- Exact and ANN result ordering:
- Timing and memory boundaries:

## 2. Data layout and exact baseline

Describe the row-major input, FP16-to-FP32 loading choice, GPU database
residency, distance kernel, CUB radix-sort baseline, and CPU reference. Explain
why the full-sort implementation is a correctness baseline rather than the
final Top-K optimization.

## 3. IVF-Flat index

Describe training sampling, center initialization, Lloyd iterations, final
assignment, list offsets/IDs, persistent index format, coarse probing, and
candidate reranking. State which phases run on CPU and GPU.

## 4. Correctness

| Metric | Dtype | N | Q | K | Recall@K | Rank ID agreement | Score error |
|---|---|---:|---:|---:|---:|---:|---:|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Include adversarial cases: duplicate scores, zero vectors for cosine, K=1 and
K=100, FP16 payloads, and index reload.

Recall checks whether the same IDs appear in the Top-K set; rank ID agreement
is stricter and checks the ID at every position. For exact CPU/GPU comparison,
report both together with the score tolerance and deterministic tie policy.
Retain the full `scripts/run_gpu_validation.py` directory: copy checked rows
from `report_table.md`, and archive `summary.csv`, `commands.txt`,
`environment.txt`, per-case logs, and the recorded database/query SHA-256.
The default validation size is only a correctness gate; the formal table must
still include at least N=1,000,000, D=128, and Q=1,000.

## 5. Recall/performance trade-off

| nprobe | Batch | Repetitions | Recall@K | Rank agreement | Mean QPS | Median QPS | Mean P50 ms | Mean P99 ms | Mean candidates |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Plot recall and latency/QPS against nprobe. Report all quality sample sizes.
State the warm-up policy, repetition count, aggregation rule, and whether the
reported latency is per query or per batch.

## 6. Exact and reference comparison

| Implementation | N | D | Q | K | QPS | P50 | P99 | Memory | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU exact | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 1.0x |
| CUDA exact/full sort | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| IVF-Flat | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| FAISS reference | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## 7. ncu/nsys analysis

Connect measured kernel time, achieved bandwidth, occupancy, launch count, and
host/device synchronization to specific changes. Keep raw profiler exports and
commands alongside the report.

## 8. Limitations and next steps

- Replace per-query full radix sort with block-local selection and hierarchical
  Top-K merge.
- Execute multiple queries concurrently and reuse candidate buffers by batch.
- Accelerate k-means assignment with tiled/shared-memory or GEMM-based distance.
- Add IVF-PQ and compare compression, recall, and memory.
- Add a FAISS benchmark using identical data, metric, K, and timing boundaries.
