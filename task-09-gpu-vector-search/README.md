# Task 09 - GPU Vector Search Engine

This project provides a correctness-first GPU exact-search baseline and a
persistent IVF-Flat approximate index. It supports L2 distance, inner product,
and cosine similarity; FP32 and FP16 input files; batched query files; and
configurable Top-K up to 100.

## Implemented contract

- GPU exact distance/similarity calculation.
- GPU Top-K baseline using CUB radix sort, with results ordered by score.
- CPU exact implementation used as the correctness and recall ground truth.
- IVF-Flat training, inverted-list construction, save/load, and `nprobe` search.
- Optional CUDA assignment during k-means training and final index assignment.
- Result file, index file, QPS, P50/P99 latency, memory, and quality logs.
- CPU-exact throughput comparison and a CUDA working-set estimate that includes
  database, candidate, score/ID, and CUB temporary buffers.
- `recall@K` and mean absolute score difference at corresponding ranks.
- K values from 1 through 100; run K=1/10/50/100 experiments from config files.

The exact CUDA baseline processes every query in the input and keeps the
database resident on the GPU for the search call. Queries are grouped by
`batch_size`, but this first baseline executes queries sequentially within each
batch so that the full distance vector can be radix-sorted in bounded memory.
This is deliberately a verifiable baseline; replacing the full sort with a
hierarchical fused Top-K kernel is the main optimization target.

## Binary formats

All integer fields are little-endian and all payloads are row-major.

Database header (28 bytes):

| Field | Type | Value |
|---|---|---|
| magic | 8 bytes | `VECDB01\0` |
| version | uint32 | 1 |
| num_vectors | int64 | N |
| dim | int32 | D |
| dtype | uint8 | 1=FP32, 2=FP16 |
| metric | uint8 | 1=L2, 2=inner product, 3=cosine |
| reserved | uint16 | 0 |

The database payload is `N * D` elements in the declared dtype.

Query header (28 bytes): `VECQRY1\0`, uint32 version, int64 query count,
int32 dimension, uint8 dtype, and three zero reserved bytes. The payload is
`num_queries * D` elements.

The IVF index stores `IVFFLT1\0`, version, N, D, `nlist`, metric, FP32 centers,
uint64 list offsets, and int64 vector IDs. Original vectors remain in the
database file, so loading an index never retrains it but does not duplicate the
vector payload.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

CUDA is enabled automatically when `nvcc` is present. Set the training-server
architecture explicitly when known, for example
`-DCMAKE_CUDA_ARCHITECTURES=75` for T4. A CPU-only build remains available for
format, index, and correctness tests.

## Generate smoke data

```bash
python3 -m pip install -r requirements.txt
python3 scripts/generate_data.py \
  --output-dir data/smoke --num-vectors 10000 --num-queries 100 \
  --dim 128 --dtype fp32 --metric l2 --clusters 64
```

For the required scale, regenerate with at least `--num-vectors 1000000` and
`--num-queries 1000`. Record data generation parameters and file hashes.

## Exact search

```bash
./build/vector_search search \
  --database data/smoke/database.bin \
  --queries data/smoke/queries.bin \
  --config configs/exact_smoke.txt \
  --backend auto \
  --output outputs/exact.results.txt \
  --performance outputs/exact.performance.log \
  --quality outputs/exact.quality.log
```

## IVF-Flat build and search

```bash
./build/vector_search build \
  --database data/smoke/database.bin \
  --config configs/ivf_smoke.txt \
  --index outputs/smoke.ivf \
  --backend auto \
  --performance outputs/ivf.build.log

./build/vector_search search \
  --database data/smoke/database.bin \
  --queries data/smoke/queries.bin \
  --config configs/ivf_smoke.txt \
  --index outputs/smoke.ivf \
  --backend auto \
  --output outputs/ivf.results.txt \
  --performance outputs/ivf.performance.log \
  --quality outputs/ivf.quality.log
```

The quality log compares the first `quality_queries` against CPU exact search.
`recall_at_k` measures set overlap and `rank_id_agreement` measures exact ID
agreement at each returned rank, so an exact backend should report 1.0 for
both. `mean_score_abs_error` additionally checks numerical score consistency.
Increase `quality_queries` for the final report after measuring the validation
cost.
Sweep both `nprobe` and `batch_size`; do not present a single operating point as
the whole latency/recall trade-off.

For repeated CPU or GPU measurements, use the sweep driver:

```bash
python3 scripts/run_sweep.py \
  --binary build/vector_search \
  --database data/smoke/database.bin \
  --queries data/smoke/queries.bin \
  --base-config configs/ivf_smoke.txt \
  --index outputs/smoke.ivf \
  --backend cpu \
  --top-k 1 10 50 100 \
  --nprobe 1 4 8 16 \
  --batch-size 1 32 128 \
  --repetitions 5 \
  --output-dir outputs/ivf_sweep
```

`summary.csv` preserves each repetition. `aggregate.csv` reports mean and
median QPS plus mean P50/P99 latency, recall, and rank agreement. Use a warm-up
outside the recorded sweep on the GPU server, keep clocks/workload fixed, and
retain raw logs rather than reporting only the best run.

## CUDA correctness matrix

Before using any GPU number in the report, build with CUDA, run CTest, then
execute the automated CPU/GPU gate:

```bash
ctest --test-dir build --output-on-failure
python3 scripts/run_gpu_validation.py \
  --binary build/vector_search \
  --output-dir outputs/gpu_validation
```

The default matrix covers L2/inner-product/cosine, FP32/FP16,
K=1/10/50/100 exact search, CUDA IVF assignment, and CPU/CUDA reranking of the
same persistent index. Each gate directly compares every CPU/GPU result ID and
score in addition to the per-backend Recall/rank metrics against CPU exact.
Repeated scores use the global smaller-vector-ID tie policy. The driver attempts
the remaining matrix after an individual failure, exits non-zero at the end,
and retains data hashes, exact commands, environment information, raw
result/quality/performance logs, `summary.csv`, and `report_table.md`. Its
default 4096-vector dataset is a correctness matrix, not the required
one-million-vector performance run.

## Profiling and next optimization

Profile exact and IVF modes separately. The initial exact path materializes N
scores and IDs and performs a full radix sort per query. Measure distance-kernel
bandwidth and radix-sort time before implementing a two-stage Top-K:

```bash
ncu --set full --target-processes all ./build/vector_search search \
  --database data/full/database.bin --queries data/full/queries.bin \
  --config configs/exact_smoke.txt --backend cuda \
  --output outputs/exact.txt --performance outputs/exact.log \
  --quality outputs/exact.quality.log

nsys profile --stats=true -o outputs/ivf_nsys ./build/vector_search search \
  --database data/full/database.bin --queries data/full/queries.bin \
  --config configs/ivf_full.txt --index outputs/full.ivf --backend cuda \
  --output outputs/ivf.txt --performance outputs/ivf.log \
  --quality outputs/ivf.quality.log
```

No GPU performance numbers are claimed until these commands run on the target
hardware. Capture the GPU, CUDA/CUB version, compiler flags, N/D/Q, dtype,
metric, K, batch size, nlist/nprobe, QPS, P50/P99, memory, and recall in
`REPORT.md`.
