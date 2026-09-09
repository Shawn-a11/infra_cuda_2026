# Task 05 Report

## 1. Problem and contract

- Target GPU and software versions:
- Option types and monitoring conventions:
- Input/output formats:
- Timing boundary:

## 2. Financial model and numerical method

Document risk-neutral GBM, full-truncation Euler Heston, and local-volatility
discretization; discounted payoff; Black-Scholes/CRR references; confidence
intervals; and the assumptions behind Asian, barrier, and American monitoring.

## 3. Parallel design

Document path-level parallelism, per-thread RNG state, memory layout, block
reduction, the five accumulated moments, and CPU/GPU responsibilities.

For American LSM, document GPU path storage, eight regression moments, the
host 3x3 solve, GPU exercise update, and the full timing boundary. For
multi-GPU, record per-device path counts and derived seeds, then explain the
weighted mean and `sum(w_i^2 * SE_i^2)` variance merge.

## 4. Correctness and convergence

Seed protocol: record `master_seed`, the deterministic per-replication seed
derivation rule, RNG implementation, replication count, and whether compared
methods share common random numbers. Do not change seeds selectively after
looking at results.
For Asian/barrier contracts, state the fixed reference path count, its
independent seed, estimate, standard error, and confidence interval.

| Option | Paths | Steps | Method | Replications | Mean estimate | Empirical SD | Mean reported SE | Reference |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Include a convergence curve over increasing path counts. Do not treat overlap
with a single noisy CPU estimate as proof; report both estimators' uncertainty.
Compare empirical across-replication SD with the mean within-run standard error
to check that the uncertainty estimator is calibrated.

## 5. Variance reduction

| Option | Mode | Replications | Mean price | Empirical SD | Mean std. error | Paths/sec | Relative error |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TBD | none | TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | antithetic | TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | control_variate | TBD | TBD | TBD | TBD | TBD | TBD |

## 6. Performance and profiling

| GPU | Option | Paths | Steps | Kernel ms | Paths/sec | CPU ms | Speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Attach the relevant ncu/nsys observations and connect each optimization claim
to a measured bottleneck.

## 7. Advanced-model and CPU/GPU accuracy gates

| Product | Model | RNG | Paths/steps | CPU price | GPU price | Joint tolerance | Pass |
|---|---|---|---:|---:|---:|---:|---|
| European call | GBM | pseudo/QMC | TBD | TBD | TBD | TBD | TBD |
| European call | Heston | pseudo/QMC | TBD | TBD | TBD | TBD | TBD |
| European call | Local Vol | pseudo/QMC | TBD | TBD | TBD | TBD | TBD |
| American put | GBM | pseudo/QMC | TBD | TBD | TBD | TBD | TBD |

| Metric | CPU | GPU | Analytic/tree reference | Tolerance | Pass |
|---|---:|---:|---:|---:|---|
| Delta | TBD | TBD | TBD | TBD | TBD |
| Gamma | TBD | TBD | TBD | TBD | TBD |
| Vega | TBD | TBD | TBD | TBD | TBD |

Retain `scripts/run_gpu_validation.py` output. For randomized Halton, report
the independent shift count and across-shift standard error. For American GBM,
use the dedicated 200,000-path pseudo-random validation configuration and
the dedicated 131,072-path/64-step Halton configuration, then compare CPU and
GPU LSM against the 2,000-step CRR price and finite-difference Greeks. Treat
exact Halton CPU/GPU agreement as implementation-parity evidence rather than,
by itself, proof of Greek accuracy. `backend_passed` and `reference_passed`
must both succeed when a reference is available. Copy verified rows from
`report_table.md`, and archive the adjacent `summary.csv` and `environment.txt`;
do not hand-transcribe GPU values from terminal output.

## 8. Limitations and next steps

- Compare FP32 path evolution with an FP64 accuracy mode.
- Replace five separate Greek pricing calls with a fused multi-scenario kernel
  only after profiler evidence shows launch or memory overhead is material.
- Compare custom RNG and cuRAND quality as well as throughput.
- Evaluate multi-GPU path partitioning for larger portfolios.
- Add Brownian bridge/PCA dimension reduction for high-dimensional QMC.
- Compare LSM bases and out-of-sample regression to quantify exercise bias.
