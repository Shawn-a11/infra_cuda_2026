# Repository Guidelines

## Project Structure & Module Organization

This repository contains three independent CMake projects. `task-05-derivatives-pricing` and `task-09-gpu-vector-search` are the primary submissions; `task-02-mxfp8-nvfp4` is the lower-priority follow-up. Each task keeps public interfaces in `include/`, implementations in `src/`, executable-level checks in `tests/`, run configurations in `configs/`, and experiment drivers in `scripts/`. CPU code is the portable correctness reference. When CUDA is unavailable, each target links its `*_cuda_stub.cpp`; when CUDA is detected, CMake compiles the matching `.cu` backend instead. Generated datasets, build trees, profiler captures, and `outputs/` are intentionally untracked.

## Build, Test, and Development Commands

Run commands from the task directory being changed:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
```

On a CUDA host, configure with `-DENABLE_CUDA=ON` and set `-DCMAKE_CUDA_ARCHITECTURES=<SM>` when the target GPU is known. Run one CTest target with `ctest --test-dir build -R pricing_tests`, `-R vector_search_tests`, or `-R low_precision_tests`. Task-specific README files document data generation, validation matrices, sweeps, and profiling commands.

## Coding Style & Naming Conventions

All targets use C++17 without compiler extensions. C++ compilation enables `-Wall -Wextra -Wpedantic`; keep new code warning-clean. Preserve the existing separation between configuration/file I/O, CPU reference algorithms, CUDA implementations, and thin CLI entry points. Public declarations belong under the task namespace in its `include/` tree. No repository formatter or linter configuration is currently defined.

## Testing Guidelines

CTest is the common test runner. CPU-only tests must remain runnable on macOS or other non-CUDA hosts. GPU claims require the task's `scripts/run_gpu_validation.py` matrix on the target server; keep raw logs and generated summary tables locally for report preparation rather than committing them. Use deterministic seeds recorded by the experiment scripts whenever comparing CPU and CUDA results.
