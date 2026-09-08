# Task 02 - MXFP8 / NVFP4 Software Quantization and Dequantization

This project is a correctness-first software implementation of MXFP8 and
NVFP4. It deliberately uses ordinary CUDA instructions rather than native
FP8/FP4 Tensor Core operations, so the baseline can run on pre-Hopper NVIDIA
GPUs such as T4 and V100. A portable CPU implementation defines the reference
encoding and lets file formats and numerical behavior be tested locally.

## Current implementation contract

- FP32 and FP16 row-major input files.
- MXFP8 E4M3 values with an E8M0 shared scale. Standards-aligned block mode
  uses 32 consecutive elements per scale.
- NVFP4 packed E2M1 values, one E4M3 local scale per 16 elements, and one FP32
  global tensor scale.
- Per-block and per-tensor comparison modes. Per-tensor mode is an experimental
  ablation rather than the standard MXFP8/NVFP4 storage recipe.
- Round-to-nearest-ties-to-even and deterministic stochastic rounding.
- Two E2M1 values packed per byte, with even element in the low nibble.
- FP16, BF16, or FP32 dequantized output files.
- Max absolute error, MAE, MSE, data-only compression ratio, kernel time,
  effective bandwidth, and CPU-reference comparison logs.
- CPU and CUDA software-emulation backends with a conditional CPU/CUDA CTest
  gate.

MXFP8 follows the OCP microscaling definition: E4M3 is one permitted MXFP8
element type, the standard block contains 32 values, and the shared scale is
E8M0. The baseline conversion uses the OCP recommended power-of-two scale and
saturating E4M3 conversion.

NVFP4 follows NVIDIA's hierarchical representation:

```text
x_hat = x_e2m1 * s_block_e4m3 * s_global_fp32
s_global = tensor_amax / (448 * 6)
s_block  = quantize_e4m3((block_amax / 6) / s_global)
```

Zero tensors use `s_global=1` and zero local scales to avoid division by zero.
Both public backends reject NaN/Inf before allocation or kernel launch. Direct
API calls and file-driven calls also share the same fixed block-size checks.

## Binary formats

All integer fields are little-endian.

Input tensor header (32 bytes):

| Field | Type | Meaning |
|---|---|---|
| magic | 8 bytes | `LPTENS1\0` |
| version | uint32 | `1` |
| rows | int64 | matrix rows |
| cols | int64 | matrix columns |
| dtype | uint8 | `1=FP32`, `2=FP16` |
| reserved | 3 bytes | zero |

The row-major payload immediately follows the header.

The quantized file uses `LPQNT01\0` and stores dimensions, format, scale mode,
block size, element/packed/scale counts, FP32 global scale, packed values, then
the one-byte scale array. MXFP8 stores one byte per value. NVFP4 stores two
values per byte.

The dequantized file uses `LPDEQ01\0`, followed by version, dimensions, output
type and the row-major FP16/BF16/FP32 payload.

## Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
```

On a CUDA server, enable CUDA and specify the device architecture:

```bash
cmake -S . -B build-gpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build-gpu -j
ctest --test-dir build-gpu --output-on-failure
```

The default architecture list includes Pascal, Volta, Turing, and Ampere, and
the kernels do not use native FP8/FP4 instructions.
When CUDA is compiled and a device is visible, CTest expands its gate across
both formats, block/tensor scaling, nearest/stochastic rounding, partial
blocks, and odd NVFP4 payload lengths.

## Generate the required data families

The generator uses only the Python standard library:

```bash
python3 scripts/generate_data.py \
  --output-dir data/generated \
  --rows 1024 --cols 1024 --dtype fp32
```

It writes deterministic uniform, normal, and outlier-heavy matrices.

## Run one case

```bash
./build/low_precision_tool \
  --input data/generated/normal_fp32.bin \
  --config configs/mxfp8_block.txt \
  --backend cpu \
  --quantized outputs/normal_mxfp8.bin \
  --dequantized outputs/normal_mxfp8.dequantized.bin \
  --metrics outputs/normal_mxfp8.metrics.log \
  --performance outputs/normal_mxfp8.performance.log
```

Run all four configurations over all generated matrices:

```bash
python3 scripts/run_matrix.py \
  --binary build/low_precision_tool \
  --backend cpu \
  --output-dir outputs/cpu_matrix
```

Use `--backend cuda` on the server. A CUDA run also executes the CPU reference
and records whether packed values and scales match exactly.

## Next optimization gates

1. Pass the CUDA CTest and CPU/CUDA result matrix on T4/V100/A100.
2. Replace one-thread-per-scale scans with warp/block reductions.
3. Vectorize packed NVFP4 load/store and coalesce scale access.
4. Separate H2D/D2H, kernel-only, and end-to-end timing in the final report.
5. Profile uniform, normal, and outlier inputs with ncu/nsys before making
   bandwidth claims.
6. Cross-check encoded bytes against an independent reference implementation.

## Primary format references

- [OCP Microscaling Formats (MX) Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [NVIDIA Transformer Engine FP8/NVFP4 guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
- [NVIDIA NVFP4 data-format documentation](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.15/user-guide/features/low_precision_training/nvfp4/nvfp4.html)
