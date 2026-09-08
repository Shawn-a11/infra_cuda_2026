# Task 02 Report

## 1. Problem and compliance scope

- Target GPU, compute capability, CUDA/compiler versions:
- Dataset rows/columns, input dtype, distribution, seed and file hash:
- Software-emulation boundary:
- Kernel-only and end-to-end timing boundaries:

State explicitly that the baseline does not require native FP8/FP4 Tensor Core
support. Distinguish standards-aligned block mode from the per-tensor ablation.

## 2. Numerical formats

Describe E4M3, E8M0 and E2M1 bit layouts, subnormals, saturation, ties-to-even,
and stochastic rounding. Include the exact MXFP8 and NVFP4 reconstruction
formulae and zero-tensor behavior.

## 3. Scaling and packing layout

Document:

- MXFP8 block size 32 and E8M0 scale selection;
- NVFP4 block size 16, E4M3 local scale and FP32 global scale;
- per-tensor versus block-wise scale counts;
- NVFP4 low/high nibble ordering and odd-element padding;
- binary headers and row-major payloads.

## 4. CPU/CUDA design

Describe CPU reference tables/search, CUDA scale computation, element encoding,
packed store, dequantization, memory traffic, launch geometry, and CPU/GPU
responsibilities. Record every optimization only after profiler evidence.

## 5. Correctness

| Format | Scale mode | Rounding | Distribution | Dtype | Elements | Packed match | Scale match | Max CPU/GPU diff | Pass |
|---|---|---|---|---|---:|---|---|---:|---|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Include scalar encoding tests, exact representable values, zero tensors,
partial blocks, odd NVFP4 element counts, FP16 input, all output dtypes, and
the explicit policy that NaN/Inf requests are rejected before either backend
allocates working buffers.

## 6. Quantization error and compression

| Format | Scale mode | Distribution | Max abs error | MAE | MSE | Payload compression |
|---|---|---|---:|---:|---:|---:|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Report uniform, normal, and outlier-heavy matrices separately. Define whether
headers and scales are included in each compression figure.

## 7. Performance

| GPU | Format | Elements | Quantize ms | Dequantize ms | Quantize GB/s | Dequantize GB/s | CPU ms | Speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## 8. ncu/nsys analysis

Record achieved memory bandwidth, occupancy, instruction mix, launch count,
scale-kernel cost, packed-store efficiency, and synchronization overhead.

## 9. Limitations and next steps

- Independently cross-check OCP/NVIDIA encoded bytes.
- Decide whether a later compatibility mode should propagate/saturate NaN/Inf;
  the current strict mode rejects them and has regression coverage.
- Replace scalar scale scans with warp reductions.
- Add vectorized packed loads/stores.
- Compare native/library paths only as optional architecture-specific results.
