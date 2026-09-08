# Idea Report

## Part 1 — Motivation

任务二需要一个不依赖原生 FP8/FP4 Tensor Core 的可验证软件基线，使格式、缩放、打包和误差分析能够先在 CPU 固定，再在 CUDA 上逐项对齐。

### Research Questions

- RQ1：MXFP8 与 NVFP4 的编码、缩放和反量化能否在 CPU/CUDA 上保持一致？
- RQ2：块级缩放相对张量级消融，在不同输入分布上的误差和存储代价如何变化？

### Key Works

| 来源 | 用途 |
|---|---|
| OCP Microscaling Formats Specification | MXFP8 E4M3/E8M0 与 32 元素块定义 |
| NVIDIA Transformer Engine NVFP4 guide | E2M1、E4M3 局部 scale 与 FP32 全局 scale 的两级缩放 |

## Part 2 — Method

CPU reference 使用显式标量编码表、确定性舍入和文件回读定义真值；CUDA 软件模拟复用同一格式语义。标准块模式分别固定 MXFP8=32、NVFP4=16，张量模式仅作为消融。

## Part 3 — Experiment Design

数据覆盖 uniform、normal、outlier，FP16/FP32 输入和四组格式/缩放配置。正确性门禁覆盖标量编码、有限值策略、零张量、尾块、NVFP4 奇数打包、随机舍入复现性以及 CPU/CUDA packed/scale/dequantized 对照。

## References

- OCP, *Microscaling Formats (MX) Specification*, v1.0.
- NVIDIA, *NVFP4 Training Recipe* documentation.

