# Idea Report

## Part 1 — Motivation

任务九需要先建立可证明正确的 GPU exact baseline，再以 CPU exact 为 ground truth 评估 IVF-Flat 的 Recall、延迟、吞吐和显存占用。

### Research Questions

- RQ1：CUDA exact 在 L2/IP/Cosine、FP16/FP32 和不同 K 下能否与 CPU 排序一致？
- RQ2：`nprobe` 与 batch size 如何形成 Recall/QPS/P99 trade-off？
- RQ3：全量 radix sort 的瓶颈是否支持后续 hierarchical Top-K 优化？

### Key Works

| 方法 | 工程用途 |
|---|---|
| IVF-Flat | coarse partition 与候选精排基线 |
| CUB radix sort | CUDA exact 全排序正确性基线 |
| FAISS | 最终同数据/口径的外部参考 |

## Part 2 — Method

CPU 使用固定大小 heap 和 score/id 确定性排序；CUDA 先物化 score/id，再用 CUB 排序。IVF 训练、持久化、coarse probe 和候选 rerank 共享 CPU/CUDA assignment 接口。

## Part 3 — Experiment Design

正确性覆盖三种度量、两种输入 dtype、K=1/10/50/100、cosine 零向量、重复分数、索引回读和 IVF `nprobe=nlist`。服务器矩阵对 exact 与 IVF 分别执行 CPU/GPU gate；正式规模至少 N=1,000,000、D=128、Q=1,000。

## References

- Johnson, Douze and Jégou (2017), *Billion-scale similarity search with GPUs*.
- NVIDIA CUB, *DeviceRadixSort* documentation.

