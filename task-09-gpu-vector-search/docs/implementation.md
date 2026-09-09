# Implementation Design

## Current Files

| 文件 | 责任 |
|---|---|
| `src/search_cpu.cpp` | exact/candidate CPU ground truth 与质量指标 |
| `src/search_cuda.cu` | CUDA score、CUB sort 与 IVF assignment |
| `src/ivf.cpp` | IVF 训练、索引组织、probe 与 rerank |
| `tests/test_vector_search.cpp` | CPU/I/O/索引及条件式 CUDA 门禁 |
| `scripts/run_gpu_validation.py` | 服务器端 exact/IVF CPU-GPU 验收矩阵 |

## Iteration 1 Design

1. 为 CUDA exact、candidate rerank 和 IVF assignment 增加条件式 CTest，对照 CPU 的 Recall、rank agreement 与 score tolerance。
2. 新增服务器验证脚本，覆盖三种 metric、FP32/FP16、多个 K，以及 IVF build/search；失败时非零退出。
3. 自动保存环境、summary CSV 与 Markdown 报告草表；README/REPORT 明确 smoke 与正式规模边界。

## Iteration 2 Design

1. IVF coarse probe 完成后对 candidate IDs 排序，令 CPU heap 与 CUDA 稳定 radix sort 在重复分数处共享全局 ID tie-break。
2. 验收脚本解析 CPU/GPU result 文件，记录逐 rank ID agreement、mean/max score error，并将其纳入 exact 与 IVF 最终 gate；quality log 仍用于各后端相对 CPU exact 的 Recall 证据。
3. 每次运行先删除该案例的陈旧输出；单项命令失败时保留 execution failure 行并继续剩余矩阵，最终统一非零退出，使第一次服务器调试也能留下完整失败分布。

## Result Format

每个案例保留 database/query/config/index/result/performance/quality；总目录保存 `summary.csv`、`environment.txt`、`commands.txt` 与 `report_table.md`。汇总同时记录 CPU/GPU 直接 rank agreement、mean/max score error 和各自相对 CPU exact 的质量指标。

## Implementation Order

设计文档 → CUDA CTest → GPU validation driver → README/REPORT → 本地 CPU/UBSan/Python/CUDA 静态验证。
