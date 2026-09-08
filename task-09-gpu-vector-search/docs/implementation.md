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

## Result Format

每个案例保留 database/query/config/index/result/performance/quality；总目录保存 `summary.csv`、`environment.txt` 与 `report_table.md`。

## Implementation Order

设计文档 → CUDA CTest → GPU validation driver → README/REPORT → 本地 CPU/UBSan/Python/CUDA 静态验证。

