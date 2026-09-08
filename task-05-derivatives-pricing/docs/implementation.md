# Implementation Design

## Current Files

| 文件 | 责任 |
|---|---|
| `src/pricing_cpu.cpp` | CPU reference、解析/树基准、MC/QMC、Greeks、LSM |
| `src/pricing_cuda.cu` | CUDA 路径、归约、LSM、Greeks、多 GPU |
| `tests/test_pricing.cpp` | CPU 回归与条件式 CPU/CUDA 门禁 |
| `scripts/compare_cpu_gpu.py` | 单案例统计正确性门禁 |
| `scripts/run_gpu_validation.py` | 全功能服务器矩阵 |

## Iteration 1 Design

保留算法实现不变，增强验收与报告闭环：GPU 验证脚本除 `summary.csv` 外，自动记录 GPU/CUDA 环境、每个案例的配置和报告可直接引用的 Markdown 表；任何数值 gate 失败仍以非零状态退出。

## Result Format

`summary.csv` 是机器可读门禁，`environment.txt` 保存硬件/工具链，`report_table.md` 提供报告草表，案例目录保存 CPU/GPU result、performance 与 comparison。

## Implementation Order

设计文档 → 验证脚本 → README/REPORT → Python 静态检查 → 本地 CPU CTest；真实 GPU 结果由服务器补齐。

