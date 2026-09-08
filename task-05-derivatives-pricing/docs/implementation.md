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

## Iteration 2 Design

服务器实测表明 American pseudo 在 50,000 路径时仅 Gamma 越过门限，而 200,000 路径时 CPU/GPU 的 Gamma 绝对差从约 `0.01176` 降至 `0.000783`；American Halton 的四项输出逐项一致。因此不修改算法或放宽容差，而是增加 American 专用正式验证配置（200,000 路径、1% spot bump）。验证驱动器在单案例失败后继续执行，尽可能产出全部案例的 `summary.csv` 和 `report_table.md`，最后仍以非零退出码保持严格门禁。

## Result Format

`summary.csv` 是机器可读门禁，`environment.txt` 保存硬件/工具链，`report_table.md` 提供报告草表，案例目录保存 CPU/GPU result、performance 与 comparison。若某个案例失败，汇总中保留其已生成的比较行，并在全部案例执行完成后统一报告失败列表。

## Implementation Order

设计文档 → 验证脚本 → README/REPORT → Python 静态检查 → 本地 CPU CTest；真实 GPU 结果由服务器补齐。
