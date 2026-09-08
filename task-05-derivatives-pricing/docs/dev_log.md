# Development Log

## Project Overview

CUDA 衍生品定价、风险估计与 CPU reference 工程。

## Implementation Progress

- 初版及功能扩展：基础/路径依赖期权、Greeks、American LSM、Heston、Local Vol、QMC、多 GPU 的 CPU/CUDA 路径已完成。
- 当前：迭代 #1，完成服务器验收产物与报告证据闭环。

## Development Log Entries

### 2026-08-25 13:31 — 迭代 #1：诊断与设计冻结

**改动原因**：本地 CPU CTest 已通过，但 GPU 矩阵只生成 CSV，环境与报告表仍需人工整理。
**改动内容**：
- `docs/user_requirements.md`：固定 CPU/GPU 双结果与本地/远端边界。
- `docs/idea_report.md`：记录方法和实验问题。
- `docs/implementation.md`：定义报告型验收产物。
**预期效果**：一次服务器验证即可形成数值 gate、环境清单和报告草表，减少人工转录错误。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 否

## Known Issues

- 当前机器无 CUDA，GPU 数值与性能结论仍待服务器实测。
- REPORT 中正式数值表尚未填充。

### 2026-08-25 13:57 — 迭代 #1：生成服务器环境与报告草表

**改动原因**：原 GPU 矩阵只汇总数值 gate，无法自动追溯硬件/工具链，也需要人工把 CSV 转入报告。
**改动内容**：
- `scripts/run_gpu_validation.py`：解析绝对 binary 路径，采集 platform/NVIDIA/nvcc 环境，为每行加入配置名，并生成 `environment.txt` 与 `report_table.md`。
**预期效果**：一次服务器运行同时完成数值验收和报告证据整理，降低错配配置或抄写数字的风险。
**文档同步**：idea_report.md 否 | implementation.md 是 | configs/ 否

### 2026-08-25 14:00 — 迭代 #1：同步验收产物说明

**改动原因**：新生成的环境与 Markdown 产物需要进入用户运行说明和报告流程。
**改动内容**：
- `README.md`：记录三个汇总产物和逐案例日志保留规则。
- `REPORT.md`：指定从自动生成草表填充正式结果，并禁止从终端手工抄录 GPU 数值。
**预期效果**：服务器运行到最终报告之间形成明确、可追溯的交接。
**文档同步**：idea_report.md 否 | implementation.md 是 | configs/ 否

### 2026-08-25 14:04 — 迭代 #1 结果

| 指标 | 改动前 | 改动后 | 变化 |
|---|---:|---:|---|
| Release CPU CTest | 1/1 通过 | 1/1 通过 | 保持 |
| UBSan CPU CTest | 1/1 通过 | 1/1 通过 | 保持 |
| GPU 验收汇总 | `summary.csv` | CSV + 环境 + Markdown 草表 | 增强 |
| Python 静态检查 | 通过 | 通过 | 保持 |

**结论**：迭代有效，未改变算法结果或接口；本地完整 CPU/UBSan 回归通过。GPU 数值表只能在服务器执行 `run_gpu_validation.py` 后确认。

## Run Instructions

- 本地 CPU：`cmake --build build -j && ctest --test-dir build --output-on-failure`
- 服务器 GPU：`python3 scripts/run_gpu_validation.py --binary build/derivatives_pricer --output-dir outputs/gpu_validation`
- 多 GPU：在至少两张卡可见时追加 `--include-multi-gpu`。
