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

### 2026-09-09 — 迭代 #2：American Gamma 诊断与实验设计更新

**改动原因**：RTX 5090 实测中，50,000 路径的 American pseudo 仅 Gamma 失败；提高到 200,000 路径后所有指标通过，且 American Halton 的 CPU/GPU 输出逐项一致。
**诊断证据**：
- 50k pseudo Gamma：CPU `0.0351127`、GPU `0.0233494`，绝对差约 `0.01176`，略高于 `0.01` 门限。
- 200k pseudo Gamma：CPU `0.0225246`、GPU `0.0233072`，绝对差约 `0.000783`，通过门限。
- Halton：price、Delta、Gamma、Vega 的 CPU/GPU 绝对差均为 `0`；这证明实现路径一致，但负 Gamma 不单独作为绝对精度结论。
**改动范围**：只调整 American 正式验证配置与矩阵容错/汇总行为，不修改定价模型、LSM 或容差。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 待完成

### 2026-09-09 — 迭代 #2：正式矩阵配置与失败汇总实现

**改动原因**：50k American pseudo 的二阶有限差分噪声会偶发触发 Gamma 门禁，且原驱动器在首个失败后中止，无法得到完整矩阵。
**改动内容**：
- `configs/simulation_american_validation.txt`：固定 200,000 路径、seed `20260824` 与 1% spot bump，专供 American pseudo 正式 gate。
- `scripts/run_gpu_validation.py`：American pseudo 使用专用配置；单案例失败后继续；删除陈旧 comparison；总表保留执行失败行；全部完成后统一非零退出。
- `README.md`、`REPORT.md`：同步正式运行口径，并区分 Halton 实现一致性证据与 CRR 绝对精度证据。
**预期效果**：不放宽统计容差的前提下稳定验证 American Gamma，并保证失败矩阵也能留下完整、可审计的报告产物。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 是

### 2026-09-09 — 迭代 #2 结果

| 指标 | 改动前 | 改动后 | 变化 |
|---|---:|---:|---|
| American pseudo 通过路径数 | 50,000（Gamma 失败） | 200,000（四项通过） | 稳定通过 |
| American pseudo Gamma 绝对差 | 约 0.01176 | 约 0.000783 | 降低约 93.3% |
| American Halton CPU/GPU 差 | 未执行 | 四项均为 0 | 实现一致 |
| Release CPU CTest | 1/1 通过 | 1/1 通过 | 保持 |
| 无 CUDA 失败路径覆盖 | 首个 case 后中止 | 8/8 case 均执行并汇总 | 完整 |

**结论**：迭代有效。服务器证据支持将原失败归因于 50k 路径下 American LSM Gamma 的统计波动；没有放宽门限或更改算法。本地无 CUDA 演练确认矩阵会保留完整失败清单和汇总产物。更新后的真实 CUDA 全矩阵仍需在 RTX 5090 上复跑后归档。

### 2026-09-09 — 迭代 #3：独立 reference gate 诊断与设计

**改动原因**：RTX 5090 矩阵已达到 8/8 cases、32/32 CPU/GPU 指标通过，但现有汇总只证明后端一致性，未展示 Black–Scholes/CRR 的独立价格与 Greeks reference。当前 American QMC Gamma 为 `-0.0031627`，虽然 CPU/GPU 完全相同，仍不能作为绝对精度证据。
**改动范围**：新增解析/CRR reference 输出与门禁，并为 American QMC 选择更稳定的正式配置；不修改 Monte Carlo、LSM 或 CUDA 算法。
**预期效果**：报告表能够直接区分 backend parity 和 financial-reference accuracy，防止共同误差被一致性 gate 掩盖。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 待完成

### 2026-09-09 — 迭代 #3：解析/CRR reference gate 实现

**改动内容**：
- `include/pricing/pricing.hpp`、`src/pricing_cpu.cpp`：新增 American 2,000 步 CRR 中心差分 Greeks reference，使用与估值端一致的 bump。
- `src/main.cpp`：结果文件新增 `reference_delta/gamma/vega` 与 reference method；European 使用 Black–Scholes 解析 Greeks，American 使用 CRR Greeks。
- `scripts/compare_cpu_gpu.py`：分别记录 backend parity 与每个后端的 reference error/tolerance，并合并为 overall gate。
- `scripts/run_gpu_validation.py`：报告草表展示 reference 误差与状态，American Halton 改用专用正式配置。
- `configs/simulation_american_qmc_validation.txt`：固定 131,072 路径、64 步、8 个 randomized-Halton replications 和 1% spot bump。
- `tests/test_pricing.cpp`、`README.md`、`REPORT.md`：补 reference API/config 回归与报告口径。
**预期效果**：CPU/GPU 即使共同偏离 Black–Scholes/CRR，也不能通过正式矩阵；无独立 reference 的模型明确标记为 `n/a`。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 是

### 2026-09-09 — 迭代 #3 本地结果

| 指标 | 旧 American QMC smoke | 新 American QMC validation | Reference |
|---|---:|---:|---:|
| Paths / steps | 16,384 / 32 | 131,072 / 64 | — |
| Price | 6.7995046 | 6.7413820 | CRR 6.7425208 |
| Price 绝对误差 | 0.0569838 | 0.0011388 | — |
| Gamma | -0.0031627 | 0.0133855 | CRR 0.0225197 |
| Gamma 绝对误差 | 0.0256824 | 0.0091342 | 门限 0.01 |

**验证**：Release CPU CTest 1/1 通过，UBSan CPU CTest 1/1 通过，Python compileall 通过；本机无 CUDA 的矩阵演练仍执行并记录全部 8 个失败 case。对新 American QMC 本地结果模拟同值 CPU/GPU 时，price/Delta/Gamma/Vega 四个 reference gates 全部通过。

**结论**：新 QMC 正式配置显著减小 American LSM 的价格与二阶差分偏差，并通过当前独立 CRR 门限。真实 CUDA 后端仍需在 RTX 5090 上重新构建并执行矩阵，确认 GPU reference gates。
