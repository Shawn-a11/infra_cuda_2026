# Development Log

## Project Overview

MXFP8/NVFP4 软件量化、反量化与 CPU/CUDA 对照工程。

## Implementation Progress

- 初版：CPU/CUDA 软件模拟、二进制 I/O、四组配置、实验矩阵和基础 CTest 已完成。
- 当前：迭代 #1，补齐边界正确性与服务器验收门禁。

## Development Log Entries

### 2026-08-25 13:31 — 迭代 #1：诊断与设计冻结

**改动原因**：现有 24 个 CPU 矩阵案例通过，但 README 的 NaN/Inf 描述滞后，边界条件未形成显式回归门禁。
**改动内容**：
- `docs/user_requirements.md`：记录 CPU/GPU 双实现、本地/远端验证边界和中文注释要求。
- `docs/idea_report.md`：固定方法与实验问题。
- `docs/implementation.md`：定义统一校验和边界测试迭代。
**预期效果**：直接 API 与 CLI 对非法输入行为一致，CPU/CUDA 边界对照可在 CTest 中复现。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 否

## Known Issues

- 尚未在真实 NVIDIA GPU 上运行 CUDA CTest 和实验矩阵。
- 尚未执行独立第三方 encoded-byte 交叉验证。

### 2026-08-25 13:36 — 迭代 #1：声明公共请求校验

**改动原因**：CPU/CUDA 入口原先各自维护部分校验，直接 API 可能绕过配置文件的标准块约束。
**改动内容**：
- `include/low_precision/quantization.hpp`：新增 `validate_quantization_request` 公共声明。
**预期效果**：两个后端共享同一输入契约。
**文档同步**：idea_report.md 否 | implementation.md 是 | configs/ 否

### 2026-08-25 13:38 — 迭代 #1：实现统一请求校验

**改动原因**：直接构造 `QuantizationConfig` 时，错误块大小和非法枚举此前可能进入热路径。
**改动内容**：
- `src/quantization_cpu.cpp`：实现形状溢出保护、枚举检查、标准块约束和 NaN/Inf 拒绝，并接入 CPU 入口。
**预期效果**：非法输入在分配和索引之前稳定失败，CPU/CUDA 可复用完全相同的策略。
**文档同步**：idea_report.md 否 | implementation.md 是 | configs/ 否

### 2026-08-25 13:40 — 迭代 #1：CUDA 接入公共校验

**改动原因**：CUDA 入口原有校验缺少块大小与枚举约束，且与 CPU 错误消息不一致。
**改动内容**：
- `src/quantization_cuda.cu`：删除重复校验并在任何 CUDA 分配前调用公共请求校验。
**预期效果**：CPU/GPU 对同一非法请求具有一致失败语义。
**文档同步**：idea_report.md 否 | implementation.md 是 | configs/ 否

### 2026-08-25 13:45 — 迭代 #1：扩展边界与 CUDA 回归测试

**改动原因**：原测试未显式覆盖零张量、部分尾块、奇数 NVFP4 padding、非法块和非有限输入，CUDA gate 只覆盖默认 block/nearest。
**改动内容**：
- `tests/test_quantization.cpp`：新增边界拒绝测试，并把 CUDA 矩阵扩为两种格式 × 两种 scale mode × 两种 rounding，使用 65 元素触发尾块。
**预期效果**：本地固定 CPU 边界语义，服务器 CTest 自动验证完整 CPU/CUDA 编码一致性矩阵。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 否

### 2026-08-25 13:47 — 迭代 #1：同步用户文档

**改动原因**：README/REPORT 仍把 NaN/Inf 策略列为未来工作，与已实现的严格拒绝策略矛盾。
**改动内容**：
- `README.md`：记录公共校验和扩展后的 CUDA CTest 矩阵。
- `REPORT.md`：把有限值策略改为已实现验收项并保留可选兼容模式为后续工作。
**预期效果**：实现、测试和报告口径一致。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 否

### 2026-08-25 13:49 — 迭代 #1：统一代码格式

**改动原因**：新增声明、校验与测试需要保持项目既有 C++/CUDA 风格。
**改动内容**：
- `include/low_precision/quantization.hpp`、`src/quantization_cpu.cpp`、`src/quantization_cuda.cu`、`tests/test_quantization.cpp`：执行 clang-format。
**预期效果**：减少无关格式差异并保持静态检查可读性。
**文档同步**：idea_report.md 否 | implementation.md 否 | configs/ 否

### 2026-08-25 13:52 — 迭代 #1 结果

| 指标 | 改动前 | 改动后 | 变化 |
|---|---:|---:|---|
| Release CTest | 1/1 通过 | 1/1 通过 | 保持 |
| UBSan CTest | 1/1 通过 | 1/1 通过 | 保持 |
| CPU 数据/配置矩阵 | 24/24 完成 | 24/24 完成 | 保持 |
| 显式边界门禁 | 基础格式/I/O | 新增零值、尾块、奇数 packing、非法块、NaN/Inf | 增强 |
| CUDA host-only 语法检查 | 通过 | 通过 | 保持 |

**结论**：迭代有效。量化误差数值保持不变，公共输入契约和边界回归得到补强；真实 GPU CTest/矩阵仍需在服务器执行。

## Run Instructions

- 本地：`cmake --build build -j && ctest --test-dir build --output-on-failure`
- 服务器：使用 `ENABLE_CUDA=ON` 重新配置后运行 CTest，再执行 `scripts/run_matrix.py --backend cuda`。
