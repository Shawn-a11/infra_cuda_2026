# Development Log

## Project Overview

GPU exact 与 IVF-Flat 向量检索、CPU ground truth 和性能评估工程。

## Implementation Progress

- 初版：CPU exact、CUDA exact、IVF build/load/search、数据生成和 sweep 已完成。
- 当前：迭代 #1，补 CUDA CTest 与服务器 CPU/GPU 验收矩阵。

## Development Log Entries

### 2026-08-25 13:31 — 迭代 #1：诊断与设计冻结

**改动原因**：本地 CPU CTest 已通过，但测试文件没有 CUDA gate，现有 outputs 仅为 10k × 128 smoke。
**改动内容**：
- `docs/user_requirements.md`：记录双后端、验证位置和最终规模要求。
- `docs/idea_report.md`：固定检索方法与实验问题。
- `docs/implementation.md`：定义 CUDA CTest 和服务器验证矩阵。
**预期效果**：CUDA 实现进入自动化正确性门禁，服务器运行可直接形成报告证据。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 否

## Known Issues

- 尚未在真实 GPU 上运行 CUDA CTest、1M 数据实验和 ncu/nsys。
- 当前 CUDA exact 仍逐 query 全量排序，不是最终性能实现。

### 2026-08-25 14:12 — 迭代 #1：CUDA 正确性进入 CTest

**改动原因**：CUDA exact、IVF assignment 与 candidate rerank 原先没有自动化测试入口。
**改动内容**：
- `tests/test_vector_search.cpp`：新增条件式 CUDA gate，覆盖 L2/IP/Cosine exact、非整 batch、GPU-built IVF index 和 CPU/CUDA rerank 对照。
**预期效果**：CUDA 服务器上的 CTest 会同时约束 Recall@K、rank ID agreement 与 score error。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 否

### 2026-08-25 14:20 — 迭代 #1：新增服务器 CPU/GPU 验收矩阵

**改动原因**：现有 sweep 面向性能探索，不会自动生成跨 metric/dtype/mode 的硬性正确性结论。
**改动内容**：
- `scripts/run_gpu_validation.py`：生成 3 metrics × 2 dtypes 数据，执行 K=1/10/50/100 exact 和 GPU-built IVF 的 CPU/GPU gate，保存哈希、命令、环境、CSV 与 Markdown 草表。
**预期效果**：服务器一条命令即可验证 exact、GPU assignment 与 IVF rerank，并为报告保留完整证据链。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 否

### 2026-08-25 14:23 — 迭代 #1：同步服务器验收说明

**改动原因**：新增脚本需要明确运行顺序、产物和 smoke/formal scale 边界。
**改动内容**：
- `README.md`：新增 CUDA CTest + CPU/GPU matrix 命令、覆盖范围和产物说明。
- `REPORT.md`：规定从自动产物填表并保留哈希/环境/命令，同时强调正式 1M 规模仍需另跑。
**预期效果**：不会把 4096-vector correctness gate 误写成任务要求的最终性能实验。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 否

### 2026-08-25 14:25 — 迭代 #1：格式化 CUDA gate 测试

**改动原因**：新增条件测试需保持项目 C++ 风格一致。
**改动内容**：
- `tests/test_vector_search.cpp`：执行 clang-format。
**预期效果**：降低格式噪声并便于服务器失败时定位。
**文档同步**：idea_report.md 否 | implementation.md 否 | configs/ 否

### 2026-08-25 14:26 — 迭代 #1：保护 IVF 训练样本下界

**改动原因**：自定义 `--clusters` 大于 2048 时，固定训练样本数可能小于 nlist。
**改动内容**：
- `scripts/run_gpu_validation.py`：令 training samples 至少等于 nlist，同时不超过数据库规模。
**预期效果**：合法的自定义矩阵参数不会因脚本内部默认值产生无效 IVF 配置。
**文档同步**：idea_report.md 否 | implementation.md 是 | configs/ 否

### 2026-08-25 14:29 — 迭代 #1：补 CUDA 排序 tie 与零向量门禁

**改动原因**：报告要求的重复分数确定性策略和 cosine 零向量此前只在 CPU 测试中覆盖。
**改动内容**：
- `tests/test_vector_search.cpp`：CUDA gate 新增 L2 重复分数排序对照，以及 cosine 零向量的有限 score 检查。
**预期效果**：GPU exact 的 tie policy 与零范数语义被明确纳入服务器 CTest。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 否

### 2026-08-25 14:30 — 迭代 #1：格式化新增边界 gate

**改动原因**：保持条件测试的 C++ 排版一致。
**改动内容**：
- `tests/test_vector_search.cpp`：再次执行 clang-format。
**预期效果**：新增 tie/zero-vector 代码保持可读。
**文档同步**：idea_report.md 否 | implementation.md 否 | configs/ 否

### 2026-08-25 14:34 — 迭代 #1 结果

| 指标 | 改动前 | 改动后 | 变化 |
|---|---:|---:|---|
| Release CPU CTest | 1/1 通过 | 1/1 通过 | 保持 |
| UBSan CPU CTest | 1/1 通过 | 1/1 通过 | 保持 |
| CUDA 条件测试语法 | 无 gate | host C++ 条件分支检查通过 | 新增 |
| 10k×128 CPU exact smoke | Recall@10=1 | Recall@10=1，QPS≈1570 | 正确性保持 |
| 10k×128 CPU IVF smoke | Recall@10=1 | Recall@10=1，QPS≈3902 | 正确性保持 |
| 服务器验收矩阵 | 无 | 30 个默认 gate | 新增，待 GPU 运行 |

**结论**：迭代有效。CPU exact/IVF 与索引回读回归通过，CUDA gate 已进入构建路径且验证脚本静态检查通过；真实 CUDA CTest、30-case matrix、1M 正式实验和 profiler 仍必须在服务器执行。上述 smoke QPS 仅用于确认程序运行，不用于最终性能结论。

## Run Instructions

- 本地 CPU：`cmake --build build -j && ctest --test-dir build --output-on-failure`
- 服务器 gate：`python3 scripts/run_gpu_validation.py --binary build/vector_search --output-dir outputs/gpu_validation`
- 正式实验：另行生成 N≥1,000,000、D=128、Q≥1,000 数据，再运行 sweep 与 ncu/nsys。

### 2026-09-09 — 迭代 #2：GPU 首跑前验证链审计

**改动原因**：现有 IVF gate 只比较 CPU/GPU 各自相对 ground truth 的聚合质量值；两个聚合误差可能相同，但具体 ID/score 仍可能不同。IVF 候选按 coarse-list 顺序拼接，也没有显式保证 CUDA 稳定排序在重复 score 时采用全局较小 ID。
**改动范围**：增强 tie policy 和验收证据，不改变距离公式、IVF 候选集合、K-means 或性能容差。
**预期效果**：CPU/GPU 输出逐项相同时才通过 parity gate；任何单项执行失败仍能得到完整矩阵汇总，便于 RTX 5090 首轮定位。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 否

### 2026-09-09 — 迭代 #2：直接结果门禁与失败汇总实现

**改动内容**：
- `src/ivf.cpp`：coarse probe 后按 vector ID 整理候选，固定 CUDA stable radix sort 的重复分数 tie-break。
- `scripts/run_gpu_validation.py`：逐 query/rank 解析 CPU/GPU result，新增 direct rank agreement、mean/max score error；清理陈旧产物；单项异常转为结构化 failure row 后继续。
- `tests/test_vector_search.cpp`：新增跨倒排表、乱序 ID、全重复 score 的 IVF tie 回归。
- `README.md`、`REPORT.md`：同步直接门禁和报告证据要求。
**预期效果**：30 项默认矩阵既验证相对 CPU exact 的质量，也验证两后端实际输出逐项一致。
**文档同步**：idea_report.md 是 | implementation.md 是 | configs/ 否

### 2026-09-09 — 迭代 #2 本地结果

| 指标 | 结果 |
|---|---:|
| Release CPU CTest | 1/1 通过 |
| UBSan CPU CTest | 1/1 通过 |
| Python py_compile/compileall | 通过 |
| 直接 result helper | identical file = rank 1.0、mean/max error 0 |
| CPU-as-CUDA 成功编排演练 | 30/30 case 通过，direct rank 1.0、max score error 0 |
| 无 CUDA 失败矩阵 | 30/30 case 均执行并写入 failure row |

**结论**：CPU/索引回归通过，验证驱动器在本机无 CUDA 的预期失败条件下仍生成 30 行、28 列完整汇总。ASan 在当前 macOS 沙箱未完成，不计为通过证据。真实 CUDA compile、CTest 和 30 项数值门禁仍需在 RTX 5090 上执行。
