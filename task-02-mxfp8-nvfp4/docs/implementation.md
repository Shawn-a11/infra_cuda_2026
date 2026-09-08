# Implementation Design

## Current Files

| 文件 | 责任 |
|---|---|
| `src/quantization_cpu.cpp` | 格式编解码、CPU reference、统一输入校验、误差指标 |
| `src/quantization_cuda.cu` | CUDA 缩放、量化、打包和反量化 |
| `tests/test_quantization.cpp` | 格式、I/O、边界条件和条件式 CUDA 门禁 |
| `scripts/run_matrix.py` | 数据集 × 配置实验矩阵和汇总 |

## Iteration 1 Design

1. 抽出 CPU/CUDA 共用的请求校验，直接 API 与配置文件使用相同的标准块约束，并统一拒绝 NaN/Inf。
2. 增加零张量、部分尾块、NVFP4 奇数元素 padding、非法块和非有限输入回归测试。
3. CUDA CTest 同时覆盖 block/tensor、nearest/stochastic 和边界长度；服务器矩阵保留 packed、scale 与反量化差异。
4. README/REPORT 同步有限值策略与验收范围。

## Result Format

`outputs/*/summary.csv` 保留每个数据/配置组合的误差、压缩、计时及 CPU reference 匹配字段；CUDA 最终结论必须来自服务器运行。

## Implementation Order

设计文档 → 公共校验 → CPU/CUDA 调用点 → 单元测试 → 文档 → CPU/UBSan/CUDA 静态验证。

