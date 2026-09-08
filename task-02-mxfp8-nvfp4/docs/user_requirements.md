# User Requirements

- 以训练营任务二原始要求为主，提供 MXFP8/NVFP4 量化与反量化基线。
- 所有公开功能必须同时具备 CPU reference 与 CUDA 路径，并能做准确性对照。
- 本地仅要求完成 CPU 构建、测试和实验；CUDA 运行由用户通过 SSH 在服务器完成。
- 关键函数保留中文注释，明确数值口径、数据布局和 CPU/GPU 职责。
- 不得把未在真实 GPU 上运行的静态检查写成 GPU 性能或正确性结论。

### Document Preferences

- 语言：中文正文 + English 标题。
- Data Format：保留。
- 实验表格：单表优先，原始日志必须可追溯。
- README/REPORT 是对外文档；`docs/` 记录设计和追加式迭代历史。

