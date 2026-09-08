# User Requirements

- 以训练营任务九为主，提供 exact 与 IVF-Flat GPU 向量检索及 CPU ground truth。
- 所有公开检索功能必须能在 CPU/GPU 上准确对照，最终覆盖任务要求的数据规模。
- 本地只要求 CPU 构建与测试；CUDA 正确性、性能和 profiler 在 SSH 服务器运行。
- 关键函数保留中文注释，明确度量、排序、索引和内存/计时边界。

### Document Preferences

- 语言：中文正文 + English 标题。
- 正确性、Recall/QPS trade-off、exact/reference 和 profiler 分节。
- 所有最终数字保留配置与原始日志。

