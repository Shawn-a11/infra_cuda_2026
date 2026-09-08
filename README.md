# 2026 Summer CUDA Projects

Current priority:

1. `task-05-derivatives-pricing` - primary project.
2. `task-09-gpu-vector-search` - primary project.
3. `task-02-mxfp8-nvfp4` - retained as a lower-priority follow-up.

Tasks 02, 05, and 09 are independent CMake projects. Each includes a portable
CPU reference, an optional CUDA backend enabled automatically by CMake, tests,
sample configurations, benchmark helpers, and a report template. Task 02 is
now an initial correctness-first MXFP8/NVFP4 implementation; tasks 05 and 09
remain the submission priorities. GPU performance claims must be filled only
after running on the training-server hardware.
