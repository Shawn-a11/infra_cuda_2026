#!/usr/bin/env python3
"""Run exact/IVF CPU-GPU correctness gates on a CUDA server."""

from __future__ import annotations

import argparse
import csv
import hashlib
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


METRICS = ("l2", "inner_product", "cosine")
DTYPES = ("fp32", "fp16")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("build/vector_search"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/gpu_validation")
    )
    parser.add_argument("--num-vectors", type=int, default=4096)
    parser.add_argument("--num-queries", type=int, default=32)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--clusters", type=int, default=32)
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 10, 50, 100])
    parser.add_argument("--batch-size", type=int, default=17)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--score-tolerance", type=float, default=2.0e-4)
    return parser.parse_args()


def optional_command(command: list[str]) -> str:
    """环境工具缺失时记录 unavailable，但不替代真正的 CUDA 数值 gate。"""
    executable = shutil.which(command[0])
    if executable is None:
        return f"unavailable: {command[0]}"
    completed = subprocess.run(
        [executable, *command[1:]], capture_output=True, text=True, check=False
    )
    output = (completed.stdout or completed.stderr).strip()
    return output if completed.returncode == 0 else (
        f"failed ({completed.returncode}): {output}"
    )


def write_environment(output_dir: Path) -> None:
    """保存硬件和工具链，供 REPORT.md 追溯每一组 GPU 结果。"""
    sections = {
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "nvidia-smi -L": optional_command(["nvidia-smi", "-L"]),
        "nvidia-smi inventory": optional_command(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "nvcc --version": optional_command(["nvcc", "--version"]),
    }
    text = "".join(f"[{name}]\n{value}\n\n" for name, value in sections.items())
    (output_dir / "environment.txt").write_text(text, encoding="utf-8")


def run(command: list[str], command_log: Path) -> None:
    """记录并执行无 shell 命令，保证服务器验收步骤可以逐条复现。"""
    with command_log.open("a", encoding="utf-8") as stream:
        stream.write(shlex.join(command) + "\n")
    subprocess.run(command, check=True)


def read_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def write_config(
    path: Path,
    *,
    mode: str,
    top_k: int,
    batch_size: int,
    nlist: int,
    nprobe: int,
    training_samples: int,
    quality_queries: int,
    seed: int,
) -> None:
    """为每个 gate 固化完整配置，避免命令行矩阵与报告参数脱节。"""
    values = {
        "top_k": top_k,
        "search_mode": mode,
        "batch_size": batch_size,
        "nlist": nlist,
        "nprobe": nprobe,
        "pq_m": 16,
        "kmeans_iterations": 3,
        "training_samples": training_samples,
        "quality_queries": quality_queries,
        "seed": seed,
    }
    path.write_text(
        "".join(f"{key} = {value}\n" for key, value in values.items()),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run_search(
    *,
    binary: Path,
    database: Path,
    queries: Path,
    config: Path,
    backend: str,
    case_dir: Path,
    command_log: Path,
    index: Path | None = None,
) -> dict[str, str]:
    case_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(binary),
        "search",
        "--database",
        str(database),
        "--queries",
        str(queries),
        "--config",
        str(config),
        "--backend",
        backend,
        "--output",
        str(case_dir / f"{backend}.results.txt"),
        "--performance",
        str(case_dir / f"{backend}.performance.log"),
        "--quality",
        str(case_dir / f"{backend}.quality.log"),
    ]
    if index is not None:
        command.extend(["--index", str(index)])
    run(command, command_log)
    values = read_key_values(case_dir / f"{backend}.performance.log")
    values.update(read_key_values(case_dir / f"{backend}.quality.log"))
    return values


def gate_row(
    *,
    mode: str,
    metric: str,
    dtype: str,
    top_k: int,
    cpu: dict[str, str],
    gpu: dict[str, str],
    database_hash: str,
    query_hash: str,
    score_tolerance: float,
) -> dict[str, str]:
    """Exact 使用绝对真值 gate；IVF 比较同一索引上的 CPU/GPU 质量统计。"""
    cpu_recall = float(cpu["recall_at_k"])
    gpu_recall = float(gpu["recall_at_k"])
    cpu_rank = float(cpu["rank_id_agreement"])
    gpu_rank = float(gpu["rank_id_agreement"])
    cpu_score_error = float(cpu["mean_rank_score_absolute_error"])
    gpu_score_error = float(gpu["mean_rank_score_absolute_error"])
    recall_difference = abs(cpu_recall - gpu_recall)
    rank_difference = abs(cpu_rank - gpu_rank)
    score_error_difference = abs(cpu_score_error - gpu_score_error)
    if mode == "exact":
        passed = (
            gpu_recall >= 1.0 - 1.0e-12
            and gpu_rank >= 1.0 - 1.0e-12
            and gpu_score_error <= score_tolerance
        )
    else:
        passed = (
            recall_difference <= 1.0e-12
            and rank_difference <= 1.0e-12
            and score_error_difference <= score_tolerance
        )
    cpu_qps = float(cpu["qps"])
    gpu_qps = float(gpu["qps"])
    return {
        "mode": mode,
        "metric": metric,
        "dtype": dtype,
        "top_k": str(top_k),
        "num_vectors": gpu["num_vectors"],
        "dim": gpu["dim"],
        "num_queries": gpu["num_queries"],
        "batch_size": gpu["batch_size"],
        "nlist": gpu["nlist"],
        "nprobe": gpu["nprobe"],
        "cpu_qps": f"{cpu_qps:.17g}",
        "gpu_qps": f"{gpu_qps:.17g}",
        "speedup": f"{gpu_qps / cpu_qps:.17g}" if cpu_qps > 0.0 else "0",
        "cpu_recall_at_k": f"{cpu_recall:.17g}",
        "gpu_recall_at_k": f"{gpu_recall:.17g}",
        "recall_difference": f"{recall_difference:.17g}",
        "cpu_rank_agreement": f"{cpu_rank:.17g}",
        "gpu_rank_agreement": f"{gpu_rank:.17g}",
        "rank_difference": f"{rank_difference:.17g}",
        "score_error_difference": f"{score_error_difference:.17g}",
        "score_tolerance": f"{score_tolerance:.17g}",
        "database_sha256": database_hash,
        "queries_sha256": query_hash,
        "passed": str(passed).lower(),
    }


def write_report_table(rows: list[dict[str, str]], output_path: Path) -> None:
    """生成适合粘贴到 REPORT.md 的核心正确性/性能草表。"""
    fields = [
        "mode",
        "metric",
        "dtype",
        "top_k",
        "cpu_qps",
        "gpu_qps",
        "speedup",
        "gpu_recall_at_k",
        "gpu_rank_agreement",
        "score_error_difference",
        "passed",
    ]
    labels = [
        "Mode",
        "Metric",
        "Dtype",
        "K",
        "CPU QPS",
        "GPU QPS",
        "Speedup",
        "GPU Recall@K",
        "GPU Rank agree.",
        "CPU/GPU score-error diff",
        "Pass",
    ]
    lines = [
        "| " + " | ".join(labels) + " |\n",
        "|" + "|".join("---" for _ in labels) + "|\n",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[field] for field in fields) + " |\n")
    output_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if (
        min(
            args.num_vectors,
            args.num_queries,
            args.dim,
            args.clusters,
            args.batch_size,
        )
        <= 0
        or args.clusters > args.num_vectors
        or args.score_tolerance < 0.0
        or any(top_k <= 0 or top_k > 100 for top_k in args.top_k)
        or max(args.top_k) > args.num_vectors
    ):
        raise ValueError("invalid validation dimensions, Top-K, or tolerance")

    root = Path(__file__).resolve().parents[1]
    binary = args.binary.resolve()
    if not binary.is_file():
        raise SystemExit(f"binary does not exist: {binary}")
    generator = root / "scripts" / "generate_data.py"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command_log = output_dir / "commands.txt"
    command_log.write_text("", encoding="utf-8")
    write_environment(output_dir)

    rows: list[dict[str, str]] = []
    nlist = min(args.clusters, args.num_vectors)
    nprobe = min(8, nlist)
    training_samples = min(max(2048, nlist), args.num_vectors)
    ivf_top_k = min(10, args.num_vectors)

    for metric_index, metric in enumerate(METRICS):
        for dtype_index, dtype in enumerate(DTYPES):
            case = f"{metric}_{dtype}"
            case_dir = output_dir / case
            data_dir = case_dir / "data"
            case_seed = args.seed + metric_index * 100 + dtype_index
            run(
                [
                    sys.executable,
                    str(generator),
                    "--output-dir",
                    str(data_dir),
                    "--num-vectors",
                    str(args.num_vectors),
                    "--num-queries",
                    str(args.num_queries),
                    "--dim",
                    str(args.dim),
                    "--dtype",
                    dtype,
                    "--metric",
                    metric,
                    "--clusters",
                    str(args.clusters),
                    "--seed",
                    str(case_seed),
                ],
                command_log,
            )
            database = data_dir / "database.bin"
            queries = data_dir / "queries.bin"
            database_hash = sha256(database)
            query_hash = sha256(queries)

            for top_k in args.top_k:
                exact_dir = case_dir / "exact" / f"k{top_k}"
                exact_config = exact_dir / "config.txt"
                exact_dir.mkdir(parents=True, exist_ok=True)
                write_config(
                    exact_config,
                    mode="exact",
                    top_k=top_k,
                    batch_size=args.batch_size,
                    nlist=nlist,
                    nprobe=nprobe,
                    training_samples=training_samples,
                    quality_queries=args.num_queries,
                    seed=case_seed,
                )
                cpu = run_search(
                    binary=binary,
                    database=database,
                    queries=queries,
                    config=exact_config,
                    backend="cpu",
                    case_dir=exact_dir,
                    command_log=command_log,
                )
                gpu = run_search(
                    binary=binary,
                    database=database,
                    queries=queries,
                    config=exact_config,
                    backend="cuda",
                    case_dir=exact_dir,
                    command_log=command_log,
                )
                rows.append(
                    gate_row(
                        mode="exact",
                        metric=metric,
                        dtype=dtype,
                        top_k=top_k,
                        cpu=cpu,
                        gpu=gpu,
                        database_hash=database_hash,
                        query_hash=query_hash,
                        score_tolerance=args.score_tolerance,
                    )
                )

            ivf_dir = case_dir / "ivf"
            ivf_dir.mkdir(parents=True, exist_ok=True)
            ivf_config = ivf_dir / "config.txt"
            ivf_index = ivf_dir / "gpu_built.ivf"
            write_config(
                ivf_config,
                mode="ivf_flat",
                top_k=ivf_top_k,
                batch_size=args.batch_size,
                nlist=nlist,
                nprobe=nprobe,
                training_samples=training_samples,
                quality_queries=args.num_queries,
                seed=case_seed,
            )
            run(
                [
                    str(binary),
                    "build",
                    "--database",
                    str(database),
                    "--config",
                    str(ivf_config),
                    "--index",
                    str(ivf_index),
                    "--backend",
                    "cuda",
                    "--performance",
                    str(ivf_dir / "cuda.build.log"),
                ],
                command_log,
            )
            cpu = run_search(
                binary=binary,
                database=database,
                queries=queries,
                config=ivf_config,
                backend="cpu",
                case_dir=ivf_dir,
                command_log=command_log,
                index=ivf_index,
            )
            gpu = run_search(
                binary=binary,
                database=database,
                queries=queries,
                config=ivf_config,
                backend="cuda",
                case_dir=ivf_dir,
                command_log=command_log,
                index=ivf_index,
            )
            rows.append(
                gate_row(
                    mode="ivf_flat",
                    metric=metric,
                    dtype=dtype,
                    top_k=ivf_top_k,
                    cpu=cpu,
                    gpu=gpu,
                    database_hash=database_hash,
                    query_hash=query_hash,
                    score_tolerance=args.score_tolerance,
                )
            )

    summary = output_dir / "summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_report_table(rows, output_dir / "report_table.md")
    failed = [
        f"{row['mode']}:{row['metric']}:{row['dtype']}:k{row['top_k']}"
        for row in rows
        if row["passed"] != "true"
    ]
    if failed:
        raise SystemExit("CPU/GPU validation failed: " + ", ".join(failed))
    print(
        f"all {len(rows)} vector-search CPU/GPU gates passed; "
        f"artifacts: {summary}, {output_dir / 'environment.txt'}, "
        f"{output_dir / 'report_table.md'}"
    )


if __name__ == "__main__":
    main()
