#!/usr/bin/env python3
"""Sweep Top-K, nprobe and batch size while preserving raw logs."""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("build/vector_search"))
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, default=Path("configs/ivf_smoke.txt"))
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sweep"))
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 10, 50, 100])
    parser.add_argument("--nprobe", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--batch-size", type=int, nargs="+", default=[1, 32, 128])
    parser.add_argument("--repetitions", type=int, default=1)
    return parser.parse_args()


def read_config(path: Path) -> dict[str, str]:
    """读取 config/performance/quality 共用的 key=value 文本格式。"""
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"')
    return values


def read_log(path: Path) -> dict[str, str]:
    return read_config(path)


def write_aggregate(rows: list[dict[str, str]], output_path: Path) -> None:
    """聚合重复 benchmark，性能使用均值和中位数，质量指标保留均值。"""
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["top_k"], row["nprobe"], row["batch_size"],
                row.get("backend", ""))].append(row)

    output_rows: list[dict[str, str]] = []
    for key, group in groups.items():
        def values(field: str) -> list[float]:
            return [float(row[field]) for row in group]

        qps = values("qps")
        output_rows.append(
            {
                "top_k": key[0], "nprobe": key[1], "batch_size": key[2],
                "backend": key[3], "repetitions": str(len(group)),
                "mean_qps": f"{statistics.fmean(qps):.17g}",
                "median_qps": f"{statistics.median(qps):.17g}",
                "mean_p50_latency_ms":
                    f"{statistics.fmean(values('p50_latency_ms')):.17g}",
                "mean_p99_latency_ms":
                    f"{statistics.fmean(values('p99_latency_ms')):.17g}",
                "mean_recall_at_k":
                    f"{statistics.fmean(values('recall_at_k')):.17g}",
                "mean_rank_id_agreement":
                    f"{statistics.fmean(values('rank_id_agreement')):.17g}",
            }
        )

    columns = list(output_rows[0]) if output_rows else []
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        if columns:
            writer.writeheader()
            writer.writerows(output_rows)


def main() -> None:
    args = parse_args()
    if args.repetitions <= 0:
        raise ValueError("repetitions must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = read_config(args.base_config)
    rows: list[dict[str, str]] = []
    for top_k in args.top_k:
        for nprobe in args.nprobe:
            for batch_size in args.batch_size:
                if nprobe > int(base["nlist"]):
                    continue
                config_stem = f"k{top_k}_p{nprobe}_b{batch_size}"
                config = dict(base)
                config.update(
                    top_k=str(top_k), nprobe=str(nprobe), batch_size=str(batch_size)
                )
                config_path = args.output_dir / f"{config_stem}.config.txt"
                config_path.write_text(
                    "".join(f"{key} = {value}\n" for key, value in config.items()),
                    encoding="utf-8",
                )
                # 搜索本身是确定性的；重复执行用于估计 QPS/延迟的系统噪声。
                for repetition in range(args.repetitions):
                    stem = f"{config_stem}_r{repetition:03d}"
                    result_path = args.output_dir / f"{stem}.results.txt"
                    performance_path = args.output_dir / f"{stem}.performance.log"
                    quality_path = args.output_dir / f"{stem}.quality.log"
                    subprocess.run(
                        [
                            str(args.binary), "search", "--database", str(args.database),
                            "--queries", str(args.queries), "--config", str(config_path),
                            "--index", str(args.index), "--backend", args.backend,
                            "--output", str(result_path), "--performance",
                            str(performance_path), "--quality", str(quality_path),
                        ],
                        check=True,
                    )
                    row = {"top_k": str(top_k), "nprobe": str(nprobe),
                           "batch_size": str(batch_size),
                           "repetition": str(repetition)}
                    row.update(read_log(performance_path))
                    row.update(read_log(quality_path))
                    rows.append(row)

    columns = [
        "top_k", "nprobe", "batch_size", "repetition", "backend", "qps",
        "p50_latency_ms", "p99_latency_ms", "recall_at_k", "rank_id_agreement",
        "mean_rank_score_absolute_error",
        "mean_candidates_per_query", "peak_candidates_per_query",
        "resident_data_and_index_bytes", "backend_working_set_bytes",
        "estimated_device_working_set_bytes", "speedup_vs_cpu_exact",
    ]
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    write_aggregate(rows, args.output_dir / "aggregate.csv")


if __name__ == "__main__":
    main()
