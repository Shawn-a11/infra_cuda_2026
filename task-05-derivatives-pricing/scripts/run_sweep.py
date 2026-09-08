#!/usr/bin/env python3
"""Run convergence and variance-reduction sweeps without external packages."""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("build/derivatives_pricer"))
    parser.add_argument("--option", type=Path, default=Path("configs/european_call.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sweep"))
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--paths", type=int, nargs="+", default=[10_000, 100_000, 1_000_000])
    parser.add_argument(
        "--variance", nargs="+", choices=("none", "antithetic", "control_variate"),
        default=["none", "antithetic", "control_variate"]
    )
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--master-seed", type=int, default=20260824)
    parser.add_argument(
        "--seed", type=int, default=None,
        help="兼容旧命令：提供时覆盖 --master-seed",
    )
    parser.add_argument("--replications", type=int, default=1)
    parser.add_argument(
        "--rng", choices=("curand", "custom", "halton"), default="curand"
    )
    parser.add_argument("--qmc-replications", type=int, default=8)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--compute-greeks", action="store_true")
    parser.add_argument(
        "--reference-paths", type=int, default=200_000,
        help="Asian/Barrier 固定 CPU reference 的路径数；0 表示不计算",
    )
    return parser.parse_args()


MASK64 = (1 << 64) - 1


def derive_replication_seed(master_seed: int, replication_id: int) -> int:
    """用 SplitMix64 派生可复现的 64 位 replication seed。"""
    # 不直接使用相邻整数 seed；混合后仍可由 master seed 和 replication id 完整复现。
    value = (master_seed + 0x9E3779B97F4A7C15 * (replication_id + 1)) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def load_key_values(path: Path) -> dict[str, str]:
    """读取程序生成的 key=value 日志，保留原始字符串精度。"""
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def write_aggregate(rows: list[dict[str, str]], output_path: Path) -> None:
    """按配置聚合独立 replication，区分单次 SE 与跨 seed 经验波动。"""
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row["num_paths"], row["variance_reduction"], row.get("backend", ""),
            row.get("rng", ""), row.get("num_steps", ""),
        )
        groups[key].append(row)

    aggregate_rows: list[dict[str, str]] = []
    for key, group in groups.items():
        def values(field: str) -> list[float]:
            return [float(row[field]) for row in group]

        prices = values("price")
        runtimes = values("total_runtime_ms")
        aggregate_rows.append(
            {
                "num_paths": key[0],
                "variance_reduction": key[1],
                "backend": key[2],
                "rng": key[3],
                "num_steps": key[4],
                "replications": str(len(group)),
                "mean_price": f"{statistics.fmean(prices):.17g}",
                "empirical_price_sd":
                    f"{statistics.stdev(prices) if len(prices) > 1 else 0.0:.17g}",
                "mean_reported_standard_error":
                    f"{statistics.fmean(values('standard_error')):.17g}",
                "mean_absolute_error":
                    f"{statistics.fmean(values('absolute_error')):.17g}",
                "mean_total_runtime_ms": f"{statistics.fmean(runtimes):.17g}",
                "median_total_runtime_ms": f"{statistics.median(runtimes):.17g}",
                "mean_end_to_end_paths_per_second":
                    f"{statistics.fmean(values('end_to_end_paths_per_second')):.17g}",
            }
        )

    columns = list(aggregate_rows[0]) if aggregate_rows else []
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        if columns:
            writer.writeheader()
            writer.writerows(aggregate_rows)


def main() -> None:
    args = parse_args()
    if args.replications <= 0 or any(paths < 2 for paths in args.paths):
        raise ValueError("replications must be positive and every path count must be >= 2")
    master_seed = args.seed if args.seed is not None else args.master_seed
    if master_seed < 0 or master_seed > MASK64:
        raise ValueError("master seed must fit in uint64")
    if args.reference_paths == 1 or args.reference_paths < 0:
        raise ValueError("reference paths must be 0 or >= 2")
    if args.qmc_replications < 2 or args.num_gpus <= 0:
        raise ValueError("qmc replications must be >=2 and num_gpus must be positive")
    if args.rng == "halton" and any(
        paths % args.qmc_replications != 0 for paths in args.paths
    ):
        raise ValueError("every Halton path count must divide qmc_replications")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="pricing_sweep_") as temporary:
        config_path = Path(temporary) / "simulation.txt"
        option_values = load_key_values(args.option)
        option_type = option_values.get("option_type", "").strip("\"'")
        model = option_values.get("model", "black_scholes").strip("\"'")
        path_dependent = (
            model not in {"black_scholes", "gbm"}
            or option_type in {"asian_call", "barrier_call"}
        )
        fixed_reference: dict[str, str] | None = None
        reference_seed = derive_replication_seed(master_seed, 0x524546)
        if path_dependent and args.reference_paths >= 2:
            # 固定 reference 只计算一次；被测组合均使用另一组 replication seeds。
            reference_variance = (
                "none" if option_type in {"american_call", "american_put"}
                else "control_variate"
            )
            config_path.write_text(
                f"num_paths = {args.reference_paths}\nnum_steps = {args.steps}\n"
                f"seed = {reference_seed}\nrng = curand\n"
                f"variance_reduction = {reference_variance}\n"
                f"qmc_replications = {args.qmc_replications}\n"
                "num_gpus = 1\ncompute_greeks = false\n",
                encoding="utf-8",
            )
            reference_result = args.output_dir / "reference.result.txt"
            reference_performance = args.output_dir / "reference.performance.log"
            subprocess.run(
                [
                    str(args.binary), "--option", str(args.option),
                    "--simulation", str(config_path), "--backend", "cpu",
                    "--reference-paths", "0", "--output", str(reference_result),
                    "--performance", str(reference_performance),
                ],
                check=True,
            )
            fixed_reference = load_key_values(reference_result)
        for replication_id in range(args.replications):
            replication_seed = derive_replication_seed(master_seed, replication_id)
            # 同一 replication 的所有方法共享 seed，实现 common random numbers；
            # 不同 path count 从同一随机流前缀开始，便于绘制嵌套收敛曲线。
            for paths in args.paths:
                for variance in args.variance:
                    config_path.write_text(
                        f"num_paths = {paths}\nnum_steps = {args.steps}\n"
                        f"seed = {replication_seed}\nrng = {args.rng}\n"
                        f"variance_reduction = {variance}\n"
                        f"qmc_replications = {args.qmc_replications}\n"
                        f"num_gpus = {args.num_gpus}\n"
                        f"compute_greeks = {'true' if args.compute_greeks else 'false'}\n",
                        encoding="utf-8",
                    )
                    stem = f"rep_{replication_id:03d}_paths_{paths}_{variance}"
                    result_path = args.output_dir / f"{stem}.result.txt"
                    performance_path = args.output_dir / f"{stem}.performance.log"
                    subprocess.run(
                        [
                            str(args.binary), "--option", str(args.option),
                            "--simulation", str(config_path), "--backend", args.backend,
                            "--reference-paths", "0",
                            "--output", str(result_path), "--performance",
                            str(performance_path),
                        ],
                        check=True,
                    )
                    row = {
                        "master_seed": str(master_seed),
                        "replication_id": str(replication_id),
                        "replication_seed": str(replication_seed),
                        "num_paths": str(paths),
                        "variance_reduction": variance,
                    }
                    row.update(load_key_values(result_path))
                    row.update(load_key_values(performance_path))
                    if fixed_reference is not None:
                        reference_price = float(fixed_reference["price"])
                        row["reference_price"] = fixed_reference["price"]
                        row["reference_method"] = (
                            f"fixed_cpu_control_variate_{args.reference_paths}_samples"
                        )
                        row["reference_seed"] = str(reference_seed)
                        row["reference_standard_error"] = fixed_reference["standard_error"]
                        row["reference_confidence_95_low"] = fixed_reference["confidence_95_low"]
                        row["reference_confidence_95_high"] = fixed_reference["confidence_95_high"]
                        row["absolute_error"] = f"{abs(float(row['price']) - reference_price):.17g}"
                    rows.append(row)

    columns = [
        "master_seed", "replication_id", "replication_seed", "num_paths",
        "model", "num_steps", "variance_reduction", "rng", "rng_requested", "backend", "price",
        "reference_price", "reference_method", "reference_seed",
        "reference_standard_error", "reference_confidence_95_low",
        "reference_confidence_95_high", "absolute_error", "standard_error", "confidence_95_low",
        "confidence_95_high", "delta", "gamma", "vega", "greeks_runtime_ms",
        "total_runtime_ms", "simulation_kernel_ms",
        "reduction_ms", "physical_paths", "kernel_paths_per_second",
        "end_to_end_paths_per_second", "throughput_speedup_vs_cpu_single_thread",
    ]
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    write_aggregate(rows, args.output_dir / "aggregate.csv")


if __name__ == "__main__":
    main()
