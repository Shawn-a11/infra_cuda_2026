#!/usr/bin/env python3
"""Run one configuration on CPU/GPU and enforce numerical comparison gates."""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("build/derivatives_pricer"))
    parser.add_argument("--option", type=Path, required=True)
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--z-score", type=float, default=5.0)
    parser.add_argument("--price-absolute-tolerance", type=float, default=0.02)
    parser.add_argument("--greek-relative-tolerance", type=float, default=0.10)
    parser.add_argument("--greek-absolute-tolerance", type=float, default=0.01)
    return parser.parse_args()


def read_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if "=" in raw_line:
            key, value = raw_line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def run_backend(args: argparse.Namespace, backend: str) -> dict[str, str]:
    """执行同一输入，仅切换 backend，确保比较边界一致。"""
    result = args.output_dir / f"{backend}.result.txt"
    performance = args.output_dir / f"{backend}.performance.log"
    subprocess.run(
        [
            str(args.binary), "--option", str(args.option),
            "--simulation", str(args.simulation), "--backend", backend,
            "--reference-paths", "0", "--output", str(result),
            "--performance", str(performance),
        ],
        check=True,
    )
    values = read_key_values(result)
    values.update(read_key_values(performance))
    return values


def main() -> None:
    args = parse_args()
    if args.z_score <= 0.0 or args.price_absolute_tolerance < 0.0:
        raise ValueError("comparison tolerances must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cpu = run_backend(args, "cpu")
    gpu = run_backend(args, "cuda")

    price_difference = abs(float(cpu["price"]) - float(gpu["price"]))
    combined_standard_error = math.hypot(
        float(cpu["standard_error"]), float(gpu["standard_error"])
    )
    price_tolerance = (
        args.z_score * combined_standard_error + args.price_absolute_tolerance
    )
    rows: list[dict[str, str]] = [
        {
            "metric": "price",
            "cpu": cpu["price"],
            "gpu": gpu["price"],
            "absolute_difference": f"{price_difference:.17g}",
            "tolerance": f"{price_tolerance:.17g}",
            "passed": str(price_difference <= price_tolerance).lower(),
            "rule": "difference <= z*joint_standard_error + absolute_tolerance",
        }
    ]

    if cpu.get("greeks_computed") == "true" and gpu.get("greeks_computed") == "true":
        for metric in ("delta", "gamma", "vega"):
            cpu_value = float(cpu[metric])
            gpu_value = float(gpu[metric])
            difference = abs(cpu_value - gpu_value)
            tolerance = max(
                args.greek_absolute_tolerance,
                args.greek_relative_tolerance * max(abs(cpu_value), abs(gpu_value)),
            )
            rows.append(
                {
                    "metric": metric,
                    "cpu": f"{cpu_value:.17g}",
                    "gpu": f"{gpu_value:.17g}",
                    "absolute_difference": f"{difference:.17g}",
                    "tolerance": f"{tolerance:.17g}",
                    "passed": str(difference <= tolerance).lower(),
                    "rule": "difference <= max(absolute, relative*scale)",
                }
            )

    comparison_path = args.output_dir / "comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    failed = [row["metric"] for row in rows if row["passed"] != "true"]
    if failed:
        raise SystemExit("CPU/GPU comparison failed: " + ", ".join(failed))
    print("CPU/GPU comparison passed: " + ", ".join(row["metric"] for row in rows))


if __name__ == "__main__":
    main()
