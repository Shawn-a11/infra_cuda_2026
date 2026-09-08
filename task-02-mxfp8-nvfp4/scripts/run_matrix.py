#!/usr/bin/env python3
"""Run all generated datasets against MXFP8/NVFP4 block/tensor configs."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("build/low_precision_tool"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/matrix"))
    return parser.parse_args()


def read_log(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            key, value = line.split("=", 1)
            values[key] = value
    return values


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    configs = sorted((root / "configs").glob("*.txt"))
    datasets = sorted(args.data_dir.glob("*.bin"))
    if not datasets:
        raise SystemExit("no datasets found; run scripts/generate_data.py first")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for dataset in datasets:
        for config in configs:
            stem = f"{dataset.stem}_{config.stem}"
            case_dir = args.output_dir / stem
            case_dir.mkdir(parents=True, exist_ok=True)
            metrics = case_dir / "metrics.log"
            performance = case_dir / "performance.log"
            subprocess.run(
                [
                    str(args.binary), "--input", str(dataset),
                    "--config", str(config), "--backend", args.backend,
                    "--quantized", str(case_dir / "quantized.bin"),
                    "--dequantized", str(case_dir / "dequantized.bin"),
                    "--metrics", str(metrics), "--performance", str(performance),
                ],
                check=True,
            )
            rows.append({"dataset": dataset.name, "config": config.name,
                         **read_log(metrics), **read_log(performance)})
    columns = sorted({key for row in rows for key in row})
    with (args.output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
