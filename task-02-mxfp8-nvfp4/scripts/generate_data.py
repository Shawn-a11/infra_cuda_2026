#!/usr/bin/env python3
"""Generate deterministic FP32/FP16 matrices in the task-02 binary format."""

from __future__ import annotations

import argparse
import math
import random
import struct
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--cols", type=int, default=256)
    parser.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def write_tensor(path: Path, rows: int, cols: int, dtype: str,
                 values: list[float]) -> None:
    """写入固定 32-byte header 和 row-major payload。"""
    dtype_code = 1 if dtype == "fp32" else 2
    payload_format = "<f" if dtype == "fp32" else "<e"
    with path.open("wb") as stream:
        stream.write(b"LPTENS1\0")
        stream.write(struct.pack("<IqqB3x", 1, rows, cols, dtype_code))
        for value in values:
            stream.write(struct.pack(payload_format, value))


def main() -> None:
    args = parse_args()
    if args.rows <= 0 or args.cols <= 0:
        raise ValueError("rows and cols must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    count = args.rows * args.cols
    datasets = {
        "uniform": [rng.uniform(-8.0, 8.0) for _ in range(count)],
        "normal": [rng.gauss(0.0, 2.0) for _ in range(count)],
    }
    outliers = [rng.gauss(0.0, 1.0) for _ in range(count)]
    stride = max(1, count // 100)
    for index in range(0, count, stride):
        outliers[index] = math.copysign(64.0 + index % 17, -1.0 if index & 1 else 1.0)
    datasets["outliers"] = outliers
    for name, values in datasets.items():
        path = args.output_dir / f"{name}_{args.dtype}.bin"
        write_tensor(path, args.rows, args.cols, args.dtype, values)
        print(path)


if __name__ == "__main__":
    main()
