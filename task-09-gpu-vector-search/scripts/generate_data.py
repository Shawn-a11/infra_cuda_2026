#!/usr/bin/env python3
"""Generate deterministic vector-database and query files for this project."""

from __future__ import annotations

import argparse
import random
import struct
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:  # A pure-Python smoke-data fallback is provided.
    np = None


DTYPE_CODES = {"fp32": 1, "fp16": 2}
METRICS = {"l2": 1, "inner_product": 2, "cosine": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/smoke"))
    parser.add_argument("--num-vectors", type=int, default=10_000)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--dtype", choices=DTYPE_CODES, default="fp32")
    parser.add_argument("--metric", choices=METRICS, default="l2")
    parser.add_argument("--clusters", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def generate_numpy(args: argparse.Namespace) -> tuple[object, object]:
    rng = np.random.default_rng(args.seed)
    centers = rng.normal(size=(args.clusters, args.dim)).astype(np.float32)
    assignments = rng.integers(0, args.clusters, size=args.num_vectors)
    database = centers[assignments] + 0.15 * rng.normal(
        size=(args.num_vectors, args.dim)
    ).astype(np.float32)
    query_sources = rng.integers(0, args.num_vectors, size=args.num_queries)
    queries = database[query_sources] + 0.02 * rng.normal(
        size=(args.num_queries, args.dim)
    ).astype(np.float32)
    return database, queries


def write_numpy(args: argparse.Namespace, database: object, queries: object) -> None:
    storage_dtype = np.dtype("<f4" if args.dtype == "fp32" else "<f2")
    database_path = args.output_dir / "database.bin"
    query_path = args.output_dir / "queries.bin"
    with database_path.open("wb") as stream:
        stream.write(
            struct.pack(
                "<8sIqibbh",
                b"VECDB01\0",
                1,
                args.num_vectors,
                args.dim,
                DTYPE_CODES[args.dtype],
                METRICS[args.metric],
                0,
            )
        )
        stream.write(database.astype(storage_dtype, copy=False).tobytes(order="C"))
    with query_path.open("wb") as stream:
        stream.write(
            struct.pack(
                "<8sIqib3x",
                b"VECQRY1\0",
                1,
                args.num_queries,
                args.dim,
                DTYPE_CODES[args.dtype],
            )
        )
        stream.write(queries.astype(storage_dtype, copy=False).tobytes(order="C"))


def write_pure_python(args: argparse.Namespace) -> None:
    """Stream smoke data without retaining the whole database in Python lists."""
    if args.num_vectors * args.dim > 10_000_000:
        raise RuntimeError(
            "NumPy is required for datasets above 10 million elements; "
            "install requirements.txt for the full-scale generator"
        )
    rng = random.Random(args.seed)
    centers = [
        [rng.gauss(0.0, 1.0) for _ in range(args.dim)]
        for _ in range(args.clusters)
    ]
    query_sources = [rng.randrange(args.num_vectors) for _ in range(args.num_queries)]
    needed_sources: dict[int, list[int]] = {}
    for query_id, vector_id in enumerate(query_sources):
        needed_sources.setdefault(vector_id, []).append(query_id)
    query_bases: list[list[float] | None] = [None] * args.num_queries
    element_format = "f" if args.dtype == "fp32" else "e"
    row_format = "<" + element_format * args.dim

    database_path = args.output_dir / "database.bin"
    query_path = args.output_dir / "queries.bin"
    with database_path.open("wb") as stream:
        stream.write(
            struct.pack(
                "<8sIqibbh",
                b"VECDB01\0",
                1,
                args.num_vectors,
                args.dim,
                DTYPE_CODES[args.dtype],
                METRICS[args.metric],
                0,
            )
        )
        for vector_id in range(args.num_vectors):
            center = centers[rng.randrange(args.clusters)]
            vector = [value + 0.15 * rng.gauss(0.0, 1.0) for value in center]
            stream.write(struct.pack(row_format, *vector))
            for query_id in needed_sources.get(vector_id, []):
                query_bases[query_id] = vector
    with query_path.open("wb") as stream:
        stream.write(
            struct.pack(
                "<8sIqib3x",
                b"VECQRY1\0",
                1,
                args.num_queries,
                args.dim,
                DTYPE_CODES[args.dtype],
            )
        )
        for query_id, base in enumerate(query_bases):
            if base is None:
                raise RuntimeError(f"query source {query_id} was not generated")
            query = [value + 0.02 * rng.gauss(0.0, 1.0) for value in base]
            stream.write(struct.pack(row_format, *query))


def main() -> None:
    args = parse_args()
    if min(args.num_vectors, args.num_queries, args.dim, args.clusters) <= 0:
        raise ValueError("all dimensions must be positive")
    if args.clusters > args.num_vectors:
        raise ValueError("clusters cannot exceed num_vectors")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if np is None:
        print("NumPy not found; using the standard-library smoke-data generator")
        write_pure_python(args)
    else:
        database, queries = generate_numpy(args)
        write_numpy(args, database, queries)

    database_path = args.output_dir / "database.bin"
    query_path = args.output_dir / "queries.bin"
    print(database_path)
    print(query_path)


if __name__ == "__main__":
    main()
