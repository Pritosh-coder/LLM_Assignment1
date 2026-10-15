#!/usr/bin/env python3
"""Combine independently written Part A rows in the mandated A1-A6 order."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import PART_A_RESULT_COLUMNS, PART_A_SWEEP


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    return parser


def combine(input_dir: Path, output_file: Path) -> None:
    if output_file.exists():
        raise ValueError(f"output already exists: {output_file}")

    rows: list[dict[str, str]] = []
    for run_id in PART_A_SWEEP:
        path = input_dir / f"partA_{run_id}.csv"
        if not path.is_file():
            raise ValueError(f"missing result file: {path}")
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != PART_A_RESULT_COLUMNS:
                raise ValueError(f"incompatible header: {path}")
            values = list(reader)
        if len(values) != 1 or values[0].get("run_id") != run_id:
            raise ValueError(f"expected exactly one {run_id} row in {path}")
        rows.append(values[0])

    output_file.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_file.name}.", suffix=".tmp", dir=output_file.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PART_A_RESULT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, output_file)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    args = build_parser().parse_args()
    try:
        combine(args.input_dir, args.output_file)
    except ValueError as error:
        raise SystemExit(f"ERROR: {error}") from error
    print(output_file := args.output_file)


if __name__ == "__main__":
    main()
