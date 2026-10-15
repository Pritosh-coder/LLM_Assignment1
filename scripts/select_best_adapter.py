#!/usr/bin/env python3
"""Select the Part A adapter by a documented validation-only tie-break."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import PART_A_RESULT_COLUMNS, PART_A_SWEEP  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select the best Part A adapter.")
    parser.add_argument("--results_csv", type=Path, required=True)
    parser.add_argument("--adapters_dir", type=Path, required=True)
    parser.add_argument("--selection_file", type=Path, required=True)
    parser.add_argument(
        "--alias_path",
        type=Path,
        help="optional symlink matching the grader-facing best-adapter path",
    )
    return parser


def rank_sweep_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    if {row.get("run_id") for row in rows} != set(PART_A_SWEEP):
        raise ValueError("sweep CSV must contain exactly one row for A1-A6")
    if len(rows) != len(PART_A_SWEEP):
        raise ValueError("sweep CSV contains duplicate or extra rows")
    parsed: list[dict[str, Any]] = []
    numeric_columns = (
        "val_entity_f1",
        "val_leak_rate",
        "val_loss",
        "trainable_pct",
    )
    for row in rows:
        item: dict[str, Any] = dict(row)
        for column in numeric_columns:
            try:
                value = float(row[column])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"run {row.get('run_id')}: invalid {column}"
                ) from error
            if not math.isfinite(value):
                raise ValueError(f"run {row.get('run_id')}: non-finite {column}")
            item[column] = value
        parsed.append(item)
    return sorted(
        parsed,
        key=lambda row: (
            -row["val_entity_f1"],
            row["val_leak_rate"],
            row["val_loss"],
            row["trainable_pct"],
            row["run_id"],
        ),
    )


def select_best(results_csv: Path, adapters_dir: Path) -> dict[str, Any]:
    if not results_csv.is_file():
        raise ValueError(f"results CSV does not exist: {results_csv}")
    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PART_A_RESULT_COLUMNS:
            raise ValueError("sweep CSV header differs from the required schema")
        ranking = rank_sweep_rows(list(reader))
    winner = ranking[0]
    adapter_path = adapters_dir / winner["run_id"]
    if not (adapter_path / "adapter_config.json").is_file():
        raise ValueError(f"winning adapter is missing: {adapter_path}")
    return {
        "selection_split": "partA_validation",
        "selection_rule": (
            "maximize overlap-IoU50 micro-F1; then minimize leak rate, "
            "validation loss, trainable percentage, and run ID"
        ),
        "winner": winner["run_id"],
        "adapter_path": str(adapter_path),
        "winner_metrics": {
            "val_entity_f1": winner["val_entity_f1"],
            "val_leak_rate": winner["val_leak_rate"],
            "val_loss": winner["val_loss"],
            "trainable_pct": winner["trainable_pct"],
        },
        "ranking": [row["run_id"] for row in ranking],
    }


def create_adapter_alias(adapter_path: Path, alias_path: Path) -> None:
    if os.path.lexists(alias_path):
        raise ValueError(f"adapter alias already exists: {alias_path}")
    alias_path.parent.mkdir(parents=True, exist_ok=True)
    target = os.path.relpath(adapter_path, start=alias_path.parent)
    alias_path.symlink_to(target, target_is_directory=True)
    if not (alias_path / "adapter_config.json").is_file():
        alias_path.unlink(missing_ok=True)
        raise ValueError(f"adapter alias does not resolve to a valid adapter: {alias_path}")


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    args = build_parser().parse_args()
    alias_created = False
    try:
        if args.selection_file.exists():
            raise ValueError(f"selection file already exists: {args.selection_file}")
        selection = select_best(args.results_csv, args.adapters_dir)
        if args.alias_path is not None:
            source_path = Path(selection["adapter_path"])
            create_adapter_alias(source_path, args.alias_path)
            alias_created = True
            selection["source_adapter_path"] = selection["adapter_path"]
            selection["adapter_path"] = str(args.alias_path)
        _atomic_write(args.selection_file, selection)
    except (OSError, ValueError) as error:
        if alias_created:
            args.alias_path.unlink(missing_ok=True)
        raise SystemExit(f"ERROR: {error}") from error
    print(selection["adapter_path"])


if __name__ == "__main__":
    main()
