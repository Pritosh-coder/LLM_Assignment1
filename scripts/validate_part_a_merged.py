#!/usr/bin/env python3
"""Validate a merged, frozen Part A state without training or mutating it."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import PART_A_RESULT_COLUMNS, PART_A_SWEEP


def _read_single_row(path: Path, run_id: str) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PART_A_RESULT_COLUMNS:
            raise ValueError(f"incompatible Part A row header: {path}")
        rows = list(reader)
    if len(rows) != 1 or rows[0].get("run_id") != run_id:
        raise ValueError(f"expected exactly one {run_id} row in {path}")
    return rows[0]


def validate(root: Path, *, require_sweep: bool = False) -> dict[str, Any]:
    root = root.resolve()
    row_dir = root / "results" / "detail" / "partA_rows"
    rows: list[dict[str, str]] = []
    for run_id in PART_A_SWEEP:
        adapter_config = root / "adapters" / run_id / "adapter_config.json"
        row_path = row_dir / f"partA_{run_id}.csv"
        if not adapter_config.is_file() or not row_path.is_file():
            raise ValueError(f"merged Part A is incomplete at {run_id}")
        try:
            cfg = json.loads(adapter_config.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid adapter config: {adapter_config}") from error
        if not isinstance(cfg, dict):
            raise ValueError(f"adapter config is not a JSON object: {adapter_config}")
        rows.append(_read_single_row(row_path, run_id))

    sweep_path = root / "results" / "partA_sweep.csv"
    if sweep_path.is_file():
        with sweep_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != PART_A_RESULT_COLUMNS:
                raise ValueError("partA_sweep.csv header differs from the frozen schema")
            sweep_rows = list(reader)
        if [row.get("run_id") for row in sweep_rows] != list(PART_A_SWEEP):
            raise ValueError("partA_sweep.csv must contain A1-A6 in frozen order")
        for expected, observed in zip(rows, sweep_rows, strict=True):
            if expected != observed:
                raise ValueError(
                    f"partA_sweep.csv differs from shard row {expected['run_id']}"
                )
    elif require_sweep:
        raise ValueError("merged Part A sweep CSV is missing")

    return {
        "status": "PASS",
        "runs": list(PART_A_SWEEP),
        "sweep_present": sweep_path.is_file(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--require_sweep", action="store_true")
    args = parser.parse_args()
    try:
        out = validate(args.root, require_sweep=args.require_sweep)
    except (OSError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
