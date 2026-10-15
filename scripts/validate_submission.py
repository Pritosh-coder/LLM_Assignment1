#!/usr/bin/env python3
"""Fail closed when mandatory source/result artifacts are absent or malformed."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

REQUIRED_SOURCES = (
    "README.md",
    "requirements.txt",
    "run_all.sh",
    "src/peft_sweep.py",
    "src/sanitize_eval.py",
    "src/train_victim.py",
    "src/attack_extract.py",
)
REQUIRED_RESULTS = (
    "results/partA_sweep.csv",
    "results/partA_fig.png",
    "results/partB_metrics.json",
    "results/partB_predictions.jsonl",
    "results/partC_attacks.json",
    "results/partC_exposure.png",
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def validate(root: Path, *, require_report: bool) -> dict[str, Any]:
    missing = [
        relative
        for relative in (*REQUIRED_SOURCES, *REQUIRED_RESULTS)
        if not (root / relative).is_file()
    ]
    if require_report and not (root / "report.pdf").is_file():
        missing.append("report.pdf")
    if missing:
        raise ValueError(f"missing mandatory artifacts: {missing}")

    sweep_path = root / "results/partA_sweep.csv"
    with sweep_path.open("r", encoding="utf-8", newline="") as handle:
        sweep = list(csv.DictReader(handle))
    if len(sweep) != 6 or {row.get("run_id") for row in sweep} != {
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "A6",
    }:
        raise ValueError("partA_sweep.csv does not contain exactly A1-A6")

    part_b = _load_json(root / "results/partB_metrics.json")
    required_b = {
        "system",
        "n",
        "exact",
        "overlap_iou50",
        "per_class",
        "leak_rate",
        "over_masking_rate",
        "well_formed_rate",
    }
    if int(part_b.get("n", -1)) != 2_000 or not required_b <= set(part_b):
        raise ValueError("partB_metrics.json has an invalid required schema")

    predictions_path = root / "results/partB_predictions.jsonl"
    prediction_count = 0
    required_prediction = {
        "id",
        "language",
        "source_text",
        "predicted_text",
        "predicted_spans",
        "gold_spans",
        "leaked_values",
        "latency_ms",
    }
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if not isinstance(row, dict) or not required_prediction <= set(row):
                raise ValueError(
                    f"partB_predictions.jsonl row {line_number} has invalid schema"
                )
            prediction_count += 1
    if prediction_count != 2_000:
        raise ValueError("partB_predictions.jsonl must contain exactly 2,000 rows")

    part_c = _load_json(root / "results/partC_attacks.json")
    required_c = {
        "victim",
        "heldout_ppl",
        "discoverable",
        "untargeted",
        "canary",
        "targeted",
        "models",
    }
    if not required_c <= set(part_c):
        raise ValueError("partC_attacks.json has an invalid required schema")
    if set(part_c["models"]) != {"V-base", "V-raw", "V-san"}:
        raise ValueError("partC_attacks.json must contain all three model reports")

    for relative in ("results/partA_fig.png", "results/partC_exposure.png"):
        if (root / relative).stat().st_size <= 0:
            raise ValueError(f"empty figure artifact: {relative}")
    if require_report and (root / "report.pdf").stat().st_size <= 0:
        raise ValueError("report.pdf is empty")
    return {
        "sweep_rows": len(sweep),
        "partB_predictions": prediction_count,
        "partC_models": 3,
        "report_required": require_report,
        "status": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate submission artifacts.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--require_report", action="store_true")
    args = parser.parse_args()
    try:
        report = validate(args.root.resolve(), require_report=args.require_report)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"ERROR: {error}") from error
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
