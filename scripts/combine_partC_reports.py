#!/usr/bin/env python3
"""Combine three per-model Part C reports without losing the required raw schema."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

ATTACK_KEYS = ("discoverable", "untargeted", "canary", "targeted")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Combine V-base/V-raw/V-san reports.")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--sanitized", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    return parser


def _load_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Part C report does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Part C report is not an object: {path}")
    missing = [key for key in ("victim", "heldout_ppl", *ATTACK_KEYS) if key not in value]
    if missing:
        raise ValueError(f"Part C report {path} is missing {missing}")
    if not math.isfinite(float(value["heldout_ppl"])):
        raise ValueError(f"Part C report {path} has invalid heldout_ppl")
    return value


def combine_reports(
    base: dict[str, Any],
    raw: dict[str, Any],
    sanitized: dict[str, Any],
) -> dict[str, Any]:
    exposures: dict[str, dict[str, float]] = {}
    for n_reps in ("1", "4", "16", "64"):
        raw_value = float(raw["canary"]["by_n_reps"][n_reps]["mean_exposure"])
        san_value = float(
            sanitized["canary"]["by_n_reps"][n_reps]["mean_exposure"]
        )
        exposures[n_reps] = {
            "raw": raw_value,
            "sanitized": san_value,
            "absolute_reduction": raw_value - san_value,
        }
    raw_member = float(raw["targeted"]["member"]["greedy_leak_rate"])
    san_member = float(
        sanitized["targeted"]["member"]["greedy_leak_rate"]
    )
    # Keep V-raw at the top level so the required single-victim schema remains
    # directly consumable; the complete comparison lives under the extra key.
    return {
        **raw,
        "bundle_format": "required V-raw schema plus three-model comparison",
        "models": {
            "V-base": base,
            "V-raw": raw,
            "V-san": sanitized,
        },
        "comparison": {
            "heldout_ppl": {
                "base": float(base["heldout_ppl"]),
                "raw": float(raw["heldout_ppl"]),
                "sanitized": float(sanitized["heldout_ppl"]),
                "sanitized_minus_raw": (
                    float(sanitized["heldout_ppl"])
                    - float(raw["heldout_ppl"])
                ),
            },
            "member_greedy_leak": {
                "raw": raw_member,
                "sanitized": san_member,
                "absolute_reduction": raw_member - san_member,
            },
            "canary_exposure_by_n_reps": exposures,
        },
    }


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
    if args.output_file.exists():
        raise SystemExit(f"ERROR: output already exists: {args.output_file}")
    try:
        result = combine_reports(
            _load_report(args.base),
            _load_report(args.raw),
            _load_report(args.sanitized),
        )
        _atomic_write(args.output_file, result)
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error
    print(json.dumps({"output_file": str(args.output_file)}, indent=2))


if __name__ == "__main__":
    main()
