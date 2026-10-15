#!/usr/bin/env python3
"""Read-only preflight for the instructor-provided JSONL bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import (  # noqa: E402
    EXPECTED_DATASET_ROWS,
    PART_AB_REQUIRED_KEYS,
    PART_C_CANARY_REQUIRED_KEYS,
    PART_C_CORPUS_REQUIRED_KEYS,
    PART_C_TARGETED_REQUIRED_KEYS,
)


def required_keys_for(filename: str) -> frozenset[str]:
    if filename.startswith("partA_") or filename.startswith("partB_"):
        return PART_AB_REQUIRED_KEYS
    if filename in {"partC_corpus_raw.jsonl", "partC_heldout.jsonl"}:
        return PART_C_CORPUS_REQUIRED_KEYS
    if filename == "partC_canaries.jsonl":
        return PART_C_CANARY_REQUIRED_KEYS
    if filename == "partC_targeted.jsonl":
        return PART_C_TARGETED_REQUIRED_KEYS
    raise ValueError(f"No schema contract for {filename}")


def validate_file(path: Path, expected_rows: int) -> None:
    required = required_keys_for(path.name)
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path.name}:{line_number}: blank JSONL row")
            try:
                record: Any = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path.name}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path.name}:{line_number}: row is not an object")
            missing = required.difference(record)
            if missing:
                raise ValueError(
                    f"{path.name}:{line_number}: missing keys {sorted(missing)}"
                )
            rows += 1
    if rows != expected_rows:
        raise ValueError(f"{path.name}: expected {expected_rows} rows, found {rows}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(data_dir: Path) -> None:
    manifest_path = data_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        raise SystemExit("Missing instructor-provided data/manifest.jsonl")

    entries: dict[str, dict[str, Any]] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                item: Any = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"manifest.jsonl:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(item, dict):
                raise ValueError(f"manifest.jsonl:{line_number}: row is not an object")
            required = {"file", "rows", "bytes", "sha256"}
            missing = required.difference(item)
            if missing:
                raise ValueError(
                    f"manifest.jsonl:{line_number}: missing keys {sorted(missing)}"
                )
            filename = item["file"]
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise ValueError(
                    f"manifest.jsonl:{line_number}: unsafe or invalid filename"
                )
            if filename in entries:
                raise ValueError(f"manifest.jsonl:{line_number}: duplicate {filename}")
            entries[filename] = item

    expected_names = set(EXPECTED_DATASET_ROWS)
    actual_names = set(entries)
    if actual_names != expected_names:
        raise ValueError(
            "manifest file set differs from the assignment contract: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"unexpected={sorted(actual_names - expected_names)}"
        )

    for filename, expected_rows in EXPECTED_DATASET_ROWS.items():
        item = entries[filename]
        path = data_dir / filename
        if item["rows"] != expected_rows:
            raise ValueError(
                f"manifest {filename}: expected rows={expected_rows}, "
                f"declares rows={item['rows']}"
            )
        actual_bytes = path.stat().st_size
        if item["bytes"] != actual_bytes:
            raise ValueError(
                f"{filename}: manifest bytes={item['bytes']}, actual={actual_bytes}"
            )
        actual_sha256 = file_sha256(path)
        if item["sha256"] != actual_sha256:
            raise ValueError(
                f"{filename}: SHA-256 mismatch; expected {item['sha256']}, "
                f"found {actual_sha256}"
            )


def main() -> None:
    data_dir = PROJECT_ROOT / "data"
    missing = [name for name in EXPECTED_DATASET_ROWS if not (data_dir / name).is_file()]
    if missing:
        joined = "\n  - ".join(missing)
        raise SystemExit(
            "Missing instructor-provided files under data/:\n  - " + joined
        )

    validate_manifest(data_dir)
    print("PASS manifest: filenames, row declarations, byte counts, and SHA-256")

    for filename, expected_rows in EXPECTED_DATASET_ROWS.items():
        validate_file(data_dir / filename, expected_rows)
        print(f"PASS {filename}: {expected_rows} rows")

    pii_index = data_dir / "partC_pii_index.jsonl"
    if not pii_index.exists():
        print(
            "NOTE partC_pii_index.jsonl is absent; the handout requires a Piazza "
            "clarification on whether it is supplied or constructed."
        )


if __name__ == "__main__":
    main()
