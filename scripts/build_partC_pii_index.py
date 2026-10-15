#!/usr/bin/env python3
"""Build the missing local Part C PII index with frozen independent detectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.attacks import (  # noqa: E402
    normalize_for_match,
    parse_pii_index_rows,
)
from src.common.baselines import (  # noqa: E402
    build_default_presidio_analyzer,
    presidio_sanitize,
    regex_sanitize,
)
from src.common.config import EXPECTED_DATASET_ROWS  # noqa: E402
from src.common.data import iter_jsonl  # noqa: E402
from src.common.sanitize_corpus import validate_corpus_structure  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Construct a detector-derived PII index because the required "
            "partC_pii_index.jsonl is absent from the supplied archive."
        )
    )
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--audit_file", type=Path, required=True)
    parser.add_argument(
        "--detectors",
        choices=("regex", "presidio", "both"),
        default="both",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                )
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _deduplicate_documents(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        if record["is_canary"]:
            continue
        doc_id = str(record["doc_id"])
        if doc_id in seen:
            continue
        seen.add(doc_id)
        documents.append(record)
    if len(documents) != 5_000:
        raise ValueError(
            f"expected 5,000 unique non-canary documents, found {len(documents)}"
        )
    return documents


def build_index(
    records: list[dict[str, Any]],
    *,
    detectors: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return value-bearing index rows and non-sensitive aggregate statistics."""
    documents = _deduplicate_documents(records)
    analyzer = (
        build_default_presidio_analyzer()
        if detectors in {"presidio", "both"}
        else None
    )
    output_rows: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    unique_values: set[str] = set()

    for record in documents:
        text = record["text"]
        detected: list[dict[str, Any]] = []
        if detectors in {"regex", "both"}:
            detected.extend(regex_sanitize(text)["predicted_spans"])
        if detectors in {"presidio", "both"}:
            detected.extend(
                presidio_sanitize(text, analyzer=analyzer)["predicted_spans"]
            )

        by_value: dict[str, dict[str, Any]] = {}
        for span in sorted(
            detected,
            key=lambda item: (
                item["start"],
                item["end"],
                -item["score"],
                item["label"],
                item["rule"],
            ),
        ):
            start, end = span["start"], span["end"]
            if not 0 <= start < end <= len(text):
                raise ValueError(f"detector returned an invalid span for {record['doc_id']}")
            value = text[start:end]
            normalized = normalize_for_match(value)
            if not normalized:
                continue
            existing = by_value.get(normalized)
            if existing is None:
                by_value[normalized] = {
                    "start": start,
                    "end": end,
                    "label": span["label"],
                    "value": value,
                    "detectors": [span["rule"]],
                    "max_score": span["score"],
                }
            else:
                if span["rule"] not in existing["detectors"]:
                    existing["detectors"].append(span["rule"])
                if span["score"] > existing["max_score"]:
                    existing.update(
                        {
                            "start": start,
                            "end": end,
                            "label": span["label"],
                            "value": value,
                            "max_score": span["score"],
                        }
                    )
        if not by_value:
            continue
        entries = sorted(
            by_value.values(),
            key=lambda item: (item["start"], item["end"], item["label"]),
        )
        for entry in entries:
            entry["detectors"].sort()
            label_counts[entry["label"]] += 1
            source_counts.update(entry["detectors"])
            unique_values.add(normalize_for_match(entry["value"]))
        output_rows.append({"doc_id": record["doc_id"], "pii": entries})

    if not output_rows:
        raise ValueError("detectors produced an empty PII index")
    audit = {
        "method": "detector-derived; not instructor gold",
        "detectors": detectors,
        "source_documents": len(documents),
        "indexed_documents": len(output_rows),
        "index_entries": sum(len(row["pii"]) for row in output_rows),
        "distinct_normalized_values": len(unique_values),
        "label_counts": dict(sorted(label_counts.items())),
        "detector_rule_counts": dict(sorted(source_counts.items())),
        "excluded_canaries": 20,
        "limitations": [
            "false positives and false negatives inherit the frozen detectors",
            (
                "the index is evaluation pseudo-ground-truth because the "
                "supplied bundle omitted the required file"
            ),
        ],
    }
    return output_rows, audit


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.corpus.is_file():
        raise ValueError(f"corpus does not exist: {args.corpus}")
    resolved_paths = {
        args.corpus.resolve(),
        args.output_file.resolve(),
        args.audit_file.resolve(),
    }
    if len(resolved_paths) != 3:
        raise ValueError("corpus, output_file, and audit_file must be distinct")
    for path in (args.output_file, args.audit_file):
        if path.exists() and not args.overwrite:
            raise ValueError(f"output exists; pass --overwrite explicitly: {path}")

    records = list(iter_jsonl(args.corpus))
    expected = EXPECTED_DATASET_ROWS["partC_corpus_raw.jsonl"]
    if len(records) != expected:
        raise ValueError(f"expected {expected} corpus rows, received {len(records)}")
    validate_corpus_structure(records)
    index_rows, audit = build_index(records, detectors=args.detectors)
    _atomic_write_jsonl(args.output_file, index_rows)
    persisted = list(iter_jsonl(args.output_file))
    parsed_count = len(parse_pii_index_rows(persisted))
    if parsed_count != audit["index_entries"]:
        raise RuntimeError("persisted PII index failed its schema/count check")
    final_audit = {
        **audit,
        "corpus": str(args.corpus),
        "corpus_sha256": _file_sha256(args.corpus),
        "output_file": str(args.output_file),
        "output_sha256": _file_sha256(args.output_file),
    }
    _atomic_write_json(args.audit_file, final_audit)
    return final_audit


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run(args)
    except (RuntimeError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
