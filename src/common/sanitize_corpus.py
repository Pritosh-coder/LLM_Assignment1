"""Metadata-preserving Part C corpus sanitization and lineage checks."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from src.common.config import PART_C_CORPUS_REQUIRED_KEYS


def metadata_projection(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return every field except the trainable text, preserving field order."""
    return {key: value for key, value in record.items() if key != "text"}


def metadata_digest(records: Sequence[Mapping[str, Any]]) -> str:
    """Hash ordered metadata so row reordering and field mutation are detectable."""
    digest = hashlib.sha256()
    for record in records:
        encoded = json.dumps(
            metadata_projection(record),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def validate_corpus_structure(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Validate fields and the supplied repetition contract without reading secrets."""
    if not records:
        raise ValueError("corpus must contain at least one record")
    by_document: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row_number, record in enumerate(records, start=1):
        missing = sorted(PART_C_CORPUS_REQUIRED_KEYS - set(record))
        if missing:
            raise ValueError(f"row {row_number}: missing keys {missing}")
        if not isinstance(record["text"], str) or not record["text"]:
            raise ValueError(f"row {row_number}: text must be a non-empty string")
        doc_id = record["doc_id"]
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError(f"row {row_number}: doc_id must be a non-empty string")
        n_reps = record["n_reps"]
        repetition_index = record["repetition_index"]
        if not isinstance(n_reps, int) or isinstance(n_reps, bool) or n_reps <= 0:
            raise ValueError(f"row {row_number}: n_reps must be a positive integer")
        if (
            not isinstance(repetition_index, int)
            or isinstance(repetition_index, bool)
            or repetition_index < 0
        ):
            raise ValueError(
                f"row {row_number}: repetition_index must be a non-negative integer"
            )
        by_document[doc_id].append(record)

    for doc_id, group in by_document.items():
        reference = group[0]
        n_reps = reference["n_reps"]
        if len(group) != n_reps:
            raise ValueError(
                f"doc_id={doc_id!r}: has {len(group)} rows but declares n_reps={n_reps}"
            )
        indices = sorted(record["repetition_index"] for record in group)
        if indices != list(range(n_reps)):
            raise ValueError(f"doc_id={doc_id!r}: invalid repetition indices")
        for record in group:
            if record["n_reps"] != n_reps:
                raise ValueError(f"doc_id={doc_id!r}: inconsistent n_reps")
            if record["is_canary"] != reference["is_canary"]:
                raise ValueError(f"doc_id={doc_id!r}: inconsistent is_canary")
            if record["canary_id"] != reference["canary_id"]:
                raise ValueError(f"doc_id={doc_id!r}: inconsistent canary_id")
            if record["text"] != reference["text"]:
                raise ValueError(f"doc_id={doc_id!r}: repeated raw text differs")
    return {"rows": len(records), "distinct_documents": len(by_document)}


def build_sanitized_records(
    raw_records: Sequence[Mapping[str, Any]],
    sanitized_texts: Sequence[str],
) -> list[dict[str, Any]]:
    """Replace only ``text`` in every row, without filtering or special cases."""
    if len(raw_records) != len(sanitized_texts):
        raise ValueError(
            f"received {len(sanitized_texts)} outputs for {len(raw_records)} rows"
        )
    sanitized: list[dict[str, Any]] = []
    for row_number, (record, text) in enumerate(
        zip(raw_records, sanitized_texts, strict=True), start=1
    ):
        if not isinstance(text, str) or not text:
            raise ValueError(f"row {row_number}: sanitizer returned empty/non-text output")
        output = dict(record)
        output["text"] = text
        sanitized.append(output)
    return sanitized


def validate_sanitized_lineage(
    raw_records: Sequence[Mapping[str, Any]],
    sanitized_records: Sequence[Mapping[str, Any]],
) -> dict[str, int | str | bool]:
    """Prove that only text changed and repeated documents stayed deterministic."""
    if len(raw_records) != len(sanitized_records):
        raise ValueError("sanitized corpus row count differs from raw corpus")
    raw_digest = metadata_digest(raw_records)
    sanitized_digest = metadata_digest(sanitized_records)
    if raw_digest != sanitized_digest:
        raise ValueError("ordered corpus metadata changed during sanitization")

    by_document: defaultdict[str, set[str]] = defaultdict(set)
    changed_rows = 0
    for row_number, (raw, sanitized) in enumerate(
        zip(raw_records, sanitized_records, strict=True), start=1
    ):
        if set(raw) != set(sanitized):
            raise ValueError(f"row {row_number}: field set changed")
        text = sanitized.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError(f"row {row_number}: sanitized text is invalid")
        changed_rows += text != raw["text"]
        by_document[str(raw["doc_id"])].add(text)
    inconsistent_documents = sum(len(texts) != 1 for texts in by_document.values())
    if inconsistent_documents:
        raise ValueError(
            f"{inconsistent_documents} repeated documents sanitized inconsistently"
        )
    return {
        "rows": len(raw_records),
        "distinct_documents": len(by_document),
        "changed_rows": changed_rows,
        "unchanged_rows": len(raw_records) - changed_rows,
        "metadata_sha256": raw_digest,
        "repetition_consistent": True,
    }
