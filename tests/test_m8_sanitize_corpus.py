"""Lineage gates for the Part C sanitized-corpus builder."""

from __future__ import annotations

import copy
import unittest

from src.common.sanitize_corpus import (
    build_sanitized_records,
    metadata_digest,
    validate_corpus_structure,
    validate_sanitized_lineage,
)


def _raw_corpus() -> list[dict[str, object]]:
    repeated = {
        "doc_id": "doc-repeated",
        "text": "Contact alice@example.com.",
        "n_reps": 2,
        "is_canary": True,
        "canary_id": "canary-1",
    }
    return [
        {**repeated, "repetition_index": 0},
        {**repeated, "repetition_index": 1},
        {
            "doc_id": "doc-single",
            "text": "Nothing sensitive.",
            "n_reps": 1,
            "repetition_index": 0,
            "is_canary": False,
            "canary_id": None,
        },
    ]


class M8SanitizedCorpusTests(unittest.TestCase):
    def test_builder_changes_only_text_and_preserves_lineage(self) -> None:
        raw = _raw_corpus()
        sanitized = build_sanitized_records(
            raw,
            ["Contact [EMAIL].", "Contact [EMAIL].", "Nothing sensitive."],
        )

        self.assertEqual(
            validate_corpus_structure(raw),
            {"rows": 3, "distinct_documents": 2},
        )
        audit = validate_sanitized_lineage(raw, sanitized)
        self.assertEqual(audit["rows"], 3)
        self.assertEqual(audit["distinct_documents"], 2)
        self.assertEqual(audit["changed_rows"], 2)
        self.assertEqual(audit["unchanged_rows"], 1)
        self.assertTrue(audit["repetition_consistent"])
        self.assertEqual(metadata_digest(raw), metadata_digest(sanitized))

        for before, after in zip(raw, sanitized, strict=True):
            self.assertEqual(
                {key: value for key, value in before.items() if key != "text"},
                {key: value for key, value in after.items() if key != "text"},
            )

    def test_rejects_inconsistent_repeated_outputs(self) -> None:
        raw = _raw_corpus()
        sanitized = build_sanitized_records(
            raw,
            ["Contact [EMAIL].", "Contact alice@example.com.", "Nothing sensitive."],
        )
        with self.assertRaisesRegex(ValueError, "sanitized inconsistently"):
            validate_sanitized_lineage(raw, sanitized)

    def test_rejects_metadata_mutation(self) -> None:
        raw = _raw_corpus()
        sanitized = build_sanitized_records(raw, [row["text"] for row in raw])
        mutated = copy.deepcopy(sanitized)
        mutated[0]["canary_id"] = "different-canary"
        with self.assertRaisesRegex(ValueError, "metadata changed"):
            validate_sanitized_lineage(raw, mutated)

    def test_rejects_wrong_count_and_empty_output(self) -> None:
        raw = _raw_corpus()
        with self.assertRaisesRegex(ValueError, "outputs for"):
            build_sanitized_records(raw, ["only one output"])
        with self.assertRaisesRegex(ValueError, "empty/non-text"):
            build_sanitized_records(raw, ["ok", "", "ok"])

    def test_rejects_invalid_raw_repetition_contract(self) -> None:
        raw = _raw_corpus()
        raw[1]["repetition_index"] = 0
        with self.assertRaisesRegex(ValueError, "invalid repetition indices"):
            validate_corpus_structure(raw)


if __name__ == "__main__":
    unittest.main()
