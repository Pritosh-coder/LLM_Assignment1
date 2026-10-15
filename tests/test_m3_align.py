"""M3 acceptance gate for source-offset span recovery."""

from __future__ import annotations

import itertools
import unittest
from pathlib import Path

from src.common.align import recover_spans, recover_spans_with_report
from src.common.data import iter_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = PROJECT_ROOT / "data" / "partA_train.jsonl"


def triples(spans: list[dict[str, object]]) -> list[tuple[int, int, str]]:
    return [
        (int(span["start"]), int(span["end"]), str(span["label"]))
        for span in spans
    ]


class M3AlignmentTests(unittest.TestCase):
    def test_single_replacement_uses_source_offsets(self) -> None:
        source = "Contact ada@example.org today."
        predicted = "Contact [EMAIL] today."
        spans = recover_spans(source, predicted)
        self.assertEqual(triples(spans), [(8, 23, "EMAIL")])
        self.assertEqual(spans[0]["rule"], "atomic_placeholder_replace")

    def test_adjacent_placeholders_with_unique_separator(self) -> None:
        source = "John|Smith"
        predicted = "[GIVENNAME1]|[LASTNAME1]"
        spans = recover_spans(source, predicted)
        self.assertEqual(
            triples(spans),
            [(0, 4, "GIVENNAME1"), (5, 10, "LASTNAME1")],
        )

    def test_collapsed_adjacent_placeholders_are_flagged(self) -> None:
        report = recover_spans_with_report(
            "JohnSmith", "[GIVENNAME1][LASTNAME1]"
        )
        self.assertEqual(len(report["spans"]), 2)
        self.assertTrue(
            any(
                issue["rule"] == "multi_placeholder_ambiguous_even_split"
                for issue in report["issues"]
            )
        )
        self.assertFalse(report["well_formed"])

    def test_non_placeholder_rewrite_does_not_invent_pii(self) -> None:
        report = recover_spans_with_report("Hello  world", "Hello world")
        self.assertEqual(report["spans"], [])
        self.assertTrue(
            any(issue["rule"] == "deletion_without_placeholder" for issue in report["issues"])
        )

    def test_unrecognized_placeholder_is_reported(self) -> None:
        report = recover_spans_with_report("Contact Alice", "Contact [ALIEN]")
        self.assertEqual(report["spans"], [])
        self.assertTrue(
            any(
                issue["rule"] == "unrecognized_bracketed_label"
                for issue in report["issues"]
            )
        )

    def test_ood_mode_recovers_new_labels_but_preserves_literal_brackets(self) -> None:
        generated = recover_spans_with_report(
            "Card 4111111111111111.",
            "Card [CREDITCARDNUMBER].",
            allow_unknown_labels=True,
        )
        self.assertEqual(len(generated["spans"]), 1)
        self.assertEqual(generated["spans"][0]["label"], "CREDITCARDNUMBER")

        preserved = recover_spans_with_report(
            "Keep [SECTION] exactly.",
            "Keep [SECTION] exactly.",
            allow_unknown_labels=True,
        )
        self.assertEqual(preserved["spans"], [])
        self.assertTrue(preserved["well_formed"])

    def test_gold_outputs_on_first_200_training_rows(self) -> None:
        records = list(itertools.islice(iter_jsonl(TRAIN_FILE), 200))
        exact_rows = 0
        mismatch_ids: list[str] = []
        recovered_count = 0
        gold_count = 0
        for record in records:
            recovered = triples(
                recover_spans(record["source_text"], record["target_text"])
            )
            gold = [
                (mask["start"], mask["end"], mask["label"])
                for mask in record["privacy_mask"]
            ]
            recovered_count += len(recovered)
            gold_count += len(gold)
            if recovered == gold:
                exact_rows += 1
            else:
                mismatch_ids.append(record["id"])

        self.assertEqual(len(records), 200)
        self.assertEqual(gold_count, 1_238)
        self.assertEqual(recovered_count, gold_count)
        self.assertEqual(exact_rows, 198)
        self.assertEqual(mismatch_ids, ["a_train_41284C", "a_train_41629C"])


if __name__ == "__main__":
    unittest.main()
