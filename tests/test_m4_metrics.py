"""M4 acceptance gates for entity and privacy metrics."""

from __future__ import annotations

import unittest

from src.common.metrics import (
    aggregate_entity_metrics,
    evaluate_sanitizer_records,
    leaked_gold_occurrences,
    leaked_gold_values,
    match_spans,
    over_masking_counts,
    span_iou,
)


class M4MetricTests(unittest.TestCase):
    def test_gold_scored_against_itself(self) -> None:
        source = "Email alice@example.com now."
        target = "Email [EMAIL] now."
        gold = [
            {
                "start": 6,
                "end": 23,
                "label": "EMAIL",
                "value": "alice@example.com",
            }
        ]
        report = evaluate_sanitizer_records(
            [
                {
                    "source_text": source,
                    "predicted_text": target,
                    "predicted_spans": gold,
                    "gold_spans": gold,
                }
            ],
            system="gold",
            adapter=None,
            test_set="toy",
        )
        self.assertEqual(report["exact"]["micro_f1"], 1.0)
        self.assertEqual(report["overlap_iou50"]["micro_f1"], 1.0)
        self.assertEqual(report["leak_rate"], 0.0)
        self.assertEqual(report["over_masking_rate"], 0.0)
        self.assertEqual(report["well_formed_rate"], 1.0)

    def test_identity_output_has_zero_recall_and_full_leakage(self) -> None:
        source = "Email alice@example.com now."
        gold = [
            {
                "start": 6,
                "end": 23,
                "label": "EMAIL",
                "value": "alice@example.com",
            }
        ]
        report = evaluate_sanitizer_records(
            [
                {
                    "source_text": source,
                    "predicted_text": source,
                    "predicted_spans": [],
                    "gold_spans": gold,
                }
            ],
            system="identity",
            adapter=None,
            test_set="toy",
        )
        self.assertEqual(report["exact"]["micro_r"], 0.0)
        self.assertEqual(report["overlap_iou50"]["micro_r"], 0.0)
        self.assertEqual(report["leak_rate"], 1.0)
        self.assertEqual(report["over_masking_rate"], 0.0)
        self.assertEqual(report["well_formed_rate"], 1.0)

    def test_one_prediction_cannot_match_two_gold_entities(self) -> None:
        predicted = [{"start": 0, "end": 8, "label": "EMAIL"}]
        gold = [
            {"start": 0, "end": 4, "label": "EMAIL"},
            {"start": 4, "end": 8, "label": "EMAIL"},
        ]
        matches = match_spans(
            predicted, gold, criterion="overlap_iou50", source_length=8
        )
        self.assertEqual(len(matches), 1)
        metrics = aggregate_entity_metrics(
            [(predicted, gold, 8)], criterion="overlap_iou50"
        )
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["micro_r"], 0.5)

    def test_exact_and_overlap_criteria_differ(self) -> None:
        predicted = [{"start": 12, "end": 20, "label": "EMAIL"}]
        gold = [{"start": 10, "end": 20, "label": "EMAIL"}]
        self.assertEqual(span_iou(predicted[0], gold[0]), 0.8)
        exact = aggregate_entity_metrics([(predicted, gold, 30)], criterion="exact")
        overlap = aggregate_entity_metrics(
            [(predicted, gold, 30)], criterion="overlap_iou50"
        )
        self.assertEqual(exact["micro_f1"], 0.0)
        self.assertEqual(overlap["micro_f1"], 1.0)

    def test_maximum_iou_pair_is_consumed_first(self) -> None:
        predicted = [
            {"start": 0, "end": 10, "label": "EMAIL"},
            {"start": 4, "end": 10, "label": "EMAIL"},
        ]
        gold = [
            {"start": 0, "end": 8, "label": "EMAIL"},
            {"start": 4, "end": 10, "label": "EMAIL"},
        ]
        matches = match_spans(
            predicted, gold, criterion="overlap_iou50", source_length=10
        )
        self.assertEqual(matches[0]["predicted_index"], 1)
        self.assertEqual(matches[0]["gold_index"], 1)
        self.assertEqual(matches[0]["iou"], 1.0)
        self.assertEqual(len(matches), 2)

    def test_leak_comparison_casefolds_and_collapses_whitespace(self) -> None:
        source = "Name Alice   Smith"
        gold = [
            {
                "start": 5,
                "end": 18,
                "label": "GIVENNAME1",
                "value": "Alice   Smith",
            }
        ]
        self.assertEqual(
            leaked_gold_values(source, "Unchanged: ALICE SMITH", gold),
            ["Alice   Smith"],
        )
        self.assertEqual(
            leaked_gold_occurrences(source, "Name ALICE SMITH", gold),
            ["Alice   Smith"],
        )

    def test_aligned_leak_diagnostic_rejects_unrelated_short_value_collision(self) -> None:
        source = "Sex: M; model ready."
        target = "Sex: [SEX]; model ready."
        gold = [{"start": 5, "end": 6, "label": "SEX", "value": "M"}]
        self.assertEqual(leaked_gold_values(source, target, gold), ["M"])
        self.assertEqual(leaked_gold_occurrences(source, target, gold), [])

    def test_over_masking_counts_only_non_pii_source_characters(self) -> None:
        source = "Email alice@example.com now."
        gold = [{"start": 6, "end": 23, "label": "EMAIL"}]
        self.assertEqual(over_masking_counts(source, "Email [EMAIL] now.", gold), (0, 11))
        self.assertEqual(over_masking_counts(source, "Email [EMAIL]", gold), (5, 11))

    def test_macro_includes_false_positive_only_classes(self) -> None:
        predicted = [
            {"start": 0, "end": 3, "label": "EMAIL"},
            {"start": 4, "end": 7, "label": "TEL"},
        ]
        gold = [{"start": 0, "end": 3, "label": "EMAIL"}]
        metrics = aggregate_entity_metrics([(predicted, gold, 7)], criterion="exact")
        self.assertEqual(metrics["micro_p"], 0.5)
        self.assertEqual(metrics["micro_r"], 1.0)
        self.assertEqual(metrics["per_class"]["TEL"]["support"], 0)
        self.assertEqual(metrics["macro_f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
