"""End-to-end CPU gates for the Part B evaluation entry point."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from src.common.config import SANITIZER_MODEL_ID
from src.sanitize_eval import run


class M7SanitizeEvalTests(unittest.TestCase):
    def test_regex_evaluation_writes_required_schemas(self) -> None:
        source = "Email alice@example.com now."
        row = {
            "id": "toy-1",
            "language": "English",
            "source_text": source,
            "target_text": "Email [EMAIL] now.",
            "privacy_mask": [
                {
                    "start": 6,
                    "end": 23,
                    "label": "EMAIL",
                    "value": "alice@example.com",
                }
            ],
            "split": "test",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_file = root / "test.jsonl"
            predictions = root / "predictions.jsonl"
            report = root / "metrics.json"
            test_file.write_text(json.dumps(row) + "\n", encoding="utf-8")
            args = argparse.Namespace(
                base_model=SANITIZER_MODEL_ID,
                adapter_path=None,
                test_file=test_file,
                system="regex",
                output_predictions=predictions,
                report_file=report,
                dry_run=False,
            )
            result = run(args)
            self.assertEqual(result["metrics"]["exact"]["micro_f1"], 1.0)
            self.assertEqual(result["metrics"]["leak_rate"], 0.0)
            prediction = json.loads(predictions.read_text(encoding="utf-8"))
            self.assertTrue(
                {
                    "id",
                    "language",
                    "source_text",
                    "predicted_text",
                    "predicted_spans",
                    "gold_spans",
                    "leaked_values",
                    "latency_ms",
                }
                <= set(prediction)
            )
            saved_report = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(saved_report["system"], "regex")
            self.assertEqual(saved_report["n"], 1)

    def test_evaluator_refuses_to_overwrite_its_input(self) -> None:
        row = {
            "id": "toy-2",
            "language": "English",
            "source_text": "No PII here.",
            "target_text": "No PII here.",
            "privacy_mask": [],
            "split": "test",
        }
        with tempfile.TemporaryDirectory() as directory:
            test_file = Path(directory) / "test.jsonl"
            test_file.write_text(json.dumps(row) + "\n", encoding="utf-8")
            args = argparse.Namespace(
                base_model=SANITIZER_MODEL_ID,
                adapter_path=None,
                test_file=test_file,
                system="regex",
                output_predictions=test_file,
                report_file=Path(directory) / "metrics.json",
                dry_run=False,
            )
            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                run(args)


if __name__ == "__main__":
    unittest.main()
