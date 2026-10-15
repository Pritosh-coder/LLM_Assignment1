"""Regression test for Part B per-class top-10 selection."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.common.data import iter_jsonl
from src.sanitize_eval import _top_10_labels_for_records

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PartBTop10ReportingTests(unittest.TestCase):
    def test_id_top10_is_computed_from_partb_test_gold(self) -> None:
        records = list(iter_jsonl(PROJECT_ROOT / "data" / "partB_test.jsonl"))
        self.assertEqual(
            _top_10_labels_for_records(records),
            [
                "TIME",
                "USERNAME",
                "SOCIALNUMBER",
                "EMAIL",
                "IDCARD",
                "DRIVERLICENSE",
                "LASTNAME1",
                "BOD",
                "PASSPORT",
                "SEX",
            ],
        )

    def test_ood_top10_is_computed_from_ood_gold(self) -> None:
        records = list(iter_jsonl(PROJECT_ROOT / "data" / "partB_ood_test.jsonl"))
        self.assertEqual(
            _top_10_labels_for_records(records),
            [
                "GIVENNAME1",
                "IP",
                "USERNAME",
                "LASTNAME1",
                "SEX",
                "DATE",
                "EMAIL",
                "TITLE",
                "AMOUNT",
                "JOBAREA",
            ],
        )


if __name__ == "__main__":
    unittest.main()
