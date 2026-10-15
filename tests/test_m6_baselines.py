"""CPU gates for regex and Presidio Part B baselines."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from src.common.baselines import presidio_sanitize, regex_sanitize


@dataclass
class _FakeResult:
    start: int
    end: int
    entity_type: str
    score: float


class _FakeAnalyzer:
    def __init__(self, results: list[_FakeResult]) -> None:
        self.results = results

    def analyze(self, *, text: str, language: str) -> list[_FakeResult]:
        if language != "en":
            raise AssertionError("unexpected language")
        return self.results


class M6BaselineTests(unittest.TestCase):
    def test_regex_masks_strong_entities_and_preserves_other_text(self) -> None:
        source = "Email alice@example.com from 192.168.1.4 at 09:30."
        result = regex_sanitize(source)
        self.assertEqual(
            result["predicted_text"],
            "Email [EMAIL] from [IP] at [TIME].",
        )
        self.assertEqual(
            [span["label"] for span in result["predicted_spans"]],
            ["EMAIL", "IP", "TIME"],
        )
        for span in result["predicted_spans"]:
            value = source[span["start"] : span["end"]]
            self.assertEqual(value.strip(), value)

    def test_invalid_ipv4_is_not_masked_as_an_ip(self) -> None:
        result = regex_sanitize("Version 999.999.999.999 failed.")
        self.assertNotIn("IP", [span["label"] for span in result["predicted_spans"]])

    def test_luhn_credit_card_is_detected_before_phone(self) -> None:
        result = regex_sanitize("Card 4111 1111 1111 1111 is for testing.")
        self.assertEqual(
            [span["label"] for span in result["predicted_spans"]],
            ["CREDITCARDNUMBER"],
        )

    def test_context_rules_map_to_canonical_labels(self) -> None:
        result = regex_sanitize(
            "DOB: 01/02/1990; username: river7; password: s3cret!"
        )
        self.assertEqual(
            [span["label"] for span in result["predicted_spans"]],
            ["BOD", "USERNAME", "PASS"],
        )

    def test_presidio_uses_native_spans_and_one_to_one_overlap_selection(self) -> None:
        source = "Contact alice@example.com now"
        email_start = source.index("alice")
        email_end = email_start + len("alice@example.com")
        analyzer = _FakeAnalyzer(
            [
                _FakeResult(email_start, email_end, "EMAIL_ADDRESS", 0.95),
                _FakeResult(email_start, email_start + 5, "PERSON", 0.70),
            ]
        )
        result = presidio_sanitize(source, analyzer)
        self.assertEqual(result["predicted_text"], "Contact [EMAIL] now")
        self.assertEqual(len(result["predicted_spans"]), 1)
        self.assertEqual(result["predicted_spans"][0]["label"], "EMAIL")


if __name__ == "__main__":
    unittest.main()
