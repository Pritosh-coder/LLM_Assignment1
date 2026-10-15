"""M2 acceptance gate for the shared PII-label taxonomy."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.common.data import iter_jsonl
from src.common.labels import (
    CANONICAL_LABELS,
    canonical_label_frequencies,
    normalize_gold_label,
    normalize_model_label,
    normalize_presidio_label,
    normalize_regex_label,
    top_canonical_labels,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = PROJECT_ROOT / "data" / "partA_train.jsonl"

EXPECTED_CANONICAL_LABELS = (
    "BOD",
    "BUILDING",
    "CARDISSUER",
    "CITY",
    "COUNTRY",
    "DATE",
    "DRIVERLICENSE",
    "EMAIL",
    "GEOCOORD",
    "GIVENNAME1",
    "GIVENNAME2",
    "IDCARD",
    "IP",
    "LASTNAME1",
    "LASTNAME2",
    "LASTNAME3",
    "PASS",
    "PASSPORT",
    "POSTCODE",
    "SECADDRESS",
    "SEX",
    "SOCIALNUMBER",
    "STATE",
    "STREET",
    "TEL",
    "TIME",
    "TITLE",
    "USERNAME",
)

EXPECTED_TOP_10 = [
    ("TIME", 9886),
    ("USERNAME", 7469),
    ("EMAIL", 6564),
    ("SOCIALNUMBER", 6428),
    ("IDCARD", 6242),
    ("LASTNAME1", 6092),
    ("PASSPORT", 6082),
    ("DRIVERLICENSE", 5818),
    ("BOD", 5770),
    ("IP", 5477),
]


class M2LabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = list(iter_jsonl(TRAIN_FILE))

    def test_complete_training_taxonomy(self) -> None:
        self.assertEqual(CANONICAL_LABELS, EXPECTED_CANONICAL_LABELS)
        observed = {
            mask["label"]
            for record in self.records
            for mask in record["privacy_mask"]
        }
        self.assertEqual(observed, set(CANONICAL_LABELS))
        self.assertTrue(all(normalize_gold_label(label) == label for label in observed))

    def test_top_10_is_computed_from_training_data(self) -> None:
        self.assertEqual(top_canonical_labels(self.records), EXPECTED_TOP_10)
        frequencies = canonical_label_frequencies(self.records)
        self.assertEqual(sum(frequencies.values()), 129_776)

    def test_model_placeholder_normalization(self) -> None:
        cases = {
            "[email]": "EMAIL",
            "[EMAIL_7]": "EMAIL",
            "[given_name_1]": "GIVENNAME1",
            "[GIVENNAME_2]": "GIVENNAME2",
            "[surname_3]": "LASTNAME3",
            "phone-number": "TEL",
            "date of birth": "BOD",
            "IPv6": "IP",
            "postal_code": "POSTCODE",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_model_label(raw), expected)

    def test_regex_and_presidio_mappings(self) -> None:
        self.assertEqual(normalize_regex_label("EMAIL_ADDRESS"), "EMAIL")
        self.assertEqual(normalize_regex_label("PHONE_NUMBER"), "TEL")
        self.assertEqual(normalize_regex_label("IPv4"), "IP")
        self.assertEqual(normalize_presidio_label("EMAIL_ADDRESS"), "EMAIL")
        self.assertEqual(normalize_presidio_label("PHONE_NUMBER"), "TEL")
        self.assertEqual(normalize_presidio_label("US_SSN"), "SOCIALNUMBER")
        self.assertEqual(
            normalize_presidio_label("US_DRIVER_LICENSE"), "DRIVERLICENSE"
        )

    def test_ambiguous_and_malformed_labels_are_not_invented(self) -> None:
        for raw in ("PERSON", "LOCATION", "[EMAIL] explanation", "", "[]"):
            with self.subTest(raw=raw):
                self.assertIsNone(normalize_model_label(raw))
        self.assertIsNone(normalize_presidio_label("PERSON"))
        self.assertIsNone(normalize_presidio_label("LOCATION"))
        self.assertEqual(normalize_model_label("URL", allow_unknown=True), "URL")

    def test_ood_gold_aliases_share_the_declared_taxonomy(self) -> None:
        self.assertEqual(
            normalize_gold_label("FIRSTNAME", allow_unknown=True), "GIVENNAME1"
        )
        self.assertEqual(normalize_gold_label("SSN", allow_unknown=True), "SOCIALNUMBER")
        self.assertEqual(normalize_gold_label("IPV6", allow_unknown=True), "IP")
        self.assertEqual(
            normalize_gold_label("CREDITCARDNUMBER", allow_unknown=True),
            "CREDITCARDNUMBER",
        )


if __name__ == "__main__":
    unittest.main()
