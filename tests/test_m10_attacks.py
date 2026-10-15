"""CPU gates for the four Part C defensive extraction protocols."""

from __future__ import annotations

import argparse
import math
import tempfile
import unittest
import zlib
from pathlib import Path

from src.attack_extract import _set_reproducible_evaluation_seed, run
from src.common.attacks import (
    LiteralMatcher,
    estimate_sampled_exposure,
    membership_score_row,
    parse_pii_index_rows,
    sample_canary_candidates,
    split_discoverable_tokens,
    summarize_discoverable,
    summarize_targeted,
    summarize_untargeted,
)
from src.common.config import REFERENCE_MODEL_ID, VICTIM_MODEL_ID


class M10AttackTests(unittest.TestCase):
    def test_sampled_generation_uses_seeded_warn_only_determinism(self) -> None:
        seed_calls: list[tuple[int, bool]] = []
        deterministic_calls: list[tuple[bool, bool]] = []

        class FakeTorch:
            @staticmethod
            def use_deterministic_algorithms(
                enabled: bool, *, warn_only: bool = False
            ) -> None:
                deterministic_calls.append((enabled, warn_only))

        def fake_set_seed(seed: int, *, deterministic: bool) -> None:
            seed_calls.append((seed, deterministic))

        _set_reproducible_evaluation_seed(143, fake_set_seed, FakeTorch)
        self.assertEqual(seed_calls, [(143, True)])
        self.assertEqual(deterministic_calls, [(True, True)])

    def test_discoverable_midpoint_and_one_to_one_aggregation(self) -> None:
        self.assertEqual(
            split_discoverable_tokens([1, 2, 3, 4, 5]),
            ([1, 2], [3, 4, 5]),
        )
        report = summarize_discoverable(
            [
                {
                    "n_reps": 1,
                    "expected_text": "secret suffix",
                    "generated_text": "secret suffix",
                    "exact": True,
                },
                {
                    "n_reps": 1,
                    "expected_text": "other suffix",
                    "generated_text": "wrong",
                    "exact": False,
                },
                {
                    "n_reps": 8,
                    "expected_text": "memorized",
                    "generated_text": "memorized",
                    "exact": True,
                },
            ]
        )
        self.assertAlmostEqual(report["exact_match_rate"], 2 / 3)
        self.assertEqual(report["by_n_reps"]["1"]["n"], 2)
        self.assertEqual(report["by_n_reps"]["8"]["exact_match_rate"], 1.0)

    def test_membership_scores_follow_handout_definitions(self) -> None:
        scores = membership_score_row(2.0, 4.0, 1.0, "some generated text")
        self.assertAlmostEqual(scores["ppl"], math.exp(2.0))
        self.assertAlmostEqual(scores["ratio_ref"], math.exp(-2.0))
        self.assertAlmostEqual(scores["ratio_lower"], math.exp(1.0))
        payload = "some generated text".encode("utf-8")
        entropy = 8 * len(zlib.compress(payload)) / len(payload)
        self.assertAlmostEqual(scores["ratio_zlib"], 2.0 / entropy)

    def test_pii_index_parsing_and_literal_matching(self) -> None:
        entries = parse_pii_index_rows(
            [
                {
                    "doc_id": "d1",
                    "privacy_mask": [
                        {"label": "EMAIL", "value": "Alice@Example.com"},
                        {"label": "TEL", "value": "+91 99999 99999"},
                    ],
                }
            ]
        )
        matcher = LiteralMatcher([entry.value for entry in entries])
        self.assertEqual(
            matcher.find("Recovered ALICE@example.com from the model."),
            {0},
        )
        self.assertEqual(len(entries), 2)

    def test_untargeted_reports_strict_and_pii_precision_separately(self) -> None:
        entries = parse_pii_index_rows(
            [{"doc_id": "d1", "value": "alice@example.com", "label": "EMAIL"}]
        )
        report = summarize_untargeted(
            ["Complete training document", "email alice@example.com"],
            [
                {
                    "ppl": 1.0,
                    "ratio_ref": 1.0,
                    "ratio_lower": 1.0,
                    "ratio_zlib": 1.0,
                },
                {
                    "ppl": 2.0,
                    "ratio_ref": 2.0,
                    "ratio_lower": 2.0,
                    "ratio_zlib": 2.0,
                },
            ],
            [{"doc_id": "d1", "text": "Complete training document"}],
            entries,
            k=1,
        )
        self.assertEqual(report["precision_at_100"]["ppl"], 1.0)
        self.assertEqual(report["pii_precision_at_100"]["ppl"], 0.0)
        self.assertEqual(report["verified_pii_strings"], 1)
        self.assertEqual(report["distinct_docs_recovered"], 1)

    def test_canary_sampling_is_unique_and_exposure_is_capped(self) -> None:
        candidates = sample_canary_candidates("000042", seed=42)
        self.assertEqual(candidates[0], "000042")
        self.assertEqual(len(candidates), 10_000)
        self.assertEqual(len(set(candidates)), 10_000)
        exposure = estimate_sampled_exposure([0.1, 0.2, 0.3])
        self.assertEqual(exposure["rank_in_sample"], 1)
        self.assertAlmostEqual(exposure["exposure"], math.log2(3))

    def test_targeted_separates_member_and_control_greedy_and_sampled(self) -> None:
        report = summarize_targeted(
            [
                {
                    "split": "member",
                    "gold_value": "Secret 42",
                    "greedy_text": "SECRET   42",
                    "sampled_texts": ["miss"] * 9 + ["secret 42"],
                },
                {
                    "split": "control",
                    "gold_value": "Never seen",
                    "greedy_text": "miss",
                    "sampled_texts": ["miss"] * 10,
                },
            ]
        )
        self.assertEqual(report["top1_leak_rate"], 1.0)
        self.assertEqual(report["top10_leak_rate"], 1.0)
        self.assertEqual(report["control_top1_leak_rate"], 0.0)
        self.assertEqual(report["control_top10_leak_rate"], 0.0)

    def test_official_non_pii_dry_runs_pass_and_all_flags_missing_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = dict(
                victim_model=VICTIM_MODEL_ID,
                adapter_path=None,
                ref_model=REFERENCE_MODEL_ID,
                corpus=Path("data/partC_corpus_raw.jsonl"),
                heldout=Path("data/partC_heldout.jsonl"),
                canaries=Path("data/partC_canaries.jsonl"),
                pii_index=root / "missing_pii_index.jsonl",
                targeted_file=Path("data/partC_targeted.jsonl"),
                num_samples=5_000,
                seed=42,
                dry_run=True,
                overwrite=False,
            )
            canary_plan = run(
                argparse.Namespace(
                    **common,
                    attack="canary",
                    report_file=root / "canary.json",
                )
            )
            targeted_plan = run(
                argparse.Namespace(
                    **common,
                    attack="targeted",
                    report_file=root / "targeted.json",
                )
            )
            self.assertEqual(canary_plan["canary_rows"], 20)
            self.assertEqual(targeted_plan["targeted_rows"], 400)
            with self.assertRaisesRegex(ValueError, "partC_pii_index"):
                run(
                    argparse.Namespace(
                        **common,
                        attack="all",
                        report_file=root / "all.json",
                    )
                )


if __name__ == "__main__":
    unittest.main()
