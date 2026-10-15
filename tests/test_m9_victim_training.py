"""CPU contracts for Part C victim tokenization, training, and utility scoring."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from src.common.config import (
    ALL_LORA_TARGET_MODULES,
    VICTIM_EPOCHS,
    VICTIM_MAX_SEQ_LEN,
    VICTIM_MODEL_ID,
)
from src.common.scoring import summarize_token_losses
from src.common.victim import encode_victim_text, load_full_victim_inputs
from src.train_victim import _latest_checkpoint, run


class _FakeTokenizer:
    eos_token_id = 99

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            raise AssertionError("victim encoding must not add a chat/BOS wrapper")
        return [index + 1 for index, _ in enumerate(text.split())]


class M9VictimTrainingTests(unittest.TestCase):
    def test_plain_causal_encoding_trains_every_token_and_eos(self) -> None:
        encoded = encode_victim_text(
            {"doc_id": "toy", "text": "three plain tokens"},
            _FakeTokenizer(),
        )
        self.assertEqual(encoded["input_ids"], [1, 2, 3, 99])
        self.assertEqual(encoded["labels"], encoded["input_ids"])
        self.assertEqual(encoded["attention_mask"], [1, 1, 1, 1])

    def test_victim_encoding_never_truncates(self) -> None:
        text = " ".join("x" for _ in range(VICTIM_MAX_SEQ_LEN))
        with self.assertRaisesRegex(ValueError, "truncation is prohibited"):
            encode_victim_text({"doc_id": "long", "text": text}, _FakeTokenizer())

    def test_token_weighted_perplexity_is_not_batch_averaged(self) -> None:
        result = summarize_token_losses([(2.0, 2), (1.0, 6)])
        self.assertEqual(result["predicted_tokens"], 8)
        self.assertAlmostEqual(float(result["mean_nll"]), 1.25)
        self.assertAlmostEqual(float(result["perplexity"]), 3.4903429575)
        with self.assertRaisesRegex(ValueError, "without predicted tokens"):
            summarize_token_losses([])

    def test_official_train_and_heldout_sets_are_complete_and_disjoint(self) -> None:
        corpus, heldout, summary = load_full_victim_inputs(
            Path("data/partC_corpus_raw.jsonl"),
            Path("data/partC_heldout.jsonl"),
        )
        self.assertEqual(len(corpus), 8_007)
        self.assertEqual(len(heldout), 500)
        self.assertEqual(summary["train_documents"], 5_020)
        self.assertEqual(summary["heldout_documents"], 500)

    def test_full_dry_run_freezes_base_model_and_comparison_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                model=VICTIM_MODEL_ID,
                corpus=Path("data/partC_corpus_raw.jsonl"),
                heldout=Path("data/partC_heldout.jsonl"),
                target_modules=",".join(ALL_LORA_TARGET_MODULES),
                lora_r=32,
                lora_alpha=64,
                epochs=VICTIM_EPOCHS,
                seed=42,
                output_dir=Path(directory) / "victim_raw",
                dry_run=True,
            )
            plan = run(args)
        self.assertEqual(plan["model"], "Qwen/Qwen2.5-0.5B")
        self.assertEqual(plan["epochs"], 4)
        self.assertEqual(plan["train_rows"], 8_007)
        self.assertEqual(plan["heldout_rows"], 500)
        self.assertEqual(plan["max_seq_len"], 512)


    def test_checkpoint_discovery_uses_newest_valid_trainer_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkpoint-250").mkdir()
            (root / "checkpoint-250" / "trainer_state.json").write_text("{}", encoding="utf-8")
            (root / "checkpoint-500").mkdir()
            (root / "checkpoint-500" / "trainer_state.json").write_text("{}", encoding="utf-8")
            (root / "checkpoint-999").mkdir()  # incomplete and must be ignored
            self.assertEqual(_latest_checkpoint(root), root / "checkpoint-500")

    def test_victim_training_source_has_durable_checkpoint_policy(self) -> None:
        source = Path("src/train_victim.py").read_text(encoding="utf-8")
        self.assertIn('save_strategy="steps"', source)
        self.assertIn("save_steps=250", source)
        self.assertIn("victim_plan.json", source)
        self.assertIn("resume_from_checkpoint", source)

    def test_instruct_model_and_epoch_drift_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base_args = dict(
                corpus=Path("data/partC_corpus_raw.jsonl"),
                heldout=Path("data/partC_heldout.jsonl"),
                target_modules="q_proj,v_proj",
                lora_r=32,
                lora_alpha=64,
                seed=42,
                output_dir=Path(directory) / "victim",
                dry_run=True,
            )
            with self.assertRaisesRegex(ValueError, "base model"):
                run(
                    argparse.Namespace(
                        **base_args,
                        model="Qwen/Qwen2.5-0.5B-Instruct",
                        epochs=VICTIM_EPOCHS,
                    )
                )
            with self.assertRaisesRegex(ValueError, "requires epochs=4"):
                run(
                    argparse.Namespace(
                        **base_args,
                        model=VICTIM_MODEL_ID,
                        epochs=1,
                    )
                )


if __name__ == "__main__":
    unittest.main()
