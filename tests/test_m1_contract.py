"""M1 acceptance gate: prompt formatting and completion-only masking."""

from __future__ import annotations

import itertools
import unittest
from pathlib import Path

from transformers import AutoTokenizer

from src.common import data, prompts
from src.common.config import (
    MAX_SEQ_LEN,
    SANITIZER_MODEL_ID,
    SANITIZER_MODEL_REVISION,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = PROJECT_ROOT / "data" / "partA_train.jsonl"
EXPECTED_FIRST_IDS = ["a_train_43854A", "a_train_41488A", "a_train_43341C"]


class M1ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = AutoTokenizer.from_pretrained(
            SANITIZER_MODEL_ID,
            revision=SANITIZER_MODEL_REVISION,
            use_fast=True,
        )
        cls.records = list(itertools.islice(data.iter_jsonl(TRAIN_FILE), 3))

    def test_jsonl_order_is_preserved(self) -> None:
        self.assertEqual([row["id"] for row in self.records], EXPECTED_FIRST_IDS)

    def test_prompt_contract(self) -> None:
        row = self.records[0]
        messages = prompts.build_prompt_messages(row["source_text"])
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertEqual(messages[0]["content"], prompts.SYSTEM_PROMPT)
        self.assertEqual(messages[1]["content"], row["source_text"])

    def test_completion_only_labels_on_three_rows(self) -> None:
        for row in self.records:
            with self.subTest(row_id=row["id"]):
                encoded = data.encode_training_example(row, self.tokenizer)
                self.assertEqual(
                    set(encoded), {"input_ids", "attention_mask", "labels"}
                )
                input_ids = encoded["input_ids"]
                attention_mask = encoded["attention_mask"]
                labels = encoded["labels"]
                self.assertEqual(len(input_ids), len(attention_mask))
                self.assertEqual(len(input_ids), len(labels))
                self.assertLessEqual(len(input_ids), MAX_SEQ_LEN)
                self.assertTrue(all(value == 1 for value in attention_mask))

                completion_start = next(
                    (index for index, label in enumerate(labels) if label != -100), None
                )
                self.assertIsNotNone(completion_start, "all labels are masked")
                assert completion_start is not None
                self.assertGreater(completion_start, 0)
                self.assertTrue(all(label == -100 for label in labels[:completion_start]))
                self.assertNotIn(-100, labels[completion_start:])
                self.assertEqual(labels[completion_start:], input_ids[completion_start:])
                self.assertEqual(labels[-1], self.tokenizer.eos_token_id)

                decoded = self.tokenizer.decode(
                    input_ids[completion_start:], skip_special_tokens=False
                )
                self.assertEqual(decoded, row["target_text"] + self.tokenizer.eos_token)

    def test_verify_masking_summary(self) -> None:
        summaries = data.verify_masking(self.records, self.tokenizer)
        self.assertEqual(len(summaries), 3)
        for row, summary in zip(self.records, summaries, strict=True):
            self.assertEqual(
                set(summary),
                {
                    "id",
                    "prompt_token_count",
                    "completion_token_count",
                    "decoded_completion",
                },
            )
            self.assertEqual(summary["id"], row["id"])
            self.assertGreater(summary["prompt_token_count"], 0)
            self.assertGreater(summary["completion_token_count"], 0)
            self.assertEqual(
                summary["decoded_completion"],
                row["target_text"] + self.tokenizer.eos_token,
            )


if __name__ == "__main__":
    unittest.main()
