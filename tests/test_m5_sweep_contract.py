"""CPU gates for the Part A sweep configuration and result schema."""

from __future__ import annotations

import csv
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from src.common.config import (
    ALL_LORA_TARGET_MODULES,
    HF_DETERMINISTIC_GENERATION,
    PART_A_RESULT_COLUMNS,
    PART_A_SWEEP,
    VLLM_DETERMINISTIC_GENERATION,
)
from src.peft_sweep import (
    append_sweep_result,
    gradient_accumulation_for_devices,
    load_full_part_a_inputs,
    parse_target_modules,
    validate_known_sweep_configuration,
)


class M5SweepContractTests(unittest.TestCase):
    def test_vllm_setup_uses_warn_only_determinism(self) -> None:
        calls: list[tuple[bool, bool]] = []
        fake_torch = types.ModuleType("torch")
        fake_torch.use_deterministic_algorithms = (
            lambda enabled, warn_only=False: calls.append((enabled, warn_only))
        )

        class FakeSamplingParams:
            def __init__(self, **_kwargs: object) -> None:
                pass

        class FakeLLM:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def generate(self, *_args: object, **_kwargs: object) -> list[object]:
                candidate = types.SimpleNamespace(text="sanitized")
                return [types.SimpleNamespace(outputs=[candidate], metrics=None)]

        fake_vllm = types.ModuleType("vllm")
        fake_vllm.LLM = FakeLLM
        fake_vllm.SamplingParams = FakeSamplingParams
        fake_lora = types.ModuleType("vllm.lora")
        fake_request = types.ModuleType("vllm.lora.request")
        fake_request.LoRARequest = object
        modules = {
            "torch": fake_torch,
            "vllm": fake_vllm,
            "vllm.lora": fake_lora,
            "vllm.lora.request": fake_request,
        }

        class FakeTokenizer:
            def apply_chat_template(self, *_args: object, **_kwargs: object) -> str:
                return "prompt"

        from src.common.gen import generate_with_vllm

        with mock.patch.dict(sys.modules, modules):
            generated = generate_with_vllm(
                [{"source_text": "Contact alice@example.com"}],
                FakeTokenizer(),
                base_model="test-model",
            )

        self.assertEqual(calls, [(True, True)])
        self.assertEqual(generated[0]["text"], "sanitized")

    def test_global_batch_is_preserved_on_one_or_two_gpus(self) -> None:
        self.assertEqual(gradient_accumulation_for_devices(1), 8)
        self.assertEqual(gradient_accumulation_for_devices(2), 4)
        with self.assertRaisesRegex(ValueError, "positive"):
            gradient_accumulation_for_devices(0)
        with self.assertRaisesRegex(ValueError, "not divisible"):
            gradient_accumulation_for_devices(3)

    def test_september_5_six_run_matrix_is_frozen(self) -> None:
        self.assertEqual(tuple(PART_A_SWEEP), ("A1", "A2", "A3", "A4", "A5", "A6"))
        self.assertEqual(PART_A_SWEEP["A4"]["lora_alpha"], 64)
        self.assertEqual(PART_A_SWEEP["A5"]["lora_alpha"], 16)
        self.assertEqual(PART_A_SWEEP["A6"]["quant"], "4bit")
        self.assertEqual(
            PART_A_SWEEP["A3"]["target_modules"], ALL_LORA_TARGET_MODULES
        )

    def test_generation_backends_share_non_truncating_cap(self) -> None:
        self.assertEqual(
            HF_DETERMINISTIC_GENERATION["max_new_tokens"],
            VLLM_DETERMINISTIC_GENERATION["max_tokens"],
        )
        self.assertGreaterEqual(VLLM_DETERMINISTIC_GENERATION["max_tokens"], 320)

    def test_target_modules_are_ordered_and_validated(self) -> None:
        self.assertEqual(
            parse_target_modules("q_proj, k_proj,v_proj"),
            ("q_proj", "k_proj", "v_proj"),
        )
        with self.assertRaisesRegex(ValueError, "duplicates"):
            parse_target_modules("q_proj,q_proj")
        with self.assertRaisesRegex(ValueError, "unsupported"):
            parse_target_modules("q_proj,not_a_module")

    def test_known_run_id_rejects_configuration_drift(self) -> None:
        with self.assertRaisesRegex(ValueError, "September 5 sweep"):
            validate_known_sweep_configuration(
                run_id="A4",
                target_modules=ALL_LORA_TARGET_MODULES,
                lora_r=32,
                lora_alpha=16,
                quant="none",
            )
        validate_known_sweep_configuration(
            run_id="A4",
            target_modules=ALL_LORA_TARGET_MODULES,
            lora_r=32,
            lora_alpha=64,
            quant="none",
        )

    def test_official_files_load_in_full_and_in_order(self) -> None:
        train, validation = load_full_part_a_inputs(
            Path("data/partA_train.jsonl"),
            Path("data/partA_validation.jsonl"),
        )
        self.assertEqual(len(train), 20_000)
        self.assertEqual(len(validation), 2_000)
        self.assertNotEqual(train[0]["id"], train[-1]["id"])
        self.assertNotEqual(validation[0]["id"], validation[-1]["id"])

    def test_csv_header_and_duplicate_run_protection(self) -> None:
        row = {column: 0 for column in PART_A_RESULT_COLUMNS}
        row.update(
            {
                "run_id": "A1",
                "method": "LoRA",
                "target_modules": "q_proj,v_proj",
                "quant": "none",
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partA_sweep.csv"
            append_sweep_result(path, row)
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(tuple(rows[0]), PART_A_RESULT_COLUMNS)
            self.assertEqual(rows[0]["run_id"], "A1")
            with self.assertRaisesRegex(ValueError, "already contains"):
                append_sweep_result(path, row)


if __name__ == "__main__":
    unittest.main()
