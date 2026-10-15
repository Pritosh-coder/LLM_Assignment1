"""Frozen experiment constants shared by every assignment entry point.

The values that are not dictated by the handout are starting choices. Finalize them
before Part A, then do not alter them between comparative runs.
"""

from __future__ import annotations

from typing import Final

SEED: Final[int] = 42

SANITIZER_MODEL_ID: Final[str] = "Qwen/Qwen2.5-1.5B-Instruct"
SANITIZER_MODEL_REVISION: Final[str] = (
    "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)
VICTIM_MODEL_ID: Final[str] = "Qwen/Qwen2.5-0.5B"
VICTIM_MODEL_REVISION: Final[str] = (
    "060db6499f32faf8b98477b0a26969ef7d8b9987"
)
REFERENCE_MODEL_ID: Final[str] = VICTIM_MODEL_ID

# Measured with the frozen M1 prompt and tokenizer revision: max 641 tokens
# over 20,000 training rows and 609 over 2,000 validation rows.
MAX_SEQ_LEN: Final[int] = 768
NUM_EPOCHS: Final[int] = 1
LEARNING_RATE: Final[float] = 2e-4
WEIGHT_DECAY: Final[float] = 0.0
WARMUP_RATIO: Final[float] = 0.03
PER_DEVICE_BATCH_SIZE: Final[int] = 1
GRADIENT_ACCUMULATION_STEPS: Final[int] = 8
LR_SCHEDULER_TYPE: Final[str] = "linear"

# Part C victim runs use the base (not Instruct) LM and plain causal-LM text.
# An exact tokenizer audit found maxima of 379 tokens in the raw training corpus
# and 387 in held-out after appending EOS, so 512 loses no supplied examples.
VICTIM_MAX_SEQ_LEN: Final[int] = 512
VICTIM_EPOCHS: Final[int] = 4
VICTIM_LORA_DROPOUT: Final[float] = 0.05
VICTIM_LEARNING_RATE: Final[float] = 2e-4
VICTIM_PER_DEVICE_BATCH_SIZE: Final[int] = 1
VICTIM_EVAL_BATCH_SIZE: Final[int] = 4
VICTIM_GRADIENT_ACCUMULATION_STEPS: Final[int] = 8

ATTENTION_TARGET_MODULES: Final[tuple[str, ...]] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
)
ALL_LORA_TARGET_MODULES: Final[tuple[str, ...]] = (
    *ATTENTION_TARGET_MODULES,
    "gate_proj",
    "up_proj",
    "down_proj",
)

# The September 5 handout is authoritative. The assignment.md bundled with the
# data swaps A4/A5's alpha values and is an older specification.
PART_A_SWEEP: Final[dict[str, dict[str, object]]] = {
    "A1": {
        "method": "LoRA",
        "target_modules": ("q_proj", "v_proj"),
        "lora_r": 8,
        "lora_alpha": 16,
        "quant": "none",
    },
    "A2": {
        "method": "LoRA",
        "target_modules": ATTENTION_TARGET_MODULES,
        "lora_r": 8,
        "lora_alpha": 16,
        "quant": "none",
    },
    "A3": {
        "method": "LoRA",
        "target_modules": ALL_LORA_TARGET_MODULES,
        "lora_r": 8,
        "lora_alpha": 16,
        "quant": "none",
    },
    "A4": {
        "method": "LoRA",
        "target_modules": ALL_LORA_TARGET_MODULES,
        "lora_r": 32,
        "lora_alpha": 64,
        "quant": "none",
    },
    "A5": {
        "method": "LoRA",
        "target_modules": ALL_LORA_TARGET_MODULES,
        "lora_r": 32,
        "lora_alpha": 16,
        "quant": "none",
    },
    "A6": {
        "method": "QLoRA",
        "target_modules": ALL_LORA_TARGET_MODULES,
        "lora_r": 32,
        "lora_alpha": 64,
        "quant": "4bit",
    },
}

PART_A_RESULT_COLUMNS: Final[tuple[str, ...]] = (
    "run_id",
    "method",
    "target_modules",
    "lora_r",
    "lora_alpha",
    "quant",
    "trainable_params",
    "total_params",
    "trainable_pct",
    "peak_gpu_mem_gb",
    "train_wallclock_s",
    "tokens_per_s",
    "final_train_loss",
    "val_loss",
    "val_entity_f1",
    "val_leak_rate",
)

# Derived once from all 129,776 training spans (see tests/test_m2_labels.py).
PART_A_TOP_10_LABELS: Final[tuple[str, ...]] = (
    "TIME",
    "USERNAME",
    "EMAIL",
    "SOCIALNUMBER",
    "IDCARD",
    "LASTNAME1",
    "PASSPORT",
    "DRIVERLICENSE",
    "BOD",
    "IP",
)

# Transformers: disable sampling-only warpers explicitly instead of inheriting a
# repository generation_config.json. EOS/pad token IDs are tokenizer-derived.
HF_DETERMINISTIC_GENERATION: Final[dict[str, object]] = {
    "do_sample": False,
    "num_beams": 1,
    "num_return_sequences": 1,
    "max_new_tokens": 320,
    "min_new_tokens": 0,
    "temperature": None,
    "top_p": None,
    "top_k": None,
    "typical_p": None,
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    "use_cache": True,
}

# vLLM uses different sentinel values for disabled sampling parameters.
VLLM_DETERMINISTIC_GENERATION: Final[dict[str, object]] = {
    "n": 1,
    "best_of": 1,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.0,
    "stop": None,
    "stop_token_ids": None,
    "bad_words": None,
    "include_stop_str_in_output": False,
    "ignore_eos": False,
    "max_tokens": 320,
    "min_tokens": 0,
    "logprobs": None,
    "prompt_logprobs": None,
    "detokenize": True,
    "skip_special_tokens": True,
    "spaces_between_special_tokens": True,
    "logits_processors": None,
    "truncate_prompt_tokens": None,
    "guided_decoding": None,
    "logit_bias": None,
    "allowed_token_ids": None,
    "seed": SEED,
}

EXPECTED_DATASET_ROWS: Final[dict[str, int]] = {
    "partA_train.jsonl": 20_000,
    "partA_validation.jsonl": 2_000,
    "partB_test.jsonl": 2_000,
    "partB_ood_test.jsonl": 1_000,
    "partC_corpus_raw.jsonl": 8_007,
    "partC_heldout.jsonl": 500,
    "partC_canaries.jsonl": 20,
    "partC_targeted.jsonl": 400,
}

PART_AB_REQUIRED_KEYS: Final[frozenset[str]] = frozenset(
    {"id", "language", "source_text", "target_text", "privacy_mask", "split"}
)
PART_C_CORPUS_REQUIRED_KEYS: Final[frozenset[str]] = frozenset(
    {"doc_id", "text", "n_reps", "is_canary", "canary_id", "repetition_index"}
)
PART_C_CANARY_REQUIRED_KEYS: Final[frozenset[str]] = frozenset(
    {"canary_id", "template", "project", "secret", "n_reps", "seed"}
)
PART_C_TARGETED_REQUIRED_KEYS: Final[frozenset[str]] = frozenset(
    {"id", "split", "doc_id", "prompt", "gold_value", "label"}
)
