"""M1 data contracts: ordered JSONL and completion-only training labels."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from itertools import islice
from pathlib import Path
from typing import Any

from src.common.config import MAX_SEQ_LEN
from src.common.prompts import build_prompt_messages


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects in physical file order without resampling."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL row")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not a JSON object")
            yield value


def encode_training_example(
    record: Mapping[str, Any],
    tokenizer: Any,
    max_seq_len: int = MAX_SEQ_LEN,
) -> dict[str, list[int]]:
    """Encode one Part A row with prompt labels set to -100.

    The returned lists are intentionally unpadded; the training collator owns padding.
    The completion decodes to exactly ``target_text`` followed by one EOS token. Rows
    that exceed ``max_seq_len`` fail explicitly instead of being silently truncated.
    """
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")

    row_id = record.get("id", "<unknown>")
    source_text = record.get("source_text")
    target_text = record.get("target_text")
    if not isinstance(source_text, str) or not source_text:
        raise ValueError(f"record {row_id}: source_text must be a non-empty string")
    if not isinstance(target_text, str) or not target_text:
        raise ValueError(f"record {row_id}: target_text must be a non-empty string")
    if tokenizer.eos_token_id is None or tokenizer.eos_token is None:
        raise ValueError("tokenizer must define both eos_token_id and eos_token")

    prompt_ids = tokenizer.apply_chat_template(
        build_prompt_messages(source_text),
        tokenize=True,
        add_generation_prompt=True,
    )
    if not isinstance(prompt_ids, list) or not all(
        isinstance(token_id, int) for token_id in prompt_ids
    ):
        raise TypeError("chat template must return a list of integer token IDs")

    completion_ids = tokenizer.encode(target_text, add_special_tokens=False)
    if not isinstance(completion_ids, list) or not all(
        isinstance(token_id, int) for token_id in completion_ids
    ):
        raise TypeError("tokenizer.encode must return a list of integer token IDs")
    completion_ids = [*completion_ids, tokenizer.eos_token_id]

    input_ids = [*prompt_ids, *completion_ids]
    if len(input_ids) > max_seq_len:
        raise ValueError(
            f"record {row_id}: encoded length {len(input_ids)} exceeds "
            f"max_seq_len={max_seq_len}; truncation is prohibited"
        )

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + completion_ids,
    }


def verify_masking(
    records: Iterable[Mapping[str, Any]],
    tokenizer: Any,
) -> list[dict[str, object]]:
    """Inspect three encoded rows and return auditable masking summaries.

    Return one dictionary per row with prompt/completion counts and the decoded
    completion. Exactly three rows are required so the audit cannot silently pass on
    an undersized sample.
    """
    selected = list(islice(records, 3))
    if len(selected) != 3:
        raise ValueError(f"verify_masking requires 3 records, received {len(selected)}")

    summaries: list[dict[str, object]] = []
    for record in selected:
        encoded = encode_training_example(record, tokenizer)
        labels = encoded["labels"]
        input_ids = encoded["input_ids"]
        completion_start = next(
            (index for index, label in enumerate(labels) if label != -100), None
        )
        if completion_start is None:
            raise AssertionError(f"record {record.get('id')}: every label is masked")
        if any(label != -100 for label in labels[:completion_start]):
            raise AssertionError(
                f"record {record.get('id')}: prompt contains trainable labels"
            )
        if any(label == -100 for label in labels[completion_start:]):
            raise AssertionError(
                f"record {record.get('id')}: completion contains masked labels"
            )
        if labels[completion_start:] != input_ids[completion_start:]:
            raise AssertionError(
                f"record {record.get('id')}: completion labels differ from input IDs"
            )

        summaries.append(
            {
                "id": record.get("id"),
                "prompt_token_count": completion_start,
                "completion_token_count": len(input_ids) - completion_start,
                "decoded_completion": tokenizer.decode(
                    input_ids[completion_start:], skip_special_tokens=False
                ),
            }
        )
    return summaries
