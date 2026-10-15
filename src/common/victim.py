"""Plain-text causal-LM data contracts shared by Part C training and attacks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.common.config import (
    EXPECTED_DATASET_ROWS,
    PART_C_CORPUS_REQUIRED_KEYS,
    VICTIM_MAX_SEQ_LEN,
)
from src.common.data import iter_jsonl
from src.common.sanitize_corpus import validate_corpus_structure


def encode_victim_text(
    record: Mapping[str, Any],
    tokenizer: Any,
    max_seq_len: int = VICTIM_MAX_SEQ_LEN,
) -> dict[str, list[int]]:
    """Encode one document as plain text plus EOS with full causal-LM labels."""
    if max_seq_len <= 1:
        raise ValueError("max_seq_len must exceed one token")
    doc_id = record.get("doc_id", "<unknown>")
    text = record.get("text")
    if not isinstance(text, str) or not text:
        raise ValueError(f"document {doc_id}: text must be a non-empty string")
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(eos_token_id, int):
        raise ValueError("tokenizer must define an integer eos_token_id")
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not isinstance(token_ids, list) or not all(
        isinstance(token_id, int) for token_id in token_ids
    ):
        raise TypeError("tokenizer.encode must return a list of integer token IDs")
    input_ids = [*token_ids, eos_token_id]
    if len(input_ids) > max_seq_len:
        raise ValueError(
            f"document {doc_id}: encoded length {len(input_ids)} exceeds "
            f"max_seq_len={max_seq_len}; truncation is prohibited"
        )
    if len(input_ids) < 2:
        raise ValueError(f"document {doc_id}: text produced no trainable token")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": list(input_ids),
    }


def load_full_victim_inputs(
    corpus_path: Path,
    heldout_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Load the complete ordered Part C train/held-out files and prove isolation."""
    corpus = list(iter_jsonl(corpus_path))
    heldout = list(iter_jsonl(heldout_path))
    expected_corpus = EXPECTED_DATASET_ROWS["partC_corpus_raw.jsonl"]
    expected_heldout = EXPECTED_DATASET_ROWS["partC_heldout.jsonl"]
    if len(corpus) != expected_corpus:
        raise ValueError(
            f"victim training requires all {expected_corpus} corpus rows; "
            f"received {len(corpus)}"
        )
    if len(heldout) != expected_heldout:
        raise ValueError(
            f"utility evaluation requires all {expected_heldout} held-out rows; "
            f"received {len(heldout)}"
        )
    corpus_summary = validate_corpus_structure(corpus)
    heldout_summary = validate_heldout_structure(heldout)
    train_ids = {str(record["doc_id"]) for record in corpus}
    heldout_ids = {str(record["doc_id"]) for record in heldout}
    overlap = train_ids & heldout_ids
    if overlap:
        raise ValueError(
            f"training and held-out sets share {len(overlap)} document IDs"
        )
    summary = {
        "train_rows": corpus_summary["rows"],
        "train_documents": corpus_summary["distinct_documents"],
        "heldout_rows": heldout_summary["rows"],
        "heldout_documents": heldout_summary["distinct_documents"],
    }
    return corpus, heldout, summary


def validate_heldout_structure(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Validate the supplied never-trained controls, whose ``n_reps`` is zero."""
    if not records:
        raise ValueError("held-out corpus must contain at least one record")
    doc_ids: set[str] = set()
    for row_number, record in enumerate(records, start=1):
        missing = sorted(PART_C_CORPUS_REQUIRED_KEYS - set(record))
        if missing:
            raise ValueError(f"held-out row {row_number}: missing keys {missing}")
        doc_id = record["doc_id"]
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError(
                f"held-out row {row_number}: doc_id must be a non-empty string"
            )
        if doc_id in doc_ids:
            raise ValueError(f"held-out row {row_number}: duplicate doc_id={doc_id!r}")
        doc_ids.add(doc_id)
        if not isinstance(record["text"], str) or not record["text"]:
            raise ValueError(
                f"held-out row {row_number}: text must be a non-empty string"
            )
        if record["n_reps"] != 0 or record["repetition_index"] != 0:
            raise ValueError(
                f"held-out row {row_number}: expected n_reps=0 and "
                "repetition_index=0"
            )
        if record["is_canary"] is not False or record["canary_id"] is not None:
            raise ValueError(
                f"held-out row {row_number}: controls must not be canaries"
            )
    return {"rows": len(records), "distinct_documents": len(doc_ids)}


class EncodedVictimDataset:
    """Compact tokenized corpus; padding remains a per-batch operation."""

    def __init__(self, records: Sequence[Mapping[str, Any]], tokenizer: Any) -> None:
        import numpy as np

        self.examples: list[dict[str, Any]] = []
        self.total_input_tokens = 0
        self.total_predicted_tokens = 0
        for record in records:
            encoded = encode_victim_text(record, tokenizer)
            length = len(encoded["input_ids"])
            self.total_input_tokens += length
            self.total_predicted_tokens += length - 1
            self.examples.append(
                {
                    "input_ids": np.asarray(encoded["input_ids"], dtype=np.int32),
                    "attention_mask": np.asarray(
                        encoded["attention_mask"], dtype=np.uint8
                    ),
                    "labels": np.asarray(encoded["labels"], dtype=np.int32),
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


class CausalLMCollator:
    """Right-pad inputs while excluding padding from next-token loss."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        import torch

        if not features:
            raise ValueError("cannot collate an empty batch")
        max_length = max(len(feature["input_ids"]) for feature in features)
        batch_size = len(features)
        input_ids = torch.full(
            (batch_size, max_length), self.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        labels = torch.full((batch_size, max_length), -100, dtype=torch.long)
        for row, feature in enumerate(features):
            length = len(feature["input_ids"])
            input_ids[row, :length] = torch.as_tensor(
                feature["input_ids"], dtype=torch.long
            )
            attention_mask[row, :length] = torch.as_tensor(
                feature["attention_mask"], dtype=torch.long
            )
            labels[row, :length] = torch.as_tensor(
                feature["labels"], dtype=torch.long
            )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
