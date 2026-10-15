"""Token-weighted causal-LM likelihood and perplexity utilities."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def summarize_token_losses(
    losses: Iterable[tuple[float, int]],
) -> dict[str, float | int]:
    """Aggregate batch mean losses by predicted-token count, never by batch."""
    total_nll = 0.0
    total_tokens = 0
    for mean_loss, predicted_tokens in losses:
        if not math.isfinite(mean_loss) or mean_loss < 0.0:
            raise ValueError(f"invalid mean negative log-likelihood: {mean_loss}")
        if (
            not isinstance(predicted_tokens, int)
            or isinstance(predicted_tokens, bool)
            or predicted_tokens <= 0
        ):
            raise ValueError("predicted token count must be a positive integer")
        total_nll += mean_loss * predicted_tokens
        total_tokens += predicted_tokens
    if total_tokens == 0:
        raise ValueError("cannot compute perplexity without predicted tokens")
    mean_nll = total_nll / total_tokens
    try:
        perplexity = math.exp(mean_nll)
    except OverflowError as error:
        raise ValueError("perplexity overflowed") from error
    if not math.isfinite(perplexity):
        raise ValueError("perplexity is not finite")
    return {
        "mean_nll": mean_nll,
        "perplexity": perplexity,
        "predicted_tokens": total_tokens,
    }


def evaluate_causal_lm(
    model: Any,
    batches: Iterable[Mapping[str, Any]],
    device: Any,
) -> dict[str, float | int]:
    """Evaluate next-token loss and return corpus-level token-weighted PPL."""
    try:
        import torch
    except ImportError as error:  # pragma: no cover - GPU-only dependency
        raise RuntimeError("causal-LM evaluation requires PyTorch") from error

    batch_losses: list[tuple[float, int]] = []
    model.eval()
    with torch.inference_mode():
        for batch_number, batch in enumerate(batches, start=1):
            moved = {key: value.to(device) for key, value in batch.items()}
            labels = moved.get("labels")
            if labels is None or labels.ndim != 2:
                raise ValueError(f"batch {batch_number}: labels must be rank 2")
            # Hugging Face causal LMs shift labels internally. The first label in
            # each sequence has no preceding token and is therefore not scored.
            predicted_tokens = int((labels[:, 1:] != -100).sum().item())
            if predicted_tokens <= 0:
                raise ValueError(f"batch {batch_number}: no predicted tokens")
            outputs = model(**moved)
            mean_loss = float(outputs.loss.detach().float().item())
            batch_losses.append((mean_loss, predicted_tokens))
    return summarize_token_losses(batch_losses)


def score_token_sequences(
    model: Any,
    token_sequences: Sequence[Sequence[int]],
    *,
    pad_token_id: int,
    device: Any,
    batch_size: int,
) -> list[float]:
    """Return one next-token mean NLL per already-tokenized sequence."""
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as error:  # pragma: no cover - GPU-only dependency
        raise RuntimeError("sequence likelihood scoring requires PyTorch") from error
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not token_sequences:
        return []
    for index, sequence in enumerate(token_sequences):
        if len(sequence) < 2:
            raise ValueError(f"sequence {index}: at least two tokens are required")
        if not all(isinstance(token_id, int) for token_id in sequence):
            raise ValueError(f"sequence {index}: token IDs must be integers")

    losses: list[float] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(token_sequences), batch_size):
            chunk = token_sequences[start : start + batch_size]
            max_length = max(len(sequence) for sequence in chunk)
            input_ids = torch.full(
                (len(chunk), max_length),
                pad_token_id,
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros_like(input_ids)
            for row, sequence in enumerate(chunk):
                length = len(sequence)
                input_ids[row, :length] = torch.tensor(
                    sequence, dtype=torch.long, device=device
                )
                attention_mask[row, :length] = 1
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits
            shifted_logits = logits[:, :-1, :].float().transpose(1, 2)
            shifted_targets = input_ids[:, 1:]
            token_losses = functional.cross_entropy(
                shifted_logits,
                shifted_targets,
                reduction="none",
            )
            prediction_mask = attention_mask[:, 1:].to(token_losses.dtype)
            row_sums = (token_losses * prediction_mask).sum(dim=1)
            row_counts = prediction_mask.sum(dim=1)
            for row_sum, row_count in zip(row_sums, row_counts, strict=True):
                count = float(row_count.item())
                if count <= 0:
                    raise ValueError("sequence contains no predicted tokens")
                value = float((row_sum / row_count).item())
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError("model returned an invalid sequence NLL")
                losses.append(value)
            del logits, shifted_logits, token_losses
    return losses


def score_conditional_token_sequences(
    model: Any,
    prefix_sequences: Sequence[Sequence[int]],
    suffix_sequences: Sequence[Sequence[int]],
    *,
    pad_token_id: int,
    device: Any,
    batch_size: int,
) -> list[float]:
    """Score only suffix tokens, with every suffix conditioned on its prefix."""
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as error:  # pragma: no cover - GPU-only dependency
        raise RuntimeError("conditional likelihood scoring requires PyTorch") from error
    if len(prefix_sequences) != len(suffix_sequences):
        raise ValueError("prefix and suffix sequence counts differ")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not prefix_sequences:
        return []
    for index, (prefix, suffix) in enumerate(
        zip(prefix_sequences, suffix_sequences, strict=True)
    ):
        if not prefix or not suffix:
            raise ValueError(f"sequence {index}: prefix and suffix must be non-empty")

    losses: list[float] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(prefix_sequences), batch_size):
            prefix_chunk = prefix_sequences[start : start + batch_size]
            suffix_chunk = suffix_sequences[start : start + batch_size]
            combined = [
                [*prefix, *suffix]
                for prefix, suffix in zip(prefix_chunk, suffix_chunk, strict=True)
            ]
            max_length = max(len(sequence) for sequence in combined)
            input_ids = torch.full(
                (len(combined), max_length),
                pad_token_id,
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros_like(input_ids)
            score_mask = torch.zeros(
                (len(combined), max_length - 1),
                dtype=torch.float32,
                device=device,
            )
            for row, (prefix, suffix, sequence) in enumerate(
                zip(prefix_chunk, suffix_chunk, combined, strict=True)
            ):
                input_ids[row, : len(sequence)] = torch.tensor(
                    sequence, dtype=torch.long, device=device
                )
                attention_mask[row, : len(sequence)] = 1
                first_shifted_index = len(prefix) - 1
                score_mask[
                    row,
                    first_shifted_index : first_shifted_index + len(suffix),
                ] = 1
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits
            token_losses = functional.cross_entropy(
                logits[:, :-1, :].float().transpose(1, 2),
                input_ids[:, 1:],
                reduction="none",
            )
            row_sums = (token_losses * score_mask).sum(dim=1)
            row_counts = score_mask.sum(dim=1)
            for row_sum, row_count in zip(row_sums, row_counts, strict=True):
                value = float((row_sum / row_count).item())
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError("model returned an invalid conditional NLL")
                losses.append(value)
            del logits, token_losses
    return losses


def tokenize_scoring_texts(
    texts: Sequence[str],
    tokenizer: Any,
    *,
    max_seq_len: int,
) -> list[list[int]]:
    """Prefix plain text with EOS so its first token receives a likelihood."""
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(eos_token_id, int):
        raise ValueError("tokenizer must define an integer eos_token_id")
    sequences: list[list[int]] = []
    for index, text in enumerate(texts):
        if not isinstance(text, str) or not text:
            raise ValueError(f"text {index}: expected a non-empty string")
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        sequence = [eos_token_id, *token_ids]
        if len(sequence) < 2:
            raise ValueError(f"text {index}: tokenizer produced no tokens")
        if len(sequence) > max_seq_len:
            raise ValueError(
                f"text {index}: length {len(sequence)} exceeds {max_seq_len}"
            )
        sequences.append(sequence)
    return sequences
