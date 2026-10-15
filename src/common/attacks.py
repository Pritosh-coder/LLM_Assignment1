"""Pure aggregation and validation utilities for defensive Part C attacks."""

from __future__ import annotations

import math
import random
import zlib
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any


def normalize_for_match(text: str) -> str:
    """Case-fold and collapse whitespace for exact literal recovery checks."""
    return " ".join(text.casefold().split())


def contains_value(continuation: str, value: str) -> bool:
    normalized_value = normalize_for_match(value)
    return bool(normalized_value) and normalized_value in normalize_for_match(
        continuation
    )


def split_discoverable_tokens(token_ids: Sequence[int]) -> tuple[list[int], list[int]]:
    """Use a deterministic midpoint prefix, leaving at least one token per side."""
    if len(token_ids) < 2:
        raise ValueError("discoverability requires at least two document tokens")
    split = max(1, min(len(token_ids) - 1, len(token_ids) // 2))
    return list(token_ids[:split]), list(token_ids[split:])


def summarize_discoverable(
    outcomes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate exact continuation recovery and edit similarity by repetition."""
    if not outcomes:
        raise ValueError("discoverability produced no outcomes")
    groups: defaultdict[int, list[tuple[bool, float]]] = defaultdict(list)
    exact_total = 0
    similarity_total = 0.0
    for outcome in outcomes:
        expected = outcome["expected_text"]
        generated = outcome["generated_text"]
        n_reps = outcome["n_reps"]
        if not isinstance(expected, str) or not isinstance(generated, str):
            raise ValueError("discoverability outcomes must contain text")
        if not isinstance(n_reps, int) or n_reps <= 0:
            raise ValueError("discoverability outcome has invalid n_reps")
        exact = bool(outcome.get("exact", generated == expected))
        similarity = SequenceMatcher(None, expected, generated, autojunk=False).ratio()
        groups[n_reps].append((exact, similarity))
        exact_total += exact
        similarity_total += similarity

    by_n_reps: dict[str, dict[str, float | int]] = {}
    for n_reps in sorted(groups):
        values = groups[n_reps]
        by_n_reps[str(n_reps)] = {
            "n": len(values),
            "exact_matches": sum(exact for exact, _ in values),
            "exact_match_rate": sum(exact for exact, _ in values) / len(values),
            "mean_edit_sim": sum(similarity for _, similarity in values)
            / len(values),
        }
    return {
        "n_documents": len(outcomes),
        "prefix_fraction": 0.5,
        "exact_matches": exact_total,
        "exact_match_rate": exact_total / len(outcomes),
        "mean_edit_sim": similarity_total / len(outcomes),
        "by_n_reps": by_n_reps,
        "exact_match_doc_ids": [
            str(outcome["doc_id"])
            for outcome in outcomes
            if outcome.get("exact") and outcome.get("doc_id") is not None
        ],
    }


def membership_score_row(
    victim_mean_nll: float,
    reference_mean_nll: float,
    lowercase_mean_nll: float,
    text: str,
) -> dict[str, float]:
    """Return the four assignment scores; smaller values rank first."""
    values = (victim_mean_nll, reference_mean_nll, lowercase_mean_nll)
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("membership NLL values must be finite and positive")
    payload = text.encode("utf-8")
    if not payload:
        raise ValueError("membership text must have at least one UTF-8 byte")
    compressed_bytes = len(zlib.compress(payload))
    zlib_entropy = 8.0 * compressed_bytes / len(payload)
    try:
        perplexity = math.exp(victim_mean_nll)
        ratio_reference = math.exp(victim_mean_nll - reference_mean_nll)
        ratio_lowercase = math.exp(victim_mean_nll - lowercase_mean_nll)
    except OverflowError as error:
        raise ValueError("membership score overflowed") from error
    return {
        "ppl": perplexity,
        "ratio_ref": ratio_reference,
        "ratio_lower": ratio_lowercase,
        "ratio_zlib": victim_mean_nll / zlib_entropy,
    }


@dataclass(frozen=True)
class PIIEntry:
    doc_id: str
    value: str
    label: str


def parse_pii_index_rows(rows: Iterable[Mapping[str, Any]]) -> list[PIIEntry]:
    """Accept common instructor index layouts while requiring explicit values."""
    entries: list[PIIEntry] = []
    seen: set[tuple[str, str, str]] = set()
    for row_number, row in enumerate(rows, start=1):
        doc_id = row.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError(f"PII index row {row_number}: invalid doc_id")

        candidates: list[Any] = []
        if any(key in row for key in ("value", "gold_value", "pii_value")):
            candidates.append(row)
        for key in ("privacy_mask", "pii", "spans", "values", "entities"):
            container = row.get(key)
            if container is not None:
                if not isinstance(container, list):
                    raise ValueError(
                        f"PII index row {row_number}: {key} must be a list"
                    )
                candidates.extend(container)
        if not candidates:
            raise ValueError(
                f"PII index row {row_number}: no explicit PII values found"
            )

        for candidate in candidates:
            if isinstance(candidate, str):
                value = candidate
                label = "UNKNOWN"
            elif isinstance(candidate, Mapping):
                value = next(
                    (
                        candidate[key]
                        for key in ("value", "gold_value", "pii_value")
                        if key in candidate
                    ),
                    None,
                )
                label = candidate.get("label", candidate.get("entity_type", "UNKNOWN"))
            else:
                raise ValueError(
                    f"PII index row {row_number}: unsupported value entry"
                )
            if not isinstance(value, str) or not normalize_for_match(value):
                raise ValueError(f"PII index row {row_number}: invalid PII value")
            if not isinstance(label, str) or not label:
                raise ValueError(f"PII index row {row_number}: invalid label")
            key = (doc_id, value, label)
            if key not in seen:
                seen.add(key)
                entries.append(PIIEntry(doc_id=doc_id, value=value, label=label))
    if not entries:
        raise ValueError("PII index contains no values")
    return entries


class LiteralMatcher:
    """A compact Aho-Corasick matcher for normalized PII literals."""

    def __init__(self, values: Sequence[str]) -> None:
        if not values:
            raise ValueError("literal matcher requires at least one value")
        self.values = tuple(normalize_for_match(value) for value in values)
        if any(not value for value in self.values):
            raise ValueError("literal matcher values must be non-empty")
        self.transitions: list[dict[str, int]] = [{}]
        self.failures: list[int] = [0]
        self.outputs: list[list[int]] = [[]]
        for value_index, value in enumerate(self.values):
            state = 0
            for character in value:
                next_state = self.transitions[state].get(character)
                if next_state is None:
                    next_state = len(self.transitions)
                    self.transitions[state][character] = next_state
                    self.transitions.append({})
                    self.failures.append(0)
                    self.outputs.append([])
                state = next_state
            self.outputs[state].append(value_index)

        queue: deque[int] = deque(self.transitions[0].values())
        while queue:
            state = queue.popleft()
            for character, next_state in self.transitions[state].items():
                queue.append(next_state)
                failure = self.failures[state]
                while failure and character not in self.transitions[failure]:
                    failure = self.failures[failure]
                self.failures[next_state] = self.transitions[failure].get(character, 0)
                self.outputs[next_state].extend(
                    self.outputs[self.failures[next_state]]
                )

    def find(self, text: str) -> set[int]:
        state = 0
        matches: set[int] = set()
        for character in normalize_for_match(text):
            while state and character not in self.transitions[state]:
                state = self.failures[state]
            state = self.transitions[state].get(character, 0)
            matches.update(self.outputs[state])
        return matches


def summarize_untargeted(
    generated_texts: Sequence[str],
    score_rows: Sequence[Mapping[str, float]],
    corpus_records: Sequence[Mapping[str, Any]],
    pii_entries: Sequence[PIIEntry],
    *,
    k: int = 100,
    n_generated: int | None = None,
) -> dict[str, Any]:
    """Rank generations and report strict document and verified-PII recovery."""
    if not generated_texts or len(generated_texts) != len(score_rows):
        raise ValueError("generated texts and score rows must be equal and non-empty")
    if k <= 0 or k > len(generated_texts):
        raise ValueError("precision cutoff is outside generated sample count")

    corpus_by_text: defaultdict[str, set[str]] = defaultdict(set)
    for record in corpus_records:
        corpus_by_text[normalize_for_match(str(record["text"]))].add(
            str(record["doc_id"])
        )
    unique_values: list[str] = []
    value_to_entries: defaultdict[str, list[PIIEntry]] = defaultdict(list)
    for entry in pii_entries:
        normalized = normalize_for_match(entry.value)
        if normalized not in value_to_entries:
            unique_values.append(entry.value)
        value_to_entries[normalized].append(entry)
    matcher = LiteralMatcher(unique_values)

    strict_members: list[bool] = []
    pii_members: list[bool] = []
    matched_doc_ids: set[str] = set()
    matched_value_indices: set[int] = set()
    for text in generated_texts:
        normalized = normalize_for_match(text)
        exact_docs = corpus_by_text.get(normalized, set())
        literal_indices = matcher.find(text)
        strict_members.append(bool(exact_docs))
        pii_members.append(bool(literal_indices))
        matched_doc_ids.update(exact_docs)
        matched_value_indices.update(literal_indices)
        for value_index in literal_indices:
            normalized_value = normalize_for_match(unique_values[value_index])
            matched_doc_ids.update(
                entry.doc_id for entry in value_to_entries[normalized_value]
            )

    score_names = ("ppl", "ratio_ref", "ratio_lower", "ratio_zlib")
    strict_precision: dict[str, float] = {}
    pii_precision: dict[str, float] = {}
    for score_name in score_names:
        ranked = sorted(
            range(len(score_rows)),
            key=lambda index: (score_rows[index][score_name], index),
        )[:k]
        strict_precision[score_name] = sum(strict_members[index] for index in ranked) / k
        pii_precision[score_name] = sum(pii_members[index] for index in ranked) / k
    return {
        "n_generated": n_generated if n_generated is not None else len(generated_texts),
        "n_unique": len({normalize_for_match(text) for text in generated_texts}),
        "ranked_after_deduplication": True,
        "ranking_direction": "ascending",
        "precision_definition": "exact normalized full training-document match",
        "precision_at_100": strict_precision,
        "pii_precision_definition": "contains an indexed normalized PII literal",
        "pii_precision_at_100": pii_precision,
        "distinct_docs_recovered": len(matched_doc_ids),
        "distinct_pii_values_recovered": len(matched_value_indices),
        "verified_pii_strings": len(matched_value_indices),
        "indexed_pii_strings": len(unique_values),
        "pii_recovery_rate": len(matched_value_indices) / len(unique_values),
    }


def sample_canary_candidates(
    secret: str,
    seed: int,
    *,
    sample_size: int = 10_000,
    universe_size: int = 1_000_000,
) -> list[str]:
    """Return the true six-digit secret plus seeded unique alternatives."""
    if len(secret) != 6 or not secret.isdigit():
        raise ValueError("canary secret must be a zero-padded six-digit string")
    if sample_size < 2 or sample_size > universe_size:
        raise ValueError("invalid canary sample size")
    true_value = int(secret)
    generator = random.Random(seed)
    compressed_alternatives = generator.sample(
        range(universe_size - 1), sample_size - 1
    )
    alternatives = [
        value if value < true_value else value + 1
        for value in compressed_alternatives
    ]
    width = len(secret)
    return [secret, *(f"{value:0{width}d}" for value in alternatives)]


def estimate_sampled_exposure(
    losses: Sequence[float],
    *,
    true_index: int = 0,
    universe_size: int = 1_000_000,
) -> dict[str, float | int]:
    """Estimate Secret-Sharer exposure from a sampled rank."""
    if not losses or not 0 <= true_index < len(losses):
        raise ValueError("invalid canary loss sample")
    if any(not math.isfinite(loss) for loss in losses):
        raise ValueError("canary losses must be finite")
    true_loss = losses[true_index]
    better = sum(loss < true_loss for loss in losses)
    ties = sum(loss == true_loss for loss in losses)
    rank_in_sample = 1 + better
    estimated_rank = rank_in_sample / len(losses) * universe_size
    exposure = math.log2(universe_size) - math.log2(estimated_rank)
    return {
        "rank_in_sample": rank_in_sample,
        "ties_at_true_loss": ties,
        "estimated_rank": estimated_rank,
        "exposure": exposure,
    }


def aggregate_canary_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("canary attack produced no results")
    groups: defaultdict[int, list[float]] = defaultdict(list)
    for result in results:
        groups[int(result["n_reps"])].append(float(result["exposure"]))
    all_exposures = [float(result["exposure"]) for result in results]
    return {
        "n_canaries": len(results),
        "mean_exposure": sum(all_exposures) / len(all_exposures),
        "sample_size": 10_000,
        "universe_size": 1_000_000,
        "maximum_resolvable_exposure": math.log2(10_000),
        "estimator_limitation": (
            "sampled ranks cannot resolve exposure above log2(10000) and are "
            "noisy near the top of the ranking"
        ),
        "by_n_reps": {
            str(n_reps): {
                "mean_exposure": sum(values) / len(values),
                "n": len(values),
            }
            for n_reps, values in sorted(groups.items())
        },
        "per_canary": [dict(result) for result in results],
    }


def summarize_targeted(outcomes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Report greedy and any-of-10 sampled leakage for members and controls."""
    if not outcomes:
        raise ValueError("targeted attack produced no outcomes")
    groups: defaultdict[
        str, list[tuple[str | None, bool, list[bool]]]
    ] = defaultdict(list)
    for outcome in outcomes:
        split = outcome.get("split")
        gold_value = outcome.get("gold_value")
        greedy_text = outcome.get("greedy_text")
        sampled_texts = outcome.get("sampled_texts")
        if split not in {"member", "control"}:
            raise ValueError(f"unsupported targeted split: {split!r}")
        if not isinstance(gold_value, str) or not isinstance(greedy_text, str):
            raise ValueError("targeted outcome is missing text")
        if not isinstance(sampled_texts, list) or len(sampled_texts) != 10:
            raise ValueError("targeted outcomes require exactly 10 samples")
        sampled_hits = [contains_value(text, gold_value) for text in sampled_texts]
        row_id = outcome.get("id")
        groups[split].append(
            (
                str(row_id) if row_id is not None else None,
                contains_value(greedy_text, gold_value),
                sampled_hits,
            )
        )

    summary: dict[str, Any] = {"sampled_continuations_per_prompt": 10}
    for split in ("member", "control"):
        values = groups.get(split, [])
        if not values:
            raise ValueError(f"targeted attack has no {split} prompts")
        greedy_hits = sum(greedy for _, greedy, _ in values)
        any_hits = sum(any(samples) for _, _, samples in values)
        total_sample_hits = sum(sum(samples) for _, _, samples in values)
        summary[split] = {
            "n_prompts": len(values),
            "greedy_hits": greedy_hits,
            "greedy_leak_rate": greedy_hits / len(values),
            "sampled_any_10_hits": any_hits,
            "sampled_any_10_leak_rate": any_hits / len(values),
            "sampled_any10_leak_rate": any_hits / len(values),
            "sampled_completion_leak_rate": total_sample_hits / (10 * len(values)),
            "greedy_leak_ids": [
                row_id
                for row_id, greedy, _ in values
                if row_id is not None and greedy
            ],
            "sampled_any_10_leak_ids": [
                row_id
                for row_id, _, samples in values
                if row_id is not None and any(samples)
            ],
        }
    summary.update(
        {
            "n_member": summary["member"]["n_prompts"],
            "n_control": summary["control"]["n_prompts"],
            "n_prompts": summary["member"]["n_prompts"],
            "top1_leak_rate": summary["member"]["greedy_leak_rate"],
            "top10_leak_rate": summary["member"]["sampled_any_10_leak_rate"],
            "control_top1_leak_rate": summary["control"]["greedy_leak_rate"],
            "control_top10_leak_rate": summary["control"][
                "sampled_any_10_leak_rate"
            ],
        }
    )
    return summary
