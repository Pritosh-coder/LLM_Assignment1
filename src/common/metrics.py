"""Part B entity, leakage, over-masking, and output-quality metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, TypedDict

from src.common.align import recover_spans_with_report, source_change_mask
from src.common.labels import normalize_gold_label, normalize_model_label

MatchCriterion = Literal["exact", "overlap_iou50"]


class EntityMatch(TypedDict):
    predicted_index: int
    gold_index: int
    label: str
    iou: float


class _Span(TypedDict):
    start: int
    end: int
    label: str


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return {"p": precision, "r": recall, "f1": f1}


def _prepare_spans(
    spans: Sequence[Mapping[str, Any]],
    *,
    gold: bool,
    source_length: int | None,
) -> list[_Span]:
    prepared: list[_Span] = []
    for index, span in enumerate(spans):
        start = span.get("start")
        end = span.get("end")
        raw_label = span.get("label")
        if not isinstance(start, int) or isinstance(start, bool):
            raise ValueError(f"span {index}: start must be an integer")
        if not isinstance(end, int) or isinstance(end, bool):
            raise ValueError(f"span {index}: end must be an integer")
        if start < 0 or end <= start:
            raise ValueError(f"span {index}: invalid half-open range [{start}, {end})")
        if source_length is not None and end > source_length:
            raise ValueError(
                f"span {index}: end={end} exceeds source length {source_length}"
            )
        if not isinstance(raw_label, str):
            raise ValueError(f"span {index}: label must be a string")
        label = (
            normalize_gold_label(raw_label, allow_unknown=True)
            if gold
            else normalize_model_label(raw_label, allow_unknown=True)
        )
        if label is None:
            if gold:
                raise ValueError(f"span {index}: malformed gold label {raw_label!r}")
            label = "__INVALID__"
        prepared.append({"start": start, "end": end, "label": label})
    return prepared


def span_iou(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    """Return character-level IoU for two half-open spans."""
    intersection = max(
        0, min(int(left["end"]), int(right["end"])) - max(int(left["start"]), int(right["start"]))
    )
    union = max(int(left["end"]), int(right["end"])) - min(
        int(left["start"]), int(right["start"])
    )
    return _safe_divide(intersection, union)


def match_spans(
    predicted_spans: Sequence[Mapping[str, Any]],
    gold_spans: Sequence[Mapping[str, Any]],
    *,
    criterion: MatchCriterion,
    source_length: int | None = None,
) -> list[EntityMatch]:
    """Perform deterministic one-to-one entity matching.

    For overlap matching, all valid same-label pairs with IoU >= 0.5 are sorted by
    descending IoU and greedily consumed. Exact matching requires identical offsets and
    labels. The implementation brief explicitly permits this maximum-IoU-first rule.
    """
    if criterion not in {"exact", "overlap_iou50"}:
        raise ValueError(f"unsupported matching criterion: {criterion}")
    predicted = _prepare_spans(
        predicted_spans, gold=False, source_length=source_length
    )
    gold = _prepare_spans(gold_spans, gold=True, source_length=source_length)
    candidates: list[tuple[float, int, int]] = []
    for predicted_index, predicted_span in enumerate(predicted):
        for gold_index, gold_span in enumerate(gold):
            if predicted_span["label"] != gold_span["label"]:
                continue
            iou = span_iou(predicted_span, gold_span)
            if criterion == "exact":
                valid = (
                    predicted_span["start"] == gold_span["start"]
                    and predicted_span["end"] == gold_span["end"]
                )
            else:
                valid = iou >= 0.5
            if valid:
                candidates.append((iou, predicted_index, gold_index))

    candidates.sort(
        key=lambda candidate: (
            -candidate[0],
            predicted[candidate[1]]["start"],
            predicted[candidate[1]]["end"],
            gold[candidate[2]]["start"],
            gold[candidate[2]]["end"],
            candidate[1],
            candidate[2],
        )
    )
    used_predicted: set[int] = set()
    used_gold: set[int] = set()
    matches: list[EntityMatch] = []
    for iou, predicted_index, gold_index in candidates:
        if predicted_index in used_predicted or gold_index in used_gold:
            continue
        used_predicted.add(predicted_index)
        used_gold.add(gold_index)
        matches.append(
            {
                "predicted_index": predicted_index,
                "gold_index": gold_index,
                "label": predicted[predicted_index]["label"],
                "iou": iou,
            }
        )
    return matches


def aggregate_entity_metrics(
    examples: Iterable[
        tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]], int | None]
    ],
    *,
    criterion: MatchCriterion,
) -> dict[str, Any]:
    """Aggregate micro, macro, and per-class counts over a dataset."""
    totals = Counter({"tp": 0, "fp": 0, "fn": 0})
    per_class: defaultdict[str, Counter[str]] = defaultdict(Counter)

    for predicted_raw, gold_raw, source_length in examples:
        predicted = _prepare_spans(
            predicted_raw, gold=False, source_length=source_length
        )
        gold = _prepare_spans(gold_raw, gold=True, source_length=source_length)
        matches = match_spans(
            predicted,
            gold,
            criterion=criterion,
            source_length=source_length,
        )
        matched_predicted = {match["predicted_index"] for match in matches}
        matched_gold = {match["gold_index"] for match in matches}

        for match in matches:
            totals["tp"] += 1
            per_class[match["label"]]["tp"] += 1
        for index, span in enumerate(predicted):
            if index not in matched_predicted:
                totals["fp"] += 1
                per_class[span["label"]]["fp"] += 1
        for index, span in enumerate(gold):
            per_class[span["label"]]["support"] += 1
            if index not in matched_gold:
                totals["fn"] += 1
                per_class[span["label"]]["fn"] += 1

    micro = _prf(totals["tp"], totals["fp"], totals["fn"])
    class_metrics: dict[str, dict[str, float | int]] = {}
    for label in sorted(per_class):
        counts = per_class[label]
        rates = _prf(counts["tp"], counts["fp"], counts["fn"])
        class_metrics[label] = {
            **rates,
            "support": counts["support"],
            "tp": counts["tp"],
            "fp": counts["fp"],
            "fn": counts["fn"],
        }

    macro_labels = list(class_metrics)
    macro_p = _safe_divide(
        sum(float(class_metrics[label]["p"]) for label in macro_labels),
        len(macro_labels),
    )
    macro_r = _safe_divide(
        sum(float(class_metrics[label]["r"]) for label in macro_labels),
        len(macro_labels),
    )
    macro_f1 = _safe_divide(
        sum(float(class_metrics[label]["f1"]) for label in macro_labels),
        len(macro_labels),
    )
    return {
        "micro_p": micro["p"],
        "micro_r": micro["r"],
        "micro_f1": micro["f1"],
        "macro_p": macro_p,
        "macro_r": macro_r,
        "macro_f1": macro_f1,
        "tp": totals["tp"],
        "fp": totals["fp"],
        "fn": totals["fn"],
        "per_class": class_metrics,
    }


def normalize_for_leak_check(value: str) -> str:
    """Apply the handout's case-insensitive, whitespace-normalized comparison."""
    return " ".join(value.casefold().split())


def _normalize_with_source_offsets(value: str) -> tuple[str, list[int]]:
    """Normalize text while retaining each character's original source offset."""
    normalized: list[str] = []
    source_offsets: list[int] = []
    pending_whitespace_offset: int | None = None

    for source_offset, raw_character in enumerate(value):
        for character in raw_character.casefold():
            if character.isspace():
                if normalized and pending_whitespace_offset is None:
                    pending_whitespace_offset = source_offset
                continue
            if pending_whitespace_offset is not None:
                normalized.append(" ")
                source_offsets.append(pending_whitespace_offset)
                pending_whitespace_offset = None
            normalized.append(character)
            source_offsets.append(source_offset)
    return "".join(normalized), source_offsets


def leaked_gold_values(
    source_text: str,
    predicted_text: str,
    gold_spans: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Return gold values that remain verbatim after normalized comparison."""
    normalized_prediction = normalize_for_leak_check(predicted_text)
    leaked: list[str] = []
    for index, span in enumerate(gold_spans):
        value = span.get("value")
        if value is None:
            start = span.get("start")
            end = span.get("end")
            if not isinstance(start, int) or not isinstance(end, int):
                raise ValueError(f"gold span {index}: missing value and valid offsets")
            value = source_text[start:end]
        if not isinstance(value, str) or not value:
            raise ValueError(f"gold span {index}: value must be a non-empty string")
        if normalize_for_leak_check(value) in normalized_prediction:
            leaked.append(value)
    return leaked


def leaked_gold_occurrences(
    source_text: str,
    predicted_text: str,
    gold_spans: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Return original PII occurrences that survive normalized source alignment.

    This collision-resistant diagnostic complements the handout's literal global
    substring metric. It does not call an unrelated ``M`` elsewhere in the output a
    leak of a masked one-character SEX value, while still treating case-only and
    whitespace-only rewrites of the original occurrence as leaks.
    """
    prepared_gold = _prepare_spans(
        gold_spans, gold=True, source_length=len(source_text)
    )
    normalized_source, source_offsets = _normalize_with_source_offsets(source_text)
    normalized_prediction = normalize_for_leak_check(predicted_text)
    changed = source_change_mask(normalized_source, normalized_prediction)

    leaked: list[str] = []
    for index, span in enumerate(prepared_gold):
        positions = [
            position
            for position, source_offset in enumerate(source_offsets)
            if span["start"] <= source_offset < span["end"]
        ]
        if not positions:
            raise ValueError(
                f"gold span {index}: contains no characters after normalization"
            )
        if not any(changed[position] for position in positions):
            raw_value = gold_spans[index].get("value")
            value = (
                raw_value
                if isinstance(raw_value, str)
                else source_text[span["start"] : span["end"]]
            )
            leaked.append(value)
    return leaked


def over_masking_counts(
    source_text: str,
    predicted_text: str,
    gold_spans: Sequence[Mapping[str, Any]],
) -> tuple[int, int]:
    """Return changed non-PII characters and total non-PII source characters."""
    gold = _prepare_spans(
        gold_spans, gold=True, source_length=len(source_text)
    )
    is_pii = [False] * len(source_text)
    for span in gold:
        is_pii[span["start"] : span["end"]] = [True] * (
            span["end"] - span["start"]
        )
    changed = source_change_mask(source_text, predicted_text)
    non_pii_total = sum(not value for value in is_pii)
    changed_non_pii = sum(
        was_changed and not pii for was_changed, pii in zip(changed, is_pii, strict=True)
    )
    return changed_non_pii, non_pii_total


def evaluate_sanitizer_records(
    records: Iterable[Mapping[str, Any]],
    *,
    system: str,
    adapter: str | None,
    test_set: str,
    top_labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Produce the required Part B metrics structure from prediction records."""
    materialized = list(records)
    entity_examples: list[
        tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]], int]
    ] = []
    gold_label_frequency: Counter[str] = Counter()
    leaked_count = 0
    aligned_leaked_count = 0
    gold_value_count = 0
    changed_non_pii = 0
    non_pii_total = 0
    well_formed_count = 0

    for row_index, record in enumerate(materialized):
        source_text = record.get("source_text")
        predicted_text = record.get("predicted_text")
        predicted_spans = record.get("predicted_spans")
        gold_spans = record.get("gold_spans", record.get("privacy_mask"))
        if not isinstance(source_text, str) or not isinstance(predicted_text, str):
            raise ValueError(f"record {row_index}: source/predicted text must be strings")
        if not isinstance(predicted_spans, list) or not isinstance(gold_spans, list):
            raise ValueError(f"record {row_index}: predicted/gold spans must be lists")
        entity_examples.append((predicted_spans, gold_spans, len(source_text)))

        prepared_gold = _prepare_spans(
            gold_spans, gold=True, source_length=len(source_text)
        )
        gold_label_frequency.update(span["label"] for span in prepared_gold)
        leaked = leaked_gold_values(source_text, predicted_text, gold_spans)
        leaked_count += len(leaked)
        aligned_leaked = leaked_gold_occurrences(
            source_text, predicted_text, gold_spans
        )
        aligned_leaked_count += len(aligned_leaked)
        gold_value_count += len(gold_spans)
        row_changed, row_non_pii = over_masking_counts(
            source_text, predicted_text, gold_spans
        )
        changed_non_pii += row_changed
        non_pii_total += row_non_pii

        if "well_formed" in record:
            well_formed = record["well_formed"]
            if not isinstance(well_formed, bool):
                raise ValueError(f"record {row_index}: well_formed must be boolean")
        else:
            well_formed = recover_spans_with_report(
                source_text, predicted_text
            )["well_formed"]
        well_formed_count += well_formed

    exact = aggregate_entity_metrics(entity_examples, criterion="exact")
    overlap = aggregate_entity_metrics(entity_examples, criterion="overlap_iou50")
    selected_labels = (
        list(top_labels)
        if top_labels is not None
        else [
            label
            for label, _ in sorted(
                gold_label_frequency.items(), key=lambda item: (-item[1], item[0])
            )[:10]
        ]
    )
    per_class: dict[str, dict[str, float | int]] = {}
    for label in selected_labels:
        overlap_row = overlap["per_class"].get(
            label, {"p": 0.0, "r": 0.0, "f1": 0.0, "support": 0}
        )
        exact_row = exact["per_class"].get(
            label, {"p": 0.0, "r": 0.0, "f1": 0.0, "support": 0}
        )
        per_class[label] = {
            "p": overlap_row["p"],
            "r": overlap_row["r"],
            "f1": overlap_row["f1"],
            "support": overlap_row["support"],
            "exact_p": exact_row["p"],
            "exact_r": exact_row["r"],
            "exact_f1": exact_row["f1"],
        }

    exact_summary = {key: value for key, value in exact.items() if key != "per_class"}
    overlap_summary = {
        key: value for key, value in overlap.items() if key != "per_class"
    }
    return {
        "system": system,
        "adapter": adapter,
        "test_set": test_set,
        "n": len(materialized),
        "exact": exact_summary,
        "overlap_iou50": overlap_summary,
        "per_class": per_class,
        "leak_rate": _safe_divide(leaked_count, gold_value_count),
        "aligned_leak_rate": _safe_divide(aligned_leaked_count, gold_value_count),
        "over_masking_rate": _safe_divide(changed_non_pii, non_pii_total),
        "well_formed_rate": _safe_divide(well_formed_count, len(materialized)),
        "counts": {
            "gold_values": gold_value_count,
            "leaked_gold_values": leaked_count,
            "aligned_leaked_gold_occurrences": aligned_leaked_count,
            "non_pii_characters": non_pii_total,
            "changed_non_pii_characters": changed_non_pii,
            "well_formed_outputs": well_formed_count,
        },
    }
