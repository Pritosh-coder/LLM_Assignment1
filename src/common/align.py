"""Recover source-offset PII spans from generated sanitized text.

Recognized placeholders are replaced by unique private-use characters before running
``difflib.SequenceMatcher``. This prevents characters inside ``[EMAIL]`` or ``[DATE]``
from spuriously matching the source value. Every recovered span carries the rule that
created it; unresolved rewrites and ambiguous multi-placeholder splits are retained as
structured issues for the Part B error taxonomy.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Sequence
from typing import TypedDict

from src.common.labels import normalize_model_label

_PLACEHOLDER_RE = re.compile(r"\[([A-Za-z][A-Za-z0-9 _-]*)\]")
_PRIVATE_USE_START = 0xE000
_PRIVATE_USE_END = 0xF8FF


class RecoveredSpan(TypedDict):
    start: int
    end: int
    label: str
    rule: str
    output_start: int
    output_end: int
    raw_placeholder: str


class AlignmentIssue(TypedDict):
    rule: str
    source_start: int
    source_end: int
    output_start: int
    output_end: int
    detail: str


class AlignmentReport(TypedDict):
    spans: list[RecoveredSpan]
    issues: list[AlignmentIssue]
    similarity_ratio: float
    well_formed: bool


class _Placeholder(TypedDict):
    sentinel: str
    label: str
    raw_placeholder: str
    output_start: int
    output_end: int


def _next_sentinel(index: int, source_text: str, predicted_text: str) -> str:
    codepoint = _PRIVATE_USE_START + index
    while codepoint <= _PRIVATE_USE_END:
        candidate = chr(codepoint)
        if candidate not in source_text and candidate not in predicted_text:
            return candidate
        codepoint += 1
    raise ValueError("too many placeholders to allocate collision-free sentinels")


def _atomicize_prediction(
    source_text: str,
    predicted_text: str,
    *,
    allow_unknown_labels: bool = False,
) -> tuple[str, dict[str, _Placeholder], list[AlignmentIssue]]:
    pieces: list[str] = []
    placeholders: dict[str, _Placeholder] = {}
    issues: list[AlignmentIssue] = []
    cursor = 0
    sentinel_index = 0

    for match in _PLACEHOLDER_RE.finditer(predicted_text):
        raw_placeholder = match.group(0)
        label = normalize_model_label(match.group(1))
        if (
            label is None
            and allow_unknown_labels
            and raw_placeholder not in source_text
        ):
            label = normalize_model_label(match.group(1), allow_unknown=True)
        if label is None:
            # Leave unknown bracketed text literal. It may be legitimate source text;
            # report it only when it was newly generated rather than preserved.
            if raw_placeholder not in source_text:
                issues.append(
                    {
                        "rule": "unrecognized_bracketed_label",
                        "source_start": 0,
                        "source_end": 0,
                        "output_start": match.start(),
                        "output_end": match.end(),
                        "detail": f"not treated as a PII placeholder: {raw_placeholder}",
                    }
                )
            continue

        pieces.append(predicted_text[cursor : match.start()])
        sentinel = _next_sentinel(sentinel_index, source_text, predicted_text)
        sentinel_index += 1
        pieces.append(sentinel)
        placeholders[sentinel] = {
            "sentinel": sentinel,
            "label": label,
            "raw_placeholder": raw_placeholder,
            "output_start": match.start(),
            "output_end": match.end(),
        }
        cursor = match.end()

    pieces.append(predicted_text[cursor:])
    return "".join(pieces), placeholders, issues


def _even_boundaries(start: int, end: int, parts: int) -> list[tuple[int, int]]:
    """Deterministically divide a non-empty source range into ``parts`` ranges."""
    width = end - start
    if parts <= 0 or width < parts:
        return []
    quotient, remainder = divmod(width, parts)
    boundaries: list[tuple[int, int]] = []
    cursor = start
    for index in range(parts):
        part_width = quotient + (1 if index < remainder else 0)
        boundaries.append((cursor, cursor + part_width))
        cursor += part_width
    return boundaries


def _split_multi_placeholder_replace(
    source_text: str,
    skeleton: str,
    source_start: int,
    source_end: int,
    output_start: int,
    output_end: int,
    occurrences: Sequence[_Placeholder],
) -> tuple[list[tuple[int, int]], str]:
    """Split one replacement opcode containing multiple placeholder sentinels.

    Literal text between sentinels is used only when every separator has one unique,
    ordered occurrence in the source slice. Otherwise an even deterministic split is
    returned and marked ambiguous.
    """
    local_skeleton = skeleton[output_start:output_end]
    sentinel_positions = [
        local_skeleton.index(occurrence["sentinel"]) for occurrence in occurrences
    ]
    separators = [
        local_skeleton[left + 1 : right]
        for left, right in zip(sentinel_positions, sentinel_positions[1:])
    ]
    source_slice = source_text[source_start:source_end]

    if separators and all(separators):
        splits: list[tuple[int, int]] = []
        relative_cursor = 0
        unique = True
        for separator in separators:
            first = source_slice.find(separator, relative_cursor)
            last = source_slice.rfind(separator, relative_cursor)
            if first < 0 or first != last:
                unique = False
                break
            splits.append((relative_cursor, first))
            relative_cursor = first + len(separator)
        if unique:
            splits.append((relative_cursor, len(source_slice)))
            if len(splits) == len(occurrences) and all(a < b for a, b in splits):
                return [
                    (source_start + local_start, source_start + local_end)
                    for local_start, local_end in splits
                ], "multi_placeholder_unique_separator"

    return (
        _even_boundaries(source_start, source_end, len(occurrences)),
        "multi_placeholder_ambiguous_even_split",
    )


def recover_spans_with_report(
    source_text: str,
    predicted_text: str,
    *,
    allow_unknown_labels: bool = False,
) -> AlignmentReport:
    """Recover PII spans and return alignment diagnostics."""
    if not isinstance(source_text, str) or not isinstance(predicted_text, str):
        raise TypeError("source_text and predicted_text must be strings")

    skeleton, placeholder_map, issues = _atomicize_prediction(
        source_text,
        predicted_text,
        allow_unknown_labels=allow_unknown_labels,
    )
    matcher = difflib.SequenceMatcher(
        None, source_text, skeleton, autojunk=False
    )
    spans: list[RecoveredSpan] = []

    for tag, source_start, source_end, output_start, output_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        occurrences = [
            placeholder_map[character]
            for character in skeleton[output_start:output_end]
            if character in placeholder_map
        ]

        if tag == "replace" and len(occurrences) == 1 and source_start < source_end:
            occurrence = occurrences[0]
            spans.append(
                {
                    "start": source_start,
                    "end": source_end,
                    "label": occurrence["label"],
                    "rule": "atomic_placeholder_replace",
                    "output_start": occurrence["output_start"],
                    "output_end": occurrence["output_end"],
                    "raw_placeholder": occurrence["raw_placeholder"],
                }
            )
            continue

        if tag == "replace" and len(occurrences) > 1 and source_start < source_end:
            boundaries, rule = _split_multi_placeholder_replace(
                source_text,
                skeleton,
                source_start,
                source_end,
                output_start,
                output_end,
                occurrences,
            )
            if len(boundaries) == len(occurrences):
                for occurrence, (span_start, span_end) in zip(
                    occurrences, boundaries, strict=True
                ):
                    spans.append(
                        {
                            "start": span_start,
                            "end": span_end,
                            "label": occurrence["label"],
                            "rule": rule,
                            "output_start": occurrence["output_start"],
                            "output_end": occurrence["output_end"],
                            "raw_placeholder": occurrence["raw_placeholder"],
                        }
                    )
                if rule.endswith("even_split"):
                    issues.append(
                        {
                            "rule": rule,
                            "source_start": source_start,
                            "source_end": source_end,
                            "output_start": occurrences[0]["output_start"],
                            "output_end": occurrences[-1]["output_end"],
                            "detail": "placeholder boundaries are not identifiable from literal separators",
                        }
                    )
                continue

        if occurrences:
            issue_rule = (
                "inserted_placeholder_without_source"
                if tag == "insert"
                else "unresolved_placeholder_alignment"
            )
        elif tag == "delete":
            issue_rule = "deletion_without_placeholder"
        else:
            issue_rule = "rewrite_without_placeholder"
        issues.append(
            {
                "rule": issue_rule,
                "source_start": source_start,
                "source_end": source_end,
                "output_start": output_start,
                "output_end": output_end,
                "detail": f"SequenceMatcher opcode={tag}, placeholders={len(occurrences)}",
            }
        )

    spans.sort(key=lambda span: (span["start"], span["end"], span["output_start"]))
    non_overlapping = all(
        left["end"] <= right["start"] for left, right in zip(spans, spans[1:])
    )
    # Zero-placeholder outputs are structurally valid: some supplied rows contain no
    # PII, and recall/leakage metrics—not format validity—must penalize missed entities.
    well_formed = non_overlapping and not issues
    return {
        "spans": spans,
        "issues": issues,
        "similarity_ratio": matcher.ratio(),
        "well_formed": well_formed,
    }


def recover_spans(
    source_text: str,
    predicted_text: str,
    *,
    allow_unknown_labels: bool = False,
) -> list[RecoveredSpan]:
    """Return recovered source-offset spans for the Part B scorer."""
    return recover_spans_with_report(
        source_text,
        predicted_text,
        allow_unknown_labels=allow_unknown_labels,
    )["spans"]


def source_change_mask(source_text: str, predicted_text: str) -> list[bool]:
    """Mark source characters removed or replaced by the generated output.

    Recognized placeholders are atomized before alignment so letters inside a marker
    cannot be mistaken for preserved source characters. Insertions do not mark source
    characters because over-masking is defined over removed/replaced source content.
    """
    if not isinstance(source_text, str) or not isinstance(predicted_text, str):
        raise TypeError("source_text and predicted_text must be strings")
    skeleton, _, _ = _atomicize_prediction(
        source_text, predicted_text, allow_unknown_labels=True
    )
    changed = [True] * len(source_text)
    matcher = difflib.SequenceMatcher(None, source_text, skeleton, autojunk=False)
    for tag, source_start, source_end, _, _ in matcher.get_opcodes():
        if tag == "equal":
            changed[source_start:source_end] = [False] * (source_end - source_start)
    return changed
