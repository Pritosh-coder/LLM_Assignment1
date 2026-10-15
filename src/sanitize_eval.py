"""Evaluate fine-tuned and non-fine-tuned sanitizers on a Part B JSONL file."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# The handout invokes this file directly rather than with ``python -m``.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.align import recover_spans_with_report
from src.common.baselines import (
    build_default_presidio_analyzer,
    presidio_sanitize,
    regex_sanitize,
)
from src.common.config import (
    PART_AB_REQUIRED_KEYS,
    SANITIZER_MODEL_ID,
    SANITIZER_MODEL_REVISION,
)
from src.common.data import iter_jsonl
from src.common.gen import GeneratedOutput, generate_with_vllm
from src.common.metrics import evaluate_sanitizer_records, leaked_gold_values
from src.common.labels import normalize_gold_label

SYSTEMS = ("finetuned", "regex", "presidio", "zeroshot")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a PII sanitization system.")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_path", type=Path)
    parser.add_argument("--test_file", type=Path, required=True)
    parser.add_argument("--system", choices=SYSTEMS, required=True)
    parser.add_argument("--output_predictions", type=Path, required=True)
    parser.add_argument("--report_file", type=Path, required=True)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="validate inputs and execution requirements without inference or writes",
    )
    return parser


def _validate_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"test_file does not exist: {path}")
    records = list(iter_jsonl(path))
    if not records:
        raise ValueError("test_file must contain at least one row")
    for row_number, record in enumerate(records, start=1):
        missing = sorted(PART_AB_REQUIRED_KEYS - set(record))
        if missing:
            raise ValueError(f"row {row_number}: missing keys {missing}")
        if not isinstance(record["source_text"], str):
            raise ValueError(f"row {row_number}: source_text must be a string")
        if not isinstance(record["privacy_mask"], list):
            raise ValueError(f"row {row_number}: privacy_mask must be a list")
    return records


def _load_adapter_rank(adapter_path: Path) -> int:
    config_path = adapter_path / "adapter_config.json"
    if not config_path.is_file():
        raise ValueError(f"adapter_config.json is missing from {adapter_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rank = config.get("r")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise ValueError(f"invalid LoRA rank in {config_path}")
    return rank


def validate_cli_contract(
    args: argparse.Namespace, records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if args.base_model != SANITIZER_MODEL_ID:
        raise ValueError(
            f"Part B model systems require {SANITIZER_MODEL_ID!r}, "
            f"received {args.base_model!r}"
        )
    if args.output_predictions.resolve() == args.test_file.resolve():
        raise ValueError("output_predictions must not overwrite test_file")
    if args.report_file.resolve() == args.test_file.resolve():
        raise ValueError("report_file must not overwrite test_file")
    if args.output_predictions.resolve() == args.report_file.resolve():
        raise ValueError("prediction and report paths must be distinct")

    adapter_rank: int | None = None
    if args.system == "finetuned":
        if args.adapter_path is None or not args.adapter_path.is_dir():
            raise ValueError("finetuned system requires an existing --adapter_path")
        adapter_rank = _load_adapter_rank(args.adapter_path)
    return {
        "system": args.system,
        "test_file": str(args.test_file),
        "rows": len(records),
        "requires_gpu": args.system in {"finetuned", "zeroshot"},
        "requires_presidio": args.system == "presidio",
        "adapter_rank": adapter_rank,
    }


def _run_regex(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for record in records:
        started = time.perf_counter()
        result = regex_sanitize(record["source_text"])
        latency_ms = (time.perf_counter() - started) * 1000.0
        outputs.append({**result, "latency_ms": latency_ms})
    return outputs


def _run_presidio(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    analyzer = build_default_presidio_analyzer()
    outputs: list[dict[str, Any]] = []
    for record in records:
        started = time.perf_counter()
        result = presidio_sanitize(record["source_text"], analyzer)
        latency_ms = (time.perf_counter() - started) * 1000.0
        outputs.append({**result, "latency_ms": latency_ms})
    return outputs


def _run_model(
    args: argparse.Namespace,
    records: Sequence[Mapping[str, Any]],
    adapter_rank: int | None,
) -> list[dict[str, Any]]:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:  # pragma: no cover - GPU environment dependency
        raise RuntimeError("model evaluation requires the pinned Transformers stack") from error

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        revision=SANITIZER_MODEL_REVISION,
        trust_remote_code=False,
    )
    generated: list[GeneratedOutput] = generate_with_vllm(
        records,
        tokenizer,
        base_model=args.base_model,
        adapter_path=args.adapter_path if args.system == "finetuned" else None,
        adapter_name=args.adapter_path.name
        if args.system == "finetuned"
        else "zeroshot",
        max_lora_rank=adapter_rank or 32,
    )
    outputs: list[dict[str, Any]] = []
    for record, output in zip(records, generated, strict=True):
        alignment = recover_spans_with_report(
            record["source_text"],
            output["text"],
            allow_unknown_labels=True,
        )
        outputs.append(
            {
                "predicted_text": output["text"],
                "predicted_spans": alignment["spans"],
                "well_formed": alignment["well_formed"],
                "issues": alignment["issues"],
                "latency_ms": output["latency_ms"],
            }
        )
    return outputs


def _public_gold_spans(spans: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    public: list[dict[str, Any]] = []
    for span in spans:
        public.append(
            {
                "start": span["start"],
                "end": span["end"],
                "label": span["label"],
            }
        )
    return public


def _build_prediction_rows(
    records: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    if len(records) != len(outputs):
        raise RuntimeError("system output count differs from test row count")
    rows: list[dict[str, Any]] = []
    issues: Counter[str] = Counter()
    for record, output in zip(records, outputs, strict=True):
        output_issues = output.get("issues", [])
        if not isinstance(output_issues, list):
            raise RuntimeError("system issues must be a list")
        for issue in output_issues:
            issues[str(issue.get("rule", "unknown"))] += 1
        predicted_spans = [
            {
                "start": span["start"],
                "end": span["end"],
                "label": span["label"],
                **({"rule": span["rule"]} if "rule" in span else {}),
                **({"score": span["score"]} if "score" in span else {}),
            }
            for span in output["predicted_spans"]
        ]
        gold_spans = _public_gold_spans(record["privacy_mask"])
        rows.append(
            {
                "id": record["id"],
                "language": record["language"],
                "source_text": record["source_text"],
                "predicted_text": output["predicted_text"],
                "predicted_spans": predicted_spans,
                "gold_spans": gold_spans,
                "leaked_values": leaked_gold_values(
                    record["source_text"],
                    output["predicted_text"],
                    record["privacy_mask"],
                ),
                "latency_ms": float(output["latency_ms"]),
                "well_formed": bool(output["well_formed"]),
                "alignment_issues": output_issues,
            }
        )
    return rows, issues


def _atomic_write_json(path: Path, value: Any, *, jsonl: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            if jsonl:
                for row in value:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                json.dump(value, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _latency_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    values = sorted(float(row["latency_ms"]) for row in rows)
    p95_index = max(0, math_ceil(0.95 * len(values)) - 1)
    return {
        "mean_ms": statistics.fmean(values),
        "p95_ms": values[p95_index],
    }


def math_ceil(value: float) -> int:
    """Small local integer ceiling without adding a heavyweight dependency."""
    integer = int(value)
    return integer if value == integer else integer + 1


def _top_10_labels_for_records(
    records: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Return the 10 most frequent normalized gold labels in this eval split.

    Part B requires the per-class table to follow the current test/OOD split,
    not Part A training frequencies. OOD aliases are normalized with the same
    policy used by the metric evaluator; genuinely novel labels remain valid.
    Ties are broken deterministically by label name.
    """
    frequencies: Counter[str] = Counter()
    for row_index, record in enumerate(records, start=1):
        masks = record.get("privacy_mask")
        if not isinstance(masks, list):
            raise ValueError(f"record {row_index}: privacy_mask must be a list")
        for mask_index, mask in enumerate(masks, start=1):
            if not isinstance(mask, Mapping):
                raise ValueError(
                    f"record {row_index}, mask {mask_index}: mask must be an object"
                )
            raw_label = mask.get("label")
            if not isinstance(raw_label, str):
                raise ValueError(
                    f"record {row_index}, mask {mask_index}: label must be a string"
                )
            label = normalize_gold_label(raw_label, allow_unknown=True)
            if label is None:
                raise ValueError(
                    f"record {row_index}, mask {mask_index}: malformed label {raw_label!r}"
                )
            frequencies[label] += 1
    return [
        label
        for label, _count in sorted(
            frequencies.items(), key=lambda item: (-item[1], item[0])
        )[:10]
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    records = _validate_records(args.test_file)
    plan = validate_cli_contract(args, records)
    if args.dry_run:
        return plan

    if args.system == "regex":
        outputs = _run_regex(records)
    elif args.system == "presidio":
        outputs = _run_presidio(records)
    else:
        outputs = _run_model(args, records, plan["adapter_rank"])

    prediction_rows, issue_counts = _build_prediction_rows(records, outputs)
    report = evaluate_sanitizer_records(
        prediction_rows,
        system=args.system,
        adapter=str(args.adapter_path) if args.system == "finetuned" else None,
        test_set=args.test_file.stem,
        top_labels=_top_10_labels_for_records(records),
    )
    report["latency"] = _latency_summary(prediction_rows)
    report["alignment_issue_counts"] = dict(sorted(issue_counts.items()))
    _atomic_write_json(args.output_predictions, prediction_rows, jsonl=True)
    _atomic_write_json(args.report_file, report, jsonl=False)
    return {
        "system": args.system,
        "test_set": args.test_file.stem,
        "n": len(records),
        "predictions": str(args.output_predictions),
        "report": str(args.report_file),
        "metrics": report,
    }


def main() -> None:
    args = build_parser().parse_args()
    try:
        from src.common.hub_artifacts import resolve_adapter
        if args.system == "finetuned":
            args.adapter_path = resolve_adapter(args.adapter_path or Path("adapters/best_sanitizer"))
        result = run(args)
    except (RuntimeError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
