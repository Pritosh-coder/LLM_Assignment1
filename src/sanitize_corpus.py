"""Create the metadata-identical sanitized corpus used to train V-san."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import (
    EXPECTED_DATASET_ROWS,
    SANITIZER_MODEL_ID,
    SANITIZER_MODEL_REVISION,
    VLLM_DETERMINISTIC_GENERATION,
)
from src.common.data import iter_jsonl
from src.common.gen import generate_with_vllm
from src.common.prompts import SYSTEM_PROMPT
from src.common.sanitize_corpus import (
    build_sanitized_records,
    validate_corpus_structure,
    validate_sanitized_lineage,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sanitize every Part C raw-corpus row without changing metadata."
    )
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_path", type=Path, required=True)
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--audit_file", type=Path, required=True)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_adapter_rank(path: Path) -> int:
    config_path = path / "adapter_config.json"
    if not config_path.is_file():
        raise ValueError(f"adapter_config.json is missing from {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rank = config.get("r")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise ValueError(f"invalid adapter rank in {config_path}")
    return rank


def _atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def validate_cli_contract(args: argparse.Namespace) -> tuple[list[dict[str, Any]], int]:
    if args.base_model != SANITIZER_MODEL_ID:
        raise ValueError(
            f"corpus sanitizer requires {SANITIZER_MODEL_ID!r}, "
            f"received {args.base_model!r}"
        )
    if not args.input_file.is_file():
        raise ValueError(f"input_file does not exist: {args.input_file}")
    if not args.adapter_path.is_dir():
        raise ValueError(f"adapter_path does not exist: {args.adapter_path}")
    resolved_input = args.input_file.resolve()
    if args.output_file.resolve() == resolved_input:
        raise ValueError("output_file must not overwrite the raw corpus")
    if args.audit_file.resolve() in {resolved_input, args.output_file.resolve()}:
        raise ValueError("audit_file must be distinct from input and output")

    records = list(iter_jsonl(args.input_file))
    structure = validate_corpus_structure(records)
    expected = EXPECTED_DATASET_ROWS["partC_corpus_raw.jsonl"]
    if structure["rows"] != expected:
        raise ValueError(
            f"V-san must derive from all {expected} raw rows; "
            f"received {structure['rows']}"
        )
    return records, _read_adapter_rank(args.adapter_path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    raw_records, adapter_rank = validate_cli_contract(args)
    plan = {
        "base_model": args.base_model,
        "adapter": str(args.adapter_path),
        "input_file": str(args.input_file),
        "output_file": str(args.output_file),
        "audit_file": str(args.audit_file),
        "rows": len(raw_records),
        "adapter_rank": adapter_rank,
        "deterministic_generation": VLLM_DETERMINISTIC_GENERATION,
    }
    if args.dry_run:
        return plan

    try:
        from transformers import AutoTokenizer
    except ImportError as error:  # pragma: no cover - GPU dependency
        raise RuntimeError("corpus sanitization requires Transformers and vLLM") from error
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        revision=SANITIZER_MODEL_REVISION,
        trust_remote_code=False,
    )
    inference_records = [
        {"source_text": record["text"]} for record in raw_records
    ]
    generated = generate_with_vllm(
        inference_records,
        tokenizer,
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        adapter_name=args.adapter_path.name,
        max_lora_rank=adapter_rank,
    )
    sanitized_records = build_sanitized_records(
        raw_records, [output["text"] for output in generated]
    )
    lineage = validate_sanitized_lineage(raw_records, sanitized_records)
    _atomic_write_jsonl(args.output_file, sanitized_records)

    persisted_records = list(iter_jsonl(args.output_file))
    persisted_lineage = validate_sanitized_lineage(raw_records, persisted_records)
    if persisted_lineage != lineage:
        raise RuntimeError("persisted corpus lineage differs from the in-memory audit")
    audit = {
        **plan,
        **lineage,
        "input_sha256": _file_sha256(args.input_file),
        "output_sha256": _file_sha256(args.output_file),
        "adapter_config_sha256": _file_sha256(
            args.adapter_path / "adapter_config.json"
        ),
        "system_prompt_sha256": hashlib.sha256(
            SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
    }
    _atomic_write_json(args.audit_file, audit)
    return audit


def main() -> None:
    args = build_parser().parse_args()
    try:
        from src.common.hub_artifacts import resolve_adapter
        args.adapter_path = resolve_adapter(args.adapter_path)
        result = run(args)
    except (RuntimeError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
