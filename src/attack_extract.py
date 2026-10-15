"""Run the four defensive Part C extraction evaluations on local victims."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.attacks import (
    PIIEntry,
    aggregate_canary_results,
    estimate_sampled_exposure,
    membership_score_row,
    normalize_for_match,
    parse_pii_index_rows,
    sample_canary_candidates,
    split_discoverable_tokens,
    summarize_discoverable,
    summarize_targeted,
    summarize_untargeted,
)
from src.common.config import (
    EXPECTED_DATASET_ROWS,
    PART_C_CANARY_REQUIRED_KEYS,
    PART_C_TARGETED_REQUIRED_KEYS,
    REFERENCE_MODEL_ID,
    VICTIM_EVAL_BATCH_SIZE,
    VICTIM_MAX_SEQ_LEN,
    VICTIM_MODEL_ID,
    VICTIM_MODEL_REVISION,
)
from src.common.data import iter_jsonl
from src.common.scoring import (
    evaluate_causal_lm,
    score_conditional_token_sequences,
    score_token_sequences,
    tokenize_scoring_texts,
)
from src.common.victim import (
    CausalLMCollator,
    EncodedVictimDataset,
    load_full_victim_inputs,
)

DISCOVERABLE_BATCH_SIZE = 32
UNTARGETED_GENERATION_BATCH_SIZE = 32
TARGETED_BATCH_SIZE = 16
LIKELIHOOD_BATCH_SIZE = 16
CANARY_LIKELIHOOD_BATCH_SIZE = 64
UNTARGETED_MAX_NEW_TOKENS = 128
TARGETED_MAX_NEW_TOKENS = 64
TARGETED_SAMPLES_PER_PROMPT = 10
CANARY_SAMPLE_SIZE = 10_000
CANARY_UNIVERSE_SIZE = 1_000_000


def _set_reproducible_evaluation_seed(
    seed: int, set_seed_fn: Any, torch_module: Any
) -> None:
    """Seed evaluation while allowing unsupported deterministic CUDA kernels.

    Sampling still uses an explicit seed.  Warn-only mode is required because
    Hugging Face top-p generation uses CUDA cumsum, which PyTorch cannot run
    under strict deterministic-algorithm enforcement on Kaggle T4 GPUs.
    """
    set_seed_fn(seed, deterministic=True)
    torch_module.use_deterministic_algorithms(True, warn_only=True)




def _configure_generation_tokenizer(tokenizer: Any) -> None:
    """Use the same left-padding convention as batched prefix generation.

    _generate_from_prefix_ids already left-pads tensors explicitly.  Keeping
    tokenizer.padding_side in sync avoids decoder-only generation warnings and
    prevents future callers from accidentally switching back to right padding.
    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run defensive extraction evaluations on assignment victims."
    )
    parser.add_argument("--victim_model", required=True)
    parser.add_argument("--adapter_path", type=Path)
    parser.add_argument("--ref_model", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--canaries", type=Path, required=True)
    parser.add_argument("--pii_index", type=Path, required=True)
    parser.add_argument("--targeted_file", type=Path, required=True)
    parser.add_argument(
        "--attack",
        choices=("discoverable", "untargeted", "canary", "targeted", "all"),
        required=True,
    )
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--report_file", type=Path, required=True)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="validate protocols, complete inputs, and adapter metadata without CUDA",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly replace an existing report file",
    )
    return parser


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _read_adapter_config(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    config_path = path / "adapter_config.json"
    if not path.is_dir() or not config_path.is_file():
        raise ValueError(f"invalid victim adapter directory: {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_model = config.get("base_model_name_or_path")
    if base_model not in {VICTIM_MODEL_ID, None}:
        raise ValueError(
            f"adapter targets {base_model!r}, expected the local Part C victim"
        )
    rank = config.get("r")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise ValueError(f"invalid LoRA rank in {config_path}")
    return {
        "rank": rank,
        "base_model_name_or_path": base_model,
        "config_sha256": _file_sha256(config_path),
    }


def _load_canaries(
    path: Path,
    corpus_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = list(iter_jsonl(path))
    expected = EXPECTED_DATASET_ROWS["partC_canaries.jsonl"]
    if len(rows) != expected:
        raise ValueError(f"expected {expected} canaries, received {len(rows)}")
    ids: set[str] = set()
    repetitions: Counter[int] = Counter()
    corpus_counts = Counter(
        str(record["canary_id"])
        for record in corpus_records
        if record["is_canary"]
    )
    for row_number, row in enumerate(rows, start=1):
        missing = sorted(PART_C_CANARY_REQUIRED_KEYS - set(row))
        if missing:
            raise ValueError(f"canary row {row_number}: missing keys {missing}")
        canary_id = row["canary_id"]
        if not isinstance(canary_id, str) or not canary_id or canary_id in ids:
            raise ValueError(f"canary row {row_number}: invalid/duplicate canary_id")
        ids.add(canary_id)
        secret = row["secret"]
        template = row["template"]
        project = row["project"]
        n_reps = row["n_reps"]
        seed = row["seed"]
        if not isinstance(secret, str) or len(secret) != 6 or not secret.isdigit():
            raise ValueError(f"canary row {row_number}: secret is not six digits")
        if (
            not isinstance(template, str)
            or "{project}" not in template
            or "{secret}" not in template
        ):
            raise ValueError(
                f"canary row {row_number}: template lacks required fields"
            )
        if not isinstance(project, str) or not project:
            raise ValueError(f"canary row {row_number}: invalid project")
        if n_reps not in {1, 4, 16, 64} or not isinstance(seed, int):
            raise ValueError(f"canary row {row_number}: invalid repetition/seed")
        if corpus_counts[canary_id] != n_reps:
            raise ValueError(
                f"canary {canary_id!r}: corpus repetitions do not match metadata"
            )
        repetitions[n_reps] += 1
    if repetitions != Counter({1: 5, 4: 5, 16: 5, 64: 5}):
        raise ValueError(f"unexpected canary repetition distribution: {repetitions}")
    return rows


def _load_targeted(
    path: Path,
    corpus_records: Sequence[Mapping[str, Any]],
    heldout_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = list(iter_jsonl(path))
    expected = EXPECTED_DATASET_ROWS["partC_targeted.jsonl"]
    if len(rows) != expected:
        raise ValueError(f"expected {expected} targeted rows, received {len(rows)}")
    train_ids = {str(record["doc_id"]) for record in corpus_records}
    heldout_ids = {str(record["doc_id"]) for record in heldout_records}
    ids: set[str] = set()
    split_counts: Counter[str] = Counter()
    for row_number, row in enumerate(rows, start=1):
        missing = sorted(PART_C_TARGETED_REQUIRED_KEYS - set(row))
        if missing:
            raise ValueError(f"targeted row {row_number}: missing keys {missing}")
        row_id = row["id"]
        split = row["split"]
        doc_id = row["doc_id"]
        if not isinstance(row_id, str) or not row_id or row_id in ids:
            raise ValueError(f"targeted row {row_number}: invalid/duplicate id")
        ids.add(row_id)
        if split not in {"member", "control"}:
            raise ValueError(f"targeted row {row_number}: invalid split")
        if not isinstance(doc_id, str):
            raise ValueError(f"targeted row {row_number}: invalid doc_id")
        expected_ids = train_ids if split == "member" else heldout_ids
        if doc_id not in expected_ids:
            raise ValueError(
                f"targeted row {row_number}: doc_id is outside its declared split"
            )
        for key in ("prompt", "gold_value", "label"):
            if not isinstance(row[key], str) or not row[key]:
                raise ValueError(f"targeted row {row_number}: invalid {key}")
        split_counts[split] += 1
    if split_counts != Counter({"member": 200, "control": 200}):
        raise ValueError(f"unexpected targeted split counts: {split_counts}")
    return rows


def _load_pii_index(
    path: Path,
    corpus_records: Sequence[Mapping[str, Any]],
) -> list[PIIEntry]:
    if not path.is_file():
        raise ValueError(
            "untargeted extraction requires partC_pii_index.jsonl, which is "
            f"not present at {path}; obtain the instructor-specified file/schema"
        )
    entries = parse_pii_index_rows(iter_jsonl(path))
    corpus_ids = {str(record["doc_id"]) for record in corpus_records}
    unknown_ids = {entry.doc_id for entry in entries} - corpus_ids
    if unknown_ids:
        raise ValueError(
            f"PII index contains {len(unknown_ids)} document IDs outside the corpus"
        )
    return entries


def validate_cli_contract(
    args: argparse.Namespace,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]] | None,
    list[PIIEntry] | None,
    list[dict[str, Any]] | None,
    dict[str, Any] | None,
    dict[str, int],
]:
    if args.victim_model != VICTIM_MODEL_ID:
        raise ValueError(
            f"victim_model must be the assignment base model {VICTIM_MODEL_ID!r}"
        )
    if args.ref_model != REFERENCE_MODEL_ID:
        raise ValueError(
            f"ref_model must be the untouched base model {REFERENCE_MODEL_ID!r}"
        )
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if args.num_samples != 5_000 and args.attack in {"untargeted", "all"}:
        raise ValueError("the required untargeted evaluation uses num_samples=5000")
    for description, path in (
        ("corpus", args.corpus),
        ("held-out", args.heldout),
        ("canary metadata", args.canaries),
        ("targeted file", args.targeted_file),
    ):
        if not path.is_file():
            raise ValueError(f"{description} file does not exist: {path}")
    input_paths = {
        args.corpus.resolve(),
        args.heldout.resolve(),
        args.canaries.resolve(),
        args.targeted_file.resolve(),
    }
    if args.pii_index.exists():
        input_paths.add(args.pii_index.resolve())
    if args.report_file.resolve() in input_paths:
        raise ValueError("report_file must not overwrite an input")
    if args.report_file.exists() and not args.overwrite:
        raise ValueError(
            f"report_file already exists; pass --overwrite explicitly: {args.report_file}"
        )

    corpus, heldout, input_summary = load_full_victim_inputs(
        args.corpus, args.heldout
    )
    adapter = _read_adapter_config(args.adapter_path)
    canaries = (
        _load_canaries(args.canaries, corpus)
        if args.attack in {"canary", "all"}
        else None
    )
    targeted = (
        _load_targeted(args.targeted_file, corpus, heldout)
        if args.attack in {"targeted", "all"}
        else None
    )
    pii_entries = (
        _load_pii_index(args.pii_index, corpus)
        if args.attack in {"untargeted", "all"}
        else None
    )
    return (
        corpus,
        heldout,
        canaries,
        pii_entries,
        targeted,
        adapter,
        input_summary,
    )


def _load_model(model_id: str, adapter_path: Path | None) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=VICTIM_MODEL_REVISION,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).to(torch.device("cuda"))
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            str(adapter_path),
            is_trainable=False,
        )
    model.config.use_cache = True
    model.eval()
    return model


def _generate_from_prefix_ids(
    model: Any,
    tokenizer: Any,
    prefixes: Sequence[Sequence[int]],
    *,
    max_new_tokens: int,
    min_new_tokens: int = 0,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    num_return_sequences: int = 1,
) -> list[list[list[int]]]:
    import torch
    from transformers import GenerationConfig

    if not prefixes:
        return []
    max_length = max(len(prefix) for prefix in prefixes)
    input_ids = torch.full(
        (len(prefixes), max_length),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=torch.device("cuda"),
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, prefix in enumerate(prefixes):
        if not prefix:
            raise ValueError("generation prefix must contain at least one token")
        length = len(prefix)
        input_ids[row, -length:] = torch.tensor(
            prefix, dtype=torch.long, device=input_ids.device
        )
        attention_mask[row, -length:] = 1
    generation_config = GenerationConfig(
        do_sample=do_sample,
        num_beams=1,
        num_beam_groups=1,
        num_return_sequences=num_return_sequences,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        temperature=temperature if do_sample else 1.0,
        top_p=top_p if do_sample else 1.0,
        top_k=0,
        typical_p=1.0,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        renormalize_logits=False,
        remove_invalid_values=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        use_cache=True,
    )
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )
    continuations = output_ids[:, max_length:].tolist()
    grouped: list[list[list[int]]] = []
    for row in range(len(prefixes)):
        start = row * num_return_sequences
        grouped.append(
            continuations[start : start + num_return_sequences]
        )
    return grouped


def _run_discoverable(
    model: Any,
    tokenizer: Any,
    corpus_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    unique_records: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for record in corpus_records:
        doc_id = str(record["doc_id"])
        if record["is_canary"] or doc_id in seen:
            continue
        seen.add(doc_id)
        unique_records.append(record)
    if len(unique_records) != 5_000:
        raise ValueError(
            f"discoverability requires 5,000 non-canary documents, found {len(unique_records)}"
        )

    prepared: list[tuple[Mapping[str, Any], list[int], list[int]]] = []
    excluded_short = 0
    for record in unique_records:
        token_ids = tokenizer.encode(record["text"], add_special_tokens=False)
        if len(token_ids) < 2:
            excluded_short += 1
            continue
        prefix, suffix = split_discoverable_tokens(token_ids)
        prepared.append((record, prefix, suffix))

    outcomes: list[dict[str, Any]] = []
    for start in range(0, len(prepared), DISCOVERABLE_BATCH_SIZE):
        chunk = prepared[start : start + DISCOVERABLE_BATCH_SIZE]
        generated_groups = _generate_from_prefix_ids(
            model,
            tokenizer,
            [prefix for _, prefix, _ in chunk],
            max_new_tokens=max(len(suffix) for _, _, suffix in chunk),
        )
        for (record, _, suffix), generated_group in zip(
            chunk, generated_groups, strict=True
        ):
            generated = generated_group[0][: len(suffix)]
            outcomes.append(
                {
                    "doc_id": record["doc_id"],
                    "n_reps": record["n_reps"],
                    "exact": generated == suffix,
                    "expected_text": tokenizer.decode(
                        suffix, skip_special_tokens=True
                    ),
                    "generated_text": tokenizer.decode(
                        generated, skip_special_tokens=True
                    ),
                }
            )
    report = summarize_discoverable(outcomes)
    report["n_excluded_short"] = excluded_short
    report["generation"] = {
        "do_sample": False,
        "num_beams": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "typical_p": 1.0,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
    }
    return report


def _decoded_continuation(tokenizer: Any, token_ids: Sequence[int]) -> str:
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    if text:
        return text
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def _run_untargeted(
    model: Any,
    reference_model: Any,
    tokenizer: Any,
    corpus_records: Sequence[Mapping[str, Any]],
    pii_entries: Sequence[PIIEntry],
    *,
    num_samples: int,
) -> dict[str, Any]:
    generated_ids: list[list[int]] = []
    unconditional_prefix = [tokenizer.eos_token_id]
    for start in range(0, num_samples, UNTARGETED_GENERATION_BATCH_SIZE):
        count = min(UNTARGETED_GENERATION_BATCH_SIZE, num_samples - start)
        groups = _generate_from_prefix_ids(
            model,
            tokenizer,
            [unconditional_prefix] * count,
            max_new_tokens=UNTARGETED_MAX_NEW_TOKENS,
            min_new_tokens=16,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
        )
        generated_ids.extend(group[0] for group in groups)
    all_generated_texts = [
        _decoded_continuation(tokenizer, token_ids)
        for token_ids in generated_ids
    ]
    unique_texts: list[str] = []
    unique_ids: list[list[int]] = []
    seen: set[str] = set()
    for text, token_ids in zip(all_generated_texts, generated_ids, strict=True):
        normalized = normalize_for_match(text)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_texts.append(text)
        unique_ids.append(token_ids)
    if len(unique_texts) < 100:
        raise RuntimeError(
            f"only {len(unique_texts)} unique generations; precision@100 is undefined"
        )
    sequences = [
        [tokenizer.eos_token_id, *token_ids] for token_ids in unique_ids
    ]
    victim_losses = score_token_sequences(
        model,
        sequences,
        pad_token_id=tokenizer.pad_token_id,
        device="cuda",
        batch_size=LIKELIHOOD_BATCH_SIZE,
    )
    if reference_model is model:
        reference_losses = list(victim_losses)
    else:
        reference_losses = score_token_sequences(
            reference_model,
            sequences,
            pad_token_id=tokenizer.pad_token_id,
            device="cuda",
            batch_size=LIKELIHOOD_BATCH_SIZE,
        )
    lowercase_sequences = tokenize_scoring_texts(
        [text.lower() for text in unique_texts],
        tokenizer,
        max_seq_len=VICTIM_MAX_SEQ_LEN,
    )
    lowercase_losses = score_token_sequences(
        model,
        lowercase_sequences,
        pad_token_id=tokenizer.pad_token_id,
        device="cuda",
        batch_size=LIKELIHOOD_BATCH_SIZE,
    )
    score_rows = [
        membership_score_row(victim, reference, lowercase, text)
        for victim, reference, lowercase, text in zip(
            victim_losses,
            reference_losses,
            lowercase_losses,
            unique_texts,
            strict=True,
        )
    ]
    report = summarize_untargeted(
        unique_texts,
        score_rows,
        corpus_records,
        pii_entries,
        k=100,
        n_generated=len(all_generated_texts),
    )
    report["generation"] = {
        "prompt": "EOS only",
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 0,
        "typical_p": 1.0,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "min_new_tokens": 16,
        "max_new_tokens": UNTARGETED_MAX_NEW_TOKENS,
    }
    report["score_definitions"] = {
        "ppl": "exp(victim mean token NLL)",
        "ratio_ref": "exp(victim NLL - pretrained-reference NLL)",
        "ratio_lower": "exp(victim NLL - lowercase victim NLL)",
        "ratio_zlib": "victim NLL / (8 * compressed_bytes / raw_bytes)",
    }
    return report


def _run_canary(
    model: Any,
    tokenizer: Any,
    canary_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for row in canary_rows:
        candidates = sample_canary_candidates(
            row["secret"],
            row["seed"],
            sample_size=CANARY_SAMPLE_SIZE,
            universe_size=CANARY_UNIVERSE_SIZE,
        )
        prefix_template, suffix_template = row["template"].split("{secret}", 1)
        prefix_text = prefix_template.format(project=row["project"])
        trailing_whitespace = prefix_text[len(prefix_text.rstrip()) :]
        prefix_text = prefix_text.rstrip()
        if "{" in suffix_template or "}" in suffix_template:
            raise ValueError("unsupported format field after canary secret")
        prefix_ids = [
            tokenizer.eos_token_id,
            *tokenizer.encode(prefix_text, add_special_tokens=False),
        ]
        suffix_ids = [
            tokenizer.encode(
                trailing_whitespace + secret,
                add_special_tokens=False,
            )
            for secret in candidates
        ]
        losses = score_conditional_token_sequences(
            model,
            [prefix_ids] * len(suffix_ids),
            suffix_ids,
            pad_token_id=tokenizer.pad_token_id,
            device="cuda",
            batch_size=CANARY_LIKELIHOOD_BATCH_SIZE,
        )
        exposure = estimate_sampled_exposure(
            losses,
            true_index=0,
            universe_size=CANARY_UNIVERSE_SIZE,
        )
        results.append(
            {
                "canary_id": row["canary_id"],
                "n_reps": row["n_reps"],
                "scored_tokens": "candidate secret conditioned on template prefix",
                **exposure,
            }
        )
    return aggregate_canary_results(results)


def _run_targeted(
    model: Any,
    tokenizer: Any,
    targeted_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    outcomes: list[dict[str, Any]] = []
    for start in range(0, len(targeted_rows), TARGETED_BATCH_SIZE):
        chunk = targeted_rows[start : start + TARGETED_BATCH_SIZE]
        prefix_ids = [
            tokenizer.encode(row["prompt"], add_special_tokens=False)
            for row in chunk
        ]
        greedy = _generate_from_prefix_ids(
            model,
            tokenizer,
            prefix_ids,
            max_new_tokens=TARGETED_MAX_NEW_TOKENS,
        )
        sampled = _generate_from_prefix_ids(
            model,
            tokenizer,
            prefix_ids,
            max_new_tokens=TARGETED_MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            num_return_sequences=TARGETED_SAMPLES_PER_PROMPT,
        )
        for row, greedy_group, sampled_group in zip(
            chunk, greedy, sampled, strict=True
        ):
            outcomes.append(
                {
                    "id": row["id"],
                    "split": row["split"],
                    "gold_value": row["gold_value"],
                    "greedy_text": _decoded_continuation(
                        tokenizer, greedy_group[0]
                    ),
                    "sampled_texts": [
                        _decoded_continuation(tokenizer, token_ids)
                        for token_ids in sampled_group
                    ],
                }
            )
    report = summarize_targeted(outcomes)
    report["generation"] = {
        "greedy": {
            "do_sample": False,
            "num_beams": 1,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "typical_p": 1.0,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "max_new_tokens": TARGETED_MAX_NEW_TOKENS,
        },
        "sampled": {
            "n": TARGETED_SAMPLES_PER_PROMPT,
            "do_sample": True,
            "num_beams": 1,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 0,
            "typical_p": 1.0,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "max_new_tokens": TARGETED_MAX_NEW_TOKENS,
        },
    }
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    (
        corpus,
        heldout,
        canaries,
        pii_entries,
        targeted,
        adapter,
        input_summary,
    ) = validate_cli_contract(args)
    plan: dict[str, Any] = {
        "victim": args.adapter_path.name if args.adapter_path else "pretrained_control",
        "victim_model": args.victim_model,
        "victim_model_revision": VICTIM_MODEL_REVISION,
        "adapter_path": str(args.adapter_path) if args.adapter_path else None,
        "adapter": adapter,
        "ref_model": args.ref_model,
        "corpus": str(args.corpus),
        "corpus_sha256": _file_sha256(args.corpus),
        "heldout": str(args.heldout),
        "heldout_sha256": _file_sha256(args.heldout),
        "attack": args.attack,
        "num_samples": args.num_samples,
        "seed": args.seed,
        **input_summary,
    }
    if canaries is not None:
        plan["canary_rows"] = len(canaries)
    if pii_entries is not None:
        plan["pii_index_entries"] = len(pii_entries)
        plan["pii_index_sha256"] = _file_sha256(args.pii_index)
    if targeted is not None:
        plan["targeted_rows"] = len(targeted)
    if args.dry_run:
        return plan

    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoTokenizer, set_seed
    except ImportError as error:  # pragma: no cover - GPU-only dependency
        raise RuntimeError(
            "Install the pinned CUDA requirements before extraction evaluation"
        ) from error
    if args.adapter_path is not None:
        try:
            import peft  # noqa: F401
        except ImportError as error:  # pragma: no cover - GPU-only dependency
            raise RuntimeError("Install peft before loading a victim adapter") from error
    if not torch.cuda.is_available():
        raise RuntimeError("Part C extraction evaluation requires a CUDA GPU")

    _set_reproducible_evaluation_seed(args.seed, set_seed, torch)
    tokenizer = AutoTokenizer.from_pretrained(
        args.victim_model,
        revision=VICTIM_MODEL_REVISION,
        trust_remote_code=False,
    )
    _configure_generation_tokenizer(tokenizer)
    model = _load_model(args.victim_model, args.adapter_path)

    heldout_dataset = EncodedVictimDataset(heldout, tokenizer)
    heldout_loader = DataLoader(
        heldout_dataset,
        batch_size=VICTIM_EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=CausalLMCollator(tokenizer.pad_token_id),
        pin_memory=True,
    )
    utility = evaluate_causal_lm(model, heldout_loader, "cuda")
    report: dict[str, Any] = {
        **plan,
        "heldout_mean_nll": utility["mean_nll"],
        "heldout_ppl": utility["perplexity"],
        "heldout_predicted_tokens": utility["predicted_tokens"],
    }

    if args.attack in {"discoverable", "all"}:
        report["discoverable"] = _run_discoverable(model, tokenizer, corpus)
    if args.attack in {"untargeted", "all"}:
        if pii_entries is None:
            raise RuntimeError("validated PII index unexpectedly unavailable")
        _set_reproducible_evaluation_seed(args.seed + 101, set_seed, torch)
        reference_model = (
            model
            if args.adapter_path is None
            else _load_model(args.ref_model, None)
        )
        report["untargeted"] = _run_untargeted(
            model,
            reference_model,
            tokenizer,
            corpus,
            pii_entries,
            num_samples=args.num_samples,
        )
        if reference_model is not model:
            del reference_model
            gc.collect()
            torch.cuda.empty_cache()
    if args.attack in {"canary", "all"}:
        if canaries is None:
            raise RuntimeError("validated canary metadata unexpectedly unavailable")
        report["canary"] = _run_canary(model, tokenizer, canaries)
    if args.attack in {"targeted", "all"}:
        if targeted is None:
            raise RuntimeError("validated targeted rows unexpectedly unavailable")
        _set_reproducible_evaluation_seed(args.seed + 202, set_seed, torch)
        report["targeted"] = _run_targeted(model, tokenizer, targeted)

    _atomic_write_json(args.report_file, report)
    return report


def main() -> None:
    args = build_parser().parse_args()
    try:
        from src.common.hub_artifacts import resolve_adapter
        args.adapter_path = resolve_adapter(args.adapter_path)
        result = run(args)
    except (RuntimeError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
