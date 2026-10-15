"""Train and evaluate one required Part A LoRA/QLoRA configuration."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# The handout invokes this file directly rather than with ``python -m``.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.align import recover_spans_with_report
from src.common.config import (
    ALL_LORA_TARGET_MODULES,
    EXPECTED_DATASET_ROWS,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LR_SCHEDULER_TYPE,
    MAX_SEQ_LEN,
    NUM_EPOCHS,
    PART_A_RESULT_COLUMNS,
    PART_A_SWEEP,
    PER_DEVICE_BATCH_SIZE,
    SANITIZER_MODEL_ID,
    SANITIZER_MODEL_REVISION,
    VLLM_DETERMINISTIC_GENERATION,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from src.common.data import encode_training_example, iter_jsonl
from src.common.gen import generate_with_vllm_adapter
from src.common.labels import top_canonical_labels
from src.common.metrics import evaluate_sanitizer_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one full-data LoRA/QLoRA sanitizer experiment."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--train_file", type=Path, required=True)
    parser.add_argument("--val_file", type=Path, required=True)
    parser.add_argument("--target_modules", required=True)
    parser.add_argument("--lora_r", type=int, required=True)
    parser.add_argument("--lora_alpha", type=int, required=True)
    parser.add_argument("--lora_dropout", type=float, required=True)
    parser.add_argument("--quant", choices=("none", "4bit"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--results_csv", type=Path, required=True)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="validate the frozen configuration and full input files without a GPU",
    )
    return parser


def parse_target_modules(value: str) -> tuple[str, ...]:
    modules = tuple(part.strip() for part in value.split(","))
    if not modules or any(not module for module in modules):
        raise ValueError("target_modules must be a non-empty comma-separated list")
    if len(set(modules)) != len(modules):
        raise ValueError("target_modules contains duplicates")
    unknown = sorted(set(modules) - set(ALL_LORA_TARGET_MODULES))
    if unknown:
        raise ValueError(f"unsupported target modules: {', '.join(unknown)}")
    return modules


def validate_known_sweep_configuration(
    *,
    run_id: str,
    target_modules: Sequence[str],
    lora_r: int,
    lora_alpha: int,
    quant: str,
) -> None:
    """Reject accidental drift for the six handout-mandated run IDs."""
    expected = PART_A_SWEEP.get(run_id.upper())
    if expected is None:
        return
    actual = {
        "target_modules": tuple(target_modules),
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "quant": quant,
    }
    mismatches = [
        f"{key}={actual[key]!r} (expected {expected[key]!r})"
        for key in actual
        if actual[key] != expected[key]
    ]
    if mismatches:
        raise ValueError(
            f"{run_id.upper()} does not match the September 5 sweep: "
            + "; ".join(mismatches)
        )


def validate_cli_contract(args: argparse.Namespace) -> tuple[str, ...]:
    modules = parse_target_modules(args.target_modules)
    if args.model != SANITIZER_MODEL_ID:
        raise ValueError(
            f"Part A requires {SANITIZER_MODEL_ID!r}, received {args.model!r}"
        )
    if args.lora_r <= 0 or args.lora_alpha <= 0:
        raise ValueError("lora_r and lora_alpha must be positive")
    if not 0.0 <= args.lora_dropout < 1.0:
        raise ValueError("lora_dropout must be in [0, 1)")
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if not args.run_id.strip():
        raise ValueError("run_id must be non-empty")
    for description, path in (
        ("training file", args.train_file),
        ("validation file", args.val_file),
    ):
        if not path.is_file():
            raise ValueError(f"{description} does not exist: {path}")
    validate_known_sweep_configuration(
        run_id=args.run_id,
        target_modules=modules,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        quant=args.quant,
    )
    return modules


def load_full_part_a_inputs(
    train_file: Path, val_file: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_records = list(iter_jsonl(train_file))
    val_records = list(iter_jsonl(val_file))
    expected_train = EXPECTED_DATASET_ROWS["partA_train.jsonl"]
    expected_val = EXPECTED_DATASET_ROWS["partA_validation.jsonl"]
    if len(train_records) != expected_train:
        raise ValueError(
            f"Part A must use all {expected_train} training rows; "
            f"received {len(train_records)}"
        )
    if len(val_records) != expected_val:
        raise ValueError(
            f"Part A must use all {expected_val} validation rows; "
            f"received {len(val_records)}"
        )
    for split_name, expected_split, records in (
        ("training", "train", train_records),
        ("validation", "validation", val_records),
    ):
        bad = sum(record.get("split") != expected_split for record in records)
        if bad:
            raise ValueError(
                f"{split_name} file contains {bad} rows outside split={expected_split!r}"
            )
    return train_records, val_records


def append_sweep_result(path: Path, row: Mapping[str, Any]) -> None:
    """Append exactly one schema-checked row, refusing duplicate run IDs."""
    missing = [column for column in PART_A_RESULT_COLUMNS if column not in row]
    extra = sorted(set(row) - set(PART_A_RESULT_COLUMNS))
    if missing or extra:
        raise ValueError(f"invalid result row; missing={missing}, extra={extra}")

    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    if not write_header:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != PART_A_RESULT_COLUMNS:
                raise ValueError(f"existing CSV has an incompatible header: {path}")
            if any(existing["run_id"] == str(row["run_id"]) for existing in reader):
                raise ValueError(
                    f"results CSV already contains run_id={row['run_id']!r}"
                )

    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PART_A_RESULT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({column: row[column] for column in PART_A_RESULT_COLUMNS})


def gradient_accumulation_for_devices(device_count: int) -> int:
    """Preserve the frozen global batch size across visible data-parallel GPUs."""
    if device_count <= 0:
        raise ValueError("device_count must be positive")
    if GRADIENT_ACCUMULATION_STEPS % device_count:
        raise ValueError(
            "the frozen gradient accumulation is not divisible by "
            f"device_count={device_count}"
        )
    return GRADIENT_ACCUMULATION_STEPS // device_count


class _EncodedDataset:
    """Compact in-memory tokenized dataset; padding is deferred to each batch."""

    def __init__(self, records: Sequence[Mapping[str, Any]], tokenizer: Any) -> None:
        import numpy as np

        self.examples: list[dict[str, Any]] = []
        self.total_input_tokens = 0
        for record in records:
            encoded = encode_training_example(record, tokenizer)
            self.total_input_tokens += len(encoded["input_ids"])
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


class _CompletionOnlyCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        import torch

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
            labels[row, :length] = torch.as_tensor(feature["labels"], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def _ensure_fresh_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(
            f"output_dir is not empty; refusing to mix or overwrite a run: {path}"
        )
    path.mkdir(parents=True, exist_ok=True)


def _build_peft_model(args: argparse.Namespace, modules: Sequence[str]) -> Any:
    import torch
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    model_kwargs: dict[str, Any] = {
        "revision": SANITIZER_MODEL_REVISION,
        "torch_dtype": torch.float16,
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
        "trust_remote_code": False,
    }
    if args.quant == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_kwargs["device_map"] = {"": torch.cuda.current_device()}

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if args.quant == "4bit":
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    else:
        model.enable_input_require_grads()

    model.config.use_cache = False
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=list(modules),
        bias="none",
    )
    return get_peft_model(model, lora_config)


def _train_and_measure(
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    train_dataset: _EncodedDataset,
    val_dataset: _EncodedDataset,
) -> dict[str, float]:
    import torch
    from transformers import Trainer, TrainingArguments

    visible_gpu_count = torch.cuda.device_count()
    runtime_gradient_accumulation = gradient_accumulation_for_devices(
        visible_gpu_count
    )
    print(
        json.dumps(
            {
                "visible_gpu_count": visible_gpu_count,
                "per_device_batch_size": PER_DEVICE_BATCH_SIZE,
                "gradient_accumulation_steps": runtime_gradient_accumulation,
                "effective_batch_size": (
                    visible_gpu_count
                    * PER_DEVICE_BATCH_SIZE
                    * runtime_gradient_accumulation
                ),
            }
        )
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=runtime_gradient_accumulation,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        optim="adamw_torch",
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="no",
        # Saving happens once after the timed loop so checkpoint I/O does not
        # confound wall-clock/throughput comparisons across adapter sizes.
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=25,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        dataloader_drop_last=False,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=_CompletionOnlyCollator(tokenizer.pad_token_id),
    )

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    train_result = trainer.train()
    torch.cuda.synchronize()
    wallclock = time.perf_counter() - started
    peak_memory = torch.cuda.max_memory_allocated() / (1024**3)

    evaluation = trainer.evaluate(metric_key_prefix="val")
    val_loss = float(evaluation["val_loss"])
    final_train_loss = float(train_result.metrics["train_loss"])
    tokens_seen = train_dataset.total_input_tokens * NUM_EPOCHS

    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(args.output_dir)
    return {
        "peak_gpu_mem_gb": peak_memory,
        "train_wallclock_s": wallclock,
        "tokens_per_s": tokens_seen / wallclock,
        "final_train_loss": final_train_loss,
        "val_loss": val_loss,
    }


def _score_saved_adapter(
    args: argparse.Namespace,
    tokenizer: Any,
    val_records: Sequence[Mapping[str, Any]],
    top_labels: Sequence[str],
) -> dict[str, Any]:
    predictions = generate_with_vllm_adapter(
        val_records,
        tokenizer,
        base_model=args.model,
        adapter_path=args.output_dir,
        adapter_name=args.run_id,
        max_lora_rank=args.lora_r,
    )
    metric_records: list[dict[str, Any]] = []
    for record, prediction in zip(val_records, predictions, strict=True):
        source_text = record["source_text"]
        alignment = recover_spans_with_report(source_text, prediction)
        metric_records.append(
            {
                "source_text": source_text,
                "predicted_text": prediction,
                "predicted_spans": alignment["spans"],
                "gold_spans": record["privacy_mask"],
                "well_formed": alignment["well_formed"],
            }
        )
    report = evaluate_sanitizer_records(
        metric_records,
        system="finetuned",
        adapter=str(args.output_dir),
        test_set=args.val_file.stem,
        top_labels=top_labels,
    )
    report_path = args.output_dir / "validation_metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    modules = validate_cli_contract(args)
    train_records, val_records = load_full_part_a_inputs(
        args.train_file, args.val_file
    )
    method = "QLoRA" if args.quant == "4bit" else "LoRA"
    plan = {
        "run_id": args.run_id,
        "method": method,
        "target_modules": list(modules),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "quant": args.quant,
        "seed": args.seed,
        "train_rows": len(train_records),
        "validation_rows": len(val_records),
        "epochs": NUM_EPOCHS,
        "max_seq_len": MAX_SEQ_LEN,
        "max_new_tokens": VLLM_DETERMINISTIC_GENERATION["max_tokens"],
        "effective_batch_size": (
            PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
        ),
        "learning_rate": LEARNING_RATE,
        "lr_scheduler_type": LR_SCHEDULER_TYPE,
    }
    if args.dry_run:
        return plan

    try:
        import torch
        from transformers import AutoTokenizer, set_seed
    except ImportError as error:
        raise RuntimeError(
            "Install the pinned GPU requirements before running the sweep"
        ) from error
    try:
        import peft  # noqa: F401
    except ImportError as error:
        raise RuntimeError("Install peft before running the sweep") from error
    if not torch.cuda.is_available():
        raise RuntimeError("Part A training requires a CUDA GPU (T4/P100/V100)")
    if args.results_csv.exists():
        with args.results_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if any(row.get("run_id") == args.run_id for row in reader):
                raise ValueError(
                    f"results CSV already contains run_id={args.run_id!r}"
                )

    _ensure_fresh_output_dir(args.output_dir)
    set_seed(args.seed, deterministic=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=SANITIZER_MODEL_REVISION,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = _EncodedDataset(train_records, tokenizer)
    val_dataset = _EncodedDataset(val_records, tokenizer)
    model = _build_peft_model(args, modules)
    model.print_trainable_parameters()
    trainable_params, total_params = model.get_nb_trainable_parameters()
    if trainable_params <= 0 or total_params <= trainable_params:
        raise RuntimeError("invalid PEFT parameter counts")

    measurements = _train_and_measure(
        args, model, tokenizer, train_dataset, val_dataset
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()

    report = _score_saved_adapter(
        args,
        tokenizer,
        val_records,
        [label for label, _ in top_canonical_labels(train_records)],
    )
    val_entity_f1 = float(report["overlap_iou50"]["micro_f1"])
    val_leak_rate = float(report["leak_rate"])
    result_row = {
        "run_id": args.run_id,
        "method": method,
        "target_modules": ",".join(modules),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "quant": args.quant,
        "trainable_params": int(trainable_params),
        "total_params": int(total_params),
        "trainable_pct": 100.0 * trainable_params / total_params,
        **measurements,
        "val_entity_f1": val_entity_f1,
        "val_leak_rate": val_leak_rate,
    }
    if any(
        isinstance(value, float) and not math.isfinite(value)
        for value in result_row.values()
    ):
        raise RuntimeError("refusing to append non-finite experiment metrics")
    append_sweep_result(args.results_csv, result_row)
    return {"plan": plan, "result": result_row}


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = run(args)
    except (RuntimeError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
