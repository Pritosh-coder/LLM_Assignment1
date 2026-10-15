"""Train one Part C base causal-LM victim and measure held-out utility."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import (
    ALL_LORA_TARGET_MODULES,
    LR_SCHEDULER_TYPE,
    VICTIM_EPOCHS,
    VICTIM_EVAL_BATCH_SIZE,
    VICTIM_GRADIENT_ACCUMULATION_STEPS,
    VICTIM_LEARNING_RATE,
    VICTIM_LORA_DROPOUT,
    VICTIM_MAX_SEQ_LEN,
    VICTIM_MODEL_ID,
    VICTIM_MODEL_REVISION,
    VICTIM_PER_DEVICE_BATCH_SIZE,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from src.common.scoring import evaluate_causal_lm
from src.common.victim import (
    CausalLMCollator,
    EncodedVictimDataset,
    load_full_victim_inputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one base causal-LM victim adapter."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--target_modules", required=True)
    parser.add_argument("--lora_r", type=int, required=True)
    parser.add_argument("--lora_alpha", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--resume_if_available",
        action="store_true",
        help=(
            "resume from the newest validated Trainer checkpoint in output_dir; "
            "the saved victim_plan.json must exactly match the requested run"
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="validate the full corpus and frozen configuration without a GPU",
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
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _ensure_output_dir_contract(path: Path, *, resume_if_available: bool) -> None:
    if path.exists() and not path.is_dir():
        raise ValueError(f"output_dir exists but is not a directory: {path}")
    if path.exists() and any(path.iterdir()) and not resume_if_available:
        raise ValueError(
            f"output_dir is not empty; refusing to mix or overwrite a run: {path}"
        )


def _latest_checkpoint(path: Path) -> Path | None:
    if not path.is_dir():
        return None
    checkpoints: list[tuple[int, Path]] = []
    for candidate in path.glob("checkpoint-*"):
        if not candidate.is_dir():
            continue
        try:
            step = int(candidate.name.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        if not (candidate / "trainer_state.json").is_file():
            continue
        checkpoints.append((step, candidate))
    return max(checkpoints, default=(0, None), key=lambda item: item[0])[1]


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_cli_contract(args: argparse.Namespace) -> tuple[str, ...]:
    modules = parse_target_modules(args.target_modules)
    if args.model != VICTIM_MODEL_ID:
        raise ValueError(
            f"Part C requires the base model {VICTIM_MODEL_ID!r}, "
            f"received {args.model!r}"
        )
    if args.lora_r <= 0 or args.lora_alpha <= 0:
        raise ValueError("lora_r and lora_alpha must be positive")
    if args.epochs != VICTIM_EPOCHS:
        raise ValueError(
            f"the frozen raw/sanitized comparison requires epochs={VICTIM_EPOCHS}"
        )
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    for description, path in (
        ("corpus", args.corpus),
        ("held-out file", args.heldout),
    ):
        if not path.is_file():
            raise ValueError(f"{description} does not exist: {path}")
    if args.corpus.resolve() == args.heldout.resolve():
        raise ValueError("corpus and held-out file must be distinct")
    _ensure_output_dir_contract(
        args.output_dir, resume_if_available=getattr(args, "resume_if_available", False)
    )
    return modules


def _build_plan(
    args: argparse.Namespace,
    modules: Sequence[str],
    input_summary: dict[str, int],
) -> dict[str, Any]:
    return {
        "model": args.model,
        "model_revision": VICTIM_MODEL_REVISION,
        "corpus": str(args.corpus),
        "corpus_sha256": _file_sha256(args.corpus),
        "heldout": str(args.heldout),
        "heldout_sha256": _file_sha256(args.heldout),
        "output_dir": str(args.output_dir),
        "target_modules": list(modules),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": VICTIM_LORA_DROPOUT,
        "epochs": args.epochs,
        "seed": args.seed,
        "max_seq_len": VICTIM_MAX_SEQ_LEN,
        "learning_rate": VICTIM_LEARNING_RATE,
        "lr_scheduler_type": LR_SCHEDULER_TYPE,
        "effective_batch_size": (
            VICTIM_PER_DEVICE_BATCH_SIZE
            * VICTIM_GRADIENT_ACCUMULATION_STEPS
        ),
        **input_summary,
    }


def _load_base_model(model_id: str) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=VICTIM_MODEL_REVISION,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    return model.to(torch.device("cuda"))


def _attach_lora(model: Any, args: argparse.Namespace, modules: Sequence[str]) -> Any:
    from peft import LoraConfig, TaskType, get_peft_model

    model.enable_input_require_grads()
    model.config.use_cache = False
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=VICTIM_LORA_DROPOUT,
        target_modules=list(modules),
        bias="none",
    )
    return get_peft_model(model, config)


def _train(
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    train_dataset: EncodedVictimDataset,
    *,
    resume_checkpoint: Path | None,
) -> dict[str, float]:
    import torch
    from transformers import Trainer, TrainingArguments

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=VICTIM_PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=VICTIM_GRADIENT_ACCUMULATION_STEPS,
        learning_rate=VICTIM_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        optim="adamw_torch",
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="no",
        save_strategy="steps",
        save_steps=250,
        save_total_limit=2,
        logging_strategy="steps",
        logging_steps=25,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        full_determinism=True,
        dataloader_drop_last=False,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=CausalLMCollator(tokenizer.pad_token_id),
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = trainer.train(
        resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None
    )
    torch.cuda.synchronize()
    wallclock = time.perf_counter() - started
    peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(args.output_dir)
    return {
        "train_loss": float(result.metrics["train_loss"]),
        "train_wallclock_s": wallclock,
        "peak_gpu_mem_gb": peak_memory,
        "tokens_per_s": (
            train_dataset.total_input_tokens * args.epochs / wallclock
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    modules = validate_cli_contract(args)
    corpus, heldout, input_summary = load_full_victim_inputs(
        args.corpus, args.heldout
    )
    plan = _build_plan(args, modules, input_summary)
    if args.dry_run:
        return plan

    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoTokenizer, set_seed
    except ImportError as error:  # pragma: no cover - GPU dependency
        raise RuntimeError(
            "Install the pinned GPU requirements before training a victim"
        ) from error
    try:
        import peft  # noqa: F401
    except ImportError as error:  # pragma: no cover - GPU dependency
        raise RuntimeError("Install peft before training a victim") from error
    if not torch.cuda.is_available():
        raise RuntimeError("Part C victim training requires a CUDA GPU")

    plan_path = args.output_dir / "victim_plan.json"
    resume_if_available = getattr(args, "resume_if_available", False)
    if args.output_dir.exists():
        if not resume_if_available:
            raise RuntimeError("non-empty victim output requires --resume_if_available")
        if not plan_path.is_file():
            raise RuntimeError(
                f"cannot safely resume without a saved run plan: {plan_path}"
            )
        saved_plan = _load_json_object(plan_path)
        if saved_plan != plan:
            raise RuntimeError(
                "saved victim_plan.json does not match the requested corpus/configuration"
            )
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        _atomic_write_json(plan_path, plan)

    resume_checkpoint = (
        _latest_checkpoint(args.output_dir) if resume_if_available else None
    )
    if resume_if_available and resume_checkpoint is None and any(
        path.name != "victim_plan.json" for path in args.output_dir.iterdir()
    ):
        raise RuntimeError(
            "output_dir contains run artifacts but no valid Trainer checkpoint to resume"
        )

    set_seed(args.seed, deterministic=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=VICTIM_MODEL_REVISION,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_dataset = EncodedVictimDataset(corpus, tokenizer)
    heldout_dataset = EncodedVictimDataset(heldout, tokenizer)
    heldout_loader = DataLoader(
        heldout_dataset,
        batch_size=VICTIM_EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=CausalLMCollator(tokenizer.pad_token_id),
        pin_memory=True,
    )

    base_model = _load_base_model(args.model)
    baseline_utility = evaluate_causal_lm(
        base_model, heldout_loader, torch.device("cuda")
    )
    model = _attach_lora(base_model, args, modules)
    trainable_params, total_params = model.get_nb_trainable_parameters()
    if trainable_params <= 0 or total_params <= trainable_params:
        raise RuntimeError("invalid PEFT parameter counts")
    model.print_trainable_parameters()

    training = _train(
        args,
        model,
        tokenizer,
        train_dataset,
        resume_checkpoint=resume_checkpoint,
    )
    victim_utility = evaluate_causal_lm(
        model, heldout_loader, torch.device("cuda")
    )
    metrics: dict[str, Any] = {
        **plan,
        "trainable_params": int(trainable_params),
        "total_params": int(total_params),
        "trainable_pct": 100.0 * trainable_params / total_params,
        **training,
        "resumed_from_checkpoint": (
            str(resume_checkpoint) if resume_checkpoint is not None else None
        ),
        "pretrained_heldout_mean_nll": baseline_utility["mean_nll"],
        "pretrained_heldout_ppl": baseline_utility["perplexity"],
        "heldout_mean_nll": victim_utility["mean_nll"],
        "heldout_ppl": victim_utility["perplexity"],
        "heldout_predicted_tokens": victim_utility["predicted_tokens"],
        "heldout_ppl_delta": (
            float(victim_utility["perplexity"])
            - float(baseline_utility["perplexity"])
        ),
        "heldout_ppl_ratio": (
            float(victim_utility["perplexity"])
            / float(baseline_utility["perplexity"])
        ),
    }
    if any(
        isinstance(value, float) and not math.isfinite(value)
        for value in metrics.values()
    ):
        raise RuntimeError("refusing to save non-finite victim metrics")
    _atomic_write_json(args.output_dir / "victim_metrics.json", metrics)
    return metrics


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = run(args)
    except (RuntimeError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
