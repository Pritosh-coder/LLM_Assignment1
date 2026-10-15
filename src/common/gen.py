"""Deterministic sanitizer decoding shared by Part A validation and Part B."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import time
from typing import Any, TypedDict

from src.common.config import (
    MAX_SEQ_LEN,
    SANITIZER_MODEL_REVISION,
    VLLM_DETERMINISTIC_GENERATION,
)
from src.common.prompts import build_prompt_messages


class GeneratedOutput(TypedDict):
    text: str
    latency_ms: float


def build_inference_prompt(source_text: str, tokenizer: Any) -> str:
    """Render the exact M1 chat prompt as text for an inference engine."""
    prompt = tokenizer.apply_chat_template(
        build_prompt_messages(source_text),
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(prompt, str):
        raise TypeError("chat template must return text when tokenize=False")
    return prompt


def generate_with_vllm(
    records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    base_model: str,
    adapter_path: Path | None = None,
    adapter_name: str = "sanitizer",
    max_lora_rank: int = 32,
) -> list[GeneratedOutput]:
    """Generate one deterministic completion per record, optionally with LoRA.

    Imports stay local so CPU-only unit tests and ``--help`` do not require vLLM.
    vLLM owns a fresh inference engine; callers should release any training model
    first. Latency uses per-request engine timestamps when available and otherwise
    falls back to amortized batch wall-clock time.
    """
    try:
        import torch
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except ImportError as error:  # pragma: no cover - exercised on the GPU runtime
        raise RuntimeError(
            "vLLM is required for the full 2,000-row Part A validation pass"
        ) from error

    prompts: list[str] = []
    for index, record in enumerate(records):
        source_text = record.get("source_text")
        if not isinstance(source_text, str) or not source_text:
            raise ValueError(f"validation record {index}: invalid source_text")
        prompts.append(build_inference_prompt(source_text, tokenizer))

    sampling_params = SamplingParams(**VLLM_DETERMINISTIC_GENERATION)
    engine_kwargs: dict[str, Any] = {
        "model": base_model,
        "revision": SANITIZER_MODEL_REVISION,
        "tokenizer": base_model,
        "tokenizer_revision": SANITIZER_MODEL_REVISION,
        "dtype": "half",
        "max_model_len": MAX_SEQ_LEN
        + int(VLLM_DETERMINISTIC_GENERATION["max_tokens"]),
        "enable_lora": adapter_path is not None,
        "gpu_memory_utilization": 0.90,
        "trust_remote_code": False,
    }
    if adapter_path is not None:
        engine_kwargs["max_lora_rank"] = max_lora_rank

    # Training uses strict deterministic CUDA algorithms.  vLLM's LoRA setup
    # currently calls a CUDA cumsum kernel that has no strict deterministic
    # implementation on Kaggle T4s.  Greedy decoding remains deterministic,
    # while warn-only mode lets that unsupported setup operation proceed.
    torch.use_deterministic_algorithms(True, warn_only=True)
    engine = LLM(**engine_kwargs)
    request = (
        LoRARequest(adapter_name, 1, str(adapter_path.resolve()))
        if adapter_path is not None
        else None
    )
    started = time.perf_counter()
    outputs = engine.generate(
        prompts,
        sampling_params,
        lora_request=request,
        use_tqdm=True,
    )
    batch_latency_ms = (time.perf_counter() - started) * 1000.0
    if len(outputs) != len(records):
        raise RuntimeError(
            f"vLLM returned {len(outputs)} outputs for {len(records)} prompts"
        )

    generated: list[GeneratedOutput] = []
    fallback_latency_ms = batch_latency_ms / len(records) if records else 0.0
    for index, request_output in enumerate(outputs):
        candidates = request_output.outputs
        if len(candidates) != 1:
            raise RuntimeError(
                f"record {index}: expected one generated candidate"
            )
        metrics = getattr(request_output, "metrics", None)
        arrival = getattr(metrics, "arrival_time", None)
        finished = getattr(metrics, "finished_time", None)
        latency_ms = (
            1000.0 * (finished - arrival)
            if isinstance(arrival, (int, float))
            and isinstance(finished, (int, float))
            and finished >= arrival
            else fallback_latency_ms
        )
        generated.append({"text": candidates[0].text, "latency_ms": latency_ms})
    return generated


def generate_with_vllm_adapter(
    records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    base_model: str,
    adapter_path: Path,
    adapter_name: str,
    max_lora_rank: int,
) -> list[str]:
    """Compatibility wrapper used by the Part A sweep scorer."""
    outputs = generate_with_vllm(
        records,
        tokenizer,
        base_model=base_model,
        adapter_path=adapter_path,
        adapter_name=adapter_name,
        max_lora_rank=max_lora_rank,
    )
    return [output["text"] for output in outputs]
