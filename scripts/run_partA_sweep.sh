#!/usr/bin/env bash
set -euo pipefail

export CUBLAS_WORKSPACE_CONFIG="${PRITOSH_CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

python_bin="${PYTHON_BIN:-python}"
extra_args=()
if [[ "${1:-}" == "--dry-run" ]]; then
  extra_args+=(--dry_run)
elif [[ -n "${1:-}" ]]; then
  echo "usage: bash scripts/run_partA_sweep.sh [--dry-run]" >&2
  exit 2
fi

run_one() {
  local run_id="$1"
  local target_modules="$2"
  local lora_r="$3"
  local lora_alpha="$4"
  local quant="$5"

  local -a env_prefix=()
  if [[ "$quant" == "4bit" ]]; then
    # BitsAndBytes device_map pins QLoRA to one GPU; LoRA uses all visible GPUs.
    env_prefix=(env CUDA_VISIBLE_DEVICES="${PARTA_QLORA_CUDA_VISIBLE_DEVICES:-0}")
  fi

  "${env_prefix[@]}" "$python_bin" src/peft_sweep.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --train_file data/partA_train.jsonl \
    --val_file data/partA_validation.jsonl \
    --target_modules "$target_modules" \
    --lora_r "$lora_r" \
    --lora_alpha "$lora_alpha" \
    --lora_dropout 0.05 \
    --quant "$quant" \
    --seed 42 \
    --run_id "$run_id" \
    --output_dir "adapters/$run_id" \
    --results_csv results/partA_sweep.csv \
    "${extra_args[@]}"
}

run_one A1 q_proj,v_proj 8 16 none
run_one A2 q_proj,k_proj,v_proj,o_proj 8 16 none
run_one A3 q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj 8 16 none
run_one A4 q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj 32 64 none
run_one A5 q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj 32 16 none
run_one A6 q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj 32 64 4bit
