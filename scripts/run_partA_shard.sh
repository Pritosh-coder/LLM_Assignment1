#!/usr/bin/env bash
set -euo pipefail

export CUBLAS_WORKSPACE_CONFIG="${PRITOSH_CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
python_bin="${PYTHON_BIN:-python}"
all_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
shard="${1:-}"

if [[ "$shard" != "1" && "$shard" != "2" && "$shard" != "3" ]]; then
  echo "usage: bash scripts/run_partA_shard.sh {1|2|3}" >&2
  exit 2
fi

mkdir -p adapters results/detail/partA_rows

run_one() {
  local gpu="$1"
  local run_id="$2"
  local target_modules="$3"
  local lora_r="$4"
  local lora_alpha="$5"
  local quant="$6"
  local result_file="results/detail/partA_rows/partA_${run_id}.csv"

  if [[ -e "$result_file" ]]; then
    echo "ERROR: result already exists: $result_file" >&2
    return 1
  fi
  echo "Starting ${run_id} on physical GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" src/peft_sweep.py \
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
    --results_csv "$result_file"
}

run_pair() {
  local left_pid right_pid left_status=0 right_status=0
  run_one 0 "$1" "$2" "$3" "$4" "$5" &
  left_pid=$!
  run_one 1 "$6" "$7" "$8" "$9" "${10}" &
  right_pid=$!
  wait "$left_pid" || left_status=$?
  wait "$right_pid" || right_status=$?
  if (( left_status != 0 || right_status != 0 )); then
    echo "ERROR: shard ${shard} failed (statuses $left_status/$right_status)" >&2
    exit 1
  fi
}

case "$shard" in
  1)
    run_pair A1 q_proj,v_proj 8 16 none \
             A2 q_proj,k_proj,v_proj,o_proj 8 16 none
    ;;
  2)
    run_pair A3 "$all_modules" 8 16 none \
             A4 "$all_modules" 32 64 none
    ;;
  3)
    run_pair A5 "$all_modules" 32 16 none \
             A6 "$all_modules" 32 64 4bit
    ;;
esac

echo "Part A shard ${shard} completed."
