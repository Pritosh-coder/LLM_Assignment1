#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"
python_bin="${PYTHON_BIN:-python3}"
mode="${1:-}"
export MPLBACKEND="${PRITOSH_MPLBACKEND:-Agg}"
export CUBLAS_WORKSPACE_CONFIG="${PRITOSH_CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

"$python_bin" scripts/check_data.py

if [[ "$mode" == "--preflight" ]]; then
  "$python_bin" -m compileall -q src tests scripts
  "$python_bin" -m unittest discover -s tests -v
  echo "CPU preflight passed."
  exit 0
fi

if [[ -n "$mode" && "$mode" != "--after-part-a" && "$mode" != "--stage-c1" && "$mode" != "--stage-c2" ]]; then
  echo "usage: bash run_all.sh [--preflight|--after-part-a|--stage-c1|--stage-c2]" >&2
  exit 2
fi

all_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

archive_recovery_files() {
  "$python_bin" scripts/archive_recovery_file.py \
    --archive_dir recovery/stale "$@"
}

"$python_bin" -m compileall -q src tests scripts
"$python_bin" -m unittest discover -s tests -v
mkdir -p adapters results/detail

if [[ ! -f data/partC_pii_index.jsonl ]]; then
  "$python_bin" scripts/build_partC_pii_index.py \
    --corpus data/partC_corpus_raw.jsonl \
    --output_file data/partC_pii_index.jsonl \
    --audit_file results/partC_pii_index_build.json \
    --detectors both
fi

run_part_b() {
  local system="$1"
  local split_name="$2"
  local test_file="$3"
  local predictions="$4"
  local metrics="$5"
  local expected_n=2000
  if [[ "$split_name" == "ood" ]]; then expected_n=1000; fi
  if [[ -f "$predictions" || -f "$metrics" ]]; then
    if [[ -f "$predictions" && -f "$metrics" ]] && 
      "$python_bin" scripts/validate_part_b_output.py \
        --metrics "$metrics" --predictions "$predictions" \
        --system "$system" --n "$expected_n" >/dev/null; then
      echo "Part B already complete and validated: ${system} on ${split_name}"
      return 0
    fi
    echo "Part B existing output is incomplete/invalid; preserving it under recovery/stale before recomputing: ${system}/${split_name}"
    archive_recovery_files "$predictions" "$metrics"
  fi
  local command=(
    "$python_bin" src/sanitize_eval.py
    --base_model Qwen/Qwen2.5-1.5B-Instruct
    --test_file "$test_file"
    --system "$system"
    --output_predictions "$predictions"
    --report_file "$metrics"
  )
  if [[ "$system" == "finetuned" ]]; then
    command+=(--adapter_path "$best_adapter")
  fi
  echo "Part B: ${system} on ${split_name}"
  CUDA_VISIBLE_DEVICES=0 "${command[@]}"
}

run_victim() {
  local gpu="$1"
  local corpus="$2"
  local output_dir="$3"
  local victim_name
  victim_name="$(basename "$output_dir")"
  if [[ -d "$output_dir" ]] && 
    "$python_bin" scripts/validate_victim_output.py --root . --name "$victim_name" --corpus "$corpus" >/dev/null 2>&1; then
    echo "Part C victim already complete and validated: ${victim_name}"
    return 0
  fi
  echo "Part C victim training on GPU ${gpu}: ${corpus}"
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" src/train_victim.py \
    --model Qwen/Qwen2.5-0.5B \
    --corpus "$corpus" \
    --heldout data/partC_heldout.jsonl \
    --target_modules "$all_modules" \
    --lora_r 32 --lora_alpha 64 --epochs 4 --seed 42 \
    --resume_if_available \
    --output_dir "$output_dir"
}

run_attack() {
  local gpu="$1"
  local corpus="$2"
  local report="$3"
  local expected_victim="$4"
  local adapter="${5:-}"
  if [[ -f "$report" ]]; then
    if "$python_bin" scripts/validate_attack_report.py \
      --report "$report" --corpus "$corpus" \
      --heldout data/partC_heldout.jsonl --expected_victim "$expected_victim" >/dev/null; then
      echo "Part C attack already complete and validated: ${report}"
      return 0
    fi
    echo "Existing attack report is invalid; preserving it under recovery/stale before recomputing: ${report}"
    archive_recovery_files "$report"
  fi
  local command=(
    "$python_bin" src/attack_extract.py
    --victim_model Qwen/Qwen2.5-0.5B
    --ref_model Qwen/Qwen2.5-0.5B
    --corpus "$corpus"
    --heldout data/partC_heldout.jsonl
    --canaries data/partC_canaries.jsonl
    --pii_index data/partC_pii_index.jsonl
    --targeted_file data/partC_targeted.jsonl
    --attack all
    --num_samples 5000
    --seed 42
    --report_file "$report"
  )
  if [[ -n "$adapter" ]]; then
    command+=(--adapter_path "$adapter")
  fi
  echo "Part C attack on GPU ${gpu}: ${report}"
  CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}"
  "$python_bin" scripts/validate_attack_report.py \
    --report "$report" --corpus "$corpus" \
    --heldout data/partC_heldout.jsonl --expected_victim "$expected_victim" >/dev/null
}

run_stage_c1() {
  if [[ ! -f results/partA_sweep.csv ]]; then
    echo "ERROR: Stage C1 requires the completed Part A sweep CSV; Part A must not be rerun here." >&2
    return 1
  fi

  if [[ -f results/best_sanitizer.json ]]; then
    best_adapter="$("$python_bin" - <<'PY'
import json, os
from pathlib import Path
from scripts.select_best_adapter import select_best
selection=json.loads(Path('results/best_sanitizer.json').read_text(encoding='utf-8'))
expected=select_best(Path('results/partA_sweep.csv'),Path('adapters'))
if selection.get('winner') != expected.get('winner') or selection.get('winner') != 'A4':
    raise SystemExit('ERROR: existing best_sanitizer.json does not match frozen Part A selection')
alias=Path('adapters/best_sanitizer')
if not alias.exists():
    alias.symlink_to(selection['winner'], target_is_directory=True)
if not (alias/'adapter_config.json').is_file():
    raise SystemExit('ERROR: best sanitizer alias is invalid')
print(str(alias))
PY
    )"
  else
    best_adapter="$(
      "$python_bin" scripts/select_best_adapter.py \
        --results_csv results/partA_sweep.csv \
        --adapters_dir adapters \
        --selection_file results/best_sanitizer.json \
        --alias_path adapters/best_sanitizer
    )"
  fi

  run_part_b finetuned test data/partB_test.jsonl \
    results/partB_predictions.jsonl results/partB_metrics.json
  run_part_b finetuned ood data/partB_ood_test.jsonl \
    results/detail/finetuned_ood_predictions.jsonl \
    results/detail/finetuned_ood_metrics.json
  for system in regex presidio zeroshot; do
    run_part_b "$system" test data/partB_test.jsonl \
      "results/detail/${system}_test_predictions.jsonl" \
      "results/detail/${system}_test_metrics.json"
    run_part_b "$system" ood data/partB_ood_test.jsonl \
      "results/detail/${system}_ood_predictions.jsonl" \
      "results/detail/${system}_ood_metrics.json"
  done

  if [[ -f data/partC_corpus_sanitized.jsonl || -f results/partC_sanitization_audit.json ]]; then
    if [[ -f data/partC_corpus_sanitized.jsonl && -f results/partC_sanitization_audit.json ]] && \
      "$python_bin" scripts/validate_sanitized_output.py \
        --raw data/partC_corpus_raw.jsonl \
        --sanitized data/partC_corpus_sanitized.jsonl \
        --audit results/partC_sanitization_audit.json >/dev/null; then
      echo "Part C sanitized corpus already complete and validated."
    else
      echo "Existing sanitized-corpus pair is incomplete/invalid; preserving it under recovery/stale before recomputing."
      archive_recovery_files data/partC_corpus_sanitized.jsonl results/partC_sanitization_audit.json
      CUDA_VISIBLE_DEVICES=0 "$python_bin" src/sanitize_corpus.py \
        --base_model Qwen/Qwen2.5-1.5B-Instruct \
        --adapter_path "$best_adapter" \
        --input_file data/partC_corpus_raw.jsonl \
        --output_file data/partC_corpus_sanitized.jsonl \
        --audit_file results/partC_sanitization_audit.json
    fi
  else
    CUDA_VISIBLE_DEVICES=0 "$python_bin" src/sanitize_corpus.py \
      --base_model Qwen/Qwen2.5-1.5B-Instruct \
      --adapter_path "$best_adapter" \
      --input_file data/partC_corpus_raw.jsonl \
      --output_file data/partC_corpus_sanitized.jsonl \
      --audit_file results/partC_sanitization_audit.json
  fi

  local victim_raw_status=0
  local victim_sanitized_status=0
  run_victim 0 data/partC_corpus_raw.jsonl adapters/victim_raw &
  local victim_raw_pid=$!
  run_victim 1 data/partC_corpus_sanitized.jsonl adapters/victim_sanitized &
  local victim_sanitized_pid=$!
  wait "$victim_raw_pid" || victim_raw_status=$?
  wait "$victim_sanitized_pid" || victim_sanitized_status=$?
  if (( victim_raw_status != 0 || victim_sanitized_status != 0 )); then
    echo "ERROR: Part C victim training failed (statuses ${victim_raw_status}/${victim_sanitized_status})" >&2
    return 1
  fi
  echo "Part C parallel victim training completed."
  local active_build_id
  active_build_id="$(tr -d '\r\n' < KAGGLE_BUILD_ID)"
  "$python_bin" scripts/validate_stage_c1.py --root . --expected-build-id "$active_build_id"
  echo "STAGE_C1_OK"
}

run_stage_c2() {
  local active_build_id
  active_build_id="$(tr -d '\r\n' < KAGGLE_BUILD_ID)"
  "$python_bin" scripts/validate_stage_c1.py --root . --expected-build-id "$active_build_id"

  local attack_base_status=0
  local attack_raw_status=0
  local attack_sanitized_status=0
  run_attack 0 data/partC_corpus_raw.jsonl \
    results/detail/partC_attacks_base.json pretrained_control &
  local attack_base_pid=$!
  run_attack 1 data/partC_corpus_raw.jsonl \
    results/detail/partC_attacks_raw.json victim_raw adapters/victim_raw &
  local attack_raw_pid=$!
  wait "$attack_base_pid" || attack_base_status=$?
  if (( attack_base_status == 0 )); then
    run_attack 0 data/partC_corpus_sanitized.jsonl \
      results/detail/partC_attacks_sanitized.json victim_sanitized adapters/victim_sanitized &
    local attack_sanitized_pid=$!
  else
    attack_sanitized_status=99
  fi
  wait "$attack_raw_pid" || attack_raw_status=$?
  if (( attack_base_status == 0 )); then
    wait "$attack_sanitized_pid" || attack_sanitized_status=$?
  fi
  if (( attack_base_status != 0 || attack_raw_status != 0 || attack_sanitized_status != 0 )); then
    echo "ERROR: Part C attacks failed (statuses ${attack_base_status}/${attack_raw_status}/${attack_sanitized_status})" >&2
    return 1
  fi
  echo "Part C parallel attack schedule completed."

  if [[ ! -f results/partC_attacks.json ]]; then
    "$python_bin" scripts/combine_partC_reports.py \
      --base results/detail/partC_attacks_base.json \
      --raw results/detail/partC_attacks_raw.json \
      --sanitized results/detail/partC_attacks_sanitized.json \
      --output_file results/partC_attacks.json
  else
    echo "Reusing existing combined Part C report: results/partC_attacks.json"
  fi

  if [[ ! -f results/partA_fig.png && ! -f results/partC_exposure.png ]]; then
    "$python_bin" scripts/make_figs.py \
      --partA_csv results/partA_sweep.csv \
      --partC_raw results/detail/partC_attacks_raw.json \
      --partC_sanitized results/detail/partC_attacks_sanitized.json \
      --partA_output results/partA_fig.png \
      --partC_output results/partC_exposure.png
  elif [[ -f results/partA_fig.png && -f results/partC_exposure.png ]]; then
    echo "Reusing existing required figures."
  else
    echo "ERROR: only one required figure exists; refusing a mixed/stale figure set." >&2
    return 1
  fi

  "$python_bin" scripts/validate_submission.py --root .
  echo "STAGE_C2_OK"
}

if [[ -z "$mode" ]]; then
  PYTHON_BIN="$python_bin" bash scripts/run_partA_sweep.sh
  run_stage_c1
  run_stage_c2
elif [[ "$mode" == "--after-part-a" ]]; then
  run_stage_c1
  run_stage_c2
elif [[ "$mode" == "--stage-c1" ]]; then
  run_stage_c1
elif [[ "$mode" == "--stage-c2" ]]; then
  run_stage_c2
fi

echo "All requested computational artifacts completed. Draft and validate report.pdf next."
