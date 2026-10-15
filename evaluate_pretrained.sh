#!/usr/bin/env bash
# Evaluate the submitted trained adapters without rerunning the training sweep.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
python_bin="${PYTHON_BIN:-python3}"
"$python_bin" scripts/check_data.py
mkdir -p evaluation/detail
for split in test ood_test; do
  for system in finetuned regex presidio zeroshot; do
    "$python_bin" src/sanitize_eval.py \
      --base_model Qwen/Qwen2.5-1.5B-Instruct \
      --adapter_path adapters/best_sanitizer \
      --test_file "data/partB_${split}.jsonl" --system "$system" \
      --output_predictions "evaluation/detail/${system}_${split}_predictions.jsonl" \
      --report_file "evaluation/detail/${system}_${split}_metrics.json"
  done
done
if [[ ! -f data/partC_pii_index.jsonl ]]; then
  "$python_bin" scripts/build_partC_pii_index.py \
    --corpus data/partC_corpus_raw.jsonl --output_file data/partC_pii_index.jsonl \
    --audit_file evaluation/detail/pii_index_build.json --detectors both
fi
for variant in base raw sanitized; do
  adapter_args=()
  if [[ "$variant" != base ]]; then
    adapter_args=(--adapter_path "adapters/victim_${variant}")
  fi
  "$python_bin" src/attack_extract.py \
    --victim_model Qwen/Qwen2.5-0.5B --ref_model Qwen/Qwen2.5-0.5B \
    "${adapter_args[@]}" \
    --corpus data/partC_corpus_raw.jsonl --heldout data/partC_heldout.jsonl \
    --canaries data/partC_canaries.jsonl --pii_index data/partC_pii_index.jsonl \
    --targeted_file data/partC_targeted.jsonl --attack all --num_samples 5000 --seed 42 \
    --report_file "evaluation/detail/partC_attacks_${variant}.json"
done
