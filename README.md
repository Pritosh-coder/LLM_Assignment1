# Assignment 1 — Fine-Tune, Sanitize, Extract

ELL8299 / COL8383 / COL873 — Trustworthy Large Language Models, Semester I 2026–27

This repository implements all three parts of Assignment 1 using the supplied datasets without resampling, replacement, augmentation, reordering, truncation, or subsampling.

## Accessibility status

Source code is published in `Pritosh-coder/LLM_Assignment1`.
The automatic-download configuration targets `Pritosh1/LLM_Assignment1-adapters`.
**Author action pending:** run the Hugging Face upload in `SETUP_GUIDE.md`, verify anonymous
weight downloads, then commit the resulting pinned revision in `config/hf_artifacts.json`.
Until that publication succeeds, a fresh-clone adapter download will fail.
The existing submission archive and original report remain in the student's Drive;
text-bearing predictions and extraction dumps are excluded from this public source repository.

## Reproducibility summary

- Reference runtime: Python 3.10 on NVIDIA T4-class GPUs.
- Global seed: `42`.
- Sanitizer: `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- Victim/reference LM: base `Qwen/Qwen2.5-0.5B`, revision `060db6499f32faf8b98477b0a26969ef7d8b9987`.
- Part A uses completion-only loss; prompt labels are masked with `-100`.
- V-raw and V-san use identical PEFT/training settings; only the training corpus differs.
- Part C uses explicit seeded generation parameters and evaluates V-base, V-raw, and V-san.
- Required input rows are consumed in the supplied order.

The frozen Part A selection rule is: maximize validation overlap-IoU50 micro-F1, then minimize validation leak rate, validation loss, trainable percentage, and finally run ID. This selects **A4**.

## Required data

Place the official assignment data files under `data/` with their original names. The source submission deliberately does **not** redistribute full assignment corpora, sanitized corpus text, base models, victim weights, or generation dumps.

Expected official files:

| File | Rows |
| --- | ---: |
| `partA_train.jsonl` | 20,000 |
| `partA_validation.jsonl` | 2,000 |
| `partB_test.jsonl` | 2,000 |
| `partB_ood_test.jsonl` | 1,000 |
| `partC_corpus_raw.jsonl` | 8,007 |
| `partC_heldout.jsonl` | 500 |
| `partC_canaries.jsonl` | 20 |
| `partC_targeted.jsonl` | 400 |

The handout CLI also references `data/partC_pii_index.jsonl`. When instructor gold is unavailable, `scripts/build_partC_pii_index.py` deterministically builds the documented detector-derived local fallback used in these experiments; instructor-provided gold should supersede that fallback if supplied.

## Environment

Install the pinned environment with:

```bash
python -m pip install -r requirements.txt
```

The dependency list is restricted to the assignment-approved stack and its dependencies. The spaCy English model is installed as a dependency for Presidio.

## Required direct CLIs

The four grader-facing entry points requested in the handout are implemented.

### Part A — one PEFT run

```bash
python src/peft_sweep.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --train_file data/partA_train.jsonl \
  --val_file data/partA_validation.jsonl \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 \
  --quant none --seed 42 \
  --run_id A4 --output_dir adapters/A4 \
  --results_csv results/partA_sweep.csv
```

For A6 use `--quant 4bit`; the A1–A6 sweep definitions are frozen in `src/common/config.py` and `scripts/run_partA_sweep.sh`.

### Part B — sanitizer evaluation

```bash
python src/sanitize_eval.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path adapters/best_sanitizer \
  --test_file data/partB_test.jsonl \
  --system finetuned \
  --output_predictions results/partB_predictions.jsonl \
  --report_file results/partB_metrics.json
```

`--system` also supports `regex`, `presidio`, and `zeroshot`. Predicted spans are recovered by deterministic source/output alignment, and both exact+label and one-to-one maximum-IoU (IoU ≥ 0.5)+label matching are reported. The per-class table is computed from the **10 most frequent gold classes of the split being evaluated** (test or OOD), not from a hard-coded label list.

### Part C — victim training

```bash
python src/train_victim.py \
  --model Qwen/Qwen2.5-0.5B \
  --corpus data/partC_corpus_raw.jsonl \
  --heldout data/partC_heldout.jsonl \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --lora_r 32 --lora_alpha 64 --epochs 4 --seed 42 \
  --output_dir adapters/victim_raw
```

V-san uses the same command/configuration with `data/partC_corpus_sanitized.jsonl` and `adapters/victim_sanitized`. Held-out perplexity is token-weighted over the full held-out corpus.

### Part C — extraction attacks

```bash
python src/attack_extract.py \
  --victim_model Qwen/Qwen2.5-0.5B \
  --adapter_path adapters/victim_raw \
  --ref_model Qwen/Qwen2.5-0.5B \
  --corpus data/partC_corpus_raw.jsonl \
  --heldout data/partC_heldout.jsonl \
  --canaries data/partC_canaries.jsonl \
  --pii_index data/partC_pii_index.jsonl \
  --targeted_file data/partC_targeted.jsonl \
  --attack all --num_samples 5000 --seed 42 \
  --report_file results/partC_attacks.json
```

Implemented attacks: discoverable memorization, 5,000-sample untargeted extraction with PPL/reference/lowercase/zlib scores, canary exposure against 9,999 alternatives, and targeted greedy plus sampled-any-10 extraction for 200 members and 200 controls.

## Full pipeline

Run the CPU/data gate first:

```bash
PYTHON_BIN=python bash run_all.sh --preflight
```

A fresh end-to-end run is:

```bash
PYTHON_BIN=python bash run_all.sh
```

For two-GPU execution, the same pipeline can be staged without changing any experiment definition:

```bash
PYTHON_BIN=python bash run_all.sh --stage-c1
PYTHON_BIN=python bash run_all.sh --stage-c2
```

- `--stage-c1`: selects the frozen A4 sanitizer from completed A1–A6 artifacts, evaluates Part B + baselines + OOD, sanitizes all 8,007 Part C rows, and trains V-raw/V-san on isolated GPUs.
- `--stage-c2`: validates the C1 state, evaluates V-base/V-raw/V-san, combines Part C reports, renders figures, and validates required result schemas.
- The full default mode additionally executes the six-run Part A sweep first.

`run_all.sh` is fail-closed: it validates official row counts and schemas, rejects inconsistent/stale artifacts, preserves incompatible partial outputs under `recovery/stale/`, and never silently changes seeds, data, model revisions, attack counts, or victim hyperparameters.

## Required result files

The original submission archive contains these handout-mandated paths (text-bearing outputs are excluded from this public source repository):

```text
results/partA_sweep.csv
results/partA_fig.png
results/partB_metrics.json
results/partB_predictions.jsonl
results/partC_attacks.json
results/partC_exposure.png
```

Additional baseline/OOD/three-victim reports are under `results/detail/`.

### Measured result snapshot

These are the observed submitted-run values; no values below are synthetic placeholders.

- Part A winner A4: validation overlap-IoU50 entity F1 `0.951051`, leak rate `0.039563`, peak allocated GPU memory `4.710696 GB`.
- Part B finetuned sanitizer, ID test: exact micro-F1 `0.944275`, IoU≥0.5 micro-F1 `0.948256`, leak rate `0.037640`, over-masking rate `0.003218`, well-formed rate `0.987`.
- Part B finetuned sanitizer, OOD: exact micro-F1 `0.368808`, IoU≥0.5 micro-F1 `0.394048`, leak rate `0.379268`.
- Part C held-out perplexity: V-base `14.174821`, V-raw `9.650770`, V-san `16.630227`.
- Part C V-raw → V-san targeted member leakage: greedy `0.430 → 0.005`; sampled-any-10 `0.460 → 0.025`.
- Part C mean sampled canary exposure: V-raw `13.287712 bits`, V-san `2.131392 bits`. The sampled-rank estimator has a `log2(10000) ≈ 13.29-bit` ceiling.

Detailed values and baselines are in `results/` and `results/detail/`; the report should be treated as analysis of those files, not as an alternate source of measurements.

## Public adapters and automatic downloads

The instructor's updated accessibility notice supersedes the earlier private-adapter instructions.
The intended Hub repository is `Pritosh1/LLM_Assignment1-adapters`; it is not published by merely downloading this package.
See `SETUP_GUIDE.md` for the author upload steps.

`scripts/publish_hf.py` publishes the three verified adapters and records the uploaded commit SHA
in `config/hf_artifacts.json`. Publish to Hugging Face **before** committing the final manifest to GitHub.

The direct Part B evaluation, corpus sanitizer, and Part C attack CLIs automatically fetch
missing `adapters/best_sanitizer`, `adapters/A4`, `adapters/victim_raw`, and
`adapters/victim_sanitized` aliases from the public Hub, anonymously, with SHA-256 verification.
Existing custom local adapter directories still work. Unknown missing paths raise an error.
The Qwen base model loaders already fetch their pinned public model revisions automatically.

After the official datasets are available in `data/`, trained-adapter evaluation on a suitable
Linux CUDA runtime can be run without retraining:

```bash
python -m pip install -r requirements.txt
bash evaluate_pretrained.sh
```

This writes new metrics under `evaluation/`, leaving submitted results separate.
An optional download/integrity check is `python scripts/download_artifacts.py`.
Add `--include-base-models` to that command to prefetch both base model snapshots.
Evaluation itself requires no prior download command and no Hugging Face login.

## Submission/privacy notes

The final `[EntryNumber].zip` must be ≤500 MB and must not contain base-model weights, the full raw/sanitized corpus, generation dumps, W&B directories, or victim-model weights. Extracted PII shown in the report must be redacted. Extraction attacks are restricted to the assignment's locally fine-tuned models.

## Disclosure

```text
Student name: Pritosh Kumar
Entry number: 2025jrb2025
Student collaborators consulted/discussed with: None
LLMs/tools used: ChatGPT (OpenAI; boilerplate/debugging/analysis/report assistance), Kaggle Notebooks, Google Colab, Google Drive, Python and PowerShell integrity checks.
```

The report must include the handout's honour-code sentence **verbatim**:

> Even though I have taken help from the following students and LLMs in terms of discussing ideas and coding practices, all my code is written by me.

## Validation before submission

After the report is added as `report.pdf`, run:

```bash
python scripts/validate_submission.py --root . --require_report
```

Then inspect the final ZIP manually as well: confirm all mandatory files exist, the report has at most five pages, private/full-corpus artifacts are absent, and the archive is ≤500 MB.

## Validation status of this accessibility update

- Three original adapter config/weight pairs verified against the supplied SHA256SUMS.
- New resolver tests pass; Python compilation and shell syntax checks pass.
- Full original suite: 79 tests run, 13 errors due to absent official data, legacy notebooks, or Transformers in this CPU preparation environment. No GPU inference or network upload has been verified.
- See `SETUP_GUIDE.md` for the three legacy notebooks absent from the original source ZIP.
