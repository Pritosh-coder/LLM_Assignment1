# Assignment 1 accessibility setup — Pritosh Kumar (2025jrb2025)

Accounts: GitHub `Pritosh-coder`; Hugging Face `Pritosh1`.
Suggested new source repo: `Pritosh-coder/LLM_Assignment1`.
Configured new model repo: `Pritosh1/LLM_Assignment1-adapters`.
These names are targets, not evidence that either repository has been created.

## 1. Publish the existing trained adapters (author step, once)

Download `2025jrb2025_A1_Adapters.zip` from the adapter folder already linked in the original README:
https://drive.google.com/file/d/1buEZPTq5UPtx4nUSiM9kQVUVt1BxluyD/view

Extract this updated source package. Open a terminal in `LLM_Assignment1_ready`.
Use Python 3.10 for consistency with the original training environment.
Publication works on Windows without installing the CUDA training dependencies.

```powershell
python -m pip install -r requirements-publish.txt
huggingface-cli login
python scripts/publish_hf.py --adapter-zip "C:\path\to\2025jrb2025_A1_Adapters.zip"
python scripts/download_artifacts.py
```

Replace the example ZIP path with your actual path. Log in as `Pritosh1` with a write-capable Hugging Face token, entered privately in the terminal prompt. No token belongs in source code or chat.

The script checks the six files against their original hashes, creates the model repo,
sets it public, uploads the three adapter subfolders, checks anonymous config access,
and writes the upload commit SHA to `config/hf_artifacts.json`.
The following download command checks anonymous access and hashes of all adapter weight files.
Do not submit until both commands succeed. The initial `main` revision in the prepared
manifest becomes an immutable commit SHA only after a successful upload.

Expected Hugging Face structure:

```text
LLM_Assignment1-adapters/
  README.md
  checksums.json
  best_sanitizer/
    adapter_config.json
    adapter_model.safetensors
  victim_raw/
    adapter_config.json
    adapter_model.safetensors
  victim_sanitized/
    adapter_config.json
    adapter_model.safetensors
```

These are LoRA adapters. Do not rename adapter_model.safetensors to model.safetensors.
The upstream Qwen base weights remain in their original public repositories.

## 2. Create the GitHub source repository

In GitHub, create an empty repository named `LLM_Assignment1` under `Pritosh-coder`.
A public repository meets the notice's code-access requirement directly.
If you choose private, invite all four usernames under Settings → Collaborators:
`proadhikary`, `sudiptog81`, `vinay-kumar07`, `tkarthikeyan132`.
Ensure access has actually been accepted rather than leaving invitations pending.
GitHub personal-repository collaborator permissions may include write access;
organization repositories offer finer read-only role control.

After publishing Hugging Face, from this source directory:

```bash
git init -b main
git add .
git status --short
git commit -m "Make Assignment 1 code and public adapters reproducible"
git remote add origin https://github.com/Pritosh-coder/LLM_Assignment1.git
git push -u origin main
```

Use your normal GitHub credential manager/browser authentication for the push.
The included .gitignore excludes trained weights, datasets, local credentials,
recovery folders, predictions, and attack dumps from normal `git add .`.
Do not force-add these excluded files. Original results are retained in this
package for your review; a browser upload does not apply .gitignore automatically.

If you prefer ChatGPT to push the prepared code using the connected GitHub account,
create the empty repository first and send its link. No repo-creation operation
was exposed by the connected GitHub tools during preparation.
No authenticated Hugging Face upload connection was available either.

## 3. Check evaluator usage on Linux/CUDA

Place instructor-supplied assignment data in `data/`, retaining original names, including the official `manifest.jsonl` used by `scripts/check_data.py`.
This is assignment input, not a required manual model/adapter download.
The original package contains no official datasets; they were not recreated.

```bash
python -m pip install -r requirements.txt
python src/sanitize_eval.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path adapters/best_sanitizer \
  --test_file data/partB_test.jsonl --system finetuned \
  --output_predictions evaluation/partB_predictions.jsonl \
  --report_file evaluation/partB_metrics.json
```

On a fresh clone with no adapters folder, this command automatically downloads the
selected A4 adapter anonymously and verifies its config and weight hashes. Base
models also download automatically. No manual adapter extraction or evaluator
login is required. Existing local registered adapters must match the submitted
hashes; a stale/corrupt checkpoint raises an error.

For all ID/OOD sanitizer systems and all three Part C variants:

```bash
bash evaluate_pretrained.sh
```

This runs evaluation of the trained checkpoints; it does not retrain the six-run
sweep. The script writes new outputs under `evaluation/`. Baseline Presidio uses
the spaCy model included in requirements. The PII-index fallback remains detector
derived when the instructor's gold index is absent, as documented in the original.

## Inspection findings and limits

- Original code archive: 91 files, 4,903,412 bytes uncompressed; no model/adapter weights and no datasets.
- Separate adapter ZIP: 267,229,682 bytes compressed; six config/weight files and two metadata files.
- best_sanitizer: 147,770,496 weight bytes; Qwen/Qwen2.5-1.5B-Instruct; selected A4.
- victim_raw: 70,430,032 weight bytes; Qwen/Qwen2.5-0.5B.
- victim_sanitized: 70,430,032 weight bytes; Qwen/Qwen2.5-0.5B.
- A1/A2/A3/A5/A6 sweep checkpoints are absent from the provided archives.
  If the instructor expects all experimental checkpoints rather than only the
  three evaluation adapters, recover and publish them too. They cannot be
  reconstructed from metric JSON files. A4 is available through best_sanitizer.
- `huggingface-hub==0.28.1` is now an explicit dependency. The original requirements
  already specify the ML stack; this CPU preparation environment has no Torch,
  Transformers, vLLM or CUDA inference runtime. Full dependency installation was
  not performed, so compatibility of all original pins remains unverified here.
- The original test suite refers to three notebooks not present in the source ZIP:
  `LLM-Pritosh-Stage-C1-PartB-Train.ipynb`,
  `LLM-Pritosh-Stage-C2-Attacks-Use-VBase-FIX9.ipynb`,
  `LLM-Pritosh-Stage-C2-Attacks-Use-VBase-FIX10.ipynb`.
  Recover these if required for the complete notebook history/full original tests.
- Original `run_all.sh` runs all tests before training and therefore remains blocked
  by absent notebook fixtures. Use the documented direct CLIs or the new
  `evaluate_pretrained.sh` for checkpoint evaluation. No tests were removed to hide
  missing inputs.
- Validation: all six original adapter hashes match; new resolver tests 3/3 pass;
  Python compilation and both shell syntax checks pass.
- Broad suite in preparation environment: 79 tests run, 13 errors caused by absent
  official data, historical notebooks, or Transformers. The suite did not fully pass.
- Live Hub download, publication, and GPU inference remain unverified until the
  author upload and evaluator checks above succeed. Submitted metrics were not changed.

## Final checklist

- [ ] Complete source is visible in the intended GitHub repo.
- [ ] Public code opens while logged out, or all four TAs have accepted private access.
- [ ] All three evaluation adapter folders are public and ungated on Hugging Face.
- [ ] `python scripts/download_artifacts.py` succeeds anonymously, checking all six files.
- [ ] `config/hf_artifacts.json` contains the actual uploaded commit SHA and is committed.
- [ ] Fresh-clone Part B evaluation runs with no local adapter folder or evaluator token.
- [ ] Part C raw and sanitized variants load their respective trained adapters.
- [ ] Decide/recover missing sweep checkpoints and notebooks if required by the course.
- [ ] Share GitHub and Hugging Face repository links through the course submission channel.

Official API references used for the setup:
https://huggingface.co/docs/huggingface_hub/v0.28.1/package_reference/file_download
https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api
https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository
https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/repository-access-and-collaboration/permission-levels-for-a-personal-account-repository
