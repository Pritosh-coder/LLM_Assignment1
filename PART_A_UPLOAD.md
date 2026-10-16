# Upload Part A sweep checkpoints

All six original sweep adapters A1, A2, A3, A4, A5 and A6 are registered.
A5 and A6 were inspected from the split ZIPs Parta_3_1.zip and partA_3_2.zip.
A4 has the same config and weight hashes as the existing best_sanitizer checkpoint.

1. Download the latest source ZIP from GitHub (Code → Download ZIP), extract it,
   and open PowerShell in the directory containing `scripts/` and `config/`.
2. Keep `partA-shard-1.zip`, `partA-shard-2.zip`, `Parta_3_1.zip` and `partA_3_2.zip` in Downloads.
3. Use the existing Hugging Face login as Pritosh1. If needed, install
   `python -m pip install -r requirements-publish.txt` and run `huggingface-cli login`.
4. Run:

```powershell
python scripts/publish_part_a.py --shards "$env:USERPROFILE\Downloads\partA-shard-1.zip" "$env:USERPROFILE\Downloads\partA-shard-2.zip" "$env:USERPROFILE\Downloads\Parta_3_1.zip" "$env:USERPROFILE\Downloads\partA_3_2.zip"
python scripts/download_artifacts.py --part-a
```

The first command verifies all registered files before any upload. It adds A1–A6
folders, original tokenizer files and validation metrics to the existing public
Hugging Face repository. It does not upload `training_args.bin`, recreate the
repository, delete existing files, or replace the original evaluation checkpoints.
The optional `--check-only` flag verifies the shard ZIPs without network access.

After upload, `config/hf_part_a_artifacts.json` contains the new commit SHA.
Send that commit SHA and the verification output back so the GitHub registry can
be pinned to the published revision. A null revision means publication is pending;
the code does not claim those adapters are already downloadable.

`python scripts/download_artifacts.py --part-a` verifies anonymous downloads of
the original three adapters plus all registered Part A adapters.
The evaluation and corpus sanitizer CLIs also automatically fetch missing
`adapters/A1`, `adapters/A2`, `adapters/A3`, `adapters/A5`, and `adapters/A6` from their pinned registry once
publication is complete. `adapters/A4` continues using the verified best_sanitizer.

If you still have the original `partA-shard-3.zip`, you may supply it in place of
both split ZIPs. Use either the original third shard or its two split ZIPs, never
both in the same command (duplicate checkpoint files are rejected).
All six registered adapters must be supplied before publication can start.
