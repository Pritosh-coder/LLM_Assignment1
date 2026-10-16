"""Publish verified Part A checkpoints from the original shard ZIP files."""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import zipfile
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common.hub_artifacts import PART_A_MANIFEST, load_part_a_manifest, verify_adapter


def locate_run_file(name: str) -> tuple[str, str] | None:
    """Support original shards, run-only ZIPs and Explorer wrapper folders."""
    normalized = name.replace("\\", "/")
    parts = normalized.split("/")
    runs = [(index, part) for index, part in enumerate(parts) if re.fullmatch(r"A[1-9][0-9]*", part)]
    if not runs:
        return None
    if normalized.startswith("/") or ".." in parts or len(runs) != 1:
        raise ValueError(f"Unsafe or ambiguous archive member: {name}")
    index, run = runs[0]
    return run, "/".join(parts[index + 1:])


def stage_shards(paths: list[Path], staging: Path, manifest: dict) -> list[str]:
    """Copy only checksummed files; reject unknown runs and ambiguous duplicates."""
    seen = {}
    with ExitStack() as stack:
        for path in paths:
            archive = stack.enter_context(zipfile.ZipFile(path))
            for member in archive.infolist():
                if member.is_dir():
                    continue
                location = locate_run_file(member.filename)
                if location is None:
                    continue
                run, filename = location
                if run not in manifest["adapters"]:
                    raise ValueError(f"{run} is not in the inspected Part A manifest. Update it before publishing.")
                entry = manifest["adapters"][run]
                if filename not in entry["files"]:
                    continue
                key = (run, filename)
                if key in seen:
                    raise ValueError(f"Duplicate checkpoint file: {run}/{filename}")
                seen[key] = (archive, member)
        expected = {(run, filename) for run, entry in manifest["adapters"].items() for filename in entry["files"]}
        missing = sorted(expected - set(seen))
        if missing:
            raise ValueError(f"Missing checkpoint files; supply all registered shards: {missing}")
        # Validate the complete source set before any network mutation.
        for run, entry in manifest["adapters"].items():
            target = staging / entry["subfolder"]
            target.mkdir(parents=True)
            for filename in entry["files"]:
                if Path(filename).name != filename:
                    raise ValueError(f"Unsafe manifest filename: {filename}")
                archive, member = seen[(run, filename)]
                (target / filename).write_bytes(archive.read(member))
            verify_adapter(target, entry)
            print(f"Verified {run}")
    runs = sorted(manifest["adapters"])
    (staging / "PART_A_README.md").write_text(
        "# Part A sweep adapters\n\nAvailable runs: " + ", ".join(runs) + ".\n\n"
        "Base: Qwen/Qwen2.5-1.5B-Instruct, pinned by the source code.\n"
        "A4 config and weights are identical to the published best_sanitizer adapter.\n"
        "Each run includes its original tokenizer files and validation metrics.\n",
        encoding="utf-8",
    )
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs="+", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    manifest = load_part_a_manifest()
    if set(manifest["adapters"]) != {f"A{i}" for i in range(1, 7)}:
        raise SystemExit("Part A publication requires all six registered runs A1 through A6.")
    with tempfile.TemporaryDirectory(prefix="part-a-upload-") as directory:
        staging = Path(directory)
        runs = stage_shards(args.shards, staging, manifest)
        if args.check_only:
            print("All supplied checkpoint hashes match. Nothing uploaded.")
            return
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()
        if api.whoami()["name"].lower() != "pritosh1":
            raise SystemExit("Log in as Pritosh1 with huggingface-cli login first.")
        # The existing repository must already be public. Do not recreate it or
        # change its settings, root README, or the three evaluation checkpoints.
        info = api.model_info(manifest["repo_id"])
        if info.private or info.gated:
            raise SystemExit("The adapter repository must be public and ungated.")
        commit = api.upload_folder(
            repo_id=manifest["repo_id"], repo_type="model", folder_path=str(staging),
            commit_message="Add Part A sweep adapters: " + ", ".join(runs),
        )
        public = HfApi(token=False).model_info(manifest["repo_id"], revision=commit.oid)
        if public.private or public.gated:
            raise SystemExit("Anonymous access is required.")
        for entry in manifest["adapters"].values():
            hf_hub_download(repo_id=manifest["repo_id"], revision=commit.oid,
                            filename=f"{entry['subfolder']}/adapter_config.json", token=False)
        manifest["revision"] = commit.oid
        PART_A_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print("Published runs: " + ", ".join(runs))
        print("Published: https://huggingface.co/" + manifest["repo_id"])
        print("Pinned Part A public commit: " + commit.oid)
        print("Next: python scripts/download_artifacts.py --part-a")
        print("Commit config/hf_part_a_artifacts.json to GitHub after successful verification.")


if __name__ == "__main__":
    main()
