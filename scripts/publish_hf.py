"""Author-only publication of the three verified adapters from the Drive ZIP."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common.hub_artifacts import MANIFEST, load_manifest, verify_adapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-zip", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true", help="Verify ZIP offline; no upload")
    args = parser.parse_args()
    manifest = load_manifest()
    with tempfile.TemporaryDirectory(prefix="assignment1-hf-") as directory:
        staging = Path(directory)
        with zipfile.ZipFile(args.adapter_zip) as archive:
            # Copy exactly six expected files, never extract arbitrary ZIP paths.
            for name, entry in manifest["adapters"].items():
                target = staging / entry["subfolder"]
                target.mkdir()
                for filename in entry["files"]:
                    data = archive.read(f"adapters/{name}/{filename}")
                    (target / filename).write_bytes(data)
                verify_adapter(target, entry)
                print(f"Verified {name}")
        (staging / "README.md").write_text(
            "---\nlibrary_name: peft\ntags:\n- lora\n- coursework\n"
            "base_model:\n- Qwen/Qwen2.5-1.5B-Instruct\n- Qwen/Qwen2.5-0.5B\n---\n\n"
            "# Assignment 1 adapters — Pritosh Kumar, 2025jrb2025\n\n"
            "Published for course evaluation under the instructor's updated accessibility notice.\n\n"
            "| Subfolder | Base model | Purpose |\n|---|---|---|\n"
            "| best_sanitizer | Qwen/Qwen2.5-1.5B-Instruct | Selected A4 sanitizer |\n"
            "| victim_raw | Qwen/Qwen2.5-0.5B | Part C raw victim |\n"
            "| victim_sanitized | Qwen/Qwen2.5-0.5B | Part C sanitized victim |\n\n"
            "These are LoRA adapters, not merged standalone models. Base models are downloaded "
            "from the original Qwen repositories at revisions pinned in the source code. "
            "No assignment datasets or extraction dumps are included.\n",
            encoding="utf-8",
        )
        (staging / "checksums.json").write_text(json.dumps(manifest["adapters"], indent=2))
        if args.check_only:
            print("Offline archive integrity check passed. Nothing uploaded.")
            return
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()
        if api.whoami()["name"].lower() != "pritosh1":
            raise SystemExit("Log in as Pritosh1 using huggingface-cli login first.")
        api.create_repo(repo_id=manifest["repo_id"], repo_type="model", private=False, exist_ok=True)
        api.update_repo_settings(repo_id=manifest["repo_id"], private=False, gated=False)
        commit = api.upload_folder(
            repo_id=manifest["repo_id"], folder_path=str(staging), repo_type="model",
            commit_message="Publish verified Assignment 1 evaluation adapters",
        )
        revision = commit.oid
        public = HfApi(token=False).model_info(manifest["repo_id"], revision=revision)
        if public.private or public.gated:
            raise SystemExit("Repository must be public and ungated for evaluation.")
        for entry in manifest["adapters"].values():
            hf_hub_download(
                repo_id=manifest["repo_id"], revision=revision,
                filename=f"{entry['subfolder']}/adapter_config.json", token=False,
            )
        manifest["revision"] = revision
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Published: https://huggingface.co/{manifest['repo_id']}")
        print(f"Pinned public commit: {revision}")
        print("Next: python scripts/download_artifacts.py (checks all weight bytes anonymously).")
        print("Include config/hf_artifacts.json in your GitHub commit after this step.")


if __name__ == "__main__":
    main()
