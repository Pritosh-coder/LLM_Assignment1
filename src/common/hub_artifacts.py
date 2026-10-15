"""Resolve submitted adapters from a public Hub snapshot and verify their bytes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[2] / "config" / "hf_artifacts.json"


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def verify_adapter(path: Path, entry: dict) -> Path:
    for filename, expected in entry["files"].items():
        source = path / filename
        if not source.is_file():
            raise ValueError(f"Missing adapter file: {source}")
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"SHA256 mismatch: {source}")
    config = json.loads((path / "adapter_config.json").read_text())
    if config.get("base_model_name_or_path") != entry["base_model"]:
        raise ValueError(f"Unexpected adapter base model in {path}")
    return path


def download_adapter(name: str) -> Path:
    manifest = load_manifest()
    if name not in manifest["adapters"]:
        raise ValueError(f"Unknown submitted adapter: {name}")
    entry = manifest["adapters"][name]
    try:
        from huggingface_hub import snapshot_download
        snapshot = snapshot_download(
            repo_id=manifest["repo_id"],
            revision=manifest["revision"],
            allow_patterns=[f"{entry['subfolder']}/{f}" for f in entry["files"]],
            token=False,
        )
    except Exception as error:
        raise RuntimeError(
            f"Cannot download public adapter {manifest['repo_id']}/{name}. "
            "The author must publish it using scripts/publish_hf.py first; "
            "also check internet access. No evaluator login is needed."
        ) from error
    return verify_adapter(Path(snapshot) / entry["subfolder"], entry)


def resolve_adapter(path: Path | None) -> Path | None:
    """Preserve local custom adapters; auto-fetch only known submission aliases.

    Relative adapters/A4 and adapters/best_sanitizer both resolve to the selected
    A4 checkpoint. Unknown missing paths raise instead of silently substituting.
    """
    if path is None:
        return None
    path = Path(path)
    manifest = load_manifest()
    aliases = {Path("adapters") / name: name for name in manifest["adapters"]}
    aliases[Path("adapters/A4")] = "best_sanitizer"
    name = aliases.get(path)
    if path.is_dir():
        if name:
            return verify_adapter(path, manifest["adapters"][name])
        return path
    if name:
        return download_adapter(name)
    raise ValueError(f"Adapter directory does not exist: {path}")
