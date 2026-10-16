"""Resolve submitted adapters from a public Hub snapshot and verify their bytes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[2] / "config" / "hf_artifacts.json"
PART_A_MANIFEST = MANIFEST.with_name("hf_part_a_artifacts.json")


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def load_part_a_manifest() -> dict:
    return json.loads(PART_A_MANIFEST.read_text(encoding="utf-8"))


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
    return _download_adapter(name, load_manifest(), "scripts/publish_hf.py")


def download_part_a_adapter(name: str) -> Path:
    return _download_adapter(name, load_part_a_manifest(), "scripts/publish_part_a.py")


def _download_adapter(name: str, manifest: dict, publish_script: str) -> Path:
    if name not in manifest["adapters"]:
        raise ValueError(f"Unknown submitted adapter: {name}")
    if not manifest.get("revision"):
        raise RuntimeError(
            f"{name} has not been published and pinned yet. "
            f"The author must run {publish_script} and commit its updated manifest."
        )
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
            f"The author must publish it using {publish_script} first; "
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
    # Keep the established A4/best_sanitizer snapshot stable. Other sweep
    # checkpoints have a separate publication revision and checksum registry.
    part_a = load_part_a_manifest()
    part_a_name = next(
        (run for run in part_a["adapters"] if path == Path("adapters") / run),
        None,
    ) if name is None else None
    if path.is_dir():
        if name:
            return verify_adapter(path, manifest["adapters"][name])
        if part_a_name:
            return verify_adapter(path, part_a["adapters"][part_a_name])
        return path
    if name:
        return download_adapter(name)
    if part_a_name:
        return download_part_a_adapter(part_a_name)
    raise ValueError(f"Adapter directory does not exist: {path}")
