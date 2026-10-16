"""Optional anonymous download check; evaluation CLIs also download on demand."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.hub_artifacts import (
    download_adapter, load_manifest, download_part_a_adapter, load_part_a_manifest,
)
from src.common.config import (
    SANITIZER_MODEL_ID, SANITIZER_MODEL_REVISION,
    VICTIM_MODEL_ID, VICTIM_MODEL_REVISION,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-base-models", action="store_true")
    parser.add_argument("--part-a", action="store_true", help="Also fetch all registered Part A sweep adapters")
    args = parser.parse_args()
    for name in load_manifest()["adapters"]:
        print(f"{name}: {download_adapter(name)}")
    if args.part_a:
        for name in load_part_a_manifest()["adapters"]:
            print(f"{name}: {download_part_a_adapter(name)}")
    if args.include_base_models:
        from huggingface_hub import snapshot_download
        for model, revision in [
            (SANITIZER_MODEL_ID, SANITIZER_MODEL_REVISION),
            (VICTIM_MODEL_ID, VICTIM_MODEL_REVISION),
        ]:
            print(snapshot_download(repo_id=model, revision=revision, token=False))


if __name__ == "__main__":
    main()
