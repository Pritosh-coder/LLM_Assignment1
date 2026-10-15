#!/usr/bin/env python3
"""Preserve an invalid/partial runtime artifact before recomputing it."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def archive(path: Path, archive_dir: Path) -> Path | None:
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError(f"recovery artifact is not a file: {path}")
    digest = sha256(path)
    archive_dir.mkdir(parents=True, exist_ok=True)
    destination = archive_dir / f"{path.name}.stale-{digest[:12]}"
    if destination.exists():
        if sha256(destination) != digest:
            raise ValueError(f"recovery destination collision: {destination}")
        path.unlink()
        return destination
    os.replace(path, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive_dir", type=Path, required=True)
    parser.add_argument("paths", type=Path, nargs="+")
    args = parser.parse_args()
    archived: list[str] = []
    try:
        for path in args.paths:
            destination = archive(path, args.archive_dir)
            if destination is not None:
                archived.append(str(destination))
    except (OSError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error
    print(json.dumps({"status": "PASS", "archived": archived}, indent=2))


if __name__ == "__main__":
    main()
