#!/usr/bin/env python3
"""Validate one completed Part C attack report before reusing it."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from typing import Any

from src.common.config import VICTIM_MODEL_ID, VICTIM_MODEL_REVISION


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_obj(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'expected JSON object: {path}')
    return value


def validate(report_path: Path, corpus: Path, heldout: Path, expected_victim: str) -> dict[str, Any]:
    if not report_path.is_file():
        raise ValueError(f'report does not exist: {report_path}')
    report = load_obj(report_path)
    required = {'discoverable','untargeted','canary','targeted','heldout_ppl'}
    if not required <= set(report):
        raise ValueError(f'missing attack sections: {sorted(required-set(report))}')
    if report.get('victim') != expected_victim:
        raise ValueError(f"victim mismatch: {report.get('victim')!r} != {expected_victim!r}")
    if report.get('victim_model') != VICTIM_MODEL_ID or report.get('victim_model_revision') != VICTIM_MODEL_REVISION:
        raise ValueError('victim model/revision mismatch')
    if report.get('corpus_sha256') != sha256(corpus):
        raise ValueError('corpus hash mismatch')
    if report.get('heldout_sha256') != sha256(heldout):
        raise ValueError('held-out hash mismatch')
    if int(report.get('num_samples', -1)) != 5000 or int(report.get('seed', -1)) != 42:
        raise ValueError('attack sample count/seed mismatch')
    if int(report['untargeted'].get('n_generated', -1)) != 5000:
        raise ValueError('untargeted generation count mismatch')
    if int(report['canary'].get('n_canaries', -1)) != 20:
        raise ValueError('canary count mismatch')
    for split in ('member','control'):
        if int(report['targeted'].get(split, {}).get('n_prompts', -1)) != 200:
            raise ValueError(f'targeted {split} count mismatch')
    return {'status':'PASS','victim':expected_victim,'report':str(report_path)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--heldout', type=Path, required=True)
    p.add_argument('--expected_victim', required=True)
    args = p.parse_args()
    try:
        out = validate(args.report, args.corpus, args.heldout, args.expected_victim)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        raise SystemExit(f'ERROR: {e}') from e
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
