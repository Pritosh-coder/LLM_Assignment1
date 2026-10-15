#!/usr/bin/env python3
"""Validate the durable C1 checkpoint: Part B, sanitization, and both victims."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from typing import Any

from src.common.config import (
    ALL_LORA_TARGET_MODULES, VICTIM_EPOCHS, VICTIM_LORA_DROPOUT,
    VICTIM_MAX_SEQ_LEN, VICTIM_MODEL_ID, VICTIM_MODEL_REVISION,
)
from src.common.data import iter_jsonl
from src.common.sanitize_corpus import validate_sanitized_lineage


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def obj(path: Path) -> dict[str, Any]:
    value=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value,dict): raise ValueError(f'expected JSON object: {path}')
    return value


def jsonl_count(path: Path) -> int:
    return sum(1 for _ in iter_jsonl(path))


def validate_part_b(root: Path) -> None:
    specs=[
        (root/'results/partB_metrics.json',2000),
        (root/'results/detail/finetuned_ood_metrics.json',1000),
    ]
    for system in ('regex','presidio','zeroshot'):
        specs += [
            (root/f'results/detail/{system}_test_metrics.json',2000),
            (root/f'results/detail/{system}_ood_metrics.json',1000),
        ]
    required={'system','n','exact','overlap_iou50','per_class','leak_rate','over_masking_rate','well_formed_rate'}
    for path,n in specs:
        report=obj(path)
        if int(report.get('n',-1))!=n or not required<=set(report):
            raise ValueError(f'invalid Part B metrics: {path}')
    pred_specs=[
        (root/'results/partB_predictions.jsonl',2000),
        (root/'results/detail/finetuned_ood_predictions.jsonl',1000),
    ]
    for system in ('regex','presidio','zeroshot'):
        pred_specs += [
            (root/f'results/detail/{system}_test_predictions.jsonl',2000),
            (root/f'results/detail/{system}_ood_predictions.jsonl',1000),
        ]
    for path,n in pred_specs:
        if jsonl_count(path)!=n: raise ValueError(f'wrong prediction row count: {path}')


def validate_victim(root: Path, name: str, corpus: Path) -> dict[str, Any]:
    out=root/'adapters'/name
    for required in ('adapter_config.json','victim_plan.json','victim_metrics.json'):
        if not (out/required).is_file(): raise ValueError(f'{name} missing {required}')
    plan=obj(out/'victim_plan.json'); metrics=obj(out/'victim_metrics.json'); cfg=obj(out/'adapter_config.json')
    expected={
        'model':VICTIM_MODEL_ID,'model_revision':VICTIM_MODEL_REVISION,
        'lora_r':32,'lora_alpha':64,'lora_dropout':VICTIM_LORA_DROPOUT,
        'epochs':VICTIM_EPOCHS,'seed':42,'max_seq_len':VICTIM_MAX_SEQ_LEN,
        'train_rows':8007,'heldout_rows':500,
    }
    for k,v in expected.items():
        if plan.get(k)!=v: raise ValueError(f'{name} plan mismatch for {k}: {plan.get(k)!r}')
    if plan.get('target_modules')!=list(ALL_LORA_TARGET_MODULES): raise ValueError(f'{name} target modules mismatch')
    if plan.get('corpus_sha256')!=sha256(corpus): raise ValueError(f'{name} corpus hash mismatch')
    heldout=root/'data/partC_heldout.jsonl'
    if plan.get('heldout_sha256')!=sha256(heldout): raise ValueError(f'{name} heldout hash mismatch')
    for k,v in plan.items():
        if metrics.get(k)!=v: raise ValueError(f'{name} metrics do not preserve plan field {k}')
    if cfg.get('base_model_name_or_path') not in (VICTIM_MODEL_ID,None): raise ValueError(f'{name} adapter base mismatch')
    if int(cfg.get('r',-1))!=32 or int(cfg.get('lora_alpha',-1))!=64: raise ValueError(f'{name} adapter LoRA mismatch')
    return metrics


def validate(
    root: Path,
    *,
    expected_build_id: str = '2026-09-14-fix5-resumable-staged-part-c',
) -> dict[str, Any]:
    build=(root/'KAGGLE_BUILD_ID').read_text(encoding='utf-8').strip()
    if build != expected_build_id:
        raise ValueError(
            f'wrong build: {build!r}; expected active project build {expected_build_id!r}'
        )
    with (root/'results/partA_sweep.csv').open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=6 or [r.get('run_id') for r in rows]!=['A1','A2','A3','A4','A5','A6']:
        raise ValueError('Part A sweep must contain A1-A6 in order')
    selection=obj(root/'results/best_sanitizer.json')
    if selection.get('winner')!='A4': raise ValueError(f"expected frozen winner A4, got {selection.get('winner')!r}")
    validate_part_b(root)
    raw=list(iter_jsonl(root/'data/partC_corpus_raw.jsonl'))
    san=list(iter_jsonl(root/'data/partC_corpus_sanitized.jsonl'))
    lineage=validate_sanitized_lineage(raw,san)
    if lineage.get('rows')!=8007: raise ValueError('sanitized corpus must contain 8007 rows')
    audit=obj(root/'results/partC_sanitization_audit.json')
    if audit.get('output_sha256')!=sha256(root/'data/partC_corpus_sanitized.jsonl'):
        raise ValueError('sanitized corpus hash differs from audit')
    raw_m=validate_victim(root,'victim_raw',root/'data/partC_corpus_raw.jsonl')
    san_m=validate_victim(root,'victim_sanitized',root/'data/partC_corpus_sanitized.jsonl')
    comparable=('model','model_revision','target_modules','lora_r','lora_alpha','lora_dropout','epochs','seed','max_seq_len','learning_rate','lr_scheduler_type','effective_batch_size','heldout_sha256','heldout_rows','heldout_documents')
    for k in comparable:
        if raw_m.get(k)!=san_m.get(k): raise ValueError(f'raw/sanitized victim mismatch for {k}')
    base_raw=float(raw_m['pretrained_heldout_ppl']); base_san=float(san_m['pretrained_heldout_ppl'])
    if abs(base_raw-base_san) > max(1e-6, 1e-6*abs(base_raw)):
        raise ValueError('base heldout perplexity differs between victim runs beyond numerical tolerance')
    return {'status':'PASS','build_id':build,'sanitized_rows':lineage['rows'],'victims':['victim_raw','victim_sanitized']}


def main()->None:
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path('.')); p.add_argument('--expected-build-id',default='2026-09-14-fix5-resumable-staged-part-c'); a=p.parse_args()
    try: out=validate(a.root.resolve(), expected_build_id=a.expected_build_id)
    except (OSError,ValueError,KeyError,json.JSONDecodeError) as e: raise SystemExit(f'ERROR: {e}') from e
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
