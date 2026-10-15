#!/usr/bin/env python3
"""Validate an existing sanitized Part C corpus and its audit before reuse."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PROJECT_ROOT))
from src.common.data import iter_jsonl
from src.common.sanitize_corpus import validate_sanitized_lineage

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
 return h.hexdigest()
def main()->None:
 p=argparse.ArgumentParser(); p.add_argument('--raw',type=Path,required=True); p.add_argument('--sanitized',type=Path,required=True); p.add_argument('--audit',type=Path,required=True); a=p.parse_args()
 try:
  raw=list(iter_jsonl(a.raw)); san=list(iter_jsonl(a.sanitized)); lineage=validate_sanitized_lineage(raw,san); audit=json.loads(a.audit.read_text(encoding='utf-8'))
  if lineage['rows']!=8007: raise ValueError('sanitized row count mismatch')
  if audit.get('input_sha256')!=sha(a.raw) or audit.get('output_sha256')!=sha(a.sanitized): raise ValueError('sanitization audit hash mismatch')
  for k,v in lineage.items():
   if audit.get(k)!=v: raise ValueError(f'sanitization audit lineage mismatch for {k}')
 except (OSError,ValueError,json.JSONDecodeError) as e: raise SystemExit(f'ERROR: {e}') from e
 print(json.dumps({'status':'PASS',**lineage,'output_sha256':sha(a.sanitized)},indent=2))
if __name__=='__main__': main()
