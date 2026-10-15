#!/usr/bin/env python3
"""Validate one completed victim adapter/metrics set before skipping retraining."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PROJECT_ROOT))
from scripts.validate_stage_c1 import validate_victim

def main()->None:
 p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path('.')); p.add_argument('--name',choices=['victim_raw','victim_sanitized'],required=True); p.add_argument('--corpus',type=Path,required=True); a=p.parse_args()
 try: metrics=validate_victim(a.root.resolve(),a.name,a.corpus.resolve())
 except (OSError,ValueError,KeyError,json.JSONDecodeError) as e: raise SystemExit(f'ERROR: {e}') from e
 print(json.dumps({'status':'PASS','name':a.name,'heldout_ppl':metrics['heldout_ppl']},indent=2))
if __name__=='__main__': main()
