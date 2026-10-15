#!/usr/bin/env python3
"""Validate one Part B metrics/predictions pair before safe reuse."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PROJECT_ROOT))

def main()->None:
 p=argparse.ArgumentParser(); p.add_argument('--metrics',type=Path,required=True); p.add_argument('--predictions',type=Path,required=True); p.add_argument('--system',required=True); p.add_argument('--n',type=int,required=True); a=p.parse_args()
 try:
  m=json.loads(a.metrics.read_text(encoding='utf-8'))
  req={'system','n','exact','overlap_iou50','per_class','leak_rate','over_masking_rate','well_formed_rate'}
  if not isinstance(m,dict) or not req<=set(m): raise ValueError('invalid metrics schema')
  if m.get('system')!=a.system or int(m.get('n',-1))!=a.n: raise ValueError('system/count mismatch')
  count=0
  with a.predictions.open(encoding='utf-8') as f:
   for line in f:
    if not line.strip(): continue
    row=json.loads(line)
    if not isinstance(row,dict): raise ValueError('prediction row is not an object')
    count+=1
  if count!=a.n: raise ValueError(f'prediction count {count} != {a.n}')
 except (OSError,ValueError,json.JSONDecodeError) as e: raise SystemExit(f'ERROR: {e}') from e
 print(json.dumps({'status':'PASS','system':a.system,'n':a.n},indent=2))
if __name__=='__main__': main()
