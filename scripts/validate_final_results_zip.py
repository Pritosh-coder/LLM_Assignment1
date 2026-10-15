#!/usr/bin/env python3
"""Validate the downloaded final-results ZIP without trusting filenames or cached outputs."""
from __future__ import annotations
import argparse, csv, hashlib, json, tempfile, zipfile, sys
from pathlib import Path, PurePosixPath
from typing import Any

PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT))

REQUIRED=(
 'results/partA_sweep.csv','results/partA_fig.png',
 'results/partB_metrics.json','results/partB_predictions.jsonl',
 'results/partC_attacks.json','results/partC_exposure.png',
)
FORBIDDEN_PARTS=('wandb','checkpoint-')
FORBIDDEN_SUFFIXES=('.safetensors','.bin','.pt','.pth')
EXPECTED_BUILD_ID=(PROJECT_ROOT/'KAGGLE_BUILD_ID').read_text(encoding='utf-8').strip()

def digest(data:bytes)->str: return hashlib.sha256(data).hexdigest()
def file_sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
 return h.hexdigest()
def obj(path:Path)->dict[str,Any]:
 v=json.loads(path.read_text(encoding='utf-8'))
 if not isinstance(v,dict): raise ValueError(f'expected JSON object: {path}')
 return v

def validate_extracted(root:Path)->dict[str,Any]:
 for rel in REQUIRED:
  p=root/rel
  if not p.is_file() or p.stat().st_size<=0: raise ValueError(f'missing/empty required artifact: {rel}')
 with (root/'results/partA_sweep.csv').open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
 if len(rows)!=6 or [r.get('run_id') for r in rows]!=['A1','A2','A3','A4','A5','A6']: raise ValueError('invalid Part A sweep')
 b=obj(root/'results/partB_metrics.json')
 if int(b.get('n',-1))!=2000: raise ValueError('Part B main metrics must have n=2000')
 pred=sum(1 for line in (root/'results/partB_predictions.jsonl').open(encoding='utf-8') if line.strip())
 if pred!=2000: raise ValueError(f'Part B prediction count is {pred}, expected 2000')
 c=obj(root/'results/partC_attacks.json')
 if set(c.get('models',{}))!={'V-base','V-raw','V-san'}: raise ValueError('combined Part C report missing one or more models')
 for label in ('base','raw','sanitized'):
  p=root/f'results/detail/partC_attacks_{label}.json'
  r=obj(p)
  if int(r.get('untargeted',{}).get('n_generated',-1))!=5000: raise ValueError(f'{label}: untargeted count mismatch')
  if int(r.get('canary',{}).get('n_canaries',-1))!=20: raise ValueError(f'{label}: canary count mismatch')
  for split in ('member','control'):
   if int(r.get('targeted',{}).get(split,{}).get('n_prompts',-1))!=200: raise ValueError(f'{label}: targeted {split} mismatch')
 return {'partA_rows':6,'partB_predictions':pred,'partC_models':3}

def main()->None:
 p=argparse.ArgumentParser(); p.add_argument('--archive',type=Path,required=True); a=p.parse_args()
 zip_sha=file_sha(a.archive)
 with zipfile.ZipFile(a.archive) as z:
  names=z.namelist()
  if len(names)!=len(set(names)): raise SystemExit('ERROR: duplicate ZIP members')
  try: manifest=json.loads(z.read('FINAL_RESULTS_MANIFEST.json'))
  except KeyError as e: raise SystemExit('ERROR: FINAL_RESULTS_MANIFEST.json missing') from e
  if manifest.get('build_id')!=EXPECTED_BUILD_ID: raise SystemExit(f"ERROR: final-results build {manifest.get('build_id')!r} != expected {EXPECTED_BUILD_ID!r}")
  files=manifest.get('files')
  if not isinstance(files,dict) or not files: raise SystemExit('ERROR: invalid manifest')
  expected=set(files)|{'FINAL_RESULTS_MANIFEST.json'}
  if set(names)!=expected: raise SystemExit(f'ERROR: ZIP member set differs from manifest: extra={sorted(set(names)-expected)} missing={sorted(expected-set(names))}')
  for rel,meta in files.items():
   pp=PurePosixPath(rel)
   if pp.is_absolute() or '..' in pp.parts: raise SystemExit(f'ERROR: unsafe member {rel!r}')
   if rel.startswith(('data/','adapters/')) or rel.endswith(FORBIDDEN_SUFFIXES) or any(x in rel for x in FORBIDDEN_PARTS): raise SystemExit(f'ERROR: forbidden final-results member: {rel}')
   data=z.read(rel)
   if len(data)!=int(meta['bytes']) or digest(data)!=meta['sha256']: raise SystemExit(f'ERROR: hash/size mismatch for {rel}')
  bad=z.testzip()
  if bad: raise SystemExit(f'ERROR: ZIP CRC failure: {bad}')
  with tempfile.TemporaryDirectory() as td:
   root=Path(td)
   for rel in files:
    target=root/rel; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(z.read(rel))
   summary=validate_extracted(root)
 print(json.dumps({'status':'PASS','archive':str(a.archive),'sha256':zip_sha,'bytes':a.archive.stat().st_size,'build_id':manifest.get('build_id'),**summary},indent=2))
if __name__=='__main__': main()
