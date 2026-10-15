#!/usr/bin/env python3
"""Fail-closed restore of a C1 checkpoint into a fresh compatible project tree."""
from __future__ import annotations
import argparse, hashlib, json, zipfile
from pathlib import Path, PurePosixPath
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.validate_stage_c1 import validate

def digest(data:bytes)->str: return hashlib.sha256(data).hexdigest()

def main()->None:
 p=argparse.ArgumentParser(); p.add_argument('--archive',type=Path,required=True); p.add_argument('--root',type=Path,default=Path('.')); p.add_argument('--source-build-id'); a=p.parse_args(); root=a.root.resolve()
 build=(root/'KAGGLE_BUILD_ID').read_text(encoding='utf-8').strip()
 with zipfile.ZipFile(a.archive) as z:
  try: manifest=json.loads(z.read('C1_CHECKPOINT_MANIFEST.json'))
  except KeyError as e: raise SystemExit('ERROR: checkpoint manifest missing') from e
  expected_source_build=a.source_build_id or build
  if manifest.get('build_id')!=expected_source_build: raise SystemExit(f"ERROR: checkpoint build {manifest.get('build_id')!r} != expected source build {expected_source_build!r}")
  files=manifest.get('files');
  if not isinstance(files,dict) or not files: raise SystemExit('ERROR: invalid checkpoint manifest')
  names=set(z.namelist())
  expected=set(files)|{'C1_CHECKPOINT_MANIFEST.json'}
  if names!=expected: raise SystemExit(f'ERROR: archive members differ from manifest: extra={sorted(names-expected)} missing={sorted(expected-names)}')
  for rel,meta in files.items():
   pp=PurePosixPath(rel)
   if pp.is_absolute() or '..' in pp.parts: raise SystemExit(f'ERROR: unsafe member {rel!r}')
   data=z.read(rel)
   if len(data)!=int(meta['bytes']) or digest(data)!=meta['sha256']: raise SystemExit(f'ERROR: hash/size mismatch for {rel}')
   target=root/Path(*pp.parts)
   if target.exists(): raise SystemExit(f'ERROR: refusing to overwrite existing file: {target}')
   target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(data)
 validate(root, expected_build_id=build)
 print(json.dumps({'status':'PASS','archive':str(a.archive),'restored_files':len(files)},indent=2))
if __name__=='__main__': main()
