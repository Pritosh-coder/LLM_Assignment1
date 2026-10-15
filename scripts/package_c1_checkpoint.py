#!/usr/bin/env python3
"""Package only the private artifacts needed to continue from C1 to C2."""
from __future__ import annotations
import argparse, hashlib, json, tempfile, zipfile
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.validate_stage_c1 import validate

EXACT=(
 'results/partA_sweep.csv','results/best_sanitizer.json',
 'results/partB_metrics.json','results/partB_predictions.jsonl',
 'results/partC_sanitization_audit.json','data/partC_corpus_sanitized.jsonl',
)

def sha(path:Path)->str:
 h=hashlib.sha256();
 with path.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
 return h.hexdigest()

def main()->None:
 p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path('.')); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); root=a.root.resolve(); active_build=(root/'KAGGLE_BUILD_ID').read_text(encoding='utf-8').strip(); validate(root, expected_build_id=active_build)
 files=[root/x for x in EXACT]
 files += [x for x in (root/'results/detail').rglob('*') if x.is_file()]
 for victim in ('victim_raw','victim_sanitized'):
  base=root/'adapters'/victim
  files += [x for x in base.rglob('*') if x.is_file() and not any(part.startswith('checkpoint-') for part in x.relative_to(base).parts)]
 rels=sorted({x.relative_to(root).as_posix():x for x in files}.items())
 manifest={'build_id':(root/'KAGGLE_BUILD_ID').read_text().strip(),'files':{rel:{'bytes':path.stat().st_size,'sha256':sha(path)} for rel,path in rels}}
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile(prefix='.c1.',suffix='.zip',dir=a.output.parent,delete=False) as tmp: tmp_path=Path(tmp.name)
 try:
  with zipfile.ZipFile(tmp_path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
   z.writestr('C1_CHECKPOINT_MANIFEST.json',json.dumps(manifest,indent=2,sort_keys=True)+'\n')
   for rel,path in rels: z.write(path,rel)
  with zipfile.ZipFile(tmp_path) as z:
   bad=z.testzip()
   if bad: raise RuntimeError(f'ZIP CRC failure: {bad}')
  tmp_path.replace(a.output)
 finally:
  tmp_path.unlink(missing_ok=True)
 print(json.dumps({'output':str(a.output),'bytes':a.output.stat().st_size,'sha256':sha(a.output),'files':len(rels)},indent=2))
if __name__=='__main__': main()
