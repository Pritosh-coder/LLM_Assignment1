#!/usr/bin/env python3
"""Create a manifest-backed pritosh-final-results.zip from validated measured outputs."""
from __future__ import annotations
import argparse, hashlib, json, tempfile, zipfile, sys
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT))
from scripts.validate_submission import validate as validate_submission

EXACT=(
 'results/partA_sweep.csv','results/partA_fig.png',
 'results/partB_metrics.json','results/partB_predictions.jsonl',
 'results/partC_attacks.json','results/partC_exposure.png',
 'results/best_sanitizer.json','results/partC_sanitization_audit.json',
)

def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
 return h.hexdigest()

def main()->None:
 p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path('.')); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); root=a.root.resolve()
 validate_submission(root,require_report=False)
 files=[root/x for x in EXACT if (root/x).is_file()]
 files += [x for x in (root/'results/detail').rglob('*') if x.is_file()]
 rels=sorted({x.relative_to(root).as_posix():x for x in files}.items())
 manifest={'build_id':(root/'KAGGLE_BUILD_ID').read_text(encoding='utf-8').strip(),'files':{rel:{'bytes':path.stat().st_size,'sha256':sha(path)} for rel,path in rels}}
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile(prefix='.final.',suffix='.zip',dir=a.output.parent,delete=False) as tmp: tmp_path=Path(tmp.name)
 try:
  with zipfile.ZipFile(tmp_path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
   z.writestr('FINAL_RESULTS_MANIFEST.json',json.dumps(manifest,indent=2,sort_keys=True)+'\n')
   for rel,path in rels: z.write(path,rel)
  with zipfile.ZipFile(tmp_path) as z:
   bad=z.testzip()
   if bad: raise RuntimeError(f'ZIP CRC failure: {bad}')
  tmp_path.replace(a.output)
 finally: tmp_path.unlink(missing_ok=True)
 print(json.dumps({'output':str(a.output),'bytes':a.output.stat().st_size,'sha256':sha(a.output),'files':len(rels)},indent=2))
if __name__=='__main__': main()
