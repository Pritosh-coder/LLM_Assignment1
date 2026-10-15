import json
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

class Fix9ExtractedVBaseReuseTests(unittest.TestCase):
    def test_active_build_id_is_nonempty(self):
        self.assertTrue((ROOT/'KAGGLE_BUILD_ID').read_text().strip())
    def test_notebook_rebuilds_auto_extracted_vbase(self):
        nb=json.loads((ROOT/'LLM-Pritosh-Stage-C2-Attacks-Use-VBase-FIX9.ipynb').read_text())
        src='\n'.join(''.join(c.get('source',[])) for c in nb['cells'])
        self.assertIn("rglob('VBASE_ATTACK_MANIFEST.json')", src)
        self.assertIn("VBASE=Path('/tmp/pritosh-vbase-attack.zip')", src)
        self.assertIn("z.write(vm_path,'VBASE_ATTACK_MANIFEST.json')", src)
        self.assertIn("for rel in files", src)

if __name__=='__main__': unittest.main()
