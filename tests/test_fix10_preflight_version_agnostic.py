from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

class Fix10PreflightVersionAgnosticTests(unittest.TestCase):
    def test_notebook_build_id_matches_active_project(self):
        import json
        active = (ROOT / "KAGGLE_BUILD_ID").read_text().strip()
        nb = json.loads((ROOT / "LLM-Pritosh-Stage-C2-Attacks-Use-VBase-FIX10.ipynb").read_text(encoding="utf-8"))
        src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
        self.assertIn(f"BUILD_ID_EXPECTED = '{active}'", src)

    def test_no_stale_release_specific_final_zip_assertion(self):
        source = (ROOT / "tests" / "test_fix8_stage_c2_active_build.py").read_text(encoding="utf-8")
        self.assertNotIn('self.assertIn("2026-09-14-fix8-stage-c2-active-build-validation", source)', source)

    def test_final_zip_validator_reads_active_build_id(self):
        source = (ROOT / "scripts" / "validate_final_results_zip.py").read_text(encoding="utf-8")
        self.assertIn("EXPECTED_BUILD_ID=(PROJECT_ROOT/'KAGGLE_BUILD_ID').read_text", source)

if __name__ == "__main__":
    unittest.main()
