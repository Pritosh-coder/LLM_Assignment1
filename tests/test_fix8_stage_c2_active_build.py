from pathlib import Path
import unittest


class Fix8StageC2ActiveBuildTests(unittest.TestCase):
    def test_run_all_validates_stage_c1_against_active_build(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "run_all.sh").read_text(encoding="utf-8")
        self.assertIn("active_build_id=\"$(tr -d '\\r\\n' < KAGGLE_BUILD_ID)\"", source)
        self.assertGreaterEqual(source.count("--expected-build-id \"$active_build_id\""), 2)

    def test_final_zip_validator_tracks_active_build(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "scripts" / "validate_final_results_zip.py").read_text(encoding="utf-8")
        self.assertIn("PROJECT_ROOT/'KAGGLE_BUILD_ID'", source)
        active = (root / "KAGGLE_BUILD_ID").read_text(encoding="utf-8").strip()
        self.assertTrue(active)
        # The validator must not be pinned to any superseded release-specific build ID.
        self.assertNotIn("2026-09-14-fix8-stage-c2-active-build-validation", source)
        self.assertNotIn("2026-09-14-fix9-extracted-vbase-reuse", source)

if __name__ == "__main__":
    unittest.main()
