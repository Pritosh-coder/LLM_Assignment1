from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.archive_recovery_file import archive, sha256
from scripts.validate_part_a_merged import validate as validate_merged_part_a
from src.common.config import PART_A_RESULT_COLUMNS, PART_A_SWEEP


class Fix5ResilienceTests(unittest.TestCase):
    def _write_part_a(self, root: Path, *, with_sweep: bool = True) -> None:
        rows: list[dict[str, str]] = []
        row_dir = root / "results" / "detail" / "partA_rows"
        row_dir.mkdir(parents=True)
        for index, run_id in enumerate(PART_A_SWEEP, start=1):
            adapter = root / "adapters" / run_id
            adapter.mkdir(parents=True)
            (adapter / "adapter_config.json").write_text(
                json.dumps({"r": 8 if index < 4 else 32}), encoding="utf-8"
            )
            row = {column: "0" for column in PART_A_RESULT_COLUMNS}
            row["run_id"] = run_id
            rows.append(row)
            with (row_dir / f"partA_{run_id}.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=PART_A_RESULT_COLUMNS)
                writer.writeheader()
                writer.writerow(row)
        if with_sweep:
            with (root / "results" / "partA_sweep.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=PART_A_RESULT_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)

    def test_merged_part_a_can_be_reused_without_shard_archives(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_part_a(root)
            report = validate_merged_part_a(root, require_sweep=True)
            self.assertEqual(report["runs"], list(PART_A_SWEEP))
            self.assertTrue(report["sweep_present"])

    def test_missing_sweep_is_allowed_only_before_combination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_part_a(root, with_sweep=False)
            self.assertFalse(validate_merged_part_a(root)["sweep_present"])
            with self.assertRaisesRegex(ValueError, "sweep CSV is missing"):
                validate_merged_part_a(root, require_sweep=True)

    def test_stale_runtime_file_is_archived_without_byte_loss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "partial.json"
            source.write_bytes(b"partial-but-preserved\n")
            expected_hash = sha256(source)
            destination = archive(source, root / "recovery" / "stale")
            self.assertIsNotNone(destination)
            assert destination is not None
            self.assertFalse(source.exists())
            self.assertEqual(sha256(destination), expected_hash)
            self.assertIn(expected_hash[:12], destination.name)

    def test_stage_c1_notebook_has_partial_run_reuse_path(self) -> None:
        notebook = json.loads(
            Path("LLM-Pritosh-Stage-C1-PartB-Train.ipynb").read_text(encoding="utf-8")
        )
        source = "".join(
            "".join(cell.get("source", []))
            if isinstance(cell.get("source", []), list)
            else cell.get("source", "")
            for cell in notebook["cells"]
        )
        self.assertIn("PART_A_ALREADY_MERGED", source)
        self.assertIn("validate_part_a_merged.py", source)
        self.assertIn("PART_A_MERGED_REUSE_OK", source)
        self.assertIn("if not sweep.is_file()", source)
        self.assertIn("resume attach only the prior C1 output", source)
        self.assertIn("2026-09-14-fix5-resumable-staged-part-c", source)

    def test_recovery_directory_is_not_packaged_as_final_results(self) -> None:
        package_source = Path("scripts/package_final_results.py").read_text(encoding="utf-8")
        self.assertNotIn("recovery", package_source)
        validator_source = Path("scripts/validate_final_results_zip.py").read_text(encoding="utf-8")
        self.assertIn("EXPECTED_BUILD_ID", validator_source)


if __name__ == "__main__":
    unittest.main()
