from __future__ import annotations

import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.combine_partA_results import combine
from src.common.config import PART_A_RESULT_COLUMNS, PART_A_SWEEP


class ParallelPartATests(unittest.TestCase):
    def test_part_c_uses_two_isolated_gpu_workers(self) -> None:
        runner = Path("run_all.sh").read_text(encoding="utf-8")
        self.assertIn(
            "run_victim 0 data/partC_corpus_raw.jsonl adapters/victim_raw &", runner
        )
        self.assertIn(
            "run_victim 1 data/partC_corpus_sanitized.jsonl adapters/victim_sanitized &", runner
        )
        self.assertIn("--resume_if_available", runner)
        self.assertIn("--stage-c1", runner)
        self.assertIn("--stage-c2", runner)
        self.assertIn("run_attack 0 data/partC_corpus_raw.jsonl", runner)
        self.assertIn("run_attack 1 data/partC_corpus_raw.jsonl", runner)

        for notebook_name in (
            "LLM-Pritosh-Stage-C1-PartB-Train.ipynb",
            "LLM-Pritosh-Stage-C2-Attacks.ipynb",
        ):
            notebook = json.loads(Path(notebook_name).read_text(encoding="utf-8"))
            source = "".join("".join(cell.get("source", [])) if isinstance(cell.get("source", []), list) else cell.get("source", "") for cell in notebook["cells"])
            self.assertIn("2026-09-14-fix5-resumable-staged-part-c", source)
            self.assertIn("RUN_ENV.pop('CUDA_VISIBLE_DEVICES'", source)
        c1 = json.loads(Path("LLM-Pritosh-Stage-C1-PartB-Train.ipynb").read_text(encoding="utf-8"))
        c1_source = "".join("".join(cell.get("source", [])) if isinstance(cell.get("source", []), list) else cell.get("source", "") for cell in c1["cells"])
        self.assertIn("--stage-c1", c1_source)
        self.assertIn("pritosh-c1-checkpoint.zip", c1_source)
        c2 = json.loads(Path("LLM-Pritosh-Stage-C2-Attacks.ipynb").read_text(encoding="utf-8"))
        c2_source = "".join("".join(cell.get("source", [])) if isinstance(cell.get("source", []), list) else cell.get("source", "") for cell in c2["cells"])
        self.assertIn("--stage-c2", c2_source)
        self.assertIn("pritosh-final-results.zip", c2_source)

    def test_shard_runner_accepts_only_three_frozen_shards(self) -> None:
        result = subprocess.run(
            ["bash", "scripts/run_partA_shard.sh", "4"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("{1|2|3}", result.stderr)

    def test_combiner_preserves_mandated_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "rows"
            inputs.mkdir()
            for run_id in reversed(PART_A_SWEEP):
                with (inputs / f"partA_{run_id}.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=PART_A_RESULT_COLUMNS)
                    writer.writeheader()
                    row = {column: "0" for column in PART_A_RESULT_COLUMNS}
                    row["run_id"] = run_id
                    writer.writerow(row)
            output = root / "partA_sweep.csv"
            combine(inputs, output)
            with output.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["run_id"] for row in rows], list(PART_A_SWEEP))

    def test_combiner_rejects_missing_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "missing result"):
                combine(Path(directory), Path(directory) / "combined.csv")


if __name__ == "__main__":
    unittest.main()
