"""CPU gates for adapter selection, Part C combination, and figure inputs."""

from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from scripts.combine_partC_reports import combine_reports
from scripts.make_figs import (
    load_exposure_points,
    load_part_a_points,
    make_part_a_figure,
    make_part_c_figure,
)
from scripts.select_best_adapter import (
    create_adapter_alias,
    rank_sweep_rows,
    select_best,
)
from src.common.config import PART_A_RESULT_COLUMNS, PART_A_SWEEP


def _sweep_row(run_id: str, f1: float, leak: float) -> dict[str, object]:
    config = PART_A_SWEEP[run_id]
    row: dict[str, object] = {
        column: 1 for column in PART_A_RESULT_COLUMNS
    }
    row.update(
        {
            "run_id": run_id,
            "method": config["method"],
            "target_modules": ",".join(config["target_modules"]),
            "lora_r": config["lora_r"],
            "lora_alpha": config["lora_alpha"],
            "quant": config["quant"],
            "trainable_pct": 0.1,
            "peak_gpu_mem_gb": 2.0,
            "val_loss": 0.5,
            "val_entity_f1": f1,
            "val_leak_rate": leak,
        }
    )
    return row


def _attack_report(
    victim: str,
    ppl: float,
    exposure: float,
    leak: float,
) -> dict[str, object]:
    return {
        "victim": victim,
        "heldout_ppl": ppl,
        "discoverable": {},
        "untargeted": {},
        "canary": {
            "by_n_reps": {
                str(n_reps): {"mean_exposure": exposure, "n": 5}
                for n_reps in (1, 4, 16, 64)
            }
        },
        "targeted": {"member": {"greedy_leak_rate": leak}},
    }


class TestM11Artifacts(unittest.TestCase):
    def test_validation_only_tie_break(self) -> None:
        rows = [
            _sweep_row(run_id, 0.8 + index / 100, 0.2)
            for index, run_id in enumerate(PART_A_SWEEP)
        ]
        rows[3].update(val_entity_f1=0.9, val_leak_rate=0.12)
        rows[4].update(val_entity_f1=0.9, val_leak_rate=0.08)
        rows[5].update(val_entity_f1=0.89, val_leak_rate=0.01)
        ranking = rank_sweep_rows(
            [{key: str(value) for key, value in row.items()} for row in rows]
        )
        self.assertEqual(ranking[0]["run_id"], "A5")

    def test_selection_and_part_a_figure_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            csv_path = root / "partA_sweep.csv"
            rows = [
                _sweep_row(run_id, 0.7 + index / 100, 0.2)
                for index, run_id in enumerate(PART_A_SWEEP)
            ]
            rows[-1].update(val_entity_f1=0.95, val_leak_rate=0.04)
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=PART_A_RESULT_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            adapter_dir = root / "adapters" / "A6"
            adapter_dir.mkdir(parents=True)
            (adapter_dir / "adapter_config.json").write_text(
                "{}\n", encoding="utf-8"
            )

            selection = select_best(csv_path, root / "adapters")
            alias_path = root / "adapters" / "best_sanitizer"
            create_adapter_alias(adapter_dir, alias_path)
            points = load_part_a_points(csv_path)

            self.assertEqual(selection["winner"], "A6")
            self.assertEqual(selection["adapter_path"], str(adapter_dir))
            self.assertEqual(alias_path.resolve(), adapter_dir.resolve())
            with self.assertRaisesRegex(ValueError, "already exists"):
                create_adapter_alias(adapter_dir, alias_path)
            self.assertEqual([point["run_id"] for point in points], list(PART_A_SWEEP))

    def test_part_c_combination_and_exposure_input(self) -> None:
        base = _attack_report("V-base", 18.0, 1.0, 0.05)
        raw = _attack_report("V-raw", 12.0, 6.0, 0.4)
        sanitized = _attack_report("V-san", 14.0, 2.0, 0.1)
        combined = combine_reports(base, raw, sanitized)

        self.assertEqual(combined["victim"], "V-raw")
        self.assertEqual(set(combined["models"]), {"V-base", "V-raw", "V-san"})
        self.assertAlmostEqual(
            combined["comparison"]["member_greedy_leak"]["absolute_reduction"],
            0.3,
        )

        with tempfile.TemporaryDirectory() as temporary_dir:
            raw_path = Path(temporary_dir) / "raw.json"
            raw_path.write_text(json.dumps(raw), encoding="utf-8")
            self.assertEqual(
                load_exposure_points(raw_path),
                [(1, 6.0), (4, 6.0), (16, 6.0), (64, 6.0)],
            )

    @unittest.skipUnless(
        importlib.util.find_spec("matplotlib"),
        "matplotlib is installed only in the full assignment environment",
    )
    def test_figure_files_are_rendered(self) -> None:
        rows = [
            _sweep_row(run_id, 0.7 + index / 100, 0.2)
            for index, run_id in enumerate(PART_A_SWEEP)
        ]
        points = [
            {
                **row,
                "trainable_pct": float(row["trainable_pct"]),
                "peak_gpu_mem_gb": float(row["peak_gpu_mem_gb"]),
                "val_entity_f1": float(row["val_entity_f1"]),
            }
            for row in rows
        ]
        exposure = [(1, 1.0), (4, 2.0), (16, 3.0), (64, 4.0)]
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            part_a = root / "partA.png"
            part_c = root / "partC.pdf"
            make_part_a_figure(points, part_a)
            make_part_c_figure(exposure, exposure, part_c)
            self.assertGreater(part_a.stat().st_size, 0)
            self.assertGreater(part_c.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
