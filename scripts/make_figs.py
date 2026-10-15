#!/usr/bin/env python3
"""Create the two required figures from completed, schema-checked results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

RUN_IDS = ("A1", "A2", "A3", "A4", "A5", "A6")
PART_A_NUMERIC = (
    "trainable_pct",
    "val_entity_f1",
    "peak_gpu_mem_gb",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build required Part A/C figures.")
    parser.add_argument("--partA_csv", type=Path, required=True)
    parser.add_argument("--partC_raw", type=Path, required=True)
    parser.add_argument("--partC_sanitized", type=Path, required=True)
    parser.add_argument("--partA_output", type=Path, required=True)
    parser.add_argument("--partC_output", type=Path, required=True)
    return parser


def load_part_a_points(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"Part A CSV does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 6 or {row.get("run_id") for row in rows} != set(RUN_IDS):
        raise ValueError("Part A figure requires exactly A1-A6")
    points: list[dict[str, Any]] = []
    for row in rows:
        point: dict[str, Any] = dict(row)
        for column in PART_A_NUMERIC:
            try:
                value = float(row[column])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"run {row.get('run_id')}: invalid {column}"
                ) from error
            if not math.isfinite(value):
                raise ValueError(f"run {row.get('run_id')}: non-finite {column}")
            point[column] = value
        if point["trainable_pct"] <= 0 or point["peak_gpu_mem_gb"] <= 0:
            raise ValueError(f"run {row.get('run_id')}: plot axes must be positive")
        points.append(point)
    return sorted(points, key=lambda point: RUN_IDS.index(point["run_id"]))


def load_exposure_points(path: Path) -> list[tuple[int, float]]:
    if not path.is_file():
        raise ValueError(f"Part C report does not exist: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    try:
        by_repetitions = report["canary"]["by_n_reps"]
        points = [
            (n_reps, float(by_repetitions[str(n_reps)]["mean_exposure"]))
            for n_reps in (1, 4, 16, 64)
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid canary exposure schema: {path}") from error
    if any(not math.isfinite(value) for _, value in points):
        raise ValueError(f"non-finite canary exposure: {path}")
    return points


def _save_atomic(figure: Any, path: Path) -> None:
    if path.suffix.lower() not in {".png", ".pdf"}:
        raise ValueError("figure output must end in .png or .pdf")
    if path.exists():
        raise ValueError(f"figure output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent
    )
    os.close(descriptor)
    try:
        figure.savefig(
            temporary_name,
            dpi=220,
            bbox_inches="tight",
            facecolor="white",
        )
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def make_part_a_figure(points: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#2563EB" if point["quant"] == "none" else "#D97706" for point in points]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    first, second = axes
    first.scatter(
        [point["trainable_pct"] for point in points],
        [point["val_entity_f1"] for point in points],
        c=colors,
        s=70,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    first.set_xscale("log")
    first.set_xlabel("Trainable parameters (%) - log scale")
    first.set_ylabel("Validation entity F1 (IoU >= 0.5)")
    first.set_title("Sanitizer quality vs. adapter size")

    second.scatter(
        [point["peak_gpu_mem_gb"] for point in points],
        [point["val_entity_f1"] for point in points],
        c=colors,
        s=70,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    second.set_xlabel("Peak allocated GPU memory (GiB)")
    second.set_ylabel("Validation entity F1 (IoU >= 0.5)")
    second.set_title("Memory-quality trade-off")
    for axis, x_key in ((first, "trainable_pct"), (second, "peak_gpu_mem_gb")):
        axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
        for point in points:
            axis.annotate(
                point["run_id"],
                (point[x_key], point["val_entity_f1"]),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=9,
                fontweight="semibold",
            )
    figure.suptitle("Part A PEFT sweep", fontsize=14, fontweight="bold")
    _save_atomic(figure, output)
    plt.close(figure)


def make_part_c_figure(
    raw_points: list[tuple[int, float]],
    sanitized_points: list[tuple[int, float]],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for label, points, color, marker in (
        ("V-raw", raw_points, "#DC2626", "o"),
        ("V-san", sanitized_points, "#059669", "s"),
    ):
        axis.plot(
            [value[0] for value in points],
            [value[1] for value in points],
            label=label,
            color=color,
            marker=marker,
            linewidth=2.1,
            markersize=7,
        )
    axis.axhline(
        math.log2(10_000),
        color="#6B7280",
        linestyle=":",
        linewidth=1.4,
        label="Sampled-rank ceiling (13.29 bits)",
    )
    axis.set_xscale("log", base=2)
    axis.set_xticks([1, 4, 16, 64], labels=["1", "4", "16", "64"])
    axis.set_xlabel("Canary repetition count")
    axis.set_ylabel("Mean sampled exposure (bits)")
    axis.set_title("Canary exposure before and after sanitization")
    axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    axis.legend(frameon=False)
    _save_atomic(figure, output)
    plt.close(figure)


def main() -> None:
    args = build_parser().parse_args()
    try:
        part_a = load_part_a_points(args.partA_csv)
        raw = load_exposure_points(args.partC_raw)
        sanitized = load_exposure_points(args.partC_sanitized)
        make_part_a_figure(part_a, args.partA_output)
        make_part_c_figure(raw, sanitized, args.partC_output)
    except (ImportError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error
    print(
        json.dumps(
            {
                "partA_figure": str(args.partA_output),
                "partC_figure": str(args.partC_output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
