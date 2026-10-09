#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compare position-only and joint position-velocity GINSFGO LC on matched epochs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CLONE = Path(r"D:\Course Study\GREAT-FGO-main-v")
OUTPUT = ROOT / "plot" / "results" / "velocity_factor_ablation_20260929"
TOLERANCE_S = 0.05

CASES = [
    (
        "Data05",
        "Data05_20201128_HG4930_Vehicle_Opensky",
        "Vehicle_Opensky_LC.ins",
        "Vehicle_Opensky_LC_velocity.ins",
        "groundtruth_HG4930.txt",
    ),
    (
        "Data06",
        "Data06_20210115_HG4930_UAV_Opensky",
        "UAV_Opensky_LC.ins",
        "UAV_Opensky_LC_velocity.ins",
        "groundtruth_HG4930.txt",
    ),
    (
        "20211012",
        "FGO_20211012",
        "SEPT-RTK-TCI-ADIS-FGO_LC.ins",
        "SEPT-RTK-LC-ADIS-FGO-velocity.ins",
        "ref/groundtruth_1012_ADIS.txt",
    ),
    (
        "20250928",
        "FGO_20250928",
        "SEPT-RTK-TCI-MTI-FGO-LC.ins",
        "SEPT-RTK-LC-MTI-FGO-velocity.ins",
        "ref/groundtruth_0928_MTI.txt",
    ),
]


def evaluator_module():
    path = ROOT / "plot" / "compare_ginsfgo_lc_msf.py"
    spec = importlib.util.spec_from_file_location("lc_metrics", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_evaluator()


def rms(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(np.square(values))))


def norm_stats(*components):
    values = np.linalg.norm(np.column_stack(components), axis=1)
    return {
        "rmse": rms(values),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "mae": float(np.mean(values)),
    }


def calculate_errors(evaluator, truth, data):
    _, east, north, up = evaluator.calculate_position_errors(data, truth)
    _, ve, vn, vu = evaluator.calculate_velocity_errors(data, truth)
    _, heading, pitch, roll = evaluator.calculate_attitude_errors(data)
    return {
        "position": (east, north, up),
        "velocity": (ve, vn, vu),
        "attitude": (heading, pitch, roll),
    }


def metrics(data, errors):
    pos = norm_stats(*errors["position"])
    vel = norm_stats(*errors["velocity"])
    heading, pitch, roll = errors["attitude"]
    fixed = data["amb_status"].astype(str).str.casefold().eq("fixed").to_numpy()
    return {
        "epochs": len(data),
        "first_sow": float(data["sow_gt"].iloc[0]),
        "last_sow": float(data["sow_gt"].iloc[-1]),
        "max_abs_match_s": float(data["dt_match"].abs().max()),
        "pos_3d_rmse_m": pos["rmse"],
        "pos_3d_p95_m": pos["p95"],
        "pos_3d_max_m": pos["max"],
        "pos_3d_mean_m": pos["mae"],
        "vel_3d_rmse_mps": vel["rmse"],
        "vel_3d_p95_mps": vel["p95"],
        "vel_3d_max_mps": vel["max"],
        "heading_rmse_deg": rms(heading),
        "pitch_rmse_deg": rms(pitch),
        "roll_rmse_deg": rms(roll),
        "fixed_count": int(fixed.sum()),
        "fixed_rate_pct": float(fixed.mean() * 100),
    }


def main():
    evaluator = evaluator_module()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []

    for case, folder, old_name, new_name, truth_name in CASES:
        old_path = ROOT / "sample_data" / folder / "result" / old_name
        new_path = CLONE / "sample_data" / folder / "result" / new_name
        truth_path = ROOT / "sample_data" / folder / truth_name
        clone_truth = CLONE / "sample_data" / folder / truth_name
        for path in (old_path, new_path, truth_path, clone_truth):
            if not path.is_file():
                raise FileNotFoundError(path)
        if truth_path.read_bytes() != clone_truth.read_bytes():
            raise AssertionError(f"Truth differs between repositories: {case}")

        truth = evaluator.load_truth(str(truth_path))
        old_raw = evaluator.load_gins_result(str(old_path))
        new_raw = evaluator.load_gins_result(str(new_path))
        old = evaluator.align_by_time(old_raw, truth, TOLERANCE_S)
        new = evaluator.align_by_time(new_raw, truth, TOLERANCE_S)
        if old["sow_gt"].duplicated().any() or new["sow_gt"].duplicated().any():
            raise AssertionError(f"Duplicate truth epoch: {case}")

        shared = np.intersect1d(old["sow_gt"], new["sow_gt"])
        old = old[old["sow_gt"].isin(shared)].sort_values("sow_gt").reset_index(drop=True)
        new = new[new["sow_gt"].isin(shared)].sort_values("sow_gt").reset_index(drop=True)
        if len(old) != len(new) or not np.array_equal(old["sow_gt"], new["sow_gt"]):
            raise AssertionError(f"Epoch mismatch: {case}")
        max_est_epoch_offset = float(
            np.max(np.abs(old["sow_est"].to_numpy() - new["sow_est"].to_numpy()))
        )
        status_mismatch = int(
            (old["amb_status"].to_numpy() != new["amb_status"].to_numpy()).sum()
        )

        old_errors = calculate_errors(evaluator, truth, old)
        new_errors = calculate_errors(evaluator, truth, new)
        old_metrics = metrics(old, old_errors)
        new_metrics = metrics(new, new_errors)

        for label, data, item in (
            ("position_only", old, old_metrics),
            ("position_velocity", new, new_metrics),
        ):
            rows.append(
                {
                    "case": case,
                    "mode": label,
                    "source_file": str(old_path if label == "position_only" else new_path),
                    "truth_file": str(truth_path),
                    "raw_epochs": len(old_raw if label == "position_only" else new_raw),
                    "max_est_epoch_offset_s": max_est_epoch_offset,
                    "amb_status_mismatch_epochs": status_mismatch,
                    **item,
                }
            )

        common_fixed = (
            old["amb_status"].str.casefold().eq("fixed").to_numpy()
            & new["amb_status"].str.casefold().eq("fixed").to_numpy()
        )
        early = (old["sow_gt"].to_numpy() - old["sow_gt"].iloc[0]) < 300.0
        for label, mask in (
            ("both_fixed", common_fixed),
            ("first_300_s", early),
            ("after_300_s", ~early),
        ):
            if not mask.any():
                continue
            for method, data in (("position_only", old), ("position_velocity", new)):
                subset = data.loc[mask].reset_index(drop=True)
                err = calculate_errors(evaluator, truth, subset)
                row = metrics(subset, err)
                rows.append(
                    {
                        "case": case,
                        "mode": method,
                        "subset": label,
                        "source_file": str(old_path if method == "position_only" else new_path),
                        "truth_file": str(truth_path),
                        "raw_epochs": len(old_raw if method == "position_only" else new_raw),
                        "max_est_epoch_offset_s": max_est_epoch_offset,
                        "amb_status_mismatch_epochs": status_mismatch,
                        **row,
                    }
                )
        print(
            f"{case}: {len(shared)} common epochs, "
            f"max estimate time offset {max_est_epoch_offset:.6f} s, "
            f"status mismatches {status_mismatch}"
        )

    output = pd.DataFrame(rows)
    output["subset"] = output["subset"].fillna("all")
    output.to_csv(OUTPUT / "ablation_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.9f")
    display = output[output["subset"] == "all"][
        [
            "case",
            "mode",
            "epochs",
            "pos_3d_rmse_m",
            "pos_3d_p95_m",
            "pos_3d_max_m",
            "vel_3d_rmse_mps",
            "heading_rmse_deg",
            "pitch_rmse_deg",
            "roll_rmse_deg",
            "fixed_rate_pct",
        ]
    ]
    print(display.to_string(index=False))
    print(OUTPUT / "ablation_metrics.csv")


if __name__ == "__main__":
    main()
