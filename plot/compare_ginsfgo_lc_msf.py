#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compare GINSFGO loose coupling with GREAT-MSF loose coupling.

All metrics are recomputed on epochs shared by both methods and matched to the
same IMU-centre truth file.  Outputs are written below plot/results.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
PLOT_DIR = ROOT / "plot"
OUTPUT_DIR = PLOT_DIR / "results" / "GINSFGO_LC_vs_MSF"
TIME_TOLERANCE = 0.05


CASES = [
    {
        "case": "Data05",
        "ginsfgo": ROOT
        / "sample_data/Data05_20201128_HG4930_Vehicle_Opensky/result/Vehicle_Opensky_LC.ins",
        "msf": ROOT
        / "sample_data/Data05_20201128_HG4930_Vehicle_Opensky/result/ROVE-GB-HG4930-LCRTK.ins",
        "truth": ROOT
        / "sample_data/Data05_20201128_HG4930_Vehicle_Opensky/groundtruth_HG4930.txt",
    },
    {
        "case": "Data06",
        "ginsfgo": ROOT
        / "sample_data/Data06_20210115_HG4930_UAV_Opensky/result/UAV_Opensky_LC.ins",
        "msf": ROOT
        / "sample_data/Data06_20210115_HG4930_UAV_Opensky/result/ROVE-GB-HG4930-LCRTK.ins",
        "truth": ROOT
        / "sample_data/Data06_20210115_HG4930_UAV_Opensky/groundtruth_HG4930.txt",
    },
    {
        "case": "20211012",
        "ginsfgo": ROOT
        / "sample_data/FGO_20211012/result/SEPT-RTK-TCI-ADIS-FGO_LC.ins",
        "msf": ROOT
        / "sample_data/FGO_20211012/result/SEPT-GREC-ADIS-LCRTK.ins",
        "truth": ROOT
        / "sample_data/FGO_20211012/ref/groundtruth_1012_ADIS.txt",
    },
    {
        "case": "20250928",
        "ginsfgo": ROOT
        / "sample_data/FGO_20250928/result/SEPT-RTK-TCI-MTI-FGO-LC.ins",
        "msf": ROOT
        / "sample_data/FGO_20250928/result/SEPT-GREC-MTI-LCRTK.ins",
        "truth": ROOT
        / "sample_data/FGO_20250928/ref/groundtruth_0928_MTI.txt",
    },
]


def load_evaluator():
    evaluator_path = PLOT_DIR / "evaluate_gins(2).py"
    sys.path.insert(0, str(PLOT_DIR))
    # The evaluator's numerical functions are reused here.  Its plotting
    # module depends on matplotlib, which is not needed for this comparison.
    plot_stub = types.ModuleType("plot")
    plot_stub.plot_position_errors = lambda *args, **kwargs: None
    plot_stub.plot_velocity_errors = lambda *args, **kwargs: None
    plot_stub.plot_attitude_errors = lambda *args, **kwargs: None
    sys.modules["plot"] = plot_stub
    spec = importlib.util.spec_from_file_location("great_eval", evaluator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load evaluator: {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SKIP_FIRST_EPOCHS = 0
    module.YAW_MODE = "negative"
    return module


def rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(np.square(values))))


def mean_norm(*components: np.ndarray) -> float:
    matrix = np.column_stack(components)
    return float(np.mean(np.linalg.norm(matrix, axis=1)))


def rms_norm(*components: np.ndarray) -> float:
    matrix = np.column_stack(components)
    return float(np.sqrt(np.mean(np.sum(np.square(matrix), axis=1))))


def p95_norm(*components: np.ndarray) -> float:
    matrix = np.column_stack(components)
    return float(np.percentile(np.linalg.norm(matrix, axis=1), 95))


def max_norm(*components: np.ndarray) -> float:
    matrix = np.column_stack(components)
    return float(np.max(np.linalg.norm(matrix, axis=1)))


def common_aligned(evaluator, ginsfgo: Path, msf: Path, truth_path: Path):
    for path in (ginsfgo, msf, truth_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    truth = evaluator.load_truth(str(truth_path))
    g_est = evaluator.load_gins_result(str(ginsfgo))
    m_est = evaluator.load_gins_result(str(msf))

    g = evaluator.align_by_time(g_est, truth, TIME_TOLERANCE)
    m = evaluator.align_by_time(m_est, truth, TIME_TOLERANCE)

    g = g.sort_values("sow_gt").drop_duplicates("sow_gt")
    m = m.sort_values("sow_gt").drop_duplicates("sow_gt")
    common = np.intersect1d(g["sow_gt"].to_numpy(), m["sow_gt"].to_numpy())
    if common.size == 0:
        raise ValueError(f"No common epochs: {ginsfgo} vs {msf}")

    g = g[g["sow_gt"].isin(common)].sort_values("sow_gt").reset_index(drop=True)
    m = m[m["sow_gt"].isin(common)].sort_values("sow_gt").reset_index(drop=True)
    if not np.array_equal(g["sow_gt"].to_numpy(), m["sow_gt"].to_numpy()):
        raise AssertionError("Common truth epochs are not identical")

    return truth, g_est, m_est, g, m


def mode_metrics(evaluator, truth, aligned, method: str, case: str) -> dict:
    _, de, dn, du = evaluator.calculate_position_errors(aligned, truth)
    _, dve, dvn, dvu = evaluator.calculate_velocity_errors(aligned, truth)
    _, dh, dp, dr = evaluator.calculate_attitude_errors(aligned)
    pos_norm = np.linalg.norm(np.column_stack((de, dn, du)), axis=1)
    vel_norm = np.linalg.norm(np.column_stack((dve, dvn, dvu)), axis=1)
    worst_pos_index = int(np.argmax(pos_norm))

    fixed = aligned["amb_status"].astype(str).str.casefold().eq("fixed").to_numpy()
    return {
        "case": case,
        "method": method,
        "common_epochs": len(aligned),
        "start_sow": float(aligned["sow_gt"].iloc[0]),
        "end_sow": float(aligned["sow_gt"].iloc[-1]),
        "max_abs_time_match_s": float(aligned["dt_match"].abs().max()),
        "fixed_epochs": int(fixed.sum()),
        "fixed_rate_pct": float(fixed.mean() * 100.0),
        "pos_e_rmse_m": rms(de),
        "pos_n_rmse_m": rms(dn),
        "pos_u_rmse_m": rms(du),
        "pos_h_rmse_m": rms_norm(de, dn),
        "pos_3d_mae_m": mean_norm(de, dn, du),
        "pos_3d_rmse_m": rms_norm(de, dn, du),
        "pos_3d_p95_m": p95_norm(de, dn, du),
        "pos_3d_max_m": max_norm(de, dn, du),
        "pos_3d_worst_sow": float(aligned["sow_gt"].iloc[worst_pos_index]),
        "vel_e_rmse_mps": rms(dve),
        "vel_n_rmse_mps": rms(dvn),
        "vel_u_rmse_mps": rms(dvu),
        "vel_3d_rmse_mps": rms_norm(dve, dvn, dvu),
        "vel_3d_p95_mps": float(np.percentile(vel_norm, 95)),
        "vel_3d_max_mps": float(np.max(vel_norm)),
        "heading_rmse_deg": rms(dh),
        "heading_abs_p95_deg": float(np.percentile(np.abs(dh), 95)),
        "pitch_rmse_deg": rms(dp),
        "pitch_abs_p95_deg": float(np.percentile(np.abs(dp), 95)),
        "roll_rmse_deg": rms(dr),
        "roll_abs_p95_deg": float(np.percentile(np.abs(dr), 95)),
    }


def both_fixed_position_rmse(evaluator, truth, g, m):
    mask = (
        g["amb_status"].astype(str).str.casefold().eq("fixed").to_numpy()
        & m["amb_status"].astype(str).str.casefold().eq("fixed").to_numpy()
    )
    if not mask.any():
        return 0, math.nan, math.nan
    g_fixed = g.loc[mask].reset_index(drop=True)
    m_fixed = m.loc[mask].reset_index(drop=True)
    _, ge, gn, gu = evaluator.calculate_position_errors(g_fixed, truth)
    _, me, mn, mu = evaluator.calculate_position_errors(m_fixed, truth)
    return int(mask.sum()), rms_norm(ge, gn, gu), rms_norm(me, mn, mu)


def chart_font(size: int, bold: bool = False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    path = Path("C:/Windows/Fonts") / name
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def create_position_chart(summary: pd.DataFrame, output: Path):
    pivot = summary.pivot(index="case", columns="method", values="pos_3d_rmse_m")
    order = [item["case"] for item in CASES]
    pivot = pivot.loc[order]

    width, height = 1800, 980
    left, right, top, bottom = 300, 120, 200, 150
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = chart_font(42, bold=True)
    label_font = chart_font(28)
    small_font = chart_font(24)
    value_font = chart_font(23, bold=True)
    blue, orange, grid, ink = "#2563EB", "#D97706", "#E5E7EB", "#20242A"

    draw.text((left, 45), "GINSFGO LC vs GREAT-MSF LC", fill=ink, font=title_font)
    draw.text((left, 105), "3D position RMSE on identical common epochs; logarithmic x-axis", fill="#5B6470", font=small_font)
    draw.ellipse((left, 150, left + 24, 174), fill=blue)
    draw.text((left + 36, 145), "GINSFGO_LC", fill=ink, font=small_font)
    draw.ellipse((left + 245, 150, left + 269, 174), fill=orange)
    draw.text((left + 281, 145), "MSF_LC", fill=ink, font=small_font)

    values = pivot[["GINSFGO_LC", "MSF_LC"]].to_numpy().ravel()
    log_min = math.log10(max(values.min() / 1.8, 0.01))
    log_max = math.log10(values.max() * 1.8)
    plot_width = width - left - right
    plot_height = height - top - bottom

    def xpos(value):
        return left + (math.log10(value) - log_min) / (log_max - log_min) * plot_width

    ticks = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
    for tick in ticks:
        if log_min <= math.log10(tick) <= log_max:
            x = xpos(tick)
            draw.line((x, top, x, height - bottom), fill=grid, width=2)
            text = f"{tick:g}"
            box = draw.textbbox((0, 0), text, font=small_font)
            draw.text((x - (box[2] - box[0]) / 2, height - bottom + 22), text, fill="#5B6470", font=small_font)

    row_gap = plot_height / len(pivot)
    for idx, (case, row) in enumerate(pivot.iterrows()):
        y = top + row_gap * (idx + 0.5)
        g_value = float(row["GINSFGO_LC"])
        m_value = float(row["MSF_LC"])
        gx, mx = xpos(g_value), xpos(m_value)
        draw.text((45, y - 18), case, fill=ink, font=label_font)
        draw.line((gx, y, mx, y), fill="#A8B1BA", width=6)
        draw.ellipse((gx - 13, y - 13, gx + 13, y + 13), fill=blue)
        draw.ellipse((mx - 13, y - 13, mx + 13, y + 13), fill=orange)
        draw.text((gx - 35, y - 52), f"{g_value:.3f}", fill=blue, font=value_font)
        draw.text((mx - 35, y + 20), f"{m_value:.3f}", fill=orange, font=value_font)

    axis_label = "3D position RMSE (m)"
    box = draw.textbbox((0, 0), axis_label, font=label_font)
    draw.text(((width - (box[2] - box[0])) / 2, height - 55), axis_label, fill=ink, font=label_font)
    image.save(output)


def create_fixed_chart(summary: pd.DataFrame, output: Path):
    pivot = summary.pivot(index="case", columns="method", values="fixed_rate_pct")
    order = [item["case"] for item in CASES]
    pivot = pivot.loc[order]
    width, height = 1800, 980
    left, right, top, bottom = 300, 150, 200, 130
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = chart_font(42, bold=True)
    label_font = chart_font(28)
    small_font = chart_font(24)
    value_font = chart_font(22, bold=True)
    blue, orange, grid, ink = "#2563EB", "#D97706", "#E5E7EB", "#20242A"
    plot_width = width - left - right
    plot_height = height - top - bottom

    draw.text((left, 45), "RTK ambiguity Fixed rate", fill=ink, font=title_font)
    draw.text((left, 105), "Rates computed on identical common epochs", fill="#5B6470", font=small_font)
    draw.rectangle((left, 150, left + 28, 174), fill=blue)
    draw.text((left + 42, 145), "GINSFGO_LC", fill=ink, font=small_font)
    draw.rectangle((left + 245, 150, left + 273, 174), fill=orange)
    draw.text((left + 287, 145), "MSF_LC", fill=ink, font=small_font)

    for tick in range(0, 101, 20):
        x = left + tick / 100 * plot_width
        draw.line((x, top, x, height - bottom), fill=grid, width=2)
        draw.text((x - 16, height - bottom + 20), str(tick), fill="#5B6470", font=small_font)

    row_gap = plot_height / len(pivot)
    bar_height = 34
    for idx, (case, row) in enumerate(pivot.iterrows()):
        y = top + row_gap * (idx + 0.5)
        draw.text((45, y - 18), case, fill=ink, font=label_font)
        for offset, method, color in [(-24, "GINSFGO_LC", blue), (24, "MSF_LC", orange)]:
            value = float(row[method])
            y0 = y + offset - bar_height / 2
            x1 = left + value / 100 * plot_width
            draw.rectangle((left, y0, x1, y0 + bar_height), fill=color)
            draw.text((x1 + 12, y0 + 2), f"{value:.1f}%", fill=color, font=value_font)

    axis_label = "Fixed rate (%)"
    box = draw.textbbox((0, 0), axis_label, font=label_font)
    draw.text(((width - (box[2] - box[0])) / 2, height - 45), axis_label, fill=ink, font=label_font)
    image.save(output)


def main():
    evaluator = load_evaluator()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    pair_rows = []

    for item in CASES:
        truth, g_all, m_all, g, m = common_aligned(
            evaluator, item["ginsfgo"], item["msf"], item["truth"]
        )
        g_metrics = mode_metrics(evaluator, truth, g, "GINSFGO_LC", item["case"])
        m_metrics = mode_metrics(evaluator, truth, m, "MSF_LC", item["case"])
        rows.extend([g_metrics, m_metrics])

        both_count, g_both, m_both = both_fixed_position_rmse(evaluator, truth, g, m)
        pair_rows.append(
            {
                "case": item["case"],
                "ginsfgo_source_epochs": len(g_all),
                "msf_source_epochs": len(m_all),
                "common_epochs": len(g),
                "both_fixed_epochs": both_count,
                "ginsfgo_pos_3d_rmse_m": g_metrics["pos_3d_rmse_m"],
                "msf_pos_3d_rmse_m": m_metrics["pos_3d_rmse_m"],
                "msf_over_ginsfgo_pos_rmse": m_metrics["pos_3d_rmse_m"] / g_metrics["pos_3d_rmse_m"],
                "ginsfgo_both_fixed_pos_3d_rmse_m": g_both,
                "msf_both_fixed_pos_3d_rmse_m": m_both,
                "ginsfgo_fixed_rate_pct": g_metrics["fixed_rate_pct"],
                "msf_fixed_rate_pct": m_metrics["fixed_rate_pct"],
                "ginsfgo_vel_3d_rmse_mps": g_metrics["vel_3d_rmse_mps"],
                "msf_vel_3d_rmse_mps": m_metrics["vel_3d_rmse_mps"],
                "ginsfgo_heading_rmse_deg": g_metrics["heading_rmse_deg"],
                "msf_heading_rmse_deg": m_metrics["heading_rmse_deg"],
                "ginsfgo_pitch_rmse_deg": g_metrics["pitch_rmse_deg"],
                "msf_pitch_rmse_deg": m_metrics["pitch_rmse_deg"],
                "ginsfgo_roll_rmse_deg": g_metrics["roll_rmse_deg"],
                "msf_roll_rmse_deg": m_metrics["roll_rmse_deg"],
                "truth_file": str(item["truth"]),
                "ginsfgo_file": str(item["ginsfgo"]),
                "msf_file": str(item["msf"]),
            }
        )

    summary = pd.DataFrame(rows)
    pairwise = pd.DataFrame(pair_rows)
    summary.to_csv(OUTPUT_DIR / "comparison_summary.csv", index=False, encoding="utf-8-sig", float_format="%.9f")
    pairwise.to_csv(OUTPUT_DIR / "comparison_pairwise.csv", index=False, encoding="utf-8-sig", float_format="%.9f")
    create_position_chart(summary, OUTPUT_DIR / "position_3d_rmse_comparison.png")
    create_fixed_chart(summary, OUTPUT_DIR / "fixed_rate_comparison.png")

    display_cols = [
        "case",
        "common_epochs",
        "ginsfgo_pos_3d_rmse_m",
        "msf_pos_3d_rmse_m",
        "msf_over_ginsfgo_pos_rmse",
        "ginsfgo_fixed_rate_pct",
        "msf_fixed_rate_pct",
        "ginsfgo_vel_3d_rmse_mps",
        "msf_vel_3d_rmse_mps",
        "ginsfgo_heading_rmse_deg",
        "msf_heading_rmse_deg",
    ]
    print(pairwise[display_cols].to_string(index=False))
    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
