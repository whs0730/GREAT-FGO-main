#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按 Data05 HG4930 真值格式评定 CT/LC，输出共同历元误差和曲线。"""

import argparse
import bisect
import csv
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
TRUTH = ROOT / "sample_data/Data05_20201128_HG4930_Vehicle_Opensky/groundtruth_HG4930.txt"
TOLERANCE_S = 0.05


def wrap(angle):
    return (angle + 180.0) % 360.0 - 180.0


def validate(rows):
    if not rows or any(b["t"] <= a["t"] for a, b in zip(rows, rows[1:])):
        raise ValueError("输入必须非空且时间严格递增")
    for row in rows:
        if not math.isfinite(row["t"]) or not all(
                math.isfinite(value) for key in ("pos", "vel", "att") for value in row[key]):
            raise ValueError("输入包含非有限状态")


def read_ins(path):
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 21:
            raise ValueError(f"INS 列数错误：{path}")
        rows.append({"t": float(fields[0]), "pos": list(map(float, fields[1:4])),
                     "vel": list(map(float, fields[4:7])),
                     "att": [float(fields[7]), float(fields[8]), -float(fields[9])],
                     "fixed": fields[19] == "Fixed"})
    validate(rows)
    return rows


def read_truth(path):
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if "X-ECEF" not in lines[0] or "Heading" not in lines[0]:
        raise ValueError("真值表头不符合 Data05 HG4930 格式")
    rows = []
    for line in lines[2:]:
        if not line.strip():
            continue
        fields = line.split()
        if int(float(fields[0])) != 2133:
            raise ValueError("Data05 应使用 GPS 周 2133")
        # 经纬度为度分秒；不能将分、秒当作后续独立物理量。
        latitude = float(fields[5]) + float(fields[6]) / 60 + float(fields[7]) / 3600
        longitude = float(fields[8]) + float(fields[9]) / 60 + float(fields[10]) / 3600
        rows.append({"t": float(fields[1]), "pos": list(map(float, fields[2:5])),
                     "vel": list(map(float, fields[15:18])),
                     "att": [float(fields[25]), float(fields[26]), float(fields[24])],
                     "lat": math.radians(latitude), "lon": math.radians(longitude),
                     "fixed": fields[27] == "Fixed"})
    validate(rows)
    # 首行值来自文件本身，用于阻止列布局被误换成 ROVE 格式。
    if rows[0]["pos"] != [-2280189.184, 5008229.545, 3213982.848]:
        raise ValueError("Data05 首历元位置与预期不符")
    return rows


def same_config(ct_xml, lc_xml):
    normalized = []
    for path in (ct_xml, lc_xml):
        root = ET.parse(path).getroot()
        root.remove(root.find("outputs"))
        root.find("fgomsf_sensors/msf_type").text = "MODE"
        normalized.append([(node.tag, sorted(node.attrib.items()), (node.text or "").strip())
                           for node in root.iter()])
    if normalized[0] != normalized[1]:
        raise ValueError("CT/LC 配置除模式与输出外存在差异")


def interpolate(rows, times, t):
    index = bisect.bisect_left(times, t)
    if index < len(rows) and abs(times[index] - t) < 1e-9:
        return rows[index]
    if index == 0 or index == len(rows):
        return None
    first, second = rows[index - 1:index + 1]
    if second["t"] - first["t"] > 1.1:
        return None
    weight = (t - first["t"]) / (second["t"] - first["t"])
    row = {"t": t}
    for key in ("pos", "vel"):
        row[key] = [a + weight * (b - a) for a, b in zip(first[key], second[key])]
    # 当前只有毫秒级时间修正；姿态按最短角度差插值，并保留最近邻敏感性结果。
    row["att"] = [a + weight * wrap(b - a) for a, b in zip(first["att"], second["att"])]
    row["fixed"] = min((first, second), key=lambda item: abs(item["t"] - t))["fixed"]
    return row


def enu(vector, reference):
    sl, cl = math.sin(reference["lon"]), math.cos(reference["lon"])
    sb, cb = math.sin(reference["lat"]), math.cos(reference["lat"])
    x, y, z = vector
    return [-sl*x + cl*y, -sb*cl*x - sb*sl*y + cb*z, cb*cl*x + cb*sl*y + sb*z]


def errors(estimate, truth, reference):
    position = np.subtract(estimate["pos"], truth["pos"])
    velocity = np.subtract(estimate["vel"], truth["vel"])
    p, v = enu(position, reference), enu(velocity, reference)
    if not np.allclose([np.linalg.norm(position), np.linalg.norm(velocity)],
                       [np.linalg.norm(p), np.linalg.norm(v)], rtol=1e-12, atol=1e-12):
        raise ValueError("ECEF/ENU 误差模长不一致")
    return {"t": truth["t"], "p": p, "v": v,
            "a": [wrap(a - b) for a, b in zip(estimate["att"], truth["att"])],
            "fixed": estimate["fixed"], "truth_fixed": truth["fixed"],
            "speed": float(np.linalg.norm(truth["vel"]))}


def metrics(rows):
    result = {"epochs": len(rows)}
    if not rows:
        return result
    result.update({"first_sow": rows[0]["t"], "last_sow": rows[-1]["t"],
                   "fixed_rate_pct": 100 * sum(row["fixed"] for row in rows) / len(rows),
                   "truth_fixed_count": sum(row["truth_fixed"] for row in rows)})
    for key, prefix, unit in (("p", "pos", "m"), ("v", "vel", "mps"), ("a", "att", "deg")):
        values = np.array([row[key] for row in rows])
        for index, axis in enumerate(("pitch", "roll", "heading") if key == "a" else ("e", "n", "u")):
            column = values[:, index]
            result[f"{prefix}_{axis}_rmse_{unit}"] = float(np.sqrt(np.mean(column**2)))
            result[f"{prefix}_{axis}_bias_{unit}"] = float(np.mean(column))
            result[f"{prefix}_{axis}_maxabs_{unit}"] = float(np.max(np.abs(column)))
        if key != "a":
            norm = np.linalg.norm(values, axis=1)
            result[f"{prefix}_3d_rmse_{unit}"] = float(np.sqrt(np.mean(norm**2)))
            result[f"{prefix}_3d_p95_{unit}"] = float(np.percentile(norm, 95))
            result[f"{prefix}_3d_max_{unit}"] = float(norm.max())
            result[f"{prefix}_3d_worst_sow"] = rows[int(np.argmax(norm))]["t"]
    return result


def plot_errors(series, output, start):
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), layout="constrained", sharex=True)
    for mode, color, style in (("CT", "#0072B2", "-"), ("LC", "#D55E00", "--")):
        times = np.array([row["t"] - start for row in series[mode]])
        for column, key in enumerate(("p", "v", "a")):
            values = np.array([row[key] for row in series[mode]])
            for row in range(3):
                axis = axes[row, column]
                axis.plot(times, values[:, row], color=color, linestyle=style, linewidth=1, label=mode)
    for column, (title, labels, unit) in enumerate((
            ("Position error", ("East", "North", "Up"), "m"),
            ("Velocity error", ("East", "North", "Up"), "m/s"),
            ("Attitude error", ("Pitch", "Roll", "Heading"), "deg"))):
        axes[0, column].set_title(title)
        for row in range(3):
            axes[row, column].set_ylabel(f"{labels[row]} ({unit})")
            axes[row, column].grid(alpha=0.25)
        axes[0, column].legend()
        axes[2, column].set_xlabel(f"Seconds since GPST SOW {start:.0f}")
    fig.suptitle("Data05: CT batch and discrete LC; common IMU-centre truth epochs")
    fig.savefig(output / "errors.png", dpi=170)
    fig.savefig(output / "errors.svg")
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), layout="constrained", sharex=True)
    times = np.array([row["t"] - start for row in series["CT"]])
    axes[0].plot(times, [row["speed"] for row in series["CT"]], color="#333333")
    axes[0].set_ylabel("Truth speed (m/s)")
    for mode, color in (("CT", "#0072B2"), ("LC", "#D55E00")):
        for index, key in enumerate(("p", "v"), 1):
            axes[index].plot(times, np.linalg.norm([row[key] for row in series[mode]], axis=1),
                             color=color, linewidth=1, label=mode)
    axes[1].set_ylabel("3D position error (m)")
    axes[2].set_ylabel("3D velocity error (m/s)")
    axes[2].set_xlabel(f"Seconds since GPST SOW {start:.0f}")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[1].legend()
    fig.savefig(output / "motion_and_norms.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct", type=Path, required=True)
    parser.add_argument("--lc", type=Path, required=True)
    parser.add_argument("--ct-xml", type=Path, required=True)
    parser.add_argument("--lc-xml", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    same_config(args.ct_xml, args.lc_xml)
    ct, lc, truth = read_ins(args.ct), read_ins(args.lc), read_truth(TRUTH)
    config = ET.parse(args.ct_xml).getroot()
    if config.findtext("fgomsf_sensors/msf_type").strip() != "GINS_CT":
        raise ValueError("CT XML 模式错误")
    if ET.parse(args.lc_xml).getroot().findtext("fgomsf_sensors/msf_type").strip() != "GINS_LC":
        raise ValueError("LC XML 模式错误")
    truth_by_time = {row["t"]: row for row in truth}
    lc_times = [row["t"] for row in lc]
    series = {"CT": [], "LC": [], "LC_nearest": []}
    offsets, used = [], set()
    for row in ct:
        reference = truth_by_time.get(row["t"])
        if reference is None:
            raise ValueError("CT 输出不是 Data05 真值整秒历元")
        index = bisect.bisect_left(lc_times, row["t"])
        candidates = lc[max(0, index - 1):min(len(lc), index + 1)]
        nearest = min(candidates, key=lambda item: abs(item["t"] - row["t"]))
        interpolated = interpolate(lc, lc_times, row["t"])
        if abs(nearest["t"] - row["t"]) > TOLERANCE_S or interpolated is None:
            continue
        if nearest["t"] in used:
            raise ValueError("LC 最近邻匹配重复")
        used.add(nearest["t"])
        offsets.append(nearest["t"] - row["t"])
        for mode, estimate in (("CT", row), ("LC", interpolated), ("LC_nearest", nearest)):
            series[mode].append(errors(estimate, reference, truth[0]))
    if not offsets:
        raise ValueError("无共同历元")
    # 本脚本针对从 2020-11-28 02:00:00 GPST 开始的 Data05 验证配置。
    if config.findtext("gen/beg").strip().strip('"') != "2020-11-28 02:00:00":
        raise ValueError("验证起点应为 Data05 的 02:00:00 GPST")
    start = 525600.0
    selections = {"all": [], "common_fixed": [], "after_10s": [], "stationary": [], "dynamic": []}
    for index, row in enumerate(series["CT"]):
        selections["all"].append(index)
        if row["fixed"] and series["LC"][index]["fixed"]:
            selections["common_fixed"].append(index)
        if row["t"] >= start + 10:
            selections["after_10s"].append(index)
        if row["speed"] < 0.1:
            selections["stationary"].append(index)
        if row["speed"] >= 1.0:
            selections["dynamic"].append(index)
    report = {"truth": str(TRUTH), "gps_week": 2133,
              "standard": {"point": "IMU centre; no additional lever correction", "frame": "ECEF",
                           "enu_reference": "first truth epoch; fixed ENU",
                           "heading": "wrap360(-Yaw); all angle differences wrapped to [-180,180)",
                           "primary_time_alignment": "CT/truth exact; LC linear interpolation, no extrapolation",
                           "stationary_speed_below_mps": 0.1, "dynamic_speed_at_least_mps": 1.0},
              "counts": {"ct_raw": len(ct), "lc_raw": len(lc), "truth_raw": len(truth),
                         "common": len(offsets), "ct_excluded_alignment": len(ct) - len(offsets)},
              "lc_nearest_time_offset_s": {"min": min(offsets), "max": max(offsets)},
              "subsets": {}, "input_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (TRUTH, args.ct, args.lc, args.ct_xml, args.lc_xml)}}
    summary = []
    for subset, indices in selections.items():
        report["subsets"][subset] = {}
        for mode, rows in series.items():
            item = metrics([rows[index] for index in indices])
            report["subsets"][subset][mode] = item
            summary.append({"subset": subset, "mode": mode, **item})
    fields = ["sow", "mode", "lc_nearest_offset_s", "truth_speed_mps", "fixed", "truth_fixed",
              "pos_e_m", "pos_n_m", "pos_u_m", "vel_e_mps", "vel_n_mps", "vel_u_mps",
              "pitch_error_deg", "roll_error_deg", "heading_error_deg"]
    with (args.out / "epoch_errors.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        for mode, rows in series.items():
            for index, row in enumerate(rows):
                writer.writerow([row["t"], mode, offsets[index], row["speed"], row["fixed"], row["truth_fixed"],
                                 *row["p"], *row["v"], *row["a"]])
    with (args.out / "summary_metrics.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    # 从导出的逐历元数据独立重算三维 RMSE，检查汇总与导出一致。
    with (args.out / "epoch_errors.csv").open(encoding="utf-8-sig") as file:
        exported = list(csv.DictReader(file))
    for mode in series:
        rows = [row for row in exported if row["mode"] == mode]
        for prefix, unit, suffix in (("pos", "m", "m"), ("vel", "mps", "mps")):
            squared = math.fsum(float(row[f"{prefix}_{axis}_{suffix}"])**2
                                for row in rows for axis in ("e", "n", "u"))
            value = math.sqrt(squared / len(rows))
            if not math.isclose(value, report["subsets"]["all"][mode][f"{prefix}_3d_rmse_{unit}"], abs_tol=1e-12):
                raise ValueError("CSV 重算 RMSE 与汇总不一致")
    report["checks"] = {"config_equal_except_mode_outputs": True, "finite_monotonic_inputs": True,
                        "unique_nearest_matches": True, "enu_norm_preserved": True,
                        "exported_csv_rmse_recomputed": True}
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_errors(series, args.out, start)
    print(json.dumps({"counts": report["counts"], "all": report["subsets"]["all"],
                      "dynamic": report["subsets"]["dynamic"]}, indent=2))


if __name__ == "__main__":
    main()
