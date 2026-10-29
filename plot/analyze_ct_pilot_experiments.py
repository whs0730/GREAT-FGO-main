"""评定固定节点扫描及 GNSS 协方差与天线参考解的一致性。"""
import argparse
import contextlib
import csv
import io
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

import evaluate_ct_data05 as numerical
from run_ct_pilot_experiments import CASES, INTERVALS, OUT, ROOT, sha256
from compare_ginsfgo_lc_msf import load_evaluator


REFERENCES = {
    "Data05": ("groundtruth_HG4930.txt", "groundtruth_ROVE.txt", 2133, 525600),
    "campus": ("ref/groundtruth_0928_MTI.txt", "ref/groundtruth_0928_GNSS.txt", 2386, 47700),
}
COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_truth(path, week):
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    assert all(label in lines[0] for label in ("X-ECEF", "VX-ECEF", "Heading", "AmbStatus"))
    assert "(m^2)" in lines[1]
    rows = []
    for line in lines[2:]:
        fields = line.split()
        if not fields:
            continue
        assert len(fields) >= 31 and float(fields[0]) == week
        latitude = float(fields[5]) + float(fields[6]) / 60 + float(fields[7]) / 3600
        longitude = float(fields[8]) + float(fields[9]) / 60 + float(fields[10]) / 3600
        rows.append({"t": float(fields[1]), "pos": list(map(float, fields[2:5])),
                     "vel": list(map(float, fields[15:18])),
                     "var": list(map(float, fields[18:21])),
                     "att": [float(fields[25]), float(fields[26]), float(fields[24])],
                     "lat": math.radians(latitude), "lon": math.radians(longitude),
                     "fixed": fields[27] == "Fixed", "status": fields[27]})
    numerical.validate(rows)
    assert all(6e6 < np.linalg.norm(row["pos"]) < 7e6 for row in rows)
    assert all(np.all(np.array(row["var"]) >= 0) for row in rows)
    return rows


def body_to_enu(attitude):
    # 与 gins/gbase.cpp::a2mat 相同；参考 Heading 的符号与导航 yaw 相反。
    pitch, roll, yaw = np.deg2rad([attitude[0], attitude[1], -attitude[2]])
    sp, cp, sr, cr, sy, cy = np.sin(pitch), np.cos(pitch), np.sin(roll), np.cos(roll), np.sin(yaw), np.cos(yaw)
    return np.array([[cy*cr-sy*sp*sr, -sy*cp, cy*sr+sy*sp*cr],
                     [sy*cr+cy*sp*sr, cy*cp, sy*sr-cy*sp*cr], [-cp*sr, sp, cp*cr]])


def motion_groups(truth):
    groups = {}
    for first, second in zip(truth, truth[1:]):
        dt = second["t"] - first["t"]
        speed = float(np.linalg.norm(first["vel"]))
        acceleration = abs(np.linalg.norm(second["vel"]) - speed) / dt
        heading_rate = abs(numerical.wrap(second["att"][2] - first["att"][2])) / dt
        rotation = body_to_enu(first["att"]).T @ body_to_enu(second["att"])
        attitude_rate = math.degrees(math.acos(float(np.clip((np.trace(rotation)-1)/2, -1, 1)))) / dt
        turning = speed >= 0.1 and heading_rate >= 5
        accelerating = speed >= 0.1 and acceleration >= 0.5
        groups[first["t"]] = {"speed": speed, "heading_rate_deg_s": heading_rate,
                               "attitude_rate_deg_s": attitude_rate, "attitude_changing": attitude_rate >= 5,
                               "motion_reference_fixed": first["fixed"] and second["fixed"] and dt <= 1.1,
                               "speed_change_m_s2": float(acceleration),
                               "stationary": speed < 0.1, "turning": turning,
                               "accelerating": bool(accelerating),
                               "steady_motion": speed >= 1 and not turning and not accelerating and attitude_rate < 5}
    return groups


def covariance_metrics(rows):
    if not rows:
        return {"epochs": 0}
    error = np.array([row["error_m"] for row in rows])
    sigma = np.array([row["sigma_m"] for row in rows])
    q = np.array([row["d2_gnss"] for row in rows])
    combined = np.array([row["d2_combined"] for row in rows])
    vectors = np.array([[row["east_m"], row["north_m"], row["up_m"]] for row in rows])
    bias = vectors.mean(axis=0)
    return {"epochs": len(rows), "position_3d_rmse_m": float(np.sqrt(np.mean(error**2))),
            "position_3d_p95_m": float(np.quantile(error, .95)), "position_3d_max_m": float(error.max()),
            "reported_3d_sigma_rms_m": float(np.sqrt(np.mean(sigma**2))),
            "error_rms_over_reported_sigma_rms": float(np.sqrt(np.mean(error**2) / np.mean(sigma**2))),
            "bias_east_m": float(bias[0]), "bias_north_m": float(bias[1]), "bias_up_m": float(bias[2]),
            "bias_3d_norm_m": float(np.linalg.norm(bias)),
            "centred_3d_rms_m": float(np.sqrt(np.mean(np.sum((vectors-bias)**2, axis=1)))),
            "d2_gnss_mean": float(q.mean()), "d2_gnss_median": float(np.median(q)),
            "coverage95_gnss_pct": float(100*np.mean(q <= chi2.ppf(.95, 3))),
            "coverage95_combined_pct": float(100*np.mean(combined <= chi2.ppf(.95, 3))),
            "coverage99_combined_pct": float(100*np.mean(combined <= chi2.ppf(.99, 3)))}


def check_covariance(name, truth, antenna_truth, start, duration, config):
    folder = OUT / name
    baseline = folder / "h_1"
    console = (baseline / "console.log").read_text(encoding="utf-8", errors="replace")
    variances = {float(t): np.fromstring(values, sep=" ") for t, values in
                 re.findall(r"\[LC POS RAW\] time=([\d.]+).*?var=(.*?) PDOP=", console)}
    rejected = {float(t) for t in re.findall(r"\[LC REJECT\] time=([\d.]+)", console)}
    truth_map = {row["t"]: row for row in truth}
    antenna_map = {row["t"]: row for row in antenna_truth}
    lever_node = config.find("integration/GNSS/AntennaLever")
    assert lever_node.get("Type") == "RFU"
    lever = np.array([float(lever_node.get(key)) for key in ("X", "Y", "Z")])
    lever_differences = []
    for row in truth:
        if start <= row["t"] <= start + duration and row["t"] in antenna_map:
            enu_rotation = np.column_stack([numerical.enu(axis, row) for axis in np.eye(3)])
            expected = row["pos"] + enu_rotation.T @ body_to_enu(row["att"]) @ lever
            lever_differences.append(np.linalg.norm(expected - antenna_map[row["t"]]["pos"]))
    assert max(lever_differences) < .003, "两份参考解的杆臂关系不一致"
    rows = []
    rms_rounding = []
    for line in (baseline / "result.flt").read_text(encoding="utf-8-sig").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        assert len(fields) == 20
        t = float(fields[0])
        if not start <= t <= start + duration or t not in antenna_map:
            continue
        reference = antenna_map[t]
        variance = variances[t]
        assert variance.shape == (3,) and np.all(variance > 0)
        printed_sigma = np.array(fields[7:10], float)
        rms_rounding.append(float(np.max(np.abs(np.sqrt(variance) - printed_sigma))))
        assert rms_rounding[-1] <= 0.000051
        difference = np.array(fields[1:4], float) - reference["pos"]
        enu_error = numerical.enu(difference, truth_map[t])
        rows.append({"sow": t, "error_m": float(np.linalg.norm(difference)),
                     "east_m": enu_error[0], "north_m": enu_error[1], "up_m": enu_error[2],
                     "sigma_m": float(np.sqrt(variance.sum())),
                     "combined_sigma_m": float(np.sqrt((variance + reference["var"]).sum())),
                     "d2_gnss": float(np.sum(difference**2 / variance)),
                     "d2_combined": float(np.sum(difference**2 / (variance + reference["var"]))),
                     "gnss_fixed": fields[16] == "Fixed", "gnss_status": fields[16],
                     "truth_fixed": reference["fixed"], "accepted": t not in rejected,
                     "pdop": float(fields[14]), "nsat": int(fields[13]), "ratio": float(fields[17]),
                     "after_30s": t >= start + 30,
                     **dict(zip(("dx_m", "dy_m", "dz_m"), difference)),
                     **dict(zip(("gnss_var_x_m2", "gnss_var_y_m2", "gnss_var_z_m2"), variance)),
                     **dict(zip(("ref_var_x_m2", "ref_var_y_m2", "ref_var_z_m2"), reference["var"]))})
    masks = {
        "all_output": lambda r: True,
        "accepted": lambda r: r["accepted"],
        "accepted_reference_fixed": lambda r: r["accepted"] and r["truth_fixed"],
        "accepted_both_fixed": lambda r: r["accepted"] and r["truth_fixed"] and r["gnss_fixed"],
        "accepted_both_fixed_after30": lambda r: r["accepted"] and r["truth_fixed"] and r["gnss_fixed"] and r["after_30s"],
        "accepted_float_reference_fixed": lambda r: r["accepted"] and r["truth_fixed"] and not r["gnss_fixed"],
        "reference_non_fixed": lambda r: not r["truth_fixed"],
        "rejected": lambda r: not r["accepted"],
    }
    motion = motion_groups(truth)
    for group in ("stationary", "steady_motion", "turning", "attitude_changing", "accelerating"):
        masks[group+"_accepted_both_fixed_after30"] = lambda r, g=group: (r["accepted"] and r["truth_fixed"]
            and r["gnss_fixed"] and r["after_30s"] and r["sow"] in motion
            and motion[r["sow"]]["motion_reference_fixed"] and motion[r["sow"]][g])
    summary = {key: covariance_metrics([r for r in rows if select(r)]) for key, select in masks.items()}
    write_csv(folder / "gnss_covariance_epochs.csv", rows)
    bin_rows = []
    selected = sorted([r for r in rows if masks["accepted_both_fixed_after30"](r)], key=lambda r: r["sigma_m"])
    for index, indices in enumerate(np.array_split(np.arange(len(selected)), 5)):
        if len(indices):
            values = [selected[i] for i in indices]
            bin_rows.append({"bin": index+1, "sigma_min_m": values[0]["sigma_m"],
                             "sigma_max_m": values[-1]["sigma_m"], **covariance_metrics(values)})
    write_csv(folder / "gnss_sigma_bins.csv", bin_rows)
    key = "accepted_both_fixed_after30"
    selected_times = {r["sow"] for r in rows if masks[key](r)}
    time_blocks = []
    for offset in range(30, duration, 30):
        values = [r for r in rows if masks[key](r) and start+offset <= r["sow"] < start+offset+30]
        if values:
            time_blocks.append({"start_offset_s": offset, "end_offset_s_exclusive": offset+30,
                                **covariance_metrics(values)})
    write_csv(folder / "gnss_time_blocks.csv", time_blocks)
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, layout="constrained")
    times = np.array([r["sow"] - start for r in rows])
    axes[0].semilogy(times, [r["error_m"] for r in rows], label="GNSS vs antenna reference", color=COLORS[0])
    axes[0].semilogy(times, [r["combined_sigma_m"] for r in rows], label="sqrt(trace(Rgnss + Rref))", color=COLORS[1])
    axes[0].set_ylabel("3D difference / sigma (m)")
    axes[0].legend(fontsize=8)
    q = np.array([r["d2_combined"] for r in rows])
    axes[1].semilogy(times, q, color=COLORS[0], linewidth=.8)
    axes[1].axhline(chi2.ppf(.95, 3), linestyle="--", color="black", label="chi-square(3) 95% benchmark")
    axes[1].set_ylabel("Normalized difference squared")
    axes[1].legend(fontsize=8)
    axes[2].plot(times, [r["pdop"] for r in rows], color=COLORS[0], label="PDOP")
    axes[2].axhline(3, linestyle="--", color="black")
    axes[2].set_ylabel("PDOP")
    axes[3].step(times, [int(r["truth_fixed"]) for r in rows], where="mid", label="reference Fixed")
    axes[3].step(times, [int(r["accepted"])+2 for r in rows], where="mid", label="passed original GNSS gates")
    axes[3].set_yticks([0, 1, 2, 3], ["ref other", "ref Fixed", "rejected", "accepted"])
    axes[3].legend(fontsize=8)
    axes[3].set_xlabel(f"Seconds since GPST SOW {start}")
    for ax in axes:
        ax.grid(alpha=.25)
        for r in rows:
            if not r["truth_fixed"]:
                ax.axvspan(r["sow"]-start-.5, r["sow"]-start+.5, color="grey", alpha=.06, linewidth=0)
    fig.suptitle(f"{name}: covariance diagnostic; grey = reference not Fixed")
    fig.savefig(folder / "gnss_covariance.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    reliable = [r for r in rows if r["sow"] in selected_times]
    for field, label in (("d2_gnss", "Rgnss"), ("d2_combined", "Rgnss + Rref")):
        values = np.sort([r[field] for r in reliable])
        axes[0].semilogx(values, np.arange(1, len(values)+1)/len(values), label=label)
    grid = np.geomspace(.001, max(30, max(r["d2_combined"] for r in reliable)), 300)
    axes[0].semilogx(grid, chi2.cdf(grid, 3), "k--", label="chi-square(3) benchmark")
    axes[0].set(xlabel="Normalized difference squared", ylabel="Empirical cumulative fraction")
    axes[0].legend(fontsize=8)
    x = [r["reported_3d_sigma_rms_m"] for r in bin_rows]
    y = [r["position_3d_rmse_m"] for r in bin_rows]
    axes[1].plot(x, y, "o-", color=COLORS[0])
    lim = max(x+y)
    axes[1].plot([0, lim], [0, lim], "k--")
    axes[1].set(xlabel="Reported 3D sigma RMS (m)", ylabel="Observed 3D difference RMS (m)")
    for ax in axes:
        ax.grid(alpha=.25)
    fig.suptitle(f"{name}: accepted GNSS Fixed + reference Fixed, after first 30 s")
    fig.savefig(folder / "gnss_reliability.png", dpi=160)
    plt.close(fig)
    validation = {"matched_epochs": len(rows), "raw_variance_records": len(variances),
                  "rejected_log_epochs": len(rejected),
                  "rejected_epochs_without_flt": sorted(rejected - {r["sow"] for r in rows}),
                  "lever_check_max_m": float(max(lever_differences)),
                  "rms_print_rounding_max_m": max(rms_rounding),
                  "reference_paths_sha256": {str(ROOT/"sample_data"/CASES[name][0]/p): sha256(ROOT/"sample_data"/CASES[name][0]/p)
                                             for p in REFERENCES[name][:2]}}
    result = {"subsets": summary, "validation": validation, "time_blocks_30s": time_blocks,
              "largest_accepted_both_fixed": sorted([r for r in rows if masks["accepted_both_fixed"](r)],
                                                     key=lambda r: r["error_m"], reverse=True)[:10],
              "largest_normalized_accepted_both_fixed_after30": sorted(reliable,
                                                   key=lambda r: r["d2_combined"], reverse=True)[:5]}
    write_json(folder / "covariance_summary.json", result)
    return result


def analyze_case(name, covariance_only):
    directory, xml_name, duration = CASES[name]
    imu_reference, antenna_reference, week, start = REFERENCES[name]
    base = ROOT / "sample_data" / directory
    truth = read_truth(base / imu_reference, week)
    antenna = read_truth(base / antenna_reference, week)
    config = ET.parse(base / "xml" / xml_name)
    covariance = check_covariance(name, truth, antenna, start, duration, config)
    if covariance_only:
        return {"covariance": covariance}
    maps, runs, all_raw_times = {}, {}, {}
    for interval in INTERVALS:
        folder = OUT / name / f"h_{interval:g}"
        rows = numerical.read_ins(folder / "result.ins")
        assert all(abs(r["t"]-round(r["t"])) < 1e-8 for r in rows)
        assert all(start <= r["t"] <= start + duration for r in rows)
        maps[interval] = {r["t"]: r for r in rows}
        all_raw_times[interval] = set(maps[interval])
        run = json.loads((folder / "run.json").read_text())
        runs[interval] = {key: run[key] for key in ("wall_seconds", "window_count", "no_convergence_count", "failure_count")}
        runs[interval].update(knots_median=float(np.median(run["knot_counts"])),
                              knots_max=max(run["knot_counts"]),
                              iterations_mean=float(np.mean(run["iterations"])),
                              raw_epochs=len(rows))
    assert len({sha256(OUT/name/f"h_{h:g}/result.flt") for h in INTERVALS}) == 1, "GNSS 前端输出随节点间隔变化"
    truth_map = {r["t"]: r for r in truth}
    motion = motion_groups(truth)
    times = sorted(set.intersection(*all_raw_times.values()) & set(truth_map) & set(motion))
    masks = {"all": lambda t: True, "reference_fixed": lambda t: truth_map[t]["fixed"],
             "reference_non_fixed": lambda t: not truth_map[t]["fixed"],
             "reference_fixed_after30": lambda t: truth_map[t]["fixed"] and t >= start+30}
    for group in ("stationary", "steady_motion", "turning", "attitude_changing", "accelerating"):
        masks[group+"_reference_fixed_after30"] = lambda t, g=group: (truth_map[t]["fixed"] and t >= start+30
            and motion[t]["motion_reference_fixed"] and motion[t][g])
    series = {h: [numerical.errors(maps[h][t], truth_map[t], truth[0]) for t in times] for h in INTERVALS}
    evaluator = load_evaluator()
    independent = {}
    with contextlib.redirect_stdout(io.StringIO()):
        independent_truth = evaluator.load_truth(str(base / imu_reference))
        for interval in INTERVALS:
            aligned = evaluator.align_by_time(evaluator.load_gins_result(str(OUT/name/f"h_{interval:g}/result.ins")), independent_truth, .05)
            aligned = aligned[aligned["sow_gt"].isin(times)].sort_values("sow_gt")
            assert aligned["sow_gt"].tolist() == times
            _, pe, pn, pu = evaluator.calculate_position_errors(aligned, independent_truth)
            _, ve, vn, vu = evaluator.calculate_velocity_errors(aligned, independent_truth)
            _, heading, pitch, roll = evaluator.calculate_attitude_errors(aligned)
            largest = 0.0
            for key, values in (("p", [pe, pn, pu]), ("v", [ve, vn, vu]), ("a", [pitch, roll, heading])):
                difference = np.max(np.abs(np.column_stack(values) - np.array([r[key] for r in series[interval]])))
                largest = max(largest, float(difference))
                assert difference < 1e-8, (interval, key, difference)
            independent[str(interval)] = {"epochs": len(aligned), "max_abs_nine_axis_difference": largest}
    summaries = []
    epoch_rows = []
    for interval in INTERVALS:
        for key, select in masks.items():
            values = numerical.metrics([r for r in series[interval] if select(r["t"])])
            summaries.append({"case": name, "interval_s": interval, "subset": key, **values})
        for row in series[interval]:
            epoch_rows.append({"interval_s": interval, "sow": row["t"], "truth_fixed": row["truth_fixed"],
                               **motion[row["t"]], "position_3d_m": float(np.linalg.norm(row["p"])),
                               "velocity_3d_m_s": float(np.linalg.norm(row["v"])),
                               "heading_error_deg": row["a"][2],
                               **dict(zip(("east_m", "north_m", "up_m"), row["p"]))})
    folder = OUT / name
    write_csv(folder / "knot_epoch_errors.csv", epoch_rows)
    # 空子集只有计数，其余列用空白表示，不补零。
    columns = list(dict.fromkeys(k for r in summaries for k in r))
    write_csv(folder / "knot_summary.csv", [{k: r.get(k, "") for k in columns} for r in summaries])
    write_csv(folder / "runtime.csv", [{"interval_s": h, **runs[h]} for h in INTERVALS])
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, layout="constrained")
    x = np.array(times)-start
    axes[0].plot(x, [motion[t]["speed"] for t in times], color="#333333", label="speed (m/s)")
    axes[0].plot(x, [motion[t]["heading_rate_deg_s"] for t in times], color=COLORS[1], label="abs heading rate (deg/s)")
    axes[0].set_ylabel("Reference motion")
    axes[0].legend(fontsize=8)
    for interval, color in zip(INTERVALS, COLORS):
        for ax, key in ((axes[1], "p"), (axes[2], "v")):
            ax.plot(x, np.linalg.norm([r[key] for r in series[interval]], axis=1), color=color,
                    linewidth=.9, label=f"h={interval:g} s")
        axes[3].plot(x, [r["a"][2] for r in series[interval]], color=color, linewidth=.9)
    for ax, label in zip(axes[1:], ("3D position difference (m)", "3D velocity difference (m/s)", "Heading difference (deg)")):
        ax.set_ylabel(label)
    axes[1].legend(ncol=4, fontsize=8)
    axes[3].set_xlabel(f"Seconds since GPST SOW {start}")
    for ax in axes:
        ax.grid(alpha=.25)
        for t in times:
            if not truth_map[t]["fixed"]:
                ax.axvspan(t-start-.5, t-start+.5, color="grey", alpha=.06, linewidth=0)
    fig.suptitle(f"{name}: same common epochs; grey = reference not Fixed")
    fig.savefig(folder / "knot_scan.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, metric, label in zip(axes[:2], ("pos_3d_rmse_m", "att_heading_rmse_deg"), ("3D position RMSE (m)", "Heading RMSE (deg)")):
        for group in ("reference_fixed_after30", "stationary_reference_fixed_after30", "steady_motion_reference_fixed_after30",
                      "turning_reference_fixed_after30", "attitude_changing_reference_fixed_after30", "accelerating_reference_fixed_after30"):
            values = [r for r in summaries if r["subset"] == group and r["epochs"] >= 10]
            if values:
                group_label = group.replace("_reference_fixed_after30", "").replace("reference_fixed_after30", "all reliable")
                ax.plot([r["interval_s"] for r in values], [r[metric] for r in values], "o-", label=f"{group_label} (n={values[0]['epochs']})")
        ax.set(xlabel="Knot interval (s)", ylabel=label)
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=7)
    for h, color in zip(INTERVALS, COLORS):
        metric = next(r for r in summaries if r["interval_s"] == h and r["subset"] == "reference_fixed_after30")
        axes[2].scatter(runs[h]["wall_seconds"], metric["pos_3d_rmse_m"], color=color)
        offset = (-20, -14) if h == .5 else (4, 4)
        axes[2].annotate(f"{h:g} s", (runs[h]["wall_seconds"], metric["pos_3d_rmse_m"]), xytext=offset, textcoords="offset points")
    axes[2].set(xlabel="Whole-process wall time (s)", ylabel="3D position RMSE (m)")
    axes[2].set_ylim(0, 1.15*max(r["pos_3d_rmse_m"] for r in summaries if r["subset"] == "reference_fixed_after30"))
    axes[2].margins(x=.18)
    axes[2].grid(alpha=.25)
    fig.suptitle(f"{name}: reference Fixed, after first 30 s; single-run timing")
    fig.savefig(folder / "accuracy_cost.png", dpi=160)
    plt.close(fig)
    result = {"case": name, "duration_s": duration, "common_epochs": len(times),
              "excluded_from_common": {str(h): sorted(all_raw_times[h]-set(times)) for h in INTERVALS},
              "runtime": runs, "knot_subsets": summaries, "covariance": covariance,
              "validation": {"gnss_frontend_byte_identical_across_intervals": True,
                             "independent_evaluator": independent,
                             "attitude_order": "pitch, roll, heading; heading=-output yaw",
                             "reference": "IMU centre for knot scan, antenna for GNSS covariance"}}
    write_json(folder / "summary.json", result)
    return result


def write_report(results):
    lines = ["# 连续时间 GNSS/INS 自适应方法预实验", "", "日期：2026-10-29。两组真实数据的探索性诊断，未加入动态权重或动态节点算法。", "",
             "## 实验设置", "",
             "保留原始起点、初始化、IMU 采样、GNSS 门控、传感器噪声和先验。仅截短结束时间、改变节点间隔并隔离输出。10 秒窗口、1 秒更新、1 Hz 输出，四档间隔顺序运行。参考解仅用于事后评定和运动分组，不进入解算。", "",
             "|算例|处理区间（GPST）|输入长度|IMU|", "|---|---|---:|---:|",
             "|Data05 开阔道路|2020-11-28 02:00:00–02:10:00|600 s|100 Hz|",
             "|校园复杂环境|2025-09-28 13:15:00–13:21:00|360 s|400 Hz|", "",
             "主比较仅使用四档共同整秒历元；位置/速度为 ECEF 差值转固定 ENU，Heading=-INS 输出 yaw，姿态差归一到 [-180°,180°)。位置参考点为 IMU 中心。另列全段与参考 Fixed 且起点 30 秒后的指标，保留全部峰值。", "",
             "## 固定节点间隔扫描", "",
             "下表精度采用参考 Fixed 且排除前 30 秒的共同历元。控制点数量按单条位置样条计，姿态样条使用相同数量。耗时为从启动到退出的完整进程墙钟，包含输入读取、GNSS 前端、CT 求解、边缘化及日志；每档仅运行一次，期间有少量分析工作，不能当作严格性能基准。", "",
             "|算例|间隔/s|历元|位置 RMSE/m|速度 RMSE/(m/s)|Heading RMSE/°|总墙钟/s|控制点中位数|平均迭代|",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, result in results.items():
        for interval in INTERVALS:
            metric = next(r for r in result["knot_subsets"] if r["interval_s"] == interval and r["subset"] == "reference_fixed_after30")
            run = result["runtime"][interval]
            lines.append(f"|{name}|{interval:g}|{metric['epochs']}|{metric['pos_3d_rmse_m']:.6f}|{metric['vel_3d_rmse_mps']:.6f}|{metric['att_heading_rmse_deg']:.6f}|{run['wall_seconds']:.2f}|{run['knots_median']:g}|{run['iterations_mean']:.2f}|")
    lines += ["", "### 运动分组", "",
              "事后分组使用参考解：静止速度 <0.1 m/s；平稳运动速度 ≥1 m/s 且不属于转弯、姿态明显变化或加减速；转弯要求速度 ≥0.1 m/s 且相邻 1 秒 Heading 变化率 ≥5°/s；姿态明显变化使用三维相对旋转角，变化率 ≥5°/s；加减速要求速度 ≥0.1 m/s 且速度模长变化率 ≥0.5 m/s²。这些变化类别可以重叠；不是在线节点判据。运动子集要求差分两端参考均 Fixed、间隔不超过 1.1 s，并排除前 30 秒。小样本子集仅用于定位现象，不作泛化推断。", "",
              "|算例|分组|间隔/s|历元|位置 RMSE/m|Heading RMSE/°|", "|---|---|---:|---:|---:|---:|"]
    for name, result in results.items():
        for subset in ("stationary", "steady_motion", "turning", "attitude_changing", "accelerating"):
            for interval in INTERVALS:
                metric = next(r for r in result["knot_subsets"] if r["interval_s"] == interval and r["subset"] == subset+"_reference_fixed_after30")
                if metric["epochs"]:
                    lines.append(f"|{name}|{subset}|{interval:g}|{metric['epochs']}|{metric['pos_3d_rmse_m']:.6f}|{metric['att_heading_rmse_deg']:.6f}|")
    lines += ["", "### 全段敏感性", "", "|算例|间隔/s|共同历元|位置 RMSE/m|位置最大差/m|最大差 SOW|Heading RMSE/°|",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, result in results.items():
        for interval in INTERVALS:
            metric = next(r for r in result["knot_subsets"] if r["interval_s"] == interval and r["subset"] == "all")
            lines.append(f"|{name}|{interval:g}|{metric['epochs']}|{metric['pos_3d_rmse_m']:.6f}|{metric['pos_3d_max_m']:.6f}|{metric['pos_3d_worst_sow']:.0f}|{metric['att_heading_rmse_deg']:.6f}|")
    lines += ["", "全段包含参考非 Fixed 历元，特别是校园弱 GNSS 时段；这些差值不能直接解释为绝对定位误差。参考非 Fixed 子集另存于 knot_summary.csv。", "",
              "## GNSS 协方差可信度", "",
              "直接比较 GNSS 天线 ECEF 与对应天线参考解，避免把杆臂算进误差。使用日志中的 Rpos 对角方差，而不是把 FLT 的 RMS 当作方差。FLT 提供 0.1 mm 精度坐标；sqrt(Rpos) 与 FLT RMS 的最大差约 0.051 mm，与 FLT 舍入及日志有效位数相容。原始 PDOP/卫星数/解算失败门控按日志识别，被拒绝的观测独立统计。", "",
              "定义 D²_GNSS=Σ(dx_i²/R_i)，另算 D²_combined=Σ(dx_i²/(R_i+Rref_i))。95% 覆盖率为 D²≤χ²(3,0.95)=7.8147 的比例。协方差相加的解释依赖两套误差独立的假设，当前参考解与 GNSS 可能共享观测，交叉协方差未知；因此这只是归一化差值诊断，不是严格 NEES/NIS 检验，也不报告独立样本置信区间。参考只提供对角位置协方差。", "",
              "主表使用通过原门控、GNSS Fixed、天线参考 Fixed 且排除前 30 秒的历元。合成 sigma 为 sqrt(trace(R)) 的 RMS，既不是 95% 半径，也不是三维误差的确定上界。", "",
              "|算例|历元|三维差值 RMS/cm|报告合成 sigma RMS/cm|RMS 比值|均值偏移模/cm|去均值散布/cm|GNSS 95%覆盖率|含参考方差95%覆盖率|",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, result in results.items():
        m = result["covariance"]["subsets"]["accepted_both_fixed_after30"]
        lines.append(f"|{name}|{m['epochs']}|{100*m['position_3d_rmse_m']:.3f}|{100*m['reported_3d_sigma_rms_m']:.3f}|{m['error_rms_over_reported_sigma_rms']:.2f}|{100*m['bias_3d_norm_m']:.3f}|{100*m['centred_3d_rms_m']:.3f}|{m['coverage95_gnss_pct']:.2f}%|{m['coverage95_combined_pct']:.2f}%|")
    lines += ["", "去均值结果仅用于区分固定偏移与散布，没有校正输入或重新解算。比值和覆盖率同时受参考误差、系统偏移、时间同步及误差相关性影响，不能直接拿这个比值设 GNSS 权重。", "",
              "|算例|GNSS输出历元|通过原门控|通过且参考Fixed|通过且双方Fixed|参考非Fixed|有坐标但被拒绝|失败无FLT坐标|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, result in results.items():
        subsets = result["covariance"]["subsets"]
        counts = [subsets[k]["epochs"] for k in ("all_output", "accepted", "accepted_reference_fixed", "accepted_both_fixed", "reference_non_fixed", "rejected")]
        counts.append(len(result["covariance"]["validation"]["rejected_epochs_without_flt"]))
        lines.append("|"+name+"|"+"|".join(map(str, counts))+"|")
    lines += ["", "以下为主表子集中归一化差值最大的三个历元，均通过原门控、GNSS 与参考均 Fixed。按 D²_combined 排序，只作事件定位，不据此拟合阈值。", "",
              "|算例|SOW|三维差值/m|报告合成sigma/m|PDOP|卫星数|ratio|D²_combined|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, result in results.items():
        for row in result["covariance"]["largest_normalized_accepted_both_fixed_after30"][:3]:
            lines.append(f"|{name}|{row['sow']:.0f}|{row['error_m']:.4f}|{row['sigma_m']:.4f}|{row['pdop']:.2f}|{row['nsat']}|{row['ratio']:.2f}|{row['d2_combined']:.2f}|")
    lines += ["", "### 对方法选择的含义", ""]
    for name, result in results.items():
        metrics = {h: next(r for r in result["knot_subsets"] if r["interval_s"] == h and r["subset"] == "reference_fixed_after30") for h in (1.0, .1)}
        ratio = result["runtime"][.1]["wall_seconds"] / result["runtime"][1.0]["wall_seconds"]
        position = 100*(metrics[1.0]["pos_3d_rmse_m"]-metrics[.1]["pos_3d_rmse_m"])/metrics[1.0]["pos_3d_rmse_m"]
        heading = 100*(metrics[1.0]["att_heading_rmse_deg"]-metrics[.1]["att_heading_rmse_deg"])/metrics[1.0]["att_heading_rmse_deg"]
        lines.append(f"- {name}：0.1 s 相对 1 s，在参考 Fixed、排除前 30 s 子集中，位置 RMSE 改善 {position:+.2f}%，Heading RMSE 改善 {heading:+.2f}%；总墙钟为 {ratio:.2f} 倍。正值表示改善，负值表示变差。")
    for name, result in results.items():
        blocks = [r for r in result["covariance"]["time_blocks_30s"] if r["epochs"] >= 10]
        ratios = [r["error_rms_over_reported_sigma_rms"] for r in blocks]
        lines.append(f"- {name}：按固定 30 s 非重叠时段统计双方 Fixed 且通过门控的观测，至少 10 个有效历元的 {len(blocks)} 个时段中，三维差值 RMS / 报告 sigma RMS 比值范围为 {min(ratios):.2f}–{max(ratios):.2f}。这是时间分布的描述，不是在线可用的校准量。逐段计数和偏移/散布见 gnss_time_blocks.csv。")
    lines += ["", "老师提出的目标是按运动变化安排后续新增节点：匀速直行放宽间隔，转弯或姿态变化明显时缩短后续间隔。当前扫描为候选间隔提供依据，尚未实现该在线机制。后续触发必须使用截至当前时刻的 IMU/状态信息，不使用参考解或未来观测；已有边缘化节点不随意重新布置。", "",
              "节点扫描只能判断固定间隔的敏感性和计算取舍，不能证明任何动态节点方法有效。节点变化还会改变状态维数、数值条件及边缘化离散化，不能把全部差异直接归因于运动逼近误差。GNSS 方差诊断支持进一步核查不确定度建模，但动态调权还必须与固定尺度校准、普通鲁棒核比较，并控制系统偏移。两项机制是否适合联合，需由后续同输入消融实验判断。", "",
              "## 验证与复现", ""]
    for name, result in results.items():
        max_difference = max(r["max_abs_nine_axis_difference"] for r in result["validation"]["independent_evaluator"].values())
        lever = result["covariance"]["validation"]["lever_check_max_m"]
        lines.append(f"- {name}：四档 GNSS FLT 逐字节相同；{result['common_epochs']} 个共同评定历元；九轴误差与既有独立评定器最大绝对差 {max_difference:.3g}；两份参考解的杆臂校核最大差 {1000*lever:.3f} mm。")
        for h in INTERVALS:
            run = result["runtime"][h]
            lines.append(f"  - h={h:g} s：{run['window_count']} 窗，未收敛 {run['no_convergence_count']}，失败 {run['failure_count']}。退出码及各窗控制点/迭代数见 h_{h:g}/run.json。")
    lines += ["", "主程序、DLL、源码、原始配置和输入的 SHA-256 保存在 manifest.json，8 次运行结束时核对 3306 个文件均未变化，见 preservation.json。执行时的运行脚本存档为 run_ct_pilot_experiments.executed.py；完成运行后，仅将当前运行脚本改为拒绝覆盖已存在的输出根目录。各实验配置、控制台日志和原始 INS/FLT 全部独立保留。脚本和依赖版本见 analysis_provenance.json，方差 CSV 独立复算见 covariance_independent_check.json。", "",
              "```powershell", "python plot/run_ct_pilot_experiments.py",
              "& 'E:\\SoftWare\\Miniconda3\\envs\\modeling\\python.exe' plot/analyze_ct_pilot_experiments.py", "```", "",
              "运行脚本会拒绝覆盖已有输出根目录；重跑应先修改脚本的 OUT 为新的实验目录。分析脚本可复算已完成输出。图中的运动分组只展示至少 10 个历元的子集，完整小样本统计仍保留在表格和 CSV。", "",
              "## 图与逐历元数据", ""]
    for name in results:
        lines += [f"### {name}", "", f"![节点间隔扫描]({name}/knot_scan.png)", "", f"![精度与计算开销]({name}/accuracy_cost.png)", "",
                  f"![GNSS 协方差诊断]({name}/gnss_covariance.png)", "", f"![GNSS 可靠性]({name}/gnss_reliability.png)", "",
                  f"逐历元数据：[节点误差]({name}/knot_epoch_errors.csv)、[GNSS 方差诊断]({name}/gnss_covariance_epochs.csv)、[方差分箱]({name}/gnss_sigma_bins.csv)、[30 s 时段]({name}/gnss_time_blocks.csv)；完整统计见 [{name}/summary.json]({name}/summary.json)。", ""]
    (OUT / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--covariance-only", action="store_true")
    args = parser.parse_args()
    results = {}
    for name in ([args.case] if args.case else CASES):
        result = analyze_case(name, args.covariance_only)
        results[name] = result
        print(name, json.dumps(result["covariance"]["subsets"]["accepted_both_fixed_after30"]), flush=True)
    if not args.covariance_only and not args.case:
        write_report(results)
        write_json(OUT / "analysis_provenance.json", {
            "python": sys.version, "executable": sys.executable,
            "numpy": np.__version__, "matplotlib": matplotlib.__version__,
            "scipy": sys.modules["scipy"].__version__,
            "pandas": sys.modules["pandas"].__version__,
            "sha256": {str(p): sha256(p) for p in [Path(__file__), Path(numerical.__file__),
                ROOT/"plot/run_ct_pilot_experiments.py", ROOT/"plot/evaluate_gins(2).py",
                ROOT/"plot/compare_ginsfgo_lc_msf.py", OUT/"run_ct_pilot_experiments.executed.py"]}})


if __name__ == "__main__":
    main()
