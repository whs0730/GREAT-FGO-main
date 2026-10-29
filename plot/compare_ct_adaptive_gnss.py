"""评定四例已有 LC、CT 与自适应 CT 输出，不启动导航程序。"""

import argparse
import bisect
import contextlib
import csv
import hashlib
import io
import json
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs

import evaluate_ct_data05 as numerical
from analyze_ct_pilot_experiments import read_truth
from compare_ginsfgo_lc_msf import load_evaluator


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "results/tc_lc_ct_accuracy_20261020"
MODES = ["LC", "CT1", "CT01", "A1", "A01"]
LABELS = {"LC": "LC", "CT1": "CT 1 s", "CT01": "CT 0.1 s",
          "A1": "Adaptive CT 1 s", "A01": "Adaptive CT 0.1 s"}
COLORS = {"LC": "#777777", "CT1": "#0072B2", "CT01": "#009E73",
          "A1": "#D55E00", "A01": "#CC79A7"}
STYLES = {"LC": "--", "CT1": "--", "CT01": ":", "A1": "-", "A01": "-"}
ADAPTIVE_FILES = {
    "Data05": ["Vehicle_Opensky_CT_adaptive_gnss-1.ins", "Vehicle_Opensky_CT_adaptive_gnss-01.ins"],
    "Data06": ["UAV_Opensky_CT_adaptive_gnss-1.ins", "UAV_Opensky_CT_adaptive_gnss-01.ins"],
    "20211012": ["SEPT-CT_adaptive_gnss-ADIS-1012-1.ins", "SEPT-CT_adaptive_gnss-ADIS-1012.ins"],
    "20250928": ["SEPT-CT_adaptive_gnss-MTI-0928-1.ins", "SEPT-CT_adaptive_gnss-MTI-0928-01.ins"],
}
COLS = ["pos_e_m", "pos_n_m", "pos_u_m", "vel_e_mps", "vel_n_mps", "vel_u_mps",
        "pitch_error_deg", "roll_error_deg", "heading_error_deg"]
KEYS = ["pos_3d_rmse_m", "vel_3d_rmse_mps", "att_heading_rmse_deg",
        "att_pitch_rmse_deg", "att_roll_rmse_deg"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def file_info(path):
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size,
            "modified_local": datetime.fromtimestamp(path.stat().st_mtime,
                timezone(timedelta(hours=8))).isoformat()}


def read_result(path):
    rows = numerical.read_ins(path)
    fields = [line.split() for line in path.read_text(encoding="utf-8-sig").splitlines()
              if line.strip() and not line.lstrip().startswith("#")]
    for row, values in zip(rows, fields):
        row.update(status=values[19], meas_type=values[16])
    return rows


def metric(rows):
    result = numerical.metrics(rows)
    if rows:
        for i, name in enumerate(["pitch", "roll", "heading"]):
            values = np.abs([row["a"][i] for row in rows])
            result[f"att_{name}_p95_deg"] = float(np.percentile(values, 95))
    return result


def intervals(times, mask):
    selected = np.flatnonzero(mask)
    if not len(selected):
        return []
    groups = np.split(selected, np.where((np.diff(selected) > 1) |
                                        (np.diff(times[selected]) > 1.1))[0] + 1)
    return [(times[group[0]] - .5, times[group[-1]] + .5) for group in groups]


def plot_series(name, series, folder, start, modes, suffix=""):
    times = np.array([row["t"] for row in series["LC"]])
    spans = intervals(times, [not row["truth_fixed"] for row in series["LC"]])
    parts = []
    definitions = [("p", "position", ["East", "North", "Up"], "m"),
                   ("v", "velocity", ["East", "North", "Up"], "m/s"),
                   ("a", "attitude", ["Pitch", "Roll", "Heading"], "deg")]
    for key, filename, labels, unit in definitions:
        fig, axes = plt.subplots(3, 1, figsize=(13.5, 8.8), sharex=True, layout="constrained")
        interactive = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=.07)
        for mode in modes:
            values = np.array([row[key] for row in series[mode]])
            # 缺测区间断线，避免把两端连成不存在的误差变化。
            gaps = np.where(np.diff(times) > 1.1)[0] + 1
            x = np.insert(times - start, gaps, np.nan)
            for i, axis in enumerate(axes):
                y = np.insert(values[:, i], gaps, np.nan)
                axis.plot(x, y, label=LABELS[mode], color=COLORS[mode],
                          linestyle=STYLES[mode], linewidth=.9, alpha=.9)
                interactive.add_trace(go.Scatter(x=x.tolist(), y=y.tolist(), name=LABELS[mode],
                    legendgroup=mode, showlegend=i == 0, connectgaps=False,
                    line=dict(color=COLORS[mode], width=1.2,
                              dash={"--": "dash", ":": "dot", "-": "solid"}[STYLES[mode]]),
                    hovertemplate="t=%{x:.3f} s<br>差值=%{y:.6f}<extra>%{fullData.name}</extra>"), row=i+1, col=1)
                interactive.update_yaxes(title_text=f"{labels[i]} ({unit})", row=i+1, col=1)
        for i, axis in enumerate(axes):
            for left, right in spans:
                axis.axvspan(left-start, right-start, color="#999999", alpha=.12, linewidth=0)
            axis.axhline(0, color="#555555", linewidth=.5)
            axis.set_ylabel(f"{labels[i]} ({unit})")
            axis.grid(alpha=.22)
        axes[0].legend(ncol=len(modes), fontsize=9, loc="upper center", bbox_to_anchor=(.5, 1.16))
        axes[-1].set_xlabel(f"Time since GPST SOW {start:.0f} (s)")
        scope = " | 0.1 s comparison only" if suffix else ""
        fig.suptitle(f"{name}: {filename} errors{scope} | {len(times)} common epochs\n"
                     "All peaks retained; grey = reference not Fixed", fontsize=13)
        fig.savefig(folder / f"{filename}_errors{suffix}.png", dpi=200)
        fig.savefig(folder / f"{filename}_errors{suffix}.svg")
        plt.close(fig)
        for left, right in spans:
            interactive.add_vrect(x0=left-start, x1=right-start, fillcolor="grey", opacity=.12,
                                  line_width=0, row="all", col=1)
        interactive.update_layout(height=740, title=f"{name} — {filename} errors{scope}（全段，灰色为参考非 Fixed）",
            template="plotly_white", hovermode="x unified", legend=dict(orientation="h", y=1.09),
            margin=dict(t=120, b=55), font=dict(family="Arial, Microsoft YaHei, sans-serif"))
        interactive.update_xaxes(title_text=f"距 GPST SOW {start:.0f} 的时间 / s", row=3, col=1)
        parts.append(interactive.to_html(full_html=False, include_plotlyjs=False,
            config=dict(scrollZoom=True, displaylogo=False, responsive=True)))
    return "\n".join(parts)


def evaluate(name, out, evaluator):
    historical_path = OLD / name / "metrics.json"
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    paths = {mode: Path(historical["inputs"][mode]["path"]) for mode in MODES[:3]}
    base = paths["LC"].parent.parent
    paths.update({mode: base / "result_CT_adaptive_gnss" / filename
                  for mode, filename in zip(MODES[3:], ADAPTIVE_FILES[name])})
    truth_path = Path(historical["reference"]["path"])
    protected = list(paths.values()) + [truth_path, historical_path]
    protected += list((base / "xml").glob("*.xml"))
    flt_paths = {mode: path.with_suffix(".flt") for mode, path in paths.items()
                 if path.with_suffix(".flt").exists()}
    protected += list(flt_paths.values())
    before = {str(path): file_info(path) for path in protected}
    for mode in MODES[:3]:
        assert before[str(paths[mode])]["sha256"] == historical["inputs"][mode]["sha256"], (name, mode, "旧结果已变化")
    assert sha256(truth_path) == historical["reference"]["sha256"]
    adaptive_xml = next((base / "xml").glob("*adaptive_gnss*.xml"))
    cfg = ET.parse(adaptive_xml)
    for tag, value in [("ct_gnss_adaptive", 1), ("ct_gnss_scale_lambda", 1), ("ct_gnss_scale_max", 100)]:
        assert float(cfg.findtext("fgo/" + tag)) == value
    assert cfg.findtext("fgomsf_sensors/msf_type").strip() == "GINS_CT"
    raw = {mode: read_result(path) for mode, path in paths.items()}
    truth = read_truth(truth_path, historical["gps_week"])
    truth_map = {row["t"]: row for row in truth}
    maps = {mode: {row["t"]: row for row in raw[mode]} for mode in MODES[1:]}
    assert all(abs(t-round(t)) < 1e-9 for rows in maps.values() for t in rows)
    lc_times = [row["t"] for row in raw["LC"]]
    start, end = historical["configured_interval"]
    series = {mode: [] for mode in MODES}
    nearest_series, aligned_lc, excluded, used_lc, offsets = [], [], [], set(), []
    for time in sorted(set().union(*(set(rows) for rows in maps.values()))):
        reasons = []
        if not start <= time <= end:
            reasons.append("outside configured interval")
        if time not in truth_map:
            reasons.append("no exact reference epoch")
        for mode in maps:
            if time not in maps[mode]:
                reasons.append("no " + mode + " epoch")
        index = bisect.bisect_left(lc_times, time)
        nearest = min(raw["LC"][max(0, index-1):index+1], key=lambda row: abs(row["t"]-time))
        aligned = numerical.interpolate(raw["LC"], lc_times, time)
        if abs(nearest["t"]-time) > .05 or aligned is None:
            reasons.append("LC: nearest offset >0.05s or no bracket <=1.1s")
        if reasons:
            excluded.append(dict(sow=time, reason="; ".join(reasons)))
            continue
        assert nearest["t"] not in used_lc
        used_lc.add(nearest["t"])
        offsets.append(nearest["t"]-time)
        aligned_lc.append(aligned)
        estimates = {"LC": aligned, **{mode: maps[mode][time] for mode in maps}}
        for mode, estimate in estimates.items():
            series[mode].append(numerical.errors(estimate, truth_map[time], truth[0]))
        nearest_series.append(numerical.errors(nearest, truth_map[time], truth[0]))
    times = np.array([row["t"] for row in series["LC"]])
    assert len(times) > 0
    reference_fixed = np.array([row["truth_fixed"] for row in series["LC"]])
    common_fixed = np.array([all(series[mode][i]["fixed"] for mode in MODES) for i in range(len(times))])
    masks = {"all": np.ones(len(times), dtype=bool), "reference_fixed": reference_fixed,
             "reference_non_fixed": ~reference_fixed, "common_fixed": common_fixed,
             "common_fixed_reference_fixed": common_fixed & reference_fixed,
             "after_30s": times >= start+30,
             "after_30s_reference_fixed": (times >= start+30) & reference_fixed,
             "dynamic": np.array([row["speed"] >= 1 for row in series["LC"]]),
             "stationary": np.array([row["speed"] < .1 for row in series["LC"]])}
    for old, new in [("CT1", "A1"), ("CT01", "A01")]:
        masks[f"{old}_{new}_fixed_reference_fixed"] = reference_fixed & np.array([
            series[old][i]["fixed"] and series[new][i]["fixed"] for i in range(len(times))])
    subsets = {subset: {mode: metric([row for row, keep in zip(rows, mask) if keep])
                        for mode, rows in series.items()} for subset, mask in masks.items()}
    validation = {}
    with contextlib.redirect_stdout(io.StringIO()):
        independent_truth = evaluator.load_truth(str(truth_path))
        for mode in MODES:
            matched = evaluator.align_by_time(evaluator.load_gins_result(str(paths[mode])), independent_truth, .05)
            matched = matched[matched.sow_gt.isin(times)].sort_values("sow_gt").reset_index(drop=True)
            np.testing.assert_array_equal(matched.sow_gt.to_numpy(), times)
            expected = [np.column_stack(evaluator.calculate_position_errors(matched, independent_truth)[1:]),
                        np.column_stack(evaluator.calculate_velocity_errors(matched, independent_truth)[1:]),
                        np.column_stack(evaluator.calculate_attitude_errors(matched)[1:])[:, [1, 2, 0]]]
            target = nearest_series if mode == "LC" else series[mode]
            validation[mode] = {}
            for key, values, tolerance in zip(["p", "v", "a"], expected, [1e-7, 1e-7, 1e-10]):
                actual = np.array([row[key] for row in target])
                np.testing.assert_allclose(actual, values, atol=tolerance, rtol=0)
                validation[mode][key] = float(np.max(np.abs(actual-values)))
    for key, tolerance in [("pos", 2e-9), ("vel", 1e-11), ("att", 2e-10)]:
        values = np.array([row[key] for row in raw["LC"]])
        if key == "att":
            values = np.rad2deg(np.unwrap(np.deg2rad(values), axis=0))
        expected = np.column_stack([np.interp(times, lc_times, values[:, i]) for i in range(3)])
        delta = np.array([row[key] for row in aligned_lc])-expected
        np.testing.assert_allclose(numerical.wrap(delta) if key == "att" else delta, 0, atol=tolerance, rtol=0)
    # 相同历史输入和共同历元下，旧三组误差应与上一轮逐项一致。
    old_frame = pd.read_csv(OLD / name / "epoch_errors.csv")
    for mode in MODES[:3]:
        previous = old_frame[(old_frame["mode"] == mode) & old_frame.sow.isin(times)].sort_values("sow")
        np.testing.assert_array_equal(previous.sow.to_numpy(), times)
        current = np.array([row["p"]+row["v"]+row["a"] for row in series[mode]])
        np.testing.assert_allclose(current, previous[COLS].to_numpy(), atol=1e-10, rtol=0)
    folder = out / name
    folder.mkdir()
    epoch_rows = []
    for mode, rows in series.items():
        for i, row in enumerate(rows):
            epoch_rows.append(dict(mode=mode, sow=row["t"], elapsed_s=row["t"]-start,
                fixed=row["fixed"], reference_fixed=row["truth_fixed"], reference_speed_mps=row["speed"],
                lc_nearest_offset_s=offsets[i], **dict(zip(COLS, row["p"]+row["v"]+row["a"]))))
    write_csv(folder / "epoch_errors.csv", epoch_rows)
    write_csv(folder / "excluded_epochs.csv", excluded or [dict(sow="", reason="none")])
    frame = pd.read_csv(folder / "epoch_errors.csv")
    for subset, mask in masks.items():
        for mode in MODES:
            selected = frame[frame["mode"] == mode].iloc[np.flatnonzero(mask)]
            if len(selected):
                for column, key in zip(COLS, ["pos_e_rmse_m", "pos_n_rmse_m", "pos_u_rmse_m",
                        "vel_e_rmse_mps", "vel_n_rmse_mps", "vel_u_rmse_mps",
                        "att_pitch_rmse_deg", "att_roll_rmse_deg", "att_heading_rmse_deg"]):
                    np.testing.assert_allclose(np.sqrt(np.mean(selected[column].to_numpy()**2)),
                                               subsets[subset][mode][key], atol=1e-12, rtol=1e-12)
                for columns, key in [(COLS[:3], "pos_3d_rmse_m"), (COLS[3:6], "vel_3d_rmse_mps")]:
                    np.testing.assert_allclose(np.sqrt(np.mean(np.sum(selected[columns].to_numpy()**2, axis=1))),
                                               subsets[subset][mode][key], atol=1e-12, rtol=1e-12)
    frontend = {}
    for first, second in [("A1", "A01"), ("CT1", "A1"), ("CT01", "A01")]:
        if first in flt_paths and second in flt_paths:
            frontend[first+"_vs_"+second] = sha256(flt_paths[first]) == sha256(flt_paths[second])
    parts = plot_series(name, series, folder, start, MODES)
    if name == "20250928":
        parts += '<h3>校园 0.1 秒补充对照：保留全部历元及极值，单独展示 LC 和两份 0.1 秒 CT</h3>'
        parts += plot_series(name, series, folder, start, ["LC", "CT01", "A01"], "_0p1s")
    for path, info in before.items():
        assert sha256(Path(path)) == info["sha256"], (path, "输入在分析期间变化")
    result = dict(gps_week=historical["gps_week"], configured_interval=[start, end],
        common_interval=[times[0], times[-1]], common_count=len(times), excluded=excluded,
        inputs={mode: {**before[str(path)], "rows": len(raw[mode]),
                        "sow": [raw[mode][0]["t"], raw[mode][-1]["t"]]} for mode, path in paths.items()},
        protected_inputs=before, subsets=subsets, independent_max_abs_difference=validation,
        frontend_flt_byte_identical=frontend, lc_max_nearest_offset_s=float(np.max(np.abs(offsets))),
        checks=dict(independent_evaluator_files=5, lc_numpy_interpolation=True,
                    historical_errors_reproduced=True, csv_metrics_recomputed=True, inputs_unchanged=True))
    write_json(folder / "metrics.json", result)
    print(name, len(times), {mode: round(subsets["all"][mode]["pos_3d_rmse_m"], 6) for mode in MODES}, flush=True)
    return result, parts


def metrics_table(results, subset):
    lines = ["| 算例 | 历元 | 方法 | 位置3D / m | 速度3D / (m/s) | 航向 / ° | 俯仰 / ° | 横滚 / ° | Fixed / % |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for case, result in results.items():
        for mode in MODES:
            row = result["subsets"][subset][mode]
            if row["epochs"]:
                values = " | ".join(f"{row[key]:.6f}" for key in KEYS + ["fixed_rate_pct"])
                lines.append(f"| {case} | {row['epochs']} | {LABELS[mode]} | {values} |")
    return lines


def report(results, out):
    findings = []
    for name, result in results.items():
        rows = result["subsets"]["all"]
        old, new = rows["CT01"], rows["A01"]
        changes = [100*(old[key]-new[key])/old[key] for key in KEYS]
        findings.append(f"- {name}：0.1 s 自适应相对同间隔旧 CT 的位置/速度 RMSE 改善分别为 "
            f"{changes[0]:+.2f}% / {changes[1]:+.2f}%；航向 RMSE {old[KEYS[2]]:.3f}°→{new[KEYS[2]]:.3f}°。")
    campus = results["20250928"]["subsets"]
    bad = campus["all"]["A1"]
    findings += [
        f"- 校园 1 s 自适应出现严重退化：位置 RMSE={bad[KEYS[0]]:.3f} m、速度 RMSE={bad[KEYS[1]]:.3f} m/s、"
        f"航向 RMSE={bad[KEYS[2]]:.3f}°；位置峰值={bad['pos_3d_max_m']:.3f} m（SOW {bad['pos_3d_worst_sow']:g}）。"
        f"仅保留参考 Fixed 时位置 RMSE 仍为 {campus['reference_fixed']['A1'][KEYS[0]]:.3f} m，不能仅归因于参考非 Fixed。",
        "- 当前结果支持自适应权重在部分算例改善位置和速度，但不支持其在各场景、各节点间隔或全部姿态指标上稳定占优。校园 1 s 的退化需要另行排查，本次评定没有确定其机制根因。",
    ]
    lines = ["# 自适应 GNSS 权重：四算例 LC 与两档 CT 精度比较", "",
        "新结果为用户运行的 CT_adaptive_gnss 1 s / 0.1 s；基准为现有旧 CT 1 s / 0.1 s 和 LC。仅评定已有文件，未启动导航程序。", "",
        "## 主要结果", ""] + findings + ["", "## 评定口径", "",
        "- 每个算例的五组结果使用同一 IMU 中心参考解和完全相同的整秒历元。主表保留启动、非 Fixed 和全部峰值，没有扣除偏差或剔除异常。",
        "- CT 输出直接匹配整秒参考；LC 的毫秒偏移通过位置/速度线性插值、姿态最短角差插值修正。最近点偏移≤0.05 s、插值间隔≤1.1 s，不外推。Fixed 标签采用最近历元。",
        "- 位置/速度均为 ECEF 估计减参考，旋转到参考首历元的固定 ENU；Heading=-INS 的 Yaw，角差归一至 [-180°,180°)。3D RMSE=√mean(E²+N²+U²)。",
        "- 城区和校园参考存在非 Fixed 区间，其间的数值是与参考解的差值；另列参考 Fixed、启动 30 秒后且参考 Fixed、五组共同 Fixed 且参考 Fixed 统计。共同 Fixed 子集对五组使用同一个掩码。",
        "- 节点间隔 0.1 s 不代表输出 10 Hz，本批 INS 仍为 1 Hz。曲线灰色区域标记参考非 Fixed，所有极值完整显示。交互图可点击图例隐藏曲线、双击图例单独显示、框选缩放。", "",
        "## 文件归属与可复核范围", "",
        "- 新文件的 -1 / -01 后缀分别按用户所述两档结果和命名对应 1 s / 0.1 s；城区不带后缀的新版文件对应当前自适应 XML 的 0.1 s 输出，-1 对应 1 s。每份输入的路径、大小、时间及 SHA256 见各例 metrics.json。",
        "- 当前四份自适应 XML 均为开启、lambda=1、scale_max=100、节点间隔0.1 s。未保存两次运行各自的 XML/控制台快照，参数归属依赖用户说明、文件命名及当前配置，不将当前 XML 当作历史运行参数的完整证明。",
        "- 旧 CT/LC 和参考文件的 SHA256 与 2026-10-20 评定留档一致，旧三组共同历元误差逐项复现。本轮是在现有新旧结果上比较，未进行其他参数完全一致的重新运行核验。", "",
        "## 全部共同历元 RMSE", ""] + metrics_table(results, "all")
    lines += ["", "## 自适应相对同间隔旧 CT 的 RMSE 改善", "",
        "正值为改善，负值为变差；(旧CT−自适应CT)/旧CT。保持节点间隔相同，避免将节点加密与权重改变混为一项收益。", "",
        "| 算例 | 节点间隔 | 位置 | 速度 | 航向 | 俯仰 | 横滚 |", "|---|---|---:|---:|---:|---:|---:|"]
    for name, result in results.items():
        for old, new, interval in [("CT1", "A1", "1 s"), ("CT01", "A01", "0.1 s")]:
            rows = result["subsets"]["all"]
            changes = [100*(rows[old][key]-rows[new][key])/rows[old][key] for key in KEYS]
            lines.append(f"| {name} | {interval} | " + " | ".join(f"{v:+.2f}%" for v in changes) + " |")
    for subset, title in [("reference_fixed", "参考 Fixed"),
                          ("after_30s_reference_fixed", "启动 30 秒后且参考 Fixed"),
                          ("common_fixed_reference_fixed", "五组共同 Fixed 且参考 Fixed")]:
        lines += ["", "## " + title, ""] + metrics_table(results, subset)
    lines += ["", "## 覆盖和文件检查", "", "| 算例 | 共同历元 | SOW 起止 | 参考非Fixed | LC最大时间修正/s | 新两档FLT一致 |", "|---|---:|---|---:|---:|---|"]
    for name, result in results.items():
        lines.append(f"| {name} | {result['common_count']} | {result['common_interval'][0]:g}–{result['common_interval'][1]:g} | "
            f"{result['subsets']['reference_non_fixed']['LC']['epochs']} | {result['lc_max_nearest_offset_s']:.6f} | "
            f"{result['frontend_flt_byte_identical'].get('A1_vs_A01', '未提供')} |")
    lines += ["", "## 图表和数据", "", "- [可缩放误差序列图](index.html)：四例各含位置、速度、姿态三组图。",
        "- 各例 position_errors / velocity_errors / attitude_errors：PNG 和 SVG；所有五组结果在同图比较。",
        "- 校园另有同名 _0p1s 补充图，仅显示 LC、旧 CT 0.1 s、自适应 CT 0.1 s，使用与五组主图完全相同的历元，完整保留这三组的全部峰值。校园 1 s 异常完整保留于主图和主表。",
        "- summary.csv：各子集精度；improvements.csv：同节点间隔改善比例；各例 epoch_errors.csv：全部误差；excluded_epochs.csv：未进入共同样本的历元及原因。",
        "- 20 份 INS 的最近邻误差经独立评定器逐轴核对；LC 插值经 NumPy 复算；CSV 落盘后复算所有子集 9 轴与 3D RMSE；分析前后保护输入哈希不变。",
        "- analysis_source.py 为本轮执行脚本副本，manifest.json 记录依赖脚本哈希。静止/动态、P95、峰值及峰值时刻见各例 metrics.json。", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/ct_adaptive_gnss_accuracy_20261029")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    evaluator = load_evaluator()
    results, sections = {}, []
    summary, improvements = [], []
    for name in ADAPTIVE_FILES:
        result, figures = evaluate(name, args.out, evaluator)
        results[name] = result
        sections.append(f'<section id="case-{name}"><h2>{name}</h2>{figures}</section>')
        for subset, rows in result["subsets"].items():
            summary.extend(dict(case=name, subset=subset, mode=mode, method=LABELS[mode], **row)
                           for mode, row in rows.items())
            for old, new, interval in [("CT1", "A1", 1), ("CT01", "A01", .1)]:
                if rows[old]["epochs"]:
                    improvements.append(dict(case=name, subset=subset, knot_sec=interval,
                        epochs=rows[old]["epochs"], **{key+"_improvement_pct":
                        100*(rows[old][key]-rows[new][key])/rows[old][key] for key in KEYS}))
    write_csv(args.out / "summary.csv", summary)
    write_csv(args.out / "improvements.csv", improvements)
    report(results, args.out)
    nav = " · ".join(f'<a href="#case-{name}">{name}</a>' for name in results)
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>四例自适应 CT 精度比较</title>'
    page += '<style>body{font:16px Arial,"Microsoft YaHei",sans-serif;max-width:1500px;margin:auto;padding:24px;color:#222}nav{position:sticky;top:0;background:white;padding:14px;z-index:10;border-bottom:1px solid #ddd}section{scroll-margin-top:65px}a{color:#1261a0}</style>'
    page += '<h1>自适应 GNSS 权重：LC 与新旧 CT 精度比较</h1><p>每例五组采用相同历元，保留启动和全部峰值。灰色为参考非 Fixed，其间仅称参考差值。点击图例切换曲线，双击图例单独显示；支持缩放和平移。</p>'
    page += f'<p><a href="report.md">中文报告</a> · <a href="summary.csv">全部精度指标</a></p><nav>{nav}</nav>'
    page += '<script>' + get_plotlyjs() + '</script>' + "\n".join(sections) + '</html>'
    (args.out / "index.html").write_text(page, encoding="utf-8")
    dependencies = [Path(__file__).resolve(), ROOT/"plot/evaluate_ct_data05.py",
        ROOT/"plot/analyze_ct_pilot_experiments.py", ROOT/"plot/run_ct_pilot_experiments.py",
        ROOT/"plot/compare_ginsfgo_lc_msf.py", ROOT/"plot/evaluate_gins(2).py", ROOT/"plot/file_reader.py"]
    shutil.copyfile(__file__, args.out / "analysis_source.py")
    write_json(args.out / "manifest.json", dict(created_local=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        dependencies=[file_info(path) for path in dependencies],
        inputs={name: result["inputs"] for name, result in results.items()},
        mode_labels=LABELS, matplotlib_version=matplotlib.__version__,
        checks={name: result["checks"] for name, result in results.items()}))
    print("PASS: independent evaluator, NumPy interpolation, historical errors, CSV metrics and input hashes.")
    print(args.out)


if __name__ == "__main__":
    main()
