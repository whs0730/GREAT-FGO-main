#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GREAT-FGO GNSS/INS 精度评定脚本
--------------------------------
使用方法：
1. 只需要修改下面【用户配置区】中的文件路径；
2. 在 PyCharm / VS Code 中直接点击“运行”即可；
3. 不需要在终端输入任何参数。

适用：
- 松耦合 LC .ins
- 紧耦合 TC .ins
- Inertial Explorer 的 ROVE_GroundTruth.txt

输出：
- epoch_errors.csv        每个历元的详细误差
- summary_metrics.csv     RMSE / Bias / MAE / MaxAbs 等统计
- position_enu_error.png
- position_error_norm.png
- velocity_enu_error.png
- attitude_error.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ======================================================================
#                         用户配置区
# ======================================================================
# 只需要修改这里，其他代码一般不用动。

# GREAT-FGO 输出的 .ins 文件
RESULT_FILE = r"D:\GREAT-FGO-main\sample_data\FGO_20211012\result\SEPT-RTK-TCI-ADIS-FGO-LC.ins"

# ROVE 真值文件
TRUTH_FILE = r"D:\GREAT-FGO-main\sample_data\FGO_20211012\ref\groundtruth_1012_ADIS.txt"

# 本次结果名称：松耦合写 "LC"，紧耦合写 "TC"
LABEL = "LC"

# 输出目录
OUTPUT_DIR = r"D:\GREAT-FGO-main\plot\result_LC(1)"


# ---------- 可选设置 ----------

# 时间匹配允许的最大误差，单位：秒
TIME_TOLERANCE = 0.05

# 评定的 GPS SOW 时间范围。
# 不限制就保持 None。
START_SOW = None
END_SOW = None

# 跳过最开始几个已经匹配的历元。
# 若想排除 LC 初始化的前两个历元，可设为 2。
SKIP_FIRST_EPOCHS = 0

# GREAT-FGO 的 Yaw 与 IE Heading 的关系。
# 当前 Vehicle_Opensky 数据通常满足：
# Heading = wrap360(-Yaw)
# 所以默认使用 "negative"。
#
# 可选：
# "negative" : Heading = wrap360(-Yaw)
# "same"     : Heading = wrap360(Yaw)
YAW_MODE = "negative"

# 是否绘图
SAVE_PLOTS = True


# ======================================================================
#                         数据读取
# ======================================================================

def load_great_ins(path: str) -> pd.DataFrame:
    """读取 GREAT-FGO 的 .ins 文件。"""
    rows = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()

            if not s or s.startswith("#"):
                continue

            p = s.split()

            # 当前 GREAT-FGO .ins 至少需要这些字段
            if len(p) < 20:
                continue

            try:
                row = {
                    "sow_est": float(p[0]),

                    "x_est": float(p[1]),
                    "y_est": float(p[2]),
                    "z_est": float(p[3]),

                    "vx_est": float(p[4]),
                    "vy_est": float(p[5]),
                    "vz_est": float(p[6]),

                    "pitch_est": float(p[7]),
                    "roll_est": float(p[8]),
                    "yaw_est": float(p[9]),

                    "meas_type": p[16],
                    "nsat": int(float(p[17])),
                    "pdop": float(p[18]),
                    "amb_status": p[19],

                    "ratio": float(p[20]) if len(p) > 20 else np.nan,
                }

            except (ValueError, IndexError):
                print(f"[WARN] 跳过 INS 第 {line_no} 行：{s[:100]}")
                continue

            rows.append(row)

    if not rows:
        raise ValueError(f"没有从 .ins 中读取到有效数据：\n{path}")

    return (
        pd.DataFrame(rows)
        .sort_values("sow_est")
        .reset_index(drop=True)
    )


def load_rover_ground_truth(path: str) -> pd.DataFrame:
    """读取 Inertial Explorer 的 ROVE_GroundTruth.txt。"""
    rows = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()

            if not s:
                continue

            p = s.split()

            # IE 真值数据行列数较多
            if len(p) < 24:
                continue

            try:
                week = int(p[0])
                sow = float(p[1])
            except ValueError:
                # 表头直接跳过
                continue

            try:
                row = {
                    "week": week,
                    "sow_gt": sow,

                    "lat_gt_deg": float(p[2]),
                    "lon_gt_deg": float(p[3]),
                    "h_gt": float(p[4]),

                    "quality_gt": p[5],
                    "amb_status_gt": p[6],

                    "x_gt": float(p[9]),
                    "y_gt": float(p[10]),
                    "z_gt": float(p[11]),

                    "vx_gt": float(p[15]),
                    "vy_gt": float(p[16]),
                    "vz_gt": float(p[17]),

                    "ve_gt": float(p[18]),
                    "vn_gt": float(p[19]),
                    "vu_gt": float(p[20]),

                    "heading_gt": float(p[21]),
                    "pitch_gt": float(p[22]),
                    "roll_gt": float(p[23]),
                }

            except (ValueError, IndexError):
                print(f"[WARN] 跳过真值第 {line_no} 行：{s[:100]}")
                continue

            rows.append(row)

    if not rows:
        raise ValueError(f"没有从真值文件中读取到有效数据：\n{path}")

    return (
        pd.DataFrame(rows)
        .sort_values("sow_gt")
        .reset_index(drop=True)
    )


# ======================================================================
#                         坐标与角度转换
# ======================================================================

def ecef_vector_to_enu(dx, dy, dz, lat_deg, lon_deg):
    """
    将 ECEF 中的矢量转换到当地 ENU 坐标系。

    对位置：
        [dx,dy,dz] = P_est - P_truth
        输出 E/N/U 位置误差。

    对速度：
        [dx,dy,dz] = V_ECEF
        输出 VE/VN/VU。
    """

    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=float))

    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    dz = np.asarray(dz, dtype=float)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    east = -sin_lon * dx + cos_lon * dy

    north = (
        -sin_lat * cos_lon * dx
        - sin_lat * sin_lon * dy
        + cos_lat * dz
    )

    up = (
        cos_lat * cos_lon * dx
        + cos_lat * sin_lon * dy
        + sin_lat * dz
    )

    return east, north, up


def wrap_to_180(angle_deg):
    """角度归一化到 [-180, 180)。"""
    a = np.asarray(angle_deg, dtype=float)
    return (a + 180.0) % 360.0 - 180.0


def wrap_to_360(angle_deg):
    """角度归一化到 [0, 360)。"""
    a = np.asarray(angle_deg, dtype=float)
    return a % 360.0


# ======================================================================
#                         时间对齐
# ======================================================================

def align_by_time(ins: pd.DataFrame,
                  gt: pd.DataFrame,
                  tolerance: float) -> pd.DataFrame:
    """
    根据 GPS SOW 做最近邻时间匹配。
    """

    ins = ins.sort_values("sow_est").copy()
    gt = gt.sort_values("sow_gt").copy()

    aligned = pd.merge_asof(
        ins,
        gt,
        left_on="sow_est",
        right_on="sow_gt",
        direction="nearest",
        tolerance=tolerance,
    )

    aligned = aligned.dropna(subset=["sow_gt"]).copy()

    if aligned.empty:
        raise ValueError(
            "没有成功匹配的历元。\n"
            "请检查结果和真值的 GPS 时间是否一致，"
            "或者适当增大 TIME_TOLERANCE。"
        )

    aligned["dt_match"] = (
        aligned["sow_est"] - aligned["sow_gt"]
    )

    return aligned.reset_index(drop=True)


# ======================================================================
#                         误差计算
# ======================================================================

def calculate_errors(df: pd.DataFrame,
                     yaw_mode: str) -> pd.DataFrame:

    out = df.copy()

    # ------------------------------------------------------------------
    # 位置误差
    # ------------------------------------------------------------------

    out["dx"] = out["x_est"] - out["x_gt"]
    out["dy"] = out["y_est"] - out["y_gt"]
    out["dz"] = out["z_est"] - out["z_gt"]

    e, n, u = ecef_vector_to_enu(
        out["dx"].to_numpy(),
        out["dy"].to_numpy(),
        out["dz"].to_numpy(),
        out["lat_gt_deg"].to_numpy(),
        out["lon_gt_deg"].to_numpy(),
    )

    out["e_err"] = e
    out["n_err"] = n
    out["u_err"] = u

    out["horizontal_err"] = np.sqrt(
        e ** 2 + n ** 2
    )

    out["position_3d_err"] = np.sqrt(
        e ** 2 + n ** 2 + u ** 2
    )

    # ------------------------------------------------------------------
    # 速度误差
    # ------------------------------------------------------------------
    # GREAT-FGO 的速度为 ECEF；
    # IE 真值中已经提供 VEast / VNorth / VUp。

    ve_est, vn_est, vu_est = ecef_vector_to_enu(
        out["vx_est"].to_numpy(),
        out["vy_est"].to_numpy(),
        out["vz_est"].to_numpy(),
        out["lat_gt_deg"].to_numpy(),
        out["lon_gt_deg"].to_numpy(),
    )

    out["ve_est"] = ve_est
    out["vn_est"] = vn_est
    out["vu_est"] = vu_est

    out["ve_err"] = out["ve_est"] - out["ve_gt"]
    out["vn_err"] = out["vn_est"] - out["vn_gt"]
    out["vu_err"] = out["vu_est"] - out["vu_gt"]

    out["velocity_3d_err"] = np.sqrt(
        out["ve_err"] ** 2
        + out["vn_err"] ** 2
        + out["vu_err"] ** 2
    )

    # ------------------------------------------------------------------
    # 姿态误差
    # ------------------------------------------------------------------

    out["pitch_err"] = (
        out["pitch_est"] - out["pitch_gt"]
    )

    out["roll_err"] = (
        out["roll_est"] - out["roll_gt"]
    )

    if yaw_mode == "negative":
        # Vehicle_Opensky 当前 GREAT 输出通常满足：
        # Heading ≈ wrap360(-Yaw)
        out["heading_est"] = wrap_to_360(
            -out["yaw_est"]
        )

    elif yaw_mode == "same":
        out["heading_est"] = wrap_to_360(
            out["yaw_est"]
        )

    else:
        raise ValueError(
            'YAW_MODE 必须是 "negative" 或 "same"'
        )

    # Heading 误差需要处理 0/360 度跳变
    out["heading_err"] = wrap_to_180(
        out["heading_est"] - out["heading_gt"]
    )

    return out


# ======================================================================
#                         精度统计
# ======================================================================

def metric_row(values,
               group,
               category,
               component):

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return None

    return {
        "Group": group,
        "Category": category,
        "Component": component,
        "N": int(values.size),

        "Mean/Bias": float(
            np.mean(values)
        ),

        "Std": float(
            np.std(values, ddof=1)
        ) if values.size > 1 else 0.0,

        "MAE": float(
            np.mean(np.abs(values))
        ),

        "RMSE": float(
            np.sqrt(np.mean(values ** 2))
        ),

        "MaxAbs": float(
            np.max(np.abs(values))
        ),
    }


def summarize(df: pd.DataFrame,
              group_name: str) -> pd.DataFrame:

    rows = []

    specs = [
        ("Position (m)", "East", "e_err"),
        ("Position (m)", "North", "n_err"),
        ("Position (m)", "Up", "u_err"),
        ("Position (m)", "Horizontal", "horizontal_err"),
        ("Position (m)", "3D", "position_3d_err"),

        ("Velocity (m/s)", "East", "ve_err"),
        ("Velocity (m/s)", "North", "vn_err"),
        ("Velocity (m/s)", "Up", "vu_err"),
        ("Velocity (m/s)", "3D", "velocity_3d_err"),

        ("Attitude (deg)", "Heading", "heading_err"),
        ("Attitude (deg)", "Pitch", "pitch_err"),
        ("Attitude (deg)", "Roll", "roll_err"),
    ]

    for category, component, col in specs:
        row = metric_row(
            df[col],
            group_name,
            category,
            component
        )

        if row is not None:
            rows.append(row)

    return pd.DataFrame(rows)


# ======================================================================
#                         绘图
# ======================================================================

def make_plots(df: pd.DataFrame,
               outdir: Path,
               label: str):

    t = df["sow_gt"].to_numpy()
    t0 = t[0]
    x = t - t0

    # ------------------------------------------------------------------
    # ENU 位置误差
    # ------------------------------------------------------------------

    plt.figure(figsize=(11, 6))

    plt.plot(x, df["e_err"], label="East")
    plt.plot(x, df["n_err"], label="North")
    plt.plot(x, df["u_err"], label="Up")

    plt.xlabel(f"Time since SOW {t0:.3f} (s)")
    plt.ylabel("Position error (m)")
    plt.title(f"{label} - Position ENU Error")

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        outdir / "position_enu_error.png",
        dpi=180
    )

    plt.close()

    # ------------------------------------------------------------------
    # 水平 / 三维位置误差
    # ------------------------------------------------------------------

    plt.figure(figsize=(11, 6))

    plt.plot(
        x,
        df["horizontal_err"],
        label="Horizontal"
    )

    plt.plot(
        x,
        df["position_3d_err"],
        label="3D"
    )

    plt.xlabel(f"Time since SOW {t0:.3f} (s)")
    plt.ylabel("Position error norm (m)")
    plt.title(f"{label} - Position Error Norm")

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        outdir / "position_error_norm.png",
        dpi=180
    )

    plt.close()

    # ------------------------------------------------------------------
    # ENU 速度误差
    # ------------------------------------------------------------------

    plt.figure(figsize=(11, 6))

    plt.plot(x, df["ve_err"], label="East")
    plt.plot(x, df["vn_err"], label="North")
    plt.plot(x, df["vu_err"], label="Up")

    plt.xlabel(f"Time since SOW {t0:.3f} (s)")
    plt.ylabel("Velocity error (m/s)")
    plt.title(f"{label} - Velocity ENU Error")

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        outdir / "velocity_enu_error.png",
        dpi=180
    )

    plt.close()

    # ------------------------------------------------------------------
    # 姿态误差
    # ------------------------------------------------------------------

    plt.figure(figsize=(11, 6))

    plt.plot(
        x,
        df["heading_err"],
        label="Heading"
    )

    plt.plot(
        x,
        df["pitch_err"],
        label="Pitch"
    )

    plt.plot(
        x,
        df["roll_err"],
        label="Roll"
    )

    plt.xlabel(f"Time since SOW {t0:.3f} (s)")
    plt.ylabel("Attitude error (deg)")
    plt.title(f"{label} - Attitude Error")

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        outdir / "attitude_error.png",
        dpi=180
    )

    plt.close()


# ======================================================================
#                         终端结果打印
# ======================================================================

def print_key_summary(summary_df: pd.DataFrame,
                      label: str):

    print("\n" + "=" * 76)
    print(f"                 {label} GNSS/INS 精度评定结果")
    print("=" * 76)

    pos = summary_df[
        (summary_df["Group"] == "All")
        & (summary_df["Category"] == "Position (m)")
    ]

    print("\n【全部历元位置精度】")

    for comp in [
        "East",
        "North",
        "Up",
        "Horizontal",
        "3D"
    ]:

        row = pos[
            pos["Component"] == comp
        ]

        if not row.empty:
            r = row.iloc[0]

            print(
                f"{comp:>10s}: "
                f"Bias={r['Mean/Bias']:+.4f} m, "
                f"RMSE={r['RMSE']:.4f} m, "
                f"MAE={r['MAE']:.4f} m, "
                f"MaxAbs={r['MaxAbs']:.4f} m"
            )

    fixed = summary_df[
        (summary_df["Group"] == "Fixed")
        & (summary_df["Category"] == "Position (m)")
    ]

    if not fixed.empty:

        print("\n【仅 Fixed 历元位置精度】")

        for comp in [
            "East",
            "North",
            "Up",
            "Horizontal",
            "3D"
        ]:

            row = fixed[
                fixed["Component"] == comp
            ]

            if not row.empty:
                r = row.iloc[0]

                print(
                    f"{comp:>10s}: "
                    f"RMSE={r['RMSE']:.4f} m, "
                    f"MaxAbs={r['MaxAbs']:.4f} m"
                )


# ======================================================================
#                         主程序
# ======================================================================

def main():

    result_path = Path(RESULT_FILE)
    truth_path = Path(TRUTH_FILE)
    outdir = Path(OUTPUT_DIR)

    # ------------------------------------------------------------------
    # 路径检查
    # ------------------------------------------------------------------

    if not result_path.exists():
        raise FileNotFoundError(
            f"找不到结果文件：\n{result_path}\n\n"
            "请修改脚本顶部 RESULT_FILE。"
        )

    if not truth_path.exists():
        raise FileNotFoundError(
            f"找不到真值文件：\n{truth_path}\n\n"
            "请修改脚本顶部 TRUTH_FILE。"
        )

    outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 76)
    print("GREAT-FGO GNSS/INS 精度评定")
    print("=" * 76)

    print(f"\n结果文件：{result_path}")
    print(f"真值文件：{truth_path}")
    print(f"模式标签：{LABEL}")
    print(f"输出目录：{outdir}")

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------

    print("\n[1/5] 正在读取 GREAT-FGO 结果...")
    ins = load_great_ins(
        str(result_path)
    )

    print(
        f"      读取到 {len(ins)} 个结果历元"
    )

    print("\n[2/5] 正在读取 ROVE Ground Truth...")
    gt = load_rover_ground_truth(
        str(truth_path)
    )

    print(
        f"      读取到 {len(gt)} 个真值历元"
    )

    # ------------------------------------------------------------------
    # 时间范围
    # ------------------------------------------------------------------

    if START_SOW is not None:

        ins = ins[
            ins["sow_est"] >= START_SOW
        ].copy()

        gt = gt[
            gt["sow_gt"] >= START_SOW
        ].copy()

    if END_SOW is not None:

        ins = ins[
            ins["sow_est"] <= END_SOW
        ].copy()

        gt = gt[
            gt["sow_gt"] <= END_SOW
        ].copy()

    # ------------------------------------------------------------------
    # 时间匹配
    # ------------------------------------------------------------------

    print("\n[3/5] 正在进行时间匹配...")

    aligned = align_by_time(
        ins,
        gt,
        TIME_TOLERANCE
    )

    if SKIP_FIRST_EPOCHS > 0:

        aligned = (
            aligned
            .iloc[SKIP_FIRST_EPOCHS:]
            .reset_index(drop=True)
        )

    if aligned.empty:
        raise ValueError(
            "过滤后没有剩余历元。"
        )

    print(
        f"      成功匹配 {len(aligned)} 个历元"
    )

    print(
        "      最大时间匹配误差 = "
        f"{aligned['dt_match'].abs().max():.6f} s"
    )

    # ------------------------------------------------------------------
    # 计算误差
    # ------------------------------------------------------------------

    print("\n[4/5] 正在计算 ENU / 速度 / 姿态误差...")

    result = calculate_errors(
        aligned,
        YAW_MODE
    )

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------

    summaries = [
        summarize(
            result,
            "All"
        )
    ]

    # 另外统计 GREAT-FGO 输出为 Fixed 的历元
    fixed_mask = (
        result["amb_status"]
        .astype(str)
        .str.lower()
        == "fixed"
    )

    if fixed_mask.any():

        summaries.append(
            summarize(
                result.loc[fixed_mask],
                "Fixed"
            )
        )

    summary = pd.concat(
        summaries,
        ignore_index=True
    )

    # ------------------------------------------------------------------
    # 保存 CSV
    # ------------------------------------------------------------------

    epoch_csv = (
        outdir
        / "epoch_errors.csv"
    )

    summary_csv = (
        outdir
        / "summary_metrics.csv"
    )

    result.to_csv(
        epoch_csv,
        index=False,
        float_format="%.9f"
    )

    summary.to_csv(
        summary_csv,
        index=False,
        float_format="%.9f"
    )

    # ------------------------------------------------------------------
    # 绘图
    # ------------------------------------------------------------------

    print("\n[5/5] 正在保存结果...")

    if SAVE_PLOTS:
        make_plots(
            result,
            outdir,
            LABEL
        )

    # ------------------------------------------------------------------
    # 输出统计
    # ------------------------------------------------------------------

    print("\n模糊度状态统计：")

    print(
        result["amb_status"]
        .value_counts(dropna=False)
        .to_string()
    )

    print_key_summary(
        summary,
        LABEL
    )

    print("\n" + "=" * 76)
    print("评定完成")
    print("=" * 76)

    print(f"\n逐历元误差：{epoch_csv}")
    print(f"汇总统计：  {summary_csv}")

    if SAVE_PLOTS:
        print(
            f"位置 ENU 图：{outdir / 'position_enu_error.png'}"
        )
        print(
            f"位置模长图：{outdir / 'position_error_norm.png'}"
        )
        print(
            f"速度 ENU 图：{outdir / 'velocity_enu_error.png'}"
        )
        print(
            f"姿态误差图：{outdir / 'attitude_error.png'}"
        )


if __name__ == "__main__":
    main()
