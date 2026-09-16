#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GREAT-FGO 统一精度评定脚本
=========================

支持三种模式：
    1. PVTFGO : 无组合 GNSS/RTK-FGO，结果文件为 .fgo
    2. LC     : RTK/INS 松耦合，结果文件为 .ins
    3. TC     : RTK/INS 紧耦合，结果文件为 .ins

设计目标：
    - 在一个 evaluate_gins.py 中统一评定 PVTFGO / LC / TC；
    - 不同模式可使用不同格式的参考真值；
    - 输出格式保持与原 Evaluation.py 一致；
    - 位置误差序列图直接复用 plot.py 的 plot_position_errors()，
      因此三幅 East/North/Up 子图、颜色、阈值线、纵轴范围、
      MAE/RMSE 标题和文件命名均与 Evaluation.py 保持一致；
    - 统计结果直接复用 statistics.py 的函数。

输出：
    <结果文件名>_position.png
    <结果文件名>_velocity.png      （仅 LC / TC）
    <结果文件名>_attitude.png      （仅 LC / TC）
    <结果文件名>_statistics.txt

建议：
    将本文件放在 GREAT-FGO-main/plot/ 下，与
    file_reader.py、statistics.py、plot.py 放在同一目录。
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from file_reader import ecef2lla, ecef2enu
from statistics import (
    calculate_all_metrics,
    save_comprehensive_statistics,
    print_statistics,
)
from plot import (
    plot_position_errors,
    plot_velocity_errors,
    plot_attitude_errors,
)


# ======================================================================
#                           用户配置区
# ======================================================================

# 可选：
#   "PVTFGO" : 无组合
#   "LC"     : 松耦合
#   "TC"     : 紧耦合
MODE = "LC"


# ----------------------------------------------------------------------
# 方法一：每次只修改下面三个路径
# ----------------------------------------------------------------------

RESULT_FILE = r"D:\GREAT-FGO-main\sample_data\Data05_20201128_HG4930_Vehicle_Opensky\result\Vehicle_Opensky_LC.ins"
TRUTH_FILE = r"D:\GREAT-FGO-main\sample_data\Data05_20201128_HG4930_Vehicle_Opensky\HG4930_GroundTruth.txt"
OUTPUT_DIR = r"D:\GREAT-FGO-main\plot\results\Data05_20201128_HG4930_Vehicle_Opensky_LC"


# ----------------------------------------------------------------------
# 可选设置
# ----------------------------------------------------------------------

# 最近邻时间匹配最大允许误差，单位 s
TIME_TOLERANCE = 0.05

# 评定时间范围（GPS SOW）；不限制保持 None
START_SOW = None
END_SOW = None

# 跳过最前面的已匹配历元。
# 如果某一模式初始化阶段不稳定，可设置为 1、2 等。
SKIP_FIRST_EPOCHS = 0

# GREAT-FGO .ins 中 Yaw 与真值 Heading 的关系：
# "negative": Heading = wrap360(-Yaw)
# "same"    : Heading = wrap360(Yaw)
YAW_MODE = "negative"


# ======================================================================
#                         通用辅助函数
# ======================================================================

def wrap_to_180(angle_deg):
    """角度归一化到 [-180, 180)。"""
    a = np.asarray(angle_deg, dtype=float)
    return (a + 180.0) % 360.0 - 180.0


def wrap_to_360(angle_deg):
    """角度归一化到 [0, 360)。"""
    a = np.asarray(angle_deg, dtype=float)
    return a % 360.0


def _looks_like_ecef(x, y, z):
    """
    粗略判断三个数是否像 ECEF XYZ。
    ECEF 坐标模长通常约为地球半径量级。
    """
    try:
        xyz = np.array([float(x), float(y), float(z)], dtype=float)
    except (TypeError, ValueError):
        return False

    if not np.all(np.isfinite(xyz)):
        return False

    norm = np.linalg.norm(xyz)
    return 5.0e6 < norm < 8.0e6


# ======================================================================
#                         结果文件读取
# ======================================================================

def load_pvtfgo_result(path: str) -> pd.DataFrame:
    """
    读取 GREAT_PVTFGO 的 .fgo 文件。

    当前 ROVE-RTK.fgo / SEPT-RTK.fgo 格式：
        0  SOW
        1  X-ECEF
        2  Y-ECEF
        3  Z-ECEF
        4  Vx-ECEF
        5  Vy-ECEF
        6  Vz-ECEF
        7  X-RMS
        8  Y-RMS
        9  Z-RMS
        10 Vx-RMS
        11 Vy-RMS
        12 Vz-RMS
        13 NSat
        14 PDOP
        15 sigma0
        16 AmbStatus
        17 Ratio
        18 BL
        19 Quality

    无组合模式本脚本只使用位置进行精度评定。
    """
    rows = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()

            if not s or s.startswith("#"):
                continue

            p = s.split()
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

                    "x_rms": float(p[7]),
                    "y_rms": float(p[8]),
                    "z_rms": float(p[9]),

                    "nsat": int(float(p[13])),
                    "pdop": float(p[14]),
                    "sigma0": float(p[15]),

                    "amb_status": p[16],
                    "ratio": float(p[17]),
                    "baseline": float(p[18]),
                    "quality": int(float(p[19])),
                }

            except (ValueError, IndexError):
                print(f"[WARN] 跳过 PVTFGO 第 {line_no} 行：{s[:100]}")
                continue

            rows.append(row)

    if not rows:
        raise ValueError(f"没有从 PVTFGO 结果中读取到有效数据：\n{path}")

    return (
        pd.DataFrame(rows)
        .sort_values("sow_est")
        .reset_index(drop=True)
    )


def load_gins_result(path: str) -> pd.DataFrame:
    """
    读取 LC / TC 的 GREAT-GINSFGO .ins 文件。

    按原 evaluate_gins.py 的字段定义：
        0  SOW
        1  X-ECEF
        2  Y-ECEF
        3  Z-ECEF
        4  Vx-ECEF
        5  Vy-ECEF
        6  Vz-ECEF
        7  Pitch
        8  Roll
        9  Yaw
        ...
        16 MeasType
        17 NSat
        18 PDOP
        19 AmbStatus
        20 Ratio（若存在）
    """
    rows = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()

            if not s or s.startswith("#"):
                continue

            p = s.split()
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
                print(f"[WARN] 跳过 GINS 第 {line_no} 行：{s[:100]}")
                continue

            rows.append(row)

    if not rows:
        raise ValueError(f"没有从 LC/TC .ins 中读取到有效数据：\n{path}")

    return (
        pd.DataFrame(rows)
        .sort_values("sow_est")
        .reset_index(drop=True)
    )


# ======================================================================
#                         参考真值读取
# ======================================================================

def load_truth(path: str) -> pd.DataFrame:
    """
    统一读取不同格式的参考真值。

    支持两类位置格式：

    A. GNSS groundtruth（例如 groundtruth_0928_GNSS.txt）
       常见格式：
           p[0] GPS Week
           p[1] GPS SOW
           p[2:5] ECEF XYZ

    B. 原 evaluate_gins.py 使用的 IE / ROVE_GroundTruth 格式
       常见格式：
           p[0]     GPS Week
           p[1]     GPS SOW
           p[2:5]   Lat/Lon/H
           p[9:12]  ECEF XYZ
           p[15:18] ECEF Velocity
           p[21:24] Heading/Pitch/Roll

    另外兼容较长的 MTI 真值：
       若一行 >= 27 列，则姿态优先尝试 p[24:27]。

    位置格式通过 ECEF 数值量级自动判断，因此 PVTFGO、LC、TC
    可以使用不同参考文件，而无需改数据处理主流程。
    """
    rows = []
    position_layout = None
    attitude_layout = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()

            if not s:
                continue

            p = s.split()
            if len(p) < 5:
                continue

            try:
                week = int(float(p[0]))
                sow = float(p[1])
            except ValueError:
                # 表头
                continue

            # ----------------------------------------------------------
            # 位置：自动识别
            # ----------------------------------------------------------
            x = y = z = None
            lat = lon = h = np.nan

            # GNSS truth：第 2~4 列直接是 ECEF
            if _looks_like_ecef(p[2], p[3], p[4]):
                x = float(p[2])
                y = float(p[3])
                z = float(p[4])

                lat, lon, h = ecef2lla(x, y, z)
                current_position_layout = "GNSS: ECEF columns 2,3,4"

            # IE / ROVE GroundTruth：第 9~11 列是 ECEF
            elif len(p) >= 12 and _looks_like_ecef(p[9], p[10], p[11]):
                x = float(p[9])
                y = float(p[10])
                z = float(p[11])

                try:
                    lat = float(p[2])
                    lon = float(p[3])
                    h = float(p[4])
                except ValueError:
                    lat, lon, h = ecef2lla(x, y, z)

                current_position_layout = "IE/ROVE: ECEF columns 9,10,11"

            else:
                print(f"[WARN] 真值第 {line_no} 行无法识别 ECEF：{s[:100]}")
                continue

            if position_layout is None:
                position_layout = current_position_layout

            row = {
                "week": week,
                "sow_gt": sow,

                "lat_gt_deg": float(lat),
                "lon_gt_deg": float(lon),
                "h_gt": float(h),

                "x_gt": x,
                "y_gt": y,
                "z_gt": z,

                "vx_gt": np.nan,
                "vy_gt": np.nan,
                "vz_gt": np.nan,

                "heading_gt": np.nan,
                "pitch_gt": np.nan,
                "roll_gt": np.nan,
            }

            # ----------------------------------------------------------
            # 速度：LC / TC 常用 ECEF 速度列
            # ----------------------------------------------------------
            if len(p) >= 18:
                try:
                    row["vx_gt"] = float(p[15])
                    row["vy_gt"] = float(p[16])
                    row["vz_gt"] = float(p[17])
                except ValueError:
                    pass

            # ----------------------------------------------------------
            # 姿态：兼容两种常见布局
            # ----------------------------------------------------------

            # 原 evaluate_gins.py / ROVE_GroundTruth
            if len(p) >= 24:
                try:
                    h21 = float(p[21])
                    p22 = float(p[22])
                    r23 = float(p[23])

                    row["heading_gt"] = h21
                    row["pitch_gt"] = p22
                    row["roll_gt"] = r23

                    current_attitude_layout = "Heading/Pitch/Roll columns 21,22,23"
                except ValueError:
                    current_attitude_layout = None

                # 对 >=27 列的 MTI 文件，若末三列为有效数值，
                # 优先使用项目旧 file_reader.py 对应的 24/25/26。
                if len(p) >= 27:
                    try:
                        h24 = float(p[24])
                        p25 = float(p[25])
                        r26 = float(p[26])

                        if np.all(np.isfinite([h24, p25, r26])):
                            row["heading_gt"] = h24
                            row["pitch_gt"] = p25
                            row["roll_gt"] = r26
                            current_attitude_layout = (
                                "Heading/Pitch/Roll columns 24,25,26"
                            )
                    except ValueError:
                        pass

                if attitude_layout is None and current_attitude_layout is not None:
                    attitude_layout = current_attitude_layout

            rows.append(row)

    if not rows:
        raise ValueError(f"没有从参考真值中读取到有效数据：\n{path}")

    print(f"Truth position format: {position_layout}")
    if attitude_layout is not None:
        print(f"Truth attitude format: {attitude_layout}")

    return (
        pd.DataFrame(rows)
        .sort_values("sow_gt")
        .reset_index(drop=True)
    )


# ======================================================================
#                         时间筛选和匹配
# ======================================================================

def filter_time_range(est: pd.DataFrame,
                      gt: pd.DataFrame):
    """按 START_SOW / END_SOW 过滤。"""
    est = est.copy()
    gt = gt.copy()

    if START_SOW is not None:
        est = est[est["sow_est"] >= START_SOW].copy()
        gt = gt[gt["sow_gt"] >= START_SOW].copy()

    if END_SOW is not None:
        est = est[est["sow_est"] <= END_SOW].copy()
        gt = gt[gt["sow_gt"] <= END_SOW].copy()

    if est.empty:
        raise ValueError("时间范围过滤后没有结果历元。")

    if gt.empty:
        raise ValueError("时间范围过滤后没有真值历元。")

    return est, gt


def align_by_time(est: pd.DataFrame,
                  gt: pd.DataFrame,
                  tolerance: float) -> pd.DataFrame:
    """
    按 GPS SOW 最近邻匹配。

    相比原 Evaluation.py 的 int(t) 字典匹配，
    这里保留原 evaluate_gins.py 的容差匹配思想，
    对存在毫秒级时间差的 IE/MTI 真值更稳健。
    """
    est = est.sort_values("sow_est").copy()
    gt = gt.sort_values("sow_gt").copy()

    aligned = pd.merge_asof(
        est,
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
            "请检查结果和真值的 GPS SOW，"
            "或适当增大 TIME_TOLERANCE。"
        )

    aligned["dt_match"] = (
        aligned["sow_est"] - aligned["sow_gt"]
    )

    if SKIP_FIRST_EPOCHS > 0:
        aligned = (
            aligned
            .iloc[SKIP_FIRST_EPOCHS:]
            .reset_index(drop=True)
        )

    if aligned.empty:
        raise ValueError("SKIP_FIRST_EPOCHS 后没有剩余历元。")

    return aligned.reset_index(drop=True)


# ======================================================================
#                         误差计算
# ======================================================================

def calculate_position_errors(df: pd.DataFrame,
                              gt_full: pd.DataFrame):
    """
    位置误差定义与原 Evaluation.py 保持一致：

    1. P_est - P_truth 得到 ECEF 差值；
    2. 用参考真值第一个历元建立固定 ENU 坐标系；
    3. 所有历元都旋转到这个固定 ENU 坐标系。

    返回：
        matched_time, de, dn, du
    """
    first_ref = gt_full.iloc[0]

    ref_lat = float(first_ref["lat_gt_deg"])
    ref_lon = float(first_ref["lon_gt_deg"])
    ref_alt = float(first_ref["h_gt"])

    print(
        f"Fixed reference point: "
        f"Lat={ref_lat:.6f}°, "
        f"Lon={ref_lon:.6f}°, "
        f"Alt={ref_alt:.2f}m"
    )

    de_list = []
    dn_list = []
    du_list = []

    for row in df.itertuples(index=False):
        dx = row.x_est - row.x_gt
        dy = row.y_est - row.y_gt
        dz = row.z_est - row.z_gt

        de, dn, du = ecef2enu(
            ref_lat,
            ref_lon,
            dx,
            dy,
            dz,
        )

        de_list.append(de)
        dn_list.append(dn)
        du_list.append(du)

    # 原 Evaluation.py 中绘图时间使用 int(t)
    matched_time = df["sow_est"].astype(int).to_numpy()

    return (
        matched_time,
        np.asarray(de_list),
        np.asarray(dn_list),
        np.asarray(du_list),
    )


def calculate_velocity_errors(df: pd.DataFrame,
                              gt_full: pd.DataFrame):
    """
    LC / TC 速度误差。

    结果和真值都使用 ECEF 速度：
        dV_ECEF = V_est - V_truth

    再使用与位置相同的固定 ENU 坐标系旋转速度误差，
    从而避免依赖不同真值文件中 VE/VN/VU 的列布局。
    """
    required = ["vx_gt", "vy_gt", "vz_gt"]

    if df[required].isna().any(axis=None):
        raise ValueError(
            "参考文件缺少有效 ECEF 速度列，无法评定速度。"
        )

    first_ref = gt_full.iloc[0]
    ref_lat = float(first_ref["lat_gt_deg"])
    ref_lon = float(first_ref["lon_gt_deg"])

    dve = []
    dvn = []
    dvu = []

    for row in df.itertuples(index=False):
        dvx = row.vx_est - row.vx_gt
        dvy = row.vy_est - row.vy_gt
        dvz = row.vz_est - row.vz_gt

        e, n, u = ecef2enu(
            ref_lat,
            ref_lon,
            dvx,
            dvy,
            dvz,
        )

        dve.append(e)
        dvn.append(n)
        dvu.append(u)

    matched_time = df["sow_est"].astype(int).to_numpy()

    return (
        matched_time,
        np.asarray(dve),
        np.asarray(dvn),
        np.asarray(dvu),
    )


def calculate_attitude_errors(df: pd.DataFrame):
    """
    LC / TC 姿态误差。

    输出顺序固定为：
        Heading/Yaw, Pitch, Roll

    这样可直接传给原 plot_attitude_errors() 和
    calculate_all_metrics(..., "Attitude")。
    """
    required = [
        "heading_gt",
        "pitch_gt",
        "roll_gt",
    ]

    if df[required].isna().any(axis=None):
        raise ValueError(
            "参考文件缺少有效姿态数据，无法评定姿态。"
        )

    if YAW_MODE == "negative":
        heading_est = wrap_to_360(
            -df["yaw_est"].to_numpy()
        )
    elif YAW_MODE == "same":
        heading_est = wrap_to_360(
            df["yaw_est"].to_numpy()
        )
    else:
        raise ValueError(
            'YAW_MODE 必须是 "negative" 或 "same"'
        )

    heading_err = wrap_to_180(
        heading_est
        - df["heading_gt"].to_numpy()
    )

    pitch_err = (
        df["pitch_est"].to_numpy()
        - df["pitch_gt"].to_numpy()
    )

    roll_err = (
        df["roll_est"].to_numpy()
        - df["roll_gt"].to_numpy()
    )

    matched_time = df["sow_est"].astype(int).to_numpy()

    return (
        matched_time,
        heading_err,
        pitch_err,
        roll_err,
    )


# ======================================================================
#                              主程序
# ======================================================================

def main():
    mode = MODE.upper().strip()

    if mode not in {"PVTFGO", "LC", "TC"}:
        raise ValueError(
            'MODE 必须是 "PVTFGO"、"LC" 或 "TC"'
        )

    result_path = Path(RESULT_FILE)
    truth_path = Path(TRUTH_FILE)
    output_dir = Path(OUTPUT_DIR)

    if not result_path.exists():
        raise FileNotFoundError(
            f"找不到结果文件：\n{result_path}\n\n"
            "请修改 RESULT_FILE。"
        )

    if not truth_path.exists():
        raise FileNotFoundError(
            f"找不到真值文件：\n{truth_path}\n\n"
            "请修改 TRUTH_FILE。"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_basename = os.path.splitext(
        os.path.basename(RESULT_FILE)
    )[0]

    print("=" * 76)
    print("GREAT-FGO Unified Accuracy Evaluation")
    print("=" * 76)
    print(f"Mode      : {mode}")
    print(f"Data      : {RESULT_FILE}")
    print(f"Reference : {TRUTH_FILE}")
    print(f"Output    : {OUTPUT_DIR}")

    # ------------------------------------------------------------------
    # 1. 读取结果
    # ------------------------------------------------------------------

    print("\nReading result data...")

    if mode == "PVTFGO":
        est = load_pvtfgo_result(
            str(result_path)
        )
    else:
        est = load_gins_result(
            str(result_path)
        )

    print(f"Result epochs: {len(est)}")

    # ------------------------------------------------------------------
    # 2. 读取参考真值
    # ------------------------------------------------------------------

    print("\nReading reference data...")

    gt = load_truth(
        str(truth_path)
    )

    print(f"Reference epochs: {len(gt)}")

    # ------------------------------------------------------------------
    # 3. 时间范围 + 时间匹配
    # ------------------------------------------------------------------

    est, gt_filtered = filter_time_range(
        est,
        gt,
    )

    aligned = align_by_time(
        est,
        gt_filtered,
        TIME_TOLERANCE,
    )

    print(f"Matched epochs: {len(aligned)}")
    print(
        "Maximum time difference: "
        f"{aligned['dt_match'].abs().max():.6f} s"
    )

    # ------------------------------------------------------------------
    # Evaluation.py 风格的输出变量
    # ------------------------------------------------------------------

    pos_errors_x = pos_errors_y = pos_errors_z = None
    vel_errors_x = vel_errors_y = vel_errors_z = None
    att_errors_x = att_errors_y = att_errors_z = None

    # ------------------------------------------------------------------
    # 4. Position
    # ------------------------------------------------------------------

    print("\nProcessing position data (ENU coordinates)...")

    try:
        matched_time_pos, de, dn, du = calculate_position_errors(
            aligned,
            gt_filtered,
        )

        pos_errors_x = de
        pos_errors_y = dn
        pos_errors_z = du

        pos_metrics = calculate_all_metrics(
            de,
            dn,
            du,
            "Position",
        )

        # 直接调用原 Evaluation.py 使用的绘图函数。
        # 因此位置误差序列图格式完全复用 plot.py。
        plot_position_errors(
            matched_time_pos,
            de,
            dn,
            du,
            pos_metrics,
            file_basename,
            str(output_dir),
        )

        print_statistics(
            pos_metrics,
            "Position",
        )

    except Exception as e:
        print(f"Error processing position data: {e}")
        raise

    # ------------------------------------------------------------------
    # 5. Velocity / Attitude
    #    仅 LC / TC
    # ------------------------------------------------------------------

    if mode in {"LC", "TC"}:

        print("\nProcessing velocity data...")

        try:
            matched_time_vel, dve, dvn, dvu = calculate_velocity_errors(
                aligned,
                gt_filtered,
            )

            vel_errors_x = dve
            vel_errors_y = dvn
            vel_errors_z = dvu

            vel_metrics = calculate_all_metrics(
                dve,
                dvn,
                dvu,
                "Velocity",
            )

            plot_velocity_errors(
                matched_time_vel,
                dve,
                dvn,
                dvu,
                vel_metrics,
                file_basename,
                str(output_dir),
            )

            print_statistics(
                vel_metrics,
                "Velocity",
            )

        except Exception as e:
            print(f"Velocity data not available or error: {e}")

        print("\nProcessing attitude data...")

        try:
            (
                matched_time_att,
                dheading,
                dpitch,
                droll,
            ) = calculate_attitude_errors(
                aligned
            )

            # statistics.py / plot.py 的 Attitude 标签顺序是
            # Yaw, Pitch, Roll，因此第一组放 Heading/Yaw。
            att_errors_x = dheading
            att_errors_y = dpitch
            att_errors_z = droll

            att_metrics = calculate_all_metrics(
                dheading,
                dpitch,
                droll,
                "Attitude",
            )

            plot_attitude_errors(
                matched_time_att,
                dheading,
                dpitch,
                droll,
                att_metrics,
                file_basename,
                str(output_dir),
            )

            print_statistics(
                att_metrics,
                "Attitude",
            )

        except Exception as e:
            print(f"Attitude data not available or error: {e}")

    else:
        print(
            "\nPVTFGO is GNSS-only: "
            "velocity/attitude evaluation is skipped."
        )

    # ------------------------------------------------------------------
    # 6. Comprehensive statistics
    # ------------------------------------------------------------------

    print("\nSaving comprehensive statistics...")

    statistics_path = os.path.join(
        str(output_dir),
        f"{file_basename}_statistics.txt",
    )

    try:
        save_comprehensive_statistics(
            pos_errors_x=pos_errors_x,
            pos_errors_y=pos_errors_y,
            pos_errors_z=pos_errors_z,

            vel_errors_x=vel_errors_x,
            vel_errors_y=vel_errors_y,
            vel_errors_z=vel_errors_z,

            att_errors_x=att_errors_x,
            att_errors_y=att_errors_y,
            att_errors_z=att_errors_z,

            file_data=RESULT_FILE,
            file_ref=TRUTH_FILE,
            output_path=statistics_path,
        )

        print(
            "Comprehensive statistics "
            "saved successfully!"
        )

    except Exception as e:
        print(
            f"Error saving comprehensive statistics: {e}"
        )

    # ------------------------------------------------------------------
    # 7. 完成
    # ------------------------------------------------------------------

    print(
        f"\nAll processing completed! "
        f"Results saved to: {OUTPUT_DIR}"
    )

    print("\nOutput files:")
    print(
        f"  Position  : "
        f"{output_dir / (file_basename + '_position.png')}"
    )

    if mode in {"LC", "TC"}:
        print(
            f"  Velocity  : "
            f"{output_dir / (file_basename + '_velocity.png')}"
        )
        print(
            f"  Attitude  : "
            f"{output_dir / (file_basename + '_attitude.png')}"
        )

    print(
        f"  Statistics: {statistics_path}"
    )


if __name__ == "__main__":
    main()
