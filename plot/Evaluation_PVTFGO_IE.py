#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PVTFGO 无组合 RTK 精度评定
-------------------------
用途：
- GREAT_PVTFGO 输出的 .fgo 文件
- Inertial Explorer 的 ROVE_GroundTruth.txt 参考真值

设计原则：
1. 参考文件读取方式沿用 evaluate_gins.py 中 ROVE_GroundTruth.txt 的列定义；
2. 输出格式保持与原 Evaluation.py 一致；
3. 位置误差序列图直接调用原 plot.py 的 plot_position_errors()，
   因此图形布局、颜色、阈值线、纵轴范围、标题和文件名格式均与 Evaluation.py 一致；
4. 无组合 PVTFGO 当前只评定位置精度，不评定姿态；
5. PVTFGO 当前 .fgo 中速度为 0 时，不进行速度精度评定。

运行要求：
- 建议把本文件放到 GREAT-FGO-main/plot/ 目录；
- 与 file_reader.py、statistics.py、plot.py 放在同一目录；
- 只需修改下面“用户配置区”的 3 个路径。
"""

import os
from pathlib import Path

import numpy as np

# 复用原 Evaluation.py 的坐标转换、统计和绘图函数，
# 保证输出统计格式及位置误差图风格一致。
from file_reader import ecef2lla, ecef2enu
from statistics import (
    calculate_all_metrics,
    save_comprehensive_statistics,
    print_statistics,
)
from plot import plot_position_errors


# ======================================================================
#                           用户配置区
# ======================================================================

# GREAT_PVTFGO 无组合 RTK 输出
FILE_DATA = (
    r"D:\GREAT-FGO-main\sample_data\Data05_20201128_HG4930_Vehicle_Opensky"
    r"\result\ROVE-RTK.fgo"
)

# Inertial Explorer 真值
FILE_REF = (
    r"D:\GREAT-FGO-main\sample_data\Data05_20201128_HG4930_Vehicle_Opensky"
    r"\ROVE_GroundTruth.txt"
)

# 输出目录
OUTPUT_DIR = (
    r"D:\GREAT-FGO-main\plot\results"
    r"\Data05_20201128_HG4930_Vehicle_Opensky_PVTFGO"
)

# 最近邻时间匹配允许的最大时间差，单位：s
TIME_TOLERANCE = 0.05

# 可选：限制评定的 GPS SOW 范围；不限制时保持 None
START_SOW = None
END_SOW = None


# ======================================================================
#                         数据读取
# ======================================================================

def read_pvtfgo_position(path):
    """
    读取 GREAT_PVTFGO .fgo 文件中的位置。

    当前 .fgo 前四列：
        0: SOW
        1: X-ECEF
        2: Y-ECEF
        3: Z-ECEF

    返回：
        time_est : (N,)
        pos_est  : (N, 3)
    """
    times = []
    positions = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()

            if not s or s.startswith("#"):
                continue

            p = s.split()
            if len(p) < 4:
                continue

            try:
                sow = float(p[0])
                x = float(p[1])
                y = float(p[2])
                z = float(p[3])
            except (ValueError, IndexError):
                # 兼容可能存在的非 # 表头
                continue

            times.append(sow)
            positions.append([x, y, z])

    if not times:
        raise ValueError(f"没有从 PVTFGO 结果中读取到有效位置数据：\n{path}")

    time_est = np.asarray(times, dtype=float)
    pos_est = np.asarray(positions, dtype=float)

    order = np.argsort(time_est)
    return time_est[order], pos_est[order]


def read_rover_groundtruth_position(path):
    """
    读取 Inertial Explorer 的 ROVE_GroundTruth.txt。

    按之前 evaluate_gins.py 的列定义：
        0 : GPS Week
        1 : GPS SOW
        2 : Latitude (deg)
        3 : Longitude (deg)
        4 : Height
        ...
        9 : X-ECEF
        10: Y-ECEF
        11: Z-ECEF

    此处只读取位置精度评定所需的 SOW 和 ECEF XYZ。

    返回：
        time_ref : (M,)
        pos_ref  : (M, 3)
    """
    times = []
    positions = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()

            if not s:
                continue

            p = s.split()
            if len(p) < 12:
                continue

            try:
                # week 主要用于判断这是不是有效数据行
                int(p[0])
                sow = float(p[1])

                x = float(p[9])
                y = float(p[10])
                z = float(p[11])
            except (ValueError, IndexError):
                # 表头或其他非数据行
                continue

            times.append(sow)
            positions.append([x, y, z])

    if not times:
        raise ValueError(f"没有从真值文件中读取到有效位置数据：\n{path}")

    time_ref = np.asarray(times, dtype=float)
    pos_ref = np.asarray(positions, dtype=float)

    order = np.argsort(time_ref)
    return time_ref[order], pos_ref[order]


# ======================================================================
#                         时间匹配与 ENU 误差
# ======================================================================

def _nearest_ref_index(sorted_times, t):
    """在已排序时间数组中查找距离 t 最近的索引。"""
    i = np.searchsorted(sorted_times, t)

    candidates = []
    if i < len(sorted_times):
        candidates.append(i)
    if i > 0:
        candidates.append(i - 1)

    if not candidates:
        return None

    return min(candidates, key=lambda idx: abs(sorted_times[idx] - t))


def read_position_data_enu_ie(file_data, file_ref, tolerance=0.05):
    """
    功能对应原 Evaluation.py 中的 read_position_data_enu()，
    但针对 ROVE_GroundTruth.txt 的实际列格式读取真值。

    为保持和原 Evaluation.py 的位置误差定义一致：
    - 使用真值第一个历元的 ECEF 点确定固定 ENU 坐标系；
    - 所有 ECEF 位置差均转换到这个固定 ENU 坐标系；
    - 输出 matched_time, de, dn, du。

    时间匹配采用最近邻，并要求 |dt| <= tolerance。
    """
    time_est, pos_est = read_pvtfgo_position(file_data)
    time_ref, pos_ref = read_rover_groundtruth_position(file_ref)

    # 可选时间范围
    est_mask = np.ones(len(time_est), dtype=bool)
    ref_mask = np.ones(len(time_ref), dtype=bool)

    if START_SOW is not None:
        est_mask &= time_est >= START_SOW
        ref_mask &= time_ref >= START_SOW

    if END_SOW is not None:
        est_mask &= time_est <= END_SOW
        ref_mask &= time_ref <= END_SOW

    time_est = time_est[est_mask]
    pos_est = pos_est[est_mask]
    time_ref = time_ref[ref_mask]
    pos_ref = pos_ref[ref_mask]

    if len(time_est) == 0:
        raise ValueError("时间范围过滤后没有 PVTFGO 结果历元。")

    if len(time_ref) == 0:
        raise ValueError("时间范围过滤后没有真值历元。")

    # 与原 file_reader.read_position_data_enu() 保持一致：
    # 用第一个真值点建立固定 ENU 坐标系。
    first_ref_point = pos_ref[0]
    ref_lat, ref_lon, ref_alt = ecef2lla(
        first_ref_point[0],
        first_ref_point[1],
        first_ref_point[2],
    )

    print(
        f"Fixed reference point: "
        f"Lat={ref_lat:.6f}°, "
        f"Lon={ref_lon:.6f}°, "
        f"Alt={ref_alt:.2f}m"
    )

    matched_time = []
    de_list = []
    dn_list = []
    du_list = []
    dt_list = []

    for t, est_xyz in zip(time_est, pos_est):
        idx = _nearest_ref_index(time_ref, t)
        if idx is None:
            continue

        dt = t - time_ref[idx]
        if abs(dt) > tolerance:
            continue

        ref_xyz = pos_ref[idx]

        dx = est_xyz[0] - ref_xyz[0]
        dy = est_xyz[1] - ref_xyz[1]
        dz = est_xyz[2] - ref_xyz[2]

        de, dn, du = ecef2enu(
            ref_lat,
            ref_lon,
            dx,
            dy,
            dz,
        )

        # Evaluation.py 原来使用整数秒作为绘图横轴。
        # PVTFGO 一般也是整数 GNSS 历元，这里保持同样表现。
        matched_time.append(int(t))
        de_list.append(de)
        dn_list.append(dn)
        du_list.append(du)
        dt_list.append(dt)

    if not matched_time:
        raise ValueError(
            "没有成功匹配的结果历元。\n"
            "请检查 PVTFGO 与真值的 GPS SOW 是否一致，"
            "或适当增大 TIME_TOLERANCE。"
        )

    print(f"Matched epochs: {len(matched_time)}")
    print(f"Maximum time difference: {np.max(np.abs(dt_list)):.6f} s")

    return (
        np.asarray(matched_time),
        np.asarray(de_list),
        np.asarray(dn_list),
        np.asarray(du_list),
    )


# ======================================================================
#                              主程序
# ======================================================================

def main():
    file_data = FILE_DATA
    file_ref = FILE_REF
    output_dir = OUTPUT_DIR

    # 路径检查
    if not Path(file_data).exists():
        raise FileNotFoundError(
            f"找不到结果文件：\n{file_data}\n\n"
            "请修改脚本顶部 FILE_DATA。"
        )

    if not Path(file_ref).exists():
        raise FileNotFoundError(
            f"找不到真值文件：\n{file_ref}\n\n"
            "请修改脚本顶部 FILE_REF。"
        )

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    file_basename = os.path.splitext(os.path.basename(file_data))[0]

    # 与原 Evaluation.py 保持相同变量结构
    pos_errors_x, pos_errors_y, pos_errors_z = None, None, None
    vel_errors_x, vel_errors_y, vel_errors_z = None, None, None
    att_errors_x, att_errors_y, att_errors_z = None, None, None

    print("Processing position data (ENU coordinates)...")

    try:
        # 读取 PVTFGO + IE GroundTruth，计算固定 ENU 下的位置误差
        matched_time_pos, de, dn, du = read_position_data_enu_ie(
            file_data,
            file_ref,
            TIME_TOLERANCE,
        )

        pos_errors_x, pos_errors_y, pos_errors_z = de, dn, du

        # 与原 Evaluation.py 完全相同的统计函数
        pos_metrics = calculate_all_metrics(
            de,
            dn,
            du,
            "Position",
        )

        # 关键：直接复用 Evaluation.py 原来的绘图函数。
        # 输出仍为：
        #   <file_basename>_position.png
        # 且 3 个子图、散点颜色、±0.1 m 阈值线、ylim(-1,1)、
        # MAE/RMSE 标题等均保持一致。
        plot_position_errors(
            matched_time_pos,
            de,
            dn,
            du,
            pos_metrics,
            file_basename,
            output_dir,
        )

        print_statistics(
            pos_metrics,
            "Position",
        )

    except Exception as e:
        print(f"Error processing position data: {e}")

    # PVTFGO 无组合结果目前不进行速度/姿态精度评定。
    # 保持 None 后，save_comprehensive_statistics 只输出位置统计。

    print("\nSaving comprehensive statistics...")

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
            file_data=file_data,
            file_ref=file_ref,
            output_path=os.path.join(
                output_dir,
                f"{file_basename}_statistics.txt",
            ),
        )
        print("Comprehensive statistics saved successfully!")

    except Exception as e:
        print(f"Error saving comprehensive statistics: {e}")

    print(
        f"\nAll processing completed! "
        f"Results saved to: {output_dir}"
    )


if __name__ == "__main__":
    main()
