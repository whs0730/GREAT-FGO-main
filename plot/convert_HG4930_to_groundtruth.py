#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 Inertial Explorer 的 HG4930_GroundTruth.txt 转换成
与 groundtruth_1012_GNSS.txt 相同的列布局。

转换后可直接供原 GREAT-FGO plot/Evaluation.py 使用：
- 位置：usecols_ref=(1, 2, 3, 4)
- 速度：usecols_ref=(1, 15, 16, 17)
- 姿态：usecols_ref=(1, 24, 25, 26)

只需要修改 INPUT_FILE / OUTPUT_FILE。
"""

from pathlib import Path


INPUT_FILE = r"D:\GREAT-FGO-main\sample_data\Data06_20210115_HG4930_UAV_Opensky\ROVE_GroundTruth.txt"
OUTPUT_FILE = r"D:\GREAT-FGO-main\sample_data\Data06_20210115_HG4930_UAV_Opensky\groundtruth_ROVE.txt"


HEADER1 = (
    "      Week    GPSTime       X-ECEF       Y-ECEF       Z-ECEF"
    "         Latitude        Longitude        H-Ell      Easting     Northing       Grid-Z"
    "   VX-ECEF   VY-ECEF   VZ-ECEF          Cx11          Cx22          Cx33"
    "      Vel-Cx11      Vel-Cx22      Vel-Cx33        Heading          Pitch           Roll"
    " AmbStatus    PDOP Q NS      AccBiasX  AngRateX    GyroDriftX      AccBiasY  AngRateY"
    "    GyroDriftY      AccBiasZ  AngRateZ    GyroDriftZ AccBdyX AccBdyY AccBdyZ     UTCTime"
)

HEADER2 = (
    "   (weeks)      (sec)          (m)          (m)          (m)"
    "       (+/-D M S)       (+/-D M S)          (m)          (m)          (m)          (m)"
    "     (m/s)     (m/s)     (m/s)         (m^2)         (m^2)         (m^2)"
    "     (m^2/s^2)     (m^2/s^2)     (m^2/s^2)          (deg)          (deg)          (deg)"
    "             (dop)            (m/s^2)   (deg/s)       (deg/s)       (m/s^2)   (deg/s)"
    "       (deg/s)       (m/s^2)   (deg/s)       (deg/s) (m/s^2) (m/s^2) (m/s^2)       (HMS)"
)


def decimal_deg_to_dms(value):
    sign = -1 if value < 0 else 1
    value = abs(value)

    degree = int(value)
    minute_float = (value - degree) * 60.0
    minute = int(minute_float)
    second = (minute_float - minute) * 60.0

    degree *= sign
    return degree, minute, second


def convert(input_file, output_file):
    rows = []

    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            p = line.split()

            # HG4930 的有效数据行固定 57 列
            if len(p) != 57:
                continue

            try:
                week = int(float(p[0]))
                sow = float(p[1])

                lat = float(p[2])
                lon = float(p[3])
                h = float(p[4])

                quality = int(float(p[5]))
                amb_status = p[6]
                utc_time = p[8]

                x, y, z = map(float, p[9:12])
                vx, vy, vz = map(float, p[15:18])

                heading = float(p[21])
                pitch = float(p[22])
                roll = float(p[23])

                cx11, cx22, cx33 = map(float, p[24:27])
                vel_cx11, vel_cx22, vel_cx33 = map(float, p[30:33])

                ns = int(float(p[39]))
                pdop = float(p[47])

            except (ValueError, IndexError):
                continue

            lat_d, lat_m, lat_s = decimal_deg_to_dms(lat)
            lon_d, lon_m, lon_s = decimal_deg_to_dms(lon)

            fields = [
                f"{week:.5f}",
                f"{sow:.3f}",

                f"{x:.3f}", f"{y:.3f}", f"{z:.3f}",

                str(lat_d), str(lat_m), f"{lat_s:.5f}",
                str(lon_d), str(lon_m), f"{lon_s:.5f}",
                f"{h:.3f}",

                # groundtruth_1012_GNSS.txt 中这三列也是重复 XYZ
                f"{x:.3f}", f"{y:.3f}", f"{z:.3f}",

                f"{vx:.3f}", f"{vy:.3f}", f"{vz:.3f}",

                f"{cx11:.5E}", f"{cx22:.5E}", f"{cx33:.5E}",
                f"{vel_cx11:.5E}", f"{vel_cx22:.5E}", f"{vel_cx33:.5E}",

                f"{heading:.10f}",
                f"{pitch:.10f}",
                f"{roll:.10f}",

                amb_status,
                f"{pdop:.2f}",
                str(quality),
                str(ns),

                # 原 Evaluation.py 不读取以下字段；保持列位置即可
                "0.00000E+00", "0.0000", "0.00000E+00",
                "0.00000E+00", "0.0000", "0.00000E+00",
                "0.00000E+00", "0.0000", "0.00000E+00",
                "0.000", "0.000", "0.000",

                utc_time,
            ]

            rows.append(" ".join(fields))

    if not rows:
        raise ValueError("没有识别到 HG4930 有效数据行。")

    Path(output_file).write_text(
        HEADER1 + "\n"
        + HEADER2 + "\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )

    print(f"转换完成：{len(rows)} 个历元")
    print(f"输出：{output_file}")


if __name__ == "__main__":
    convert(INPUT_FILE, OUTPUT_FILE)
