#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot GINSFGO_LC and GREAT-MSF LC ENU position-error series.

The two methods are evaluated on identical epochs and against the same
IMU-centre truth.  Four case figures and their aligned error CSV files are
written to plot/results/GINSFGO_LC_vs_MSF（2）/ENU_error_series.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
PLOT_DIR = ROOT / "plot"
OUTPUT_DIR = (
    PLOT_DIR
    / "results"
    / "GINSFGO_LC_vs_MSF（2）"
    / "ENU_error_series"
)

BLUE = "#2563EB"
ORANGE = "#D97706"
INK = "#20242A"
MUTED = "#5B6470"
GRID = "#E5E7EB"
ZERO = "#7B8490"


def load_comparison_module():
    path = PLOT_DIR / "compare_ginsfgo_lc_msf.py"
    spec = importlib.util.spec_from_file_location("lc_compare", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load comparison module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(np.square(values))))


def text_center(draw, xy, text, font, fill):
    box = draw.textbbox((0, 0), text, font=font)
    draw.text(
        (xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2),
        text,
        font=font,
        fill=fill,
    )


def draw_marker(draw, x, y, color, shape):
    radius = 5
    if shape == "circle":
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="white", outline=color, width=3)
    else:
        draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill="white", outline=color, width=3)


def draw_subplot(
    draw,
    rect,
    elapsed,
    gins_values,
    msf_values,
    component,
    fonts,
):
    left, top, right, bottom = rect
    width = right - left
    height = bottom - top

    x_min = float(elapsed[0])
    x_max = float(elapsed[-1])
    if x_max <= x_min:
        x_max = x_min + 1.0

    combined = np.concatenate((gins_values, msf_values, np.array([0.0])))
    y_min = float(np.nanmin(combined))
    y_max = float(np.nanmax(combined))
    span = y_max - y_min
    if span < 1e-6:
        span = 1.0
    padding = span * 0.08
    y_min -= padding
    y_max += padding

    def px(value):
        return left + (float(value) - x_min) / (x_max - x_min) * width

    def py(value):
        return bottom - (float(value) - y_min) / (y_max - y_min) * height

    for index in range(6):
        x_value = x_min + (x_max - x_min) * index / 5
        x = px(x_value)
        draw.line((x, top, x, bottom), fill=GRID, width=1)
        text_center(draw, (x, bottom + 26), f"{x_value:.0f}", fonts["tick"], MUTED)

    for index in range(5):
        y_value = y_min + (y_max - y_min) * index / 4
        y = py(y_value)
        draw.line((left, y, right, y), fill=GRID, width=1)
        label = f"{y_value:.2f}" if abs(y_value) < 10 else f"{y_value:.1f}"
        box = draw.textbbox((0, 0), label, font=fonts["tick"])
        draw.text((left - 18 - (box[2] - box[0]), y - 12), label, font=fonts["tick"], fill=MUTED)

    if y_min <= 0.0 <= y_max:
        draw.line((left, py(0.0), right, py(0.0)), fill=ZERO, width=2)

    draw.line((left, top, left, bottom), fill=INK, width=2)
    draw.line((left, bottom, right, bottom), fill=INK, width=2)

    gins_points = [(px(t), py(v)) for t, v in zip(elapsed, gins_values)]
    msf_points = [(px(t), py(v)) for t, v in zip(elapsed, msf_values)]
    if len(gins_points) >= 2:
        draw.line(gins_points, fill=BLUE, width=3)
        draw.line(msf_points, fill=ORANGE, width=3)

    marker_step = max(1, len(elapsed) // 18)
    for index in range(0, len(elapsed), marker_step):
        draw_marker(draw, *gins_points[index], BLUE, "circle")
        draw_marker(draw, *msf_points[index], ORANGE, "square")

    title = f"{component} error (m)"
    draw.text((left, top - 42), title, font=fonts["subplot"], fill=INK)
    metric = f"RMSE  GINSFGO={rmse(gins_values):.3f} m   MSF={rmse(msf_values):.3f} m"
    metric_box = draw.textbbox((0, 0), metric, font=fonts["small"])
    draw.text((right - (metric_box[2] - metric_box[0]), top - 38), metric, font=fonts["small"], fill=MUTED)


def create_case_figure(case, sow, g_errors, m_errors, output, chart_font):
    width, height = 1900, 1600
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    fonts = {
        "title": chart_font(44, bold=True),
        "subtitle": chart_font(25),
        "legend": chart_font(27),
        "subplot": chart_font(29, bold=True),
        "small": chart_font(22),
        "tick": chart_font(21),
        "axis": chart_font(27),
    }

    elapsed = np.asarray(sow, dtype=float) - float(sow[0])
    draw.text((210, 38), f"{case}: ENU position-error series", font=fonts["title"], fill=INK)
    draw.text(
        (210, 98),
        f"Common epochs: {len(sow)} | SOW {sow[0]:.3f}–{sow[-1]:.3f} | Same IMU-centre truth",
        font=fonts["subtitle"],
        fill=MUTED,
    )

    draw.line((220, 160, 300, 160), fill=BLUE, width=5)
    draw_marker(draw, 260, 160, BLUE, "circle")
    draw.text((320, 143), "GINSFGO_LC", font=fonts["legend"], fill=INK)
    draw.line((570, 160, 650, 160), fill=ORANGE, width=5)
    draw_marker(draw, 610, 160, ORANGE, "square")
    draw.text((670, 143), "MSF_LC", font=fonts["legend"], fill=INK)

    left, right = 210, width - 80
    subplot_height = 320
    gap = 120
    first_top = 245
    labels = ["East", "North", "Up"]
    for index, label in enumerate(labels):
        top = first_top + index * (subplot_height + gap)
        draw_subplot(
            draw,
            (left, top, right, top + subplot_height),
            elapsed,
            np.asarray(g_errors[index]),
            np.asarray(m_errors[index]),
            label,
            fonts,
        )

    axis_label = "Elapsed time from first common epoch (s)"
    text_center(draw, (width / 2, height - 35), axis_label, fonts["axis"], INK)
    image.save(output)


def create_case_svg(case, sow, g_errors, m_errors, output):
    """Create a vector version that can be zoomed without pixelation."""
    width, height = 1900, 1600
    left, right = 210, width - 80
    subplot_height = 320
    gap = 120
    first_top = 245
    elapsed = np.asarray(sow, dtype=float) - float(sow[0])
    labels = ["East", "North", "Up"]
    svg = [
        '<?xml version="1.0" encoding="UTF-8"?>\n',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n',
        '<rect width="100%" height="100%" fill="white"/>\n',
        '<g font-family="Arial, Segoe UI, sans-serif">\n',
        f'<text x="210" y="78" font-size="44" font-weight="700" fill="{INK}">{escape(case)}: ENU position-error series</text>\n',
        f'<text x="210" y="126" font-size="25" fill="{MUTED}">Common epochs: {len(sow)} | SOW {sow[0]:.3f}–{sow[-1]:.3f} | Same IMU-centre truth</text>\n',
        f'<line x1="220" y1="160" x2="300" y2="160" stroke="{BLUE}" stroke-width="5"/>\n',
        f'<circle cx="260" cy="160" r="5" fill="white" stroke="{BLUE}" stroke-width="3"/>\n',
        f'<text x="320" y="169" font-size="27" fill="{INK}">GINSFGO_LC</text>\n',
        f'<line x1="570" y1="160" x2="650" y2="160" stroke="{ORANGE}" stroke-width="5"/>\n',
        f'<rect x="605" y="155" width="10" height="10" fill="white" stroke="{ORANGE}" stroke-width="3"/>\n',
        f'<text x="670" y="169" font-size="27" fill="{INK}">MSF_LC</text>\n',
    ]

    for index, label in enumerate(labels):
        top = first_top + index * (subplot_height + gap)
        bottom = top + subplot_height
        plot_width = right - left
        x_min = float(elapsed[0])
        x_max = float(elapsed[-1])
        if x_max <= x_min:
            x_max = x_min + 1.0

        g_values = np.asarray(g_errors[index], dtype=float)
        m_values = np.asarray(m_errors[index], dtype=float)
        combined = np.concatenate((g_values, m_values, np.array([0.0])))
        y_min = float(np.nanmin(combined))
        y_max = float(np.nanmax(combined))
        span = y_max - y_min
        if span < 1e-6:
            span = 1.0
        padding = span * 0.08
        y_min -= padding
        y_max += padding

        def px(value):
            return left + (float(value) - x_min) / (x_max - x_min) * plot_width

        def py(value):
            return bottom - (float(value) - y_min) / (y_max - y_min) * subplot_height

        svg.append(f'<text x="{left}" y="{top - 22}" font-size="29" font-weight="700" fill="{INK}">{label} error (m)</text>\n')
        metric = f"RMSE  GINSFGO={rmse(g_values):.3f} m   MSF={rmse(m_values):.3f} m"
        svg.append(f'<text x="{right}" y="{top - 22}" text-anchor="end" font-size="22" fill="{MUTED}">{metric}</text>\n')

        for tick_index in range(6):
            x_value = x_min + (x_max - x_min) * tick_index / 5
            x = px(x_value)
            svg.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" stroke="{GRID}" stroke-width="1"/>\n')
            svg.append(f'<text x="{x:.2f}" y="{bottom + 32}" text-anchor="middle" font-size="21" fill="{MUTED}">{x_value:.0f}</text>\n')

        for tick_index in range(5):
            y_value = y_min + (y_max - y_min) * tick_index / 4
            y = py(y_value)
            tick_text = f"{y_value:.2f}" if abs(y_value) < 10 else f"{y_value:.1f}"
            svg.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="{GRID}" stroke-width="1"/>\n')
            svg.append(f'<text x="{left - 18}" y="{y + 7:.2f}" text-anchor="end" font-size="21" fill="{MUTED}">{tick_text}</text>\n')

        if y_min <= 0.0 <= y_max:
            zero_y = py(0.0)
            svg.append(f'<line x1="{left}" y1="{zero_y:.2f}" x2="{right}" y2="{zero_y:.2f}" stroke="{ZERO}" stroke-width="2"/>\n')

        svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="{INK}" stroke-width="2"/>\n')
        svg.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="{INK}" stroke-width="2"/>\n')

        clip_id = f"clip{index}"
        svg.append(f'<defs><clipPath id="{clip_id}"><rect x="{left}" y="{top}" width="{plot_width}" height="{subplot_height}"/></clipPath></defs>\n')
        g_points = " ".join(f"{px(t):.2f},{py(v):.2f}" for t, v in zip(elapsed, g_values))
        m_points = " ".join(f"{px(t):.2f},{py(v):.2f}" for t, v in zip(elapsed, m_values))
        svg.append(f'<polyline points="{g_points}" fill="none" stroke="{BLUE}" stroke-width="3" clip-path="url(#{clip_id})"/>\n')
        svg.append(f'<polyline points="{m_points}" fill="none" stroke="{ORANGE}" stroke-width="3" clip-path="url(#{clip_id})"/>\n')

        marker_step = max(1, len(elapsed) // 18)
        for marker_index in range(0, len(elapsed), marker_step):
            gx, gy = px(elapsed[marker_index]), py(g_values[marker_index])
            mx, my = px(elapsed[marker_index]), py(m_values[marker_index])
            svg.append(f'<circle cx="{gx:.2f}" cy="{gy:.2f}" r="5" fill="white" stroke="{BLUE}" stroke-width="3" clip-path="url(#{clip_id})"/>\n')
            svg.append(f'<rect x="{mx - 5:.2f}" y="{my - 5:.2f}" width="10" height="10" fill="white" stroke="{ORANGE}" stroke-width="3" clip-path="url(#{clip_id})"/>\n')

    svg.append(f'<text x="{width / 2}" y="{height - 24}" text-anchor="middle" font-size="27" fill="{INK}">Elapsed time from first common epoch (s)</text>\n')
    svg.append('</g>\n</svg>\n')
    output.write_text("".join(svg), encoding="utf-8")


def main():
    comparison = load_comparison_module()
    evaluator = comparison.load_evaluator()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generated = []
    for item in comparison.CASES:
        truth, _, _, gins, msf = comparison.common_aligned(
            evaluator,
            item["ginsfgo"],
            item["msf"],
            item["truth"],
        )
        _, ge, gn, gu = evaluator.calculate_position_errors(gins, truth)
        _, me, mn, mu = evaluator.calculate_position_errors(msf, truth)
        sow = gins["sow_gt"].to_numpy(dtype=float)

        error_table = pd.DataFrame(
            {
                "sow": sow,
                "elapsed_s": sow - sow[0],
                "ginsfgo_e_m": ge,
                "ginsfgo_n_m": gn,
                "ginsfgo_u_m": gu,
                "msf_e_m": me,
                "msf_n_m": mn,
                "msf_u_m": mu,
            }
        )
        csv_path = OUTPUT_DIR / f"{item['case']}_ENU_error_series.csv"
        png_path = OUTPUT_DIR / f"{item['case']}_ENU_error_series.png"
        svg_path = OUTPUT_DIR / f"{item['case']}_ENU_error_series.svg"
        error_table.to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.9f")
        create_case_figure(
            item["case"],
            sow,
            (ge, gn, gu),
            (me, mn, mu),
            png_path,
            comparison.chart_font,
        )
        create_case_svg(
            item["case"],
            sow,
            (ge, gn, gu),
            (me, mn, mu),
            svg_path,
        )
        generated.append((item["case"], len(sow), png_path, svg_path, csv_path))

    for case, count, png_path, svg_path, csv_path in generated:
        print(f"{case}: {count} common epochs")
        print(f"  PNG: {png_path}")
        print(f"  SVG: {svg_path}")
        print(f"  CSV: {csv_path}")


if __name__ == "__main__":
    main()
