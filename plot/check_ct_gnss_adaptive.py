"""验证 CT 自适应权重的关闭回归、配置边界和两组真实数据。"""
import json
import os
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

import analyze_ct_pilot_experiments as evaluation


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/ct_gnss_adaptive_final_20261029"
PILOT = ROOT / "results/ct_adaptation_pilot_20261029"
RUNTIME = ROOT / "build_Windows/Bin/RelWithDebInfo"
BEFORE = ROOT / "build/ct_gnss_adaptive_20261029/baseline_bin"


def run_config(folder, tree, runtime):
    folder.mkdir(parents=True, exist_ok=False)
    for node in tree.find("outputs"):
        node.text = str(folder / f"result.{node.tag}")
    tree.find("outputs").set("append", "false")
    config = folder / "config.xml"
    tree.write(config, encoding="utf-8", xml_declaration=True)
    print("START", folder.relative_to(OUT), flush=True)
    started = time.perf_counter()
    environment = dict(os.environ, PATH=str(runtime)+os.pathsep+os.environ["PATH"])
    with (folder/"console.log").open("wb") as stream:
        process = subprocess.run([str(runtime/"GREAT_GINSFGO.exe"), "-x", str(config)],
                                 cwd=folder, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                 timeout=1800, creationflags=subprocess.CREATE_NO_WINDOW)
    seconds = time.perf_counter()-started
    console = (folder/"console.log").read_text(encoding="utf-8", errors="replace")
    result = {"exit_code": process.returncode, "wall_seconds": seconds,
              "ct_error": "[CT ERROR]" in console, "no_convergence_count": console.count("NO_CONVERGENCE"),
              "output_hashes": {p.name: evaluation.sha256(p) for p in folder.glob("result.*")}}
    evaluation.write_json(folder/"run.json", result)
    print("DONE", folder.relative_to(OUT), process.returncode, round(seconds, 2), flush=True)
    return result


def set_adaptive(tree, enabled, penalty=1, maximum=100):
    for name, value in (("ct_gnss_adaptive", int(enabled)), ("ct_gnss_scale_lambda", penalty),
                        ("ct_gnss_scale_max", maximum)):
        node = tree.find("fgo/"+name)
        if node is None:
            node = ET.SubElement(tree.find("fgo"), name)
        node.text = str(value)


def check_scales(folder):
    console = (folder/"console.log").read_text(encoding="utf-8", errors="replace")
    rows = []
    for window, timestamp, q, previous, scale in re.findall(
            r"\[CT GNSS WEIGHT\] window=(\d+) time=(\S+) q=(\S+) previous_scale=(\S+) scale=(\S+)", console):
        row = dict(window=int(window), time_s=float(timestamp), q=float(q), previous_scale=float(previous), scale=float(scale))
        assert all(np.isfinite(v) for v in row.values())
        assert 1 <= row["scale"] <= 100
        gradient = -row["q"]/row["scale"] + 3 + 2*np.log(row["scale"]/row["previous_scale"])
        if row["scale"] == 1:
            assert gradient >= -1e-8
        elif row["scale"] == 100:
            assert gradient <= 1e-8
        else:
            assert abs(gradient) < 1e-7
        rows.append(row)
    assert rows
    for index in np.linspace(0, len(rows)-1, min(40, len(rows)), dtype=int):
        row = rows[index]
        def objective(s):
            return row["q"]*np.exp(-s) + 3*s + (s-np.log(row["previous_scale"]))**2
        fitted = minimize_scalar(objective, bounds=(0, np.log(100)), method="bounded")
        expected = min(objective(0), objective(np.log(100)), fitted.fun)
        assert objective(np.log(row["scale"])) <= expected+1e-7*max(1, abs(expected))
    evaluation.write_csv(folder/"weights.csv", rows)
    scales = np.array([r["scale"] for r in rows])
    result = {"updates": len(rows), "scale_median": float(np.median(scales)),
              "scale_max": float(scales.max()), "downweighted_fraction": float(np.mean(scales > 1+1e-8)),
              "bound_fraction": float(np.mean(scales == 100)), "stationarity_checks_passed": True,
              "independent_scalar_minimum_checks": min(40, len(rows))}
    evaluation.write_json(folder/"weight_summary.json", result)
    return result


def compare_case(name):
    folder = OUT/name
    baseline = PILOT/name/"h_0.5"
    assert (folder/"result.flt").read_bytes() == (baseline/"result.flt").read_bytes()
    imu_reference, _, week, start = evaluation.REFERENCES[name]
    truth_path = ROOT/"sample_data"/evaluation.CASES[name][0]/imu_reference
    truth = evaluation.read_truth(truth_path, week)
    lookup = {r["t"]: r for r in truth}
    maps = {label: {r["t"]: r for r in evaluation.numerical.read_ins(path)} for label, path in (
        ("original", baseline/"result.ins"), ("adaptive", folder/"result.ins"))}
    times = sorted(set(maps["original"]) & set(maps["adaptive"]) & set(lookup))
    series = {label: [evaluation.numerical.errors(m[t], lookup[t], truth[0]) for t in times] for label,m in maps.items()}
    metrics = {subset: {label: evaluation.numerical.metrics([r for r in rows if subset == "all"
                  or r["truth_fixed"] and r["t"] >= start+30]) for label,rows in series.items()}
                  for subset in ("all", "reference_fixed_after30")}
    errors = []
    for label, rows in series.items():
        for row in rows:
            errors.append({"method": label, "sow": row["t"], "truth_fixed": row["truth_fixed"],
                           "position_3d_m": float(np.linalg.norm(row["p"])),
                           "velocity_3d_m_s": float(np.linalg.norm(row["v"])), "heading_error_deg": row["a"][2]})
    evaluation.write_csv(folder/"errors.csv", errors)
    figure, axes = plt.subplots(3, 1, figsize=(12,8), sharex=True, layout="constrained")
    for label,color in (("original", "#0072B2"), ("adaptive", "#D55E00")):
        rows = series[label]
        for axis, key in zip(axes[:2], ("p", "v")):
            axis.plot(np.array(times)-start, np.linalg.norm([r[key] for r in rows],axis=1), label=label, color=color, linewidth=.9)
        axes[2].plot(np.array(times)-start, [r["a"][2] for r in rows], label=label, color=color, linewidth=.9)
    for axis,label in zip(axes, ("3D position difference (m)", "3D velocity difference (m/s)", "Heading difference (deg)")):
        axis.set_ylabel(label)
        axis.grid(alpha=.25)
        for t in times:
            if not lookup[t]["fixed"]:
                axis.axvspan(t-start-.5,t-start+.5,color="grey",alpha=.06,linewidth=0)
    axes[0].legend()
    axes[2].set_xlabel(f"Seconds since GPST SOW {start}")
    figure.suptitle(f"{name}: 0.5 s knots, 10 s window; grey = reference not Fixed")
    figure.savefig(folder/"comparison.png",dpi=160)
    plt.close(figure)
    result = {"metrics": metrics, "common_epochs": len(times), "frontend_byte_identical": True,
              "weights": check_scales(folder), "run": json.loads((folder/"run.json").read_text()),
              "baseline_run": json.loads((baseline/"run.json").read_text())}
    evaluation.write_json(folder/"summary.json", result)
    return result


def write_report(results):
    lines = ["# CT GNSS 自适应尺度首轮验证", "", "参数：lambda=1、scale_max=100，节点0.5 s，窗口10 s、每秒更新。参数未按参考解调优。", "",
             "关闭选项的 LC、CT 批量、CT 滑窗三组 30 s INS/FLT 与修改前程序逐字节相同。三个非法配置均被拒绝。", "",
             "下表为共同历元、参考 Fixed、排除前30 s；原方法基准来自同输入、同节点配置的已保存预实验，运行库哈希已核对。耗时为单次完整进程墙钟，历史负载不同，仅供参考。", "",
             "|算例|方法|历元|位置RMSE/m|速度RMSE/(m/s)|Heading RMSE/°|墙钟/s|", "|---|---|---:|---:|---:|---:|---:|"]
    for name,result in results.items():
        for method in ("original","adaptive"):
            m=result["metrics"]["reference_fixed_after30"][method]
            seconds=result["baseline_run" if method == "original" else "run"]["wall_seconds"]
            lines.append(f"|{name}|{method}|{m['epochs']}|{m['pos_3d_rmse_m']:.6f}|{m['vel_3d_rmse_mps']:.6f}|{m['att_heading_rmse_deg']:.6f}|{seconds:.2f}|")
    lines += ["", "Data05 的位置、速度和航向 RMSE 均有小幅改善；校园位置、速度改善，但航向 RMSE 由约 1.06° 增至 2.92°。当前原型不能表述为各项精度全面提升。", ""]
    closeout_path = OUT/"closeout/validation.json"
    if closeout_path.exists():
        closeout = json.loads(closeout_path.read_text(encoding="utf-8"))
        assert closeout["algorithm_dlls_match_full_experiments"]
        lines += ["入口收尾后，六种非法参数均在读取观测前返回退出码 1，且无导航输出；关闭自适应的 LC、CT 批量、CT 滑窗三个 30 s INS/FLT 与修改前逐字节一致；启用自适应的 30 s 检查正常退出并通过尺度最优性复核。", "",
                  "两组完整算例使用入口前置校验改动之前的可执行文件，均正常退出。算法 DLL 与收尾程序的哈希一致；入口修改后补做上述短回归，未重复完整算例。旧运行版本见 manifest.json，收尾版本见 [closeout/runtime.json](closeout/runtime.json)，验证记录见 [closeout/validation.json](closeout/validation.json)。", ""]
    for name,result in results.items():
        lines += ["",f"{name} 权重统计：{json.dumps(result['weights'],ensure_ascii=False)}。GNSS 前端 FLT 与原方法逐字节一致。", "",f"![对照曲线]({name}/comparison.png)", ""]
    lines += ["实现说明见 [CT_GNSS_ADAPTIVE_WEIGHTING.md](../../doc/CT_GNSS_ADAPTIVE_WEIGHTING.md)；本轮为实现验证，不证明论文创新性或在其他数据上的精度收益。优化残差可能吸收轨迹模型误差，需继续与固定协方差放大、鲁棒核作消融比较。", ""]
    (OUT/"report.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    pilot_manifest = json.loads((PILOT/"manifest.json").read_text())
    inputs = {name:digest for name,digest in pilot_manifest["protected_sha256"].items() if "sample_data" in name}
    assert all(evaluation.sha256(Path(name)) == digest for name,digest in inputs.items())
    for name in ("GREAT_GINSFGO.exe", "LibGREATrd.dll", "LibGnutrd.dll"):
        assert evaluation.sha256(BEFORE/name) == pilot_manifest["protected_sha256"][str(RUNTIME/name)]
    runtimes = {str(p):evaluation.sha256(p) for folder in (BEFORE,RUNTIME) for p in folder.iterdir() if p.suffix == ".dll" or p.name == "GREAT_GINSFGO.exe"}
    evaluation.write_json(OUT/"manifest.json", {"inputs_sha256":inputs, "runtime_sha256":runtimes})
    source = ROOT/"sample_data/run_check/smoke_CT.xml"
    regressions = {}
    for mode,window in (("LC",0), ("CT_batch",0), ("CT_window",10)):
        hashes = []
        for label,runtime in (("before",BEFORE), ("after",RUNTIME)):
            tree = ET.parse(source)
            tree.find("fgomsf_sensors/msf_type").text = "GINS_LC" if mode == "LC" else "GINS_CT"
            tree.find("fgo/ct_window_sec").text = str(window)
            result = run_config(OUT/"regression"/mode/label,tree,runtime)
            assert result["exit_code"] == 0 and not result["ct_error"]
            hashes.append({suffix:evaluation.sha256(OUT/"regression"/mode/label/f"result.{suffix}") for suffix in ("ins","flt")})
        assert hashes[0] == hashes[1], (mode,"disabled output changed")
        regressions[mode] = {"ins_and_flt_byte_identical":True}
    evaluation.write_json(OUT/"regression.json", regressions)
    for name,tag,value in (("bad_enable","ct_gnss_adaptive",2), ("bad_lambda","ct_gnss_scale_lambda",-1), ("bad_maximum","ct_gnss_scale_max",.5)):
        tree = ET.parse(source)
        set_adaptive(tree,True)
        tree.find("fgo/"+tag).text = str(value)
        result = run_config(OUT/"invalid"/name,tree,RUNTIME)
        assert result["ct_error"] and result["exit_code"] == 1, (name,"invalid adaptive config not rejected cleanly")
        assert not (OUT/"invalid"/name/"result.ins").exists(), (name,"invalid configuration started processing")
    results = {}
    for name in evaluation.CASES:
        tree = ET.parse(PILOT/name/"h_0.5/config.xml")
        set_adaptive(tree,True)
        result = run_config(OUT/name,tree,RUNTIME)
        assert result["exit_code"] == 0 and not result["ct_error"]
        results[name] = compare_case(name)
    assert all(evaluation.sha256(Path(name)) == digest for name,digest in inputs.items())
    assert all(evaluation.sha256(Path(name)) == digest for name,digest in runtimes.items())
    evaluation.write_json(OUT/"validation.json", {"disabled_regressions":regressions,
                           "invalid_configurations_rejected":3, "inputs_and_runtimes_unchanged":True})
    write_report(results)
    print("ALL CHECKS PASSED",flush=True)


if __name__ == "__main__":
    main()
