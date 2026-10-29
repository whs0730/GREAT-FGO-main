"""在独立输出目录运行两组固定节点间隔扫描，保留原始配置和输入。"""
import hashlib
import json
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/ct_adaptation_pilot_20261029"
CASES = {
    "Data05": ("Data05_20201128_HG4930_Vehicle_Opensky",
               "Data05_20201128_HG4930_Vehicle_Opensky_CT.xml", 600),
    "campus": ("FGO_20250928", "GREAT-GINSFGO-CT-RTKINS-0928-campus.xml", 360),
}
INTERVALS = (1.0, 0.5, 0.2, 0.1)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    executable = ROOT / "build_Windows/Bin/RelWithDebInfo/GREAT_GINSFGO.exe"
    protected = [executable, *executable.parent.glob("*.dll"), *ROOT.glob("plot/*pilot*.py")]
    protected += [p for p in (ROOT / "src").rglob("*") if p.is_file() and p.suffix in (".cpp", ".h", ".hpp")]
    for directory, xml_name, _ in CASES.values():
        source = ROOT / "sample_data" / directory / "xml" / xml_name
        protected.append(source)
        config = ET.parse(source)
        for node in config.find("inputs"):
            protected.extend(Path(value) for value in node.text.split())
    hashes = {str(path): sha256(path) for path in dict.fromkeys(protected)}
    manifest = {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "protected_sha256": hashes, "intervals_s": INTERVALS,
                "scope": "Two prefix pilots with original start/initialization, 10 s windows, 1 s updates; only end, knot interval and output paths change.",
                "cases": CASES}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for name, (directory, xml_name, duration) in CASES.items():
        source = ROOT / "sample_data" / directory / "xml" / xml_name
        for interval in INTERVALS:
            folder = OUT / name / f"h_{interval:g}"
            folder.mkdir(parents=True, exist_ok=False)
            tree = ET.parse(source)
            assert tree.findtext("fgomsf_sensors/msf_type").strip() == "GINS_CT"
            assert float(tree.findtext("fgo/ct_window_sec")) == 10
            assert float(tree.findtext("fgo/ct_update_sec")) == 1
            start = datetime.strptime(tree.findtext("gen/beg").strip().strip('"'), "%Y-%m-%d %H:%M:%S")
            tree.find("gen/end").text = (start + timedelta(seconds=duration)).strftime('"%Y-%m-%d %H:%M:%S"')
            tree.find("fgo/ct_knot_sec").text = str(interval)
            for node in tree.find("outputs"):
                node.text = str(folder / f"result.{node.tag}")
            tree.find("outputs").set("append", "false")
            config_path = folder / "config.xml"
            tree.write(config_path, encoding="utf-8", xml_declaration=True)
            original = ET.parse(source)
            for config in (tree, original):
                config.getroot().remove(config.find("outputs"))
                config.find("gen/end").text = "END"
                config.find("fgo/ct_knot_sec").text = "KNOT"
            assert ET.tostring(tree.getroot()) == ET.tostring(original.getroot())
            print(f"START {name} h={interval:g}", flush=True)
            started = time.perf_counter()
            with (folder / "console.log").open("wb") as stream:
                run = subprocess.run([str(executable), "-x", str(config_path)], cwd=folder,
                                     stdout=stream, stderr=subprocess.STDOUT, timeout=1800,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
            elapsed = time.perf_counter() - started
            console = (folder / "console.log").read_text(encoding="utf-8", errors="replace")
            windows = re.findall(r"\[CT WINDOW\].*", console)
            record = {"case": name, "interval_s": interval, "exit_code": run.returncode,
                      "wall_seconds": elapsed, "window_count": len(windows),
                      "knot_counts": list(map(int, re.findall(r"\[CT WINDOW\].*?knots=(\d+)", console))),
                      "iterations": list(map(int, re.findall(r"\[CT WINDOW\].*?Iterations: (\d+)", console))),
                      "no_convergence_count": sum("NO_CONVERGENCE" in line for line in windows),
                      "failure_count": sum("FAILURE" in line for line in windows),
                      "outputs_sha256": {path.name: sha256(path) for path in folder.glob("result.*")}}
            (folder / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            print(f"DONE {name} h={interval:g} exit={run.returncode} wall={elapsed:.2f}s windows={len(windows)}", flush=True)
            assert run.returncode == 0 and windows and "[CT ERROR]" not in console
            assert "[CT] output_rows=" in console
            assert (folder / "result.ins").stat().st_size > 0
    changed = [path for path, value in hashes.items() if sha256(Path(path)) != value]
    (OUT / "preservation.json").write_text(json.dumps({"changed_protected_files": changed,
                                                       "checked_files": len(hashes)}, indent=2), encoding="utf-8")
    assert not changed, changed


if __name__ == "__main__":
    main()
