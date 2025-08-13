#!/usr/bin/env python3
import argparse
import subprocess
import statistics
import json
import re
from pathlib import Path
from typing import List, Tuple

# --- Config ---
INI_DIR   = Path("/home/ac.amillan/advection_ini")
EXE_ROOT  = Path("/home/ac.amillan/source/parallel-advection")
OUT_DIR   = EXE_ROOT / "jlse" / "out" / "parallel-adv" / "nd-range" / "manual"
N_RUNS    = 5

def build_cmd(exe: Path, ini: Path) -> List[str]:
    return [str(exe), str(ini)]

THROUGHPUT_RE = re.compile(r"\bestim(?:ated)?_throughput\s*:\s*([0-9]*\.?[0-9]+)\s*GB/s", re.IGNORECASE)
TIME_PER_ITER_RE = re.compile(r"time_per_iter\s*\(sec\)\s*:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)

def parse_metrics(stdout: str) -> Tuple[float, float]:
    tpi_m = TIME_PER_ITER_RE.search(stdout)
    thr_m = THROUGHPUT_RE.search(stdout)
    if not tpi_m or not thr_m:
        raise ValueError
    return float(tpi_m.group(1)), float(thr_m.group(1))

def stats(vals: List[float]) -> dict:
    if not vals:
        return {"mean": 0.0, "median": 0.0, "stdev": 0.0}
    if len(vals) == 1:
        return {"mean": vals[0], "median": vals[0], "stdev": 0.0}
    return {
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
        "stdev": statistics.stdev(vals),
    }

def zeros_result(runs_completed: int, status: str) -> dict:
    return {
        "time_per_iter": {"mean": 0.0, "median": 0.0, "stdev": 0.0, "unit": "sec"},
        "estimated_throughput": {"mean": 0.0, "median": 0.0, "stdev": 0.0, "unit": "GB/s"},
        "runs_completed": runs_completed,
        "status": status,
    }

def run_case_quiet(exe: Path, ini: Path, n_runs: int) -> Tuple[dict, bool]:
    tpi_vals: List[float] = []
    thr_vals: List[float] = []
    for i in range(1, n_runs + 1):
        try:
            proc = subprocess.run(
                build_cmd(exe, ini),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                return zeros_result(i - 1, "error"), False
            try:
                tpi, thr = parse_metrics(proc.stdout)
            except Exception:
                return zeros_result(len(tpi_vals), "error"), False
            tpi_vals.append(tpi)
            thr_vals.append(thr)
        except Exception:
            return zeros_result(len(tpi_vals), "error"), False

    return {
        "time_per_iter": {**stats(tpi_vals), "unit": "sec"},
        "estimated_throughput": {**stats(thr_vals), "unit": "GB/s"},
        "runs_completed": len(tpi_vals),
        "status": "ok",
    }, True

def pick_flag(name: str, flags: dict) -> str:
    chosen = [k for k, v in flags.items() if v]
    if len(chosen) != 1:
        raise ValueError(f"Specify exactly one {name} flag.")
    return chosen[0]

def main():
    p = argparse.ArgumentParser()
    comp = p.add_mutually_exclusive_group(required=True)
    comp.add_argument("--acpp",  action="store_true")
    comp.add_argument("--dpcpp", action="store_true")
    hw = p.add_mutually_exclusive_group(required=True)
    hw.add_argument("--pvc",  action="store_true")
    hw.add_argument("--mi300", action="store_true")
    hw.add_argument("--h100",  action="store_true")
    p.add_argument("--runs", type=int, default=N_RUNS)
    args = p.parse_args()

    compiler = pick_flag("compiler", {"acpp": args.acpp, "dpcpp": args.dpcpp})
    hardware = pick_flag("hardware", {"pvc": args.pvc, "mi300": args.mi300, "h100": args.h100})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = [f"case{i}.ini" for i in range(10)]
    exe = EXE_ROOT / f"build_{compiler}_{hardware}" / "src" / "advection"
    out_path = OUT_DIR / f"advection_{compiler}_{hardware}_script.json"

    results = {"compiler": compiler, "hardware": hardware, "executable": str(exe), "cases": {}}
    succeeded, failed = [], []

    if not exe.exists():
        for case in cases:
            results["cases"][case] = zeros_result(0, "error_missing_executable")
            print(f"[{compiler}/{hardware}] {case}: missing exe -> zeros written")
            failed.append(case)
    else:
        for case in cases:
            ini = INI_DIR / case
            if not ini.exists():
                results["cases"][case] = zeros_result(0, "error_missing_ini")
                print(f"[{compiler}/{hardware}] {case}: missing ini -> zeros written")
                failed.append(case)
                continue

            case_result, ok = run_case_quiet(exe, ini, args.runs)
            results["cases"][case] = case_result
            if ok:
                t = case_result["time_per_iter"]
                th = case_result["estimated_throughput"]
                print(f"[{compiler}/{hardware}] {case}: "
                      f"TPI median={t['median']:.6g}s mean={t['mean']:.6g}s stdev={t['stdev']:.6g}; "
                      f"THR median={th['median']:.6g}GB/s mean={th['mean']:.6g}GB/s stdev={th['stdev']:.6g}")
                succeeded.append(case)
            else:
                print(f"[{compiler}/{hardware}] {case}: error -> zeros written")
                failed.append(case)

    with out_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"Summary for {compiler}/{hardware} — ran: {', '.join(succeeded) or 'none'}; "
          f"did not run: {', '.join(failed) or 'none'}.")
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
