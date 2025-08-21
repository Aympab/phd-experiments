#!/usr/bin/env python3
import argparse
import subprocess
import statistics
import json
import math
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# --- Constants & Defaults ---
HYBRID_ROOT = Path("/home/ac.amillan/source/hybrid-paradv")
PARADV_ROOT = Path("/home/ac.amillan/source/parallel-advection")
OUT_DIR     = Path("/home/ac.amillan/source/phd-experiments/out/hybrid-subgroups/comparison")

DEFAULT_RUNS  = 5
WG_SIZES = [128, 256, 512, 1024]

# --- Regexes for parsing performance metrics ---
ESTIMATED_GBPS_RE = re.compile(r"estimated_throughput\s*[:=]\s*([0-9]*\.?[0-9]+)\s*GB/s", re.IGNORECASE)
ESTIMATED_BPS_RE  = re.compile(r"estimated_throughput\s*[:=]\s*([0-9]*\.?[0-9]+)\s*B/s", re.IGNORECASE)
BYTES_PER_SEC_RE  = re.compile(r"bytes?_per_sec\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)

# --- Cases ---
CASES = {
    "case0": {"n0": 32768, "n1": 64,   "n2": 1},
    "case1": {"n0": 512,   "n1": 64,   "n2": 1},
    "case2": {"n0": 512,   "n1": 6144, "n2": 64},
    "case3": {"n0": 8,     "n1": 6144, "n2": 64},
}

BEST_CONFIGS = {
    "case0": {"nsgL": 4, "nsgG": 4, "seqL": 3, "seqG": 1},
    "case1": {"nsgL": 4, "nsgG": 4, "seqL": 3, "seqG": 1},
    "case2": {"nsgL": 4, "nsgG": 4, "seqL": 3, "seqG": 1},
    "case3": {"nsgL": 4, "nsgG": 4, "seqL": 3, "seqG": 1},
}

INI_TEMPLATE = """[problem]
n0 = {n0}
n1 = {n1}
n2 = {n2}
maxIter = 50
dt  = 0.001
minRealX  = 0
maxRealX  = 1
minRealVx = -1
maxRealVx = 1

[impl]
kernelImpl  = AdaptiveWg
inplace = true

[optimization]
gpu     = true
pref_wg_size = {wg}
seq_size0 = 1
seq_size2 = 1

[io]
outputSolution = false
"""

def build_executable(impl: str, hw: str) -> Path:
    root = HYBRID_ROOT if impl == "hybrid" else PARADV_ROOT
    return root / f"build_dpcpp_{hw}" / "src" / "advection"

def write_ini(path: Path, dims: Dict[str, int], wg: int) -> None:
    path.write_text(INI_TEMPLATE.format(n0=dims['n0'], n1=dims['n1'], n2=dims['n2'], wg=wg))

def parse_perf(stdout: str) -> Optional[float]:
    m = ESTIMATED_GBPS_RE.search(stdout)
    if m: return float(m.group(1)) * 1e9
    m = ESTIMATED_BPS_RE.search(stdout)
    if m: return float(m.group(1))
    m = BYTES_PER_SEC_RE.search(stdout)
    if m: return float(m.group(1))
    return None

def run_once(exe: Path, ini: Path) -> Tuple[bool, str]:
    try:
        proc = subprocess.run([str(exe), str(ini)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        return proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    except Exception as e:
        return False, f"exception: {e}"

def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "stdev": 0.0}
    if len(values) == 1:
        v = values[0]
        return {"mean": v, "median": v, "stdev": 0.0}
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values),
    }

def benchmark_case(exe: Path, ini: Path, runs: int) -> Tuple[Dict[str, Any], float]:
    perfs = []
    for _ in range(runs):
        ok, out = run_once(exe, ini)
        if not ok:
            return ({"runs_completed": len(perfs), "status": "error", "bytes_per_sec": {**summarize([]), "unit": "B/s"}}, 0.0)
        perf = parse_perf(out)
        if perf is None:
            return ({"runs_completed": len(perfs), "status": "error_parse", "bytes_per_sec": {**summarize([]), "unit": "B/s"}}, 0.0)
        perfs.append(perf)
    summary = summarize(perfs)
    return ({"runs_completed": len(perfs), "status": "ok", "bytes_per_sec": {**summary, "unit": "B/s"}}, summary['median'])

def main():
    ap = argparse.ArgumentParser(description="Run comparison for fixed best hybrid config or WG sweep")
    ap.add_argument("--hw", required=True, choices=["mi300", "pvc", "h100"])
    ap.add_argument("--impl", required=True, choices=["hybrid", "ndrange", "adaptivewg"])
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    args = ap.parse_args()

    exe = build_executable(args.impl, args.hw)
    if not exe.exists():
        raise SystemExit(f"Executable not found: {exe}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = OUT_DIR / "tmp_configs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "impl": args.impl,
        "hardware": args.hw,
        "executable": str(exe),
        "runs_per_config": args.runs,
        "cases": {},
        "notes": {"config_source": "hybrid=best known; others=wg sweep"}
    }

    for case_name, dims in CASES.items():
        best_result = None
        best_median = -1
        best_wg = None

        for wg in (WG_SIZES if args.impl != "hybrid" else [512]):
            ini_path = tmp_dir / f"{case_name}_{args.impl}_wg{wg}.ini"
            write_ini(ini_path, dims, wg)
            print(f"[{args.impl}/{args.hw}] {case_name} with wg={wg}")
            res, median = benchmark_case(exe, ini_path, args.runs)
            if median > best_median:
                best_result = res
                best_median = median
                best_wg = wg

        results["cases"][case_name] = {
            "problem": dims,
            "wg_size": best_wg,
            "result": best_result
        }

    out_path = OUT_DIR / f"dpcpp_{args.hw}_{args.impl}.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
