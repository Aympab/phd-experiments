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
DEFAULT_MAXITERS = [50]

# --- Regexes for parsing performance metrics ---
ESTIMATED_GBPS_RE = re.compile(r"estimated_throughput\s*[:=]\s*([0-9]*\.?[0-9]+)\s*GB/s", re.IGNORECASE)
ESTIMATED_BPS_RE  = re.compile(r"estimated_throughput\s*[:=]\s*([0-9]*\.?[0-9]+)\s*B/s", re.IGNORECASE)
BYTES_PER_SEC_RE  = re.compile(r"bytes?_per_sec\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)

# --- Cases ---
CASES = {
    "case0": {"n0": 1024,  "n1": 1024, "n2": 1024},
    "case1": {"n0": 32768, "n1": 64,   "n2": 8},
    "case2": {"n0": 2048,  "n1": 6144, "n2": 1},
    "case3": {"n0": 16,    "n1": 1024, "n2": 2048},
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
maxIter = {max_iter}
dt  = 0.001
minRealX  = 0
maxRealX  = 1
minRealVx = -1
maxRealVx = 1

[impl]
kernelImpl  = {kernel_impl}
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

def write_ini(path: Path, dims: Dict[str, int], wg: int, max_iter: int, kernel_impl: str) -> None:
    path.write_text(INI_TEMPLATE.format(n0=dims['n0'], n1=dims['n1'], n2=dims['n2'], wg=wg, max_iter=max_iter, kernel_impl=kernel_impl))

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

def benchmark_case(exe: Path, ini: Path, runs: int, log_output: bool = True, tag: str = "") -> Tuple[Dict[str, Any], float]:
    perfs = []
    for i in range(runs):
        ok, out = run_once(exe, ini)
        if log_output:
            print(f"----- Output (run {i+1}/{runs}) [{tag}] -----", flush=True)
            print(out, flush=True)
        if not ok:
            if log_output:
                print("Status: NON-ZERO RETURN CODE", flush=True)
            return ({"runs_completed": len(perfs), "status": "error", "bytes_per_sec": {**summarize([]), "unit": "B/s"}}, 0.0)
        perf = parse_perf(out)
        if log_output:
            print(f"Parsed bytes/sec: {perf if perf is not None else 'NONE'}", flush=True)
        if perf is None:
            return ({"runs_completed": len(perfs), "status": "error_parse", "bytes_per_sec": {**summarize([]), "unit": "B/s"}}, 0.0)
        perfs.append(perf)
    summary = summarize(perfs)
    return ({"runs_completed": len(perfs), "status": "ok", "bytes_per_sec": {**summary, "unit": "B/s"}}, summary['median'])

# --- Helpers to parse CLI inputs ---

def parse_cases_arg(cases_arg: Optional[str]) -> Dict[str, Dict[str, int]]:
    """Return a dict of selected cases from CASES. Accepts comma-separated names or 'all'."""
    if cases_arg is None or cases_arg.lower() == 'all':
        return CASES
    names = [c.strip() for c in cases_arg.split(',') if c.strip()]
    unknown = [n for n in names if n not in CASES]
    if unknown:
        raise SystemExit(f"Unknown case(s): {', '.join(unknown)}. Available: {', '.join(sorted(CASES))}")
    return {n: CASES[n] for n in names}


def parse_maxiters_arg(arg: Optional[str], repeats: Optional[List[int]]) -> List[int]:
    """Allow --maxiter repeated and/or a comma-separated list via --maxiters."""
    values: List[int] = []
    if repeats:
        values.extend(int(x) for x in repeats)
    if arg:
        for tok in arg.split(','):
            tok = tok.strip()
            if tok:
                values.append(int(tok))
    if not values:
        values = DEFAULT_MAXITERS.copy()
    # de-dup and sort
    return sorted(set(values))


def main():
    ap = argparse.ArgumentParser(description="Run comparison for fixed best hybrid config or WG sweep; supports case and maxIter sweeps")
    ap.add_argument("--hw", required=True, choices=["mi300", "pvc", "h100"])
    ap.add_argument("--impl", required=True, choices=["hybrid", "ndrange", "adaptivewg"])
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    ap.add_argument("--cases", type=str, default="all", help="Comma-separated subset of cases to run (default: all)")
    # Two ways to specify maxIter sweeps: repeat --maxiter, or pass --maxiters 50,100,200
    ap.add_argument("--maxiter", type=int, action='append', help="Add a maxIter value to sweep; can be repeated")
    ap.add_argument("--maxiters", type=str, help="Comma-separated list of maxIter values to sweep")
    args = ap.parse_args()

    # Map CLI --impl to ini [impl].kernelImpl value
    kernel_impl = "Ndrange" if args.impl == "ndrange" else "AdaptiveWg"

    selected_cases = parse_cases_arg(args.cases)
    maxiters = parse_maxiters_arg(args.maxiters, args.maxiter)

    exe = build_executable(args.impl, args.hw)
    if not exe.exists():
        raise SystemExit(f"Executable not found: {exe}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = OUT_DIR / "tmp_configs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {
        "impl": args.impl,
        "hardware": args.hw,
        "executable": str(exe),
        "runs_per_config": args.runs,
        "cases": {},
        "maxIter_sweep": maxiters,
        "notes": {"config_source": "hybrid=best known; others=wg sweep"}
    }

    for case_name, dims in selected_cases.items():
        case_entry: Dict[str, Any] = {"problem": dims, "sweeps": {}}

        for max_iter in maxiters:
            best_result = None
            best_median = -1.0
            best_wg: Optional[int] = None

            for wg in (WG_SIZES if args.impl != "hybrid" else [512]):
                ini_path = tmp_dir / f"{case_name}_{args.impl}_wg{wg}_mi{max_iter}.ini"
                write_ini(ini_path, dims, wg, max_iter, kernel_impl)
                print(f"[{args.impl}/{args.hw}] {case_name} maxIter={max_iter} with wg={wg} (kernelImpl={kernel_impl})")
                res, median = benchmark_case(exe, ini_path, args.runs, True, f"{case_name} maxIter={max_iter} wg={wg}")
                if median > best_median:
                    best_result = res
                    best_median = median
                    best_wg = wg

            case_entry["sweeps"][str(max_iter)] = {
                "wg_size": best_wg,
                "result": best_result
            }

        results["cases"][case_name] = case_entry

    # Output file name includes impl+hw; JSON carries full sweep info
    out_path = OUT_DIR / f"dpcpp_{args.hw}_{args.impl}.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
