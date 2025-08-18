#!/usr/bin/env python3
import subprocess
import statistics
import json
import tempfile
import shutil
import re
from pathlib import Path

# --- Config ---
BASE_INI = Path("/home/ac.amillan/source/parallel-advection/build_cuda_ldg/src/advection.ini")
EXE = Path("/home/ac.amillan/source/parallel-advection/build_cuda_ldg/src/advection")
OUT_JSON = Path("/home/ac.amillan/source/phd-experiments/out/cudaldg/manual-run.json")
N_RUNS = 5
N0 = 16384*2
N1 = 1024
N2_VALUES = [2**i for i in range(11)]  # 1 to 1024
KERNEL_IMPLS = ["ndrange", "ldg"]

THROUGHPUT_RE = re.compile(r"\bestim(?:ated)?_throughput\s*:\s*([0-9]*\.?[0-9]+)\s*GB/s", re.IGNORECASE)
TIME_PER_ITER_RE = re.compile(r"time_per_iter\s*\(sec\)\s*:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)

def parse_metrics(stdout: str):
    tpi_m = TIME_PER_ITER_RE.search(stdout)
    thr_m = THROUGHPUT_RE.search(stdout)
    if not tpi_m or not thr_m:
        raise ValueError("Missing performance metrics")
    return float(tpi_m.group(1)), float(thr_m.group(1))

def stats(vals):
    return {
        "mean": statistics.mean(vals) if vals else 0.0,
        "median": statistics.median(vals) if vals else 0.0,
        "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0
    }

def run_case(temp_ini: Path):
    tpi_vals = []
    thr_vals = []
    for _ in range(N_RUNS):
        result = subprocess.run([str(EXE), str(temp_ini)], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running {temp_ini} (return code {result.returncode}):")
            print("--- STDOUT ---")
            print(result.stdout)
            print("--- STDERR ---")
            print(result.stderr)
            continue
        try:
            tpi, thr = parse_metrics(result.stdout)
            tpi_vals.append(tpi)
            thr_vals.append(thr)
        except Exception as e:
            print(f"Failed to parse metrics from output ({e}):")
            print("--- STDOUT ---")
            print(result.stdout)
            print("--- STDERR ---")
            print(result.stderr)
            continue
    return {
        "time_per_iter": {**stats(tpi_vals), "unit": "sec"},
        "estimated_throughput": {**stats(thr_vals), "unit": "GB/s"},
        "runs_completed": len(tpi_vals),
        "status": "ok" if tpi_vals else "error"
    }

def modify_ini(base_path: Path, n2: int, kernel: str) -> Path:
    temp_dir = tempfile.mkdtemp()
    temp_ini = Path(temp_dir) / f"tmp_{kernel}_{n2}.ini"
    with open(base_path) as f:
        lines = f.readlines()
    with open(temp_ini, "w") as f:
        for line in lines:
            if line.startswith("n0 ="):
                f.write(f"n0 = {N0}\n")
            elif line.startswith("n1 ="):
                f.write(f"n1 = {N1}\n")
            elif line.startswith("n2 ="):
                f.write(f"n2 = {n2}\n")
            elif line.startswith("kernelImpl"):
                f.write(f"kernelImpl = {kernel}\n")
            else:
                f.write(line)
    return temp_ini

def main():
    results = {"executable": str(EXE), "cases": {}}

    for kernel in KERNEL_IMPLS:
        for n2 in N2_VALUES:
            ini_path = modify_ini(BASE_INI, n2, kernel)
            print(f"Running: kernel={kernel}, n2={n2}")
            result = run_case(ini_path)
            results["cases"][f"{kernel}_n2_{n2}"] = {
                "n0": N0, "n1": N1, "n2": n2,
                **result
            }
            shutil.rmtree(ini_path.parent)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to: {OUT_JSON}")

if __name__ == "__main__":
    main()
