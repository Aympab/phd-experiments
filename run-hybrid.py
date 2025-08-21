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
PARADV_ROOT = Path("/home/ac.amillan/source/parallel-advection")  # in case we extend later
OUT_DIR     = Path("/home/ac.amillan/source/phd-experiments/out/hybrid-subgroups")

DEFAULT_SGRPS = [1, 2, 4, 8]
DEFAULT_SEQS  = [1, 2, 3, 4]
DEFAULT_RUNS  = 5

# --- Regexes for parsing performance metrics ---
BYTES_PER_SEC_RE = re.compile(r"\bbytes?_per_sec\b\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
THR_GB_S_RE      = re.compile(r"\bestim(?:ated)?_throughput\b\s*[:=]\s*([0-9]*\.?[0-9]+)\s*GB/s", re.IGNORECASE)
THR_B_S_RE       = re.compile(r"\bestim(?:ated)?_throughput\b\s*[:=]\s*([0-9]*\.?[0-9]+)\s*B/s", re.IGNORECASE)

# --- Cases ---
CASES = {
    "case0": {"n0": 32768, "n1": 64,   "n2": 1},
    "case1": {"n0": 512,   "n1": 64,   "n2": 1},
    "case2": {"n0": 512,   "n1": 6144, "n2": 64},
    "case3": {"n0": 8,     "n1": 6144, "n2": 64},
}

# --- Base INI template for hybrid-paradv ---
HYBRID_BASE_INI = """[problem]
n0 = {n0}
n1 = {n1}
n2 = {n2}
maxIter = 10
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
pref_wg_size = 512
seq_size0 = 1
seq_size2 = 1
nSubgroups_Local  = {nsg_local}
nSubgroups_Global = {nsg_global}
seqSize_Local  = {seq_local}
seqSize_Global = {seq_global}
"""


def build_executable(impl: str, hw: str) -> Path:
    if impl == "hybrid":
        return HYBRID_ROOT / f"build_dpcpp_{hw}" / "src" / "advection"
    else:
        # Future-proofed; not used for now
        return PARADV_ROOT / f"build_dpcpp_{hw}" / "src" / "advection"


def write_ini(tmp_dir: Path,
              case_name: str,
              n0: int, n1: int, n2: int,
              nsg_local: int, nsg_global: int,
              seq_local: int, seq_global: int) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ini_path = tmp_dir / f"{case_name}_nsgL{nsg_local}_nsgG{nsg_global}_seqL{seq_local}_seqG{seq_global}.ini"
    ini_text = HYBRID_BASE_INI.format(
        n0=n0, n1=n1, n2=n2,
        nsg_local=nsg_local, nsg_global=nsg_global,
        seq_local=seq_local, seq_global=seq_global
    )
    ini_path.write_text(ini_text)
    return ini_path


def parse_perf(stdout: str) -> Optional[float]:
    """
    Return performance in bytes_per_sec (float) if found, else None.
    Accepts:
      - 'bytes_per_sec: <num>'
      - 'estimated_throughput: <num> GB/s'  (converted to B/s)
      - 'estimated_throughput: <num> B/s'
    """
    m = BYTES_PER_SEC_RE.search(stdout)
    if m:
        return float(m.group(1))

    m = THR_GB_S_RE.search(stdout)
    if m:
        return float(m.group(1)) * 1e9

    m = THR_B_S_RE.search(stdout)
    if m:
        return float(m.group(1))

    return None


def run_once(exe: Path, ini: Path) -> Tuple[bool, str]:
    try:
        proc = subprocess.run(
            [str(exe), str(ini)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        ok = (proc.returncode == 0)
        return ok, proc.stdout + "\n" + proc.stderr
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


def sweep_subgroups(exe: Path, tmp_dir: Path, case_name: str, case_dims: Dict[str, int], sgrps: List[int], runs: int) -> Dict[str, Any]:
    """
    For each S in sgrps, test (S,0), (0,S), (S//2, S-S//2) at fixed seq sizes (keep default seqL=3, seqG=1 per given INI).
    """
    results: Dict[str, Any] = {"sgrps": sgrps, "entries": []}
    for S in sgrps:
        configs = [
            ("local_only",  S, 0),
            ("global_only", 0, S),
            ("split",       S // 2, S - (S // 2)),
        ]
        entry = {"S": S, "variants": {}}
        for label, nsgL, nsgG in configs:
            ini = write_ini(
                tmp_dir, case_name,
                case_dims["n0"], case_dims["n1"], case_dims["n2"],
                nsg_local=nsgL, nsg_global=nsgG,
                seq_local=3, seq_global=1,   # keep seq sizes from your provided INI while sweeping subgroups
            )
            perfs: List[float] = []
            for i in range(runs):
                ok, out = run_once(exe, ini)
                if not ok:
                    # record failure and stop this variant
                    entry["variants"][label] = {
                        "config": {"nSubgroups_Local": nsgL, "nSubgroups_Global": nsgG},
                        "runs_completed": len(perfs),
                        "status": "error",
                        "bytes_per_sec": {**summarize([]), "unit": "B/s"},
                    }
                    break
                perf = parse_perf(out)
                if perf is None:
                    entry["variants"][label] = {
                        "config": {"nSubgroups_Local": nsgL, "nSubgroups_Global": nsgG},
                        "runs_completed": len(perfs),
                        "status": "error_parse",
                        "bytes_per_sec": {**summarize([]), "unit": "B/s"},
                    }
                    break
                perfs.append(perf)
            else:
                # Completed all runs
                entry["variants"][label] = {
                    "config": {"nSubgroups_Local": nsgL, "nSubgroups_Global": nsgG},
                    "runs_completed": len(perfs),
                    "status": "ok",
                    "bytes_per_sec": {**summarize(perfs), "unit": "B/s"},
                }
        results["entries"].append(entry)
    return results


def sweep_seqsize(exe: Path, tmp_dir: Path, case_name: str, case_dims: Dict[str, int], seqs: List[int], runs: int) -> Dict[str, Any]:
    """
    For each Q in seqs, test (Q,1), (1,Q), (ceil(Q/2), floor(Q/2)) at fixed subgroups (keep default nsgL=4, nsgG=4 per given INI).
    """
    results: Dict[str, Any] = {"seqs": seqs, "entries": []}
    for Q in seqs:
        split_L = math.ceil(Q / 2)
        split_G = Q - split_L
        configs = [
            ("local_only",  Q, 1),
            ("global_only", 1, Q),
            ("split",       split_L, split_G),
        ]
        entry = {"Q": Q, "variants": {}}
        for label, seqL, seqG in configs:
            ini = write_ini(
                tmp_dir, case_name,
                case_dims["n0"], case_dims["n1"], case_dims["n2"],
                nsg_local=4, nsg_global=4,   # keep nSubgroups from your provided INI while sweeping seq sizes
                seq_local=seqL, seq_global=seqG,
            )
            perfs: List[float] = []
            for i in range(runs):
                ok, out = run_once(exe, ini)
                if not ok:
                    entry["variants"][label] = {
                        "config": {"seqSize_Local": seqL, "seqSize_Global": seqG},
                        "runs_completed": len(perfs),
                        "status": "error",
                        "bytes_per_sec": {**summarize([]), "unit": "B/s"},
                    }
                    break
                perf = parse_perf(out)
                if perf is None:
                    entry["variants"][label] = {
                        "config": {"seqSize_Local": seqL, "seqSize_Global": seqG},
                        "runs_completed": len(perfs),
                        "status": "error_parse",
                        "bytes_per_sec": {**summarize([]), "unit": "B/s"},
                    }
                    break
                perfs.append(perf)
            else:
                entry["variants"][label] = {
                    "config": {"seqSize_Local": seqL, "seqSize_Global": seqG},
                    "runs_completed": len(perfs),
                    "status": "ok",
                    "bytes_per_sec": {**summarize(perfs), "unit": "B/s"},
                }
        results["entries"].append(entry)
    return results


def main():
    ap = argparse.ArgumentParser(description="Hybrid-paradv subgroup/seqsize sweeper")
    ap.add_argument("--hw",   required=True, choices=["mi300", "pvc", "h100"])
    ap.add_argument("--impl", required=True, choices=["hybrid", "ndrange", "adaptivewg"])
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    ap.add_argument("--sgrps", type=str, default=",".join(map(str, DEFAULT_SGRPS)),
                    help="Comma-separated list of subgroup totals to test (default: 1,2,4,8)")
    ap.add_argument("--seqs", type=str, default=",".join(map(str, DEFAULT_SEQS)),
                    help="Comma-separated list of seq sizes to test (default: 1,2,3,4)")
    args = ap.parse_args()

    if args.impl != "hybrid":
        raise SystemExit("This benchmark currently supports only --impl hybrid (will extend later).")

    # Resolve executable
    exe = build_executable(args.impl, args.hw)
    if not exe.exists():
        raise SystemExit(f"Executable not found: {exe}")

    # Prepare output & temp config dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = OUT_DIR / "tmp_configs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Parse candidate lists
    try:
        sgrps = [int(x) for x in args.sgrps.split(",") if x.strip() != ""]
        seqs  = [int(x) for x in args.seqs.split(",")  if x.strip() != ""]
    except Exception:
        raise SystemExit("Error parsing --sgrps/--seqs; they must be comma-separated integers.")

    # Results container
    results: Dict[str, Any] = {
        "impl": args.impl,
        "hardware": args.hw,
        "executable": str(exe),
        "runs_per_config": args.runs,
        "cases": {},
        "notes": {
            "subgroups_sweep_holds_seqSize": {"seqSize_Local": 3, "seqSize_Global": 1},
            "seqsize_sweep_holds_subgroups": {"nSubgroups_Local": 4, "nSubgroups_Global": 4},
            "units": {"bytes_per_sec": "B/s"},
        },
    }

    # Run all four cases
    for case_name, dims in CASES.items():
        print(f"[{args.impl}/{args.hw}] {case_name}: n0={dims['n0']} n1={dims['n1']} n2={dims['n2']}")
        subgroups_block = sweep_subgroups(exe, tmp_dir, case_name, dims, sgrps, args.runs)
        seqsize_block   = sweep_seqsize(exe, tmp_dir, case_name, dims, seqs,  args.runs)
        results["cases"][case_name] = {
            "problem": dims,
            "subgroups_sweep": subgroups_block,
            "seqsize_sweep": seqsize_block,
        }

    # Write JSON
    out_path = OUT_DIR / f"dpcpp_{args.hw}_{args.impl}.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
