#!/usr/bin/env python3

import os
import subprocess
import re
import json
import statistics

# Sizes to test
configurations = [
    {"nx": 128, "ny": 128, "nvx": 64, "nvy": 64},
    {"nx": 1024, "ny": 1024, "nvx": 32, "nvy": 32},
    {"nx": 512, "ny": 512, "nvx": 64, "nvy": 64},
]

# Paths
conf_file = "4d-advection.ini"
executable = "./4d-advection"

def update_conf_file(cfg):
    with open(conf_file, 'r') as f:
        lines = f.readlines()

    with open(conf_file, 'w') as f:
        for line in lines:
            if line.startswith("nx"):
                f.write(f"nx  = {cfg['nx']}\n")
            elif line.startswith("ny"):
                f.write(f"ny  = {cfg['ny']}\n")
            elif line.startswith("nvx"):
                f.write(f"nvx = {cfg['nvx']}\n")
            elif line.startswith("nvy"):
                f.write(f"nvy = {cfg['nvy']}\n")
            else:
                f.write(line)

def run_simulation(cfg_str, run_id):
    log_file = f"run_{cfg_str}_{run_id}.log"
    with open(log_file, 'w') as out:
        subprocess.run([executable, conf_file], stdout=out, stderr=subprocess.STDOUT)
    return log_file

def cleanup():
    # No specific cleanup needed for now
    pass

def parse_kernel_times(log_file):
    data = {"GridX": [], "GridY": [], "GridVx": [], "GridVy": []}
    pattern = re.compile(r"(GridX|GridY|GridVx|GridVy)\s+=+ Kernel time: ([\d\.]+)")
    with open(log_file, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                key, val = match.groups()
                data[key].append(float(val))
    return data

def aggregate_stats(data):
    stats = {}
    for key, values in data.items():
        if values:
            stats[key] = {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0
            }
    return stats

if __name__ == "__main__":
    results = []

    for cfg in configurations:
        cfg_str = f"{cfg['nx']}x{cfg['nvx']}_Y{cfg['ny']}x{cfg['nvy']}"
        print(f"Running config: {cfg_str}")
        update_conf_file(cfg)
        all_data = {"GridX": [], "GridY": [], "GridVx": [], "GridVy": []}

        for i in range(50):
            log = run_simulation(cfg_str, i + 1)
            run_data = parse_kernel_times(log)
            for key in all_data:
                all_data[key].extend(run_data.get(key, []))
            cleanup()

        result_entry = {
            "nx": cfg['nx'],
            "ny": cfg['ny'],
            "nvx": cfg['nvx'],
            "nvy": cfg['nvy'],
            "stats": aggregate_stats(all_data)
        }
        results.append(result_entry)

    with open("advection_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Benchmark complete. Results saved to advection_benchmark_results.json")
