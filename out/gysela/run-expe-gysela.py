#!/usr/bin/env python

import os
import subprocess
import re
import json
import statistics
import shutil

# Sizes to test
configurations = [
    {"x": 128, "y": 128, "vx": 64, "vy": 64},
    {"x": 1024, "y": 1024, "vx": 32, "vy": 32},
    {"x": 512, "y": 512, "vx": 64, "vy": 64},
]

# Path to config file
conf_file = "/home/ac.amillan/source/gyselalibxx/build_h100/simulations/geometryXYVxVy/landau/conf.yml"
executable = "/home/ac.amillan/source/gyselalibxx/build_h100/simulations/geometryXYVxVy/landau/landau4d_fft"

def update_conf_file(cfg):
    with open(conf_file, 'r') as f:
        lines = f.readlines()

    with open(conf_file, 'w') as f:
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('x_ncells:'):
                f.write(f"  x_ncells: {cfg['x']}\n")
            elif stripped.startswith('y_ncells:'):
                f.write(f"  y_ncells: {cfg['y']}\n")
            elif stripped.startswith('vx_ncells:'):
                f.write(f"  vx_ncells: {cfg['vx']}\n")
            elif stripped.startswith('vy_ncells:'):
                f.write(f"  vy_ncells: {cfg['vy']}\n")
            else:
                f.write(line)

def run_simulation(cfg_str, run_id):
    log_file = f"run_{cfg_str}_{run_id}.log"
    with open(log_file, 'w') as out:
        subprocess.run([executable, conf_file], stdout=out, stderr=subprocess.STDOUT)
    return log_file

def cleanup():
    for f in os.listdir('.'):
        if f.startswith("GYSELALIBXX_") and f.endswith(".h5"):
            os.remove(f)

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
        cfg_str = f"{cfg['x']}x{cfg['vx']}"
        print(f"Running config: {cfg_str}")
        update_conf_file(cfg)
        all_data = {"GridX": [], "GridY": [], "GridVx": [], "GridVy": []}

        for i in range(10):  # Run twice
            log = run_simulation(cfg_str, i+1)
            run_data = parse_kernel_times(log)
            for key in all_data:
                all_data[key].extend(run_data.get(key, []))
            cleanup()

        result_entry = {
            "x": cfg['x'],
            "y": cfg['y'],
            "vx": cfg['vx'],
            "vy": cfg['vy'],
            "stats": aggregate_stats(all_data)
        }
        results.append(result_entry)

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Benchmark complete. Results saved to benchmark_results.json")
