# plotting.py
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from itertools import cycle

LOG_DIR = "results/fitness_logs"
PLOT_DIR = "results/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Automatically cycle through colors
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

def get_algo_and_benchmark(file_name):
    base = os.path.basename(file_name).replace(".csv", "")
    parts = base.split("_")
    run_idx = next(i for i, part in enumerate(parts) if part.startswith("run"))
    algo = "_".join(parts[:run_idx - 1])
    bench = parts[run_idx - 1]
    return algo, bench

def group_logs_by_benchmark():
    grouped = {}
    for filepath in glob(f"{LOG_DIR}/*.csv"):
        algo, bench = get_algo_and_benchmark(filepath)
        if bench not in grouped:
            grouped[bench] = {}
        if algo not in grouped[bench]:
            grouped[bench][algo] = []
        grouped[bench][algo].append(filepath)
    return grouped

def plot_grouped_logs():
    grouped_logs = group_logs_by_benchmark()
    for benchmark, algo_runs in grouped_logs.items():
        plt.figure(figsize=(10, 6))
        algo_colors = {algo: next(color_cycle) for algo in algo_runs}

        for algo, filepaths in algo_runs.items():
            all_runs = [np.loadtxt(fp, delimiter=",") for fp in filepaths]
            # Ensure equal length runs (in case some terminate early)
            min_len = min(len(run) for run in all_runs)
            all_runs = [run[:min_len] for run in all_runs]
            arr = np.array(all_runs)
            avg = np.mean(arr, axis=0)
            std = np.std(arr, axis=0)

            x = np.arange(len(avg))
            plt.plot(x, avg, label=algo.upper(), color=algo_colors[algo])
            plt.fill_between(x, avg - std, avg + std, alpha=0.2, color=algo_colors[algo])

        plt.title(f"Convergence on {benchmark.replace('_', ' ').upper()}")
        plt.xlabel("Iterations")
        plt.ylabel("Fitness (Lower is Better)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/{benchmark}_convergence.png")
        plt.close()
