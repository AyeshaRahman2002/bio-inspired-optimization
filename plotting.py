# plotting.py
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from itertools import cycle

LOG_DIR = "results/fitness_logs"
PLOT_DIR = "results/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Create an automatic color cycle
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

    custom_colors = {
        "PSO": "blue",
        "ADAM": "darkorange",
        "ADAM_TORCH": "green",
        "LM_IMPA": "black",
        "MPA": "purple"
    }
    
    custom_linestyles = {
        "PSO": "-",
        "ADAM": "--",
        "ADAM_TORCH": "-.",
        "LM_IMPA": ":",
        "MPA": (0, (3, 1, 1, 1))  # dotted-dashed
    }

    custom_markers = {
        "PSO": "o",
        "ADAM": "s",
        "ADAM_TORCH": "D",
        "LM_IMPA": "^",
        "MPA": "x"
    }

    for benchmark, algo_runs in grouped_logs.items():
        plt.figure(figsize=(10, 6))

        for algo, filepaths in algo_runs.items():
            all_runs = [np.loadtxt(fp, delimiter=",") for fp in filepaths]
            min_len = min(len(run) for run in all_runs)
            all_runs = [run[:min_len] for run in all_runs]
            arr = np.array(all_runs)
            avg = np.mean(arr, axis=0)
            std = np.std(arr, axis=0)

            x = np.arange(len(avg))
            algo_upper = algo.upper()
            color = custom_colors.get(algo_upper, next(color_cycle))
            linestyle = custom_linestyles.get(algo_upper, "-")
            marker = custom_markers.get(algo_upper, None)

            plt.plot(x, avg, label=algo_upper, color=color, linestyle=linestyle,
                     marker=marker, markevery=50, linewidth=2)
            plt.fill_between(x, avg - std, avg + std, alpha=0.2, color=color)

        plt.title(f"Convergence on {benchmark.upper()}")
        plt.xlabel("Iterations")
        plt.ylabel("Fitness")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/{benchmark}_convergence.png")
        plt.close()
