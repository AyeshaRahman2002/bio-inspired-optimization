import time
import numpy as np
import os
import importlib

# Import benchmark functions
from benchmarks import sphere, rastrigin, rosenbrock, ackley, griewank
from plotting import plot_grouped_logs

# === CONFIG ===
DIMENSIONS = 30
POP_SIZE = 50
ITERATIONS = 500
RUNS = 10
BOUNDS = (-5.12, 5.12)
SEED = 42
np.random.seed(SEED)

# === Benchmark functions
benchmarks = {
    "sphere": sphere,
    "rastrigin": rastrigin,
    "rosenbrock": rosenbrock,
    "ackley": ackley,
    "griewank": griewank
}

# === Dynamically try to import available algorithm modules
algorithm_modules = ["mpa", "pso", "lm_impa"]
algorithms = {}

for module_name in algorithm_modules:
    try:
        module = importlib.import_module(f"algorithms.{module_name}")
        run_func = getattr(module, f"run_{module_name}")
        algorithms[module_name] = run_func
    except (ModuleNotFoundError, AttributeError) as e:
        print(f"[SKIPPED] '{module_name}' not found or missing `run_{module_name}()`")

# === Output folder
results_dir = "results/fitness_logs"
os.makedirs(results_dir, exist_ok=True)

def run_all():
    for algo_name, algo_func in algorithms.items():
        for bench_name, bench_func in benchmarks.items():
            print(f"\nRunning {algo_name.upper()} on {bench_name.upper()}")
            for run in range(1, RUNS + 1):
                start = time.time()

                fitness_log, best_solution = algo_func(
                    benchmark_func=bench_func,
                    dim=DIMENSIONS,
                    pop_size=POP_SIZE,
                    iterations=ITERATIONS,
                    bounds=BOUNDS
                )

                end = time.time()
                duration = round(end - start, 2)

                log_path = f"{results_dir}/{algo_name}_{bench_name}_run{run}.csv"
                np.savetxt(log_path, fitness_log, delimiter=",")

                print(f"Run {run}/10 complete. Time: {duration}s. Final fitness: {fitness_log[-1]:.6f}")

if __name__ == "__main__":
    run_all()
    print("\nGenerating plots...")
    plot_grouped_logs()
    print("Plots saved in results/plots/")
