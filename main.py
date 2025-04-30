import time
import numpy as np
import os
import importlib

from benchmarks import sphere, rastrigin, rosenbrock, ackley, griewank, Schwefel
from benchmarks.breastCancer import BreastCancer
from benchmarks.listSort import ListSort
from benchmarks.listSortCoEvo import ListSortCoEvo
from plotting import plot_grouped_logs, plot_coevolution_listsort

# === CONFIG ===
POP_SIZE = 50
ITERATIONS = 500
RUNS = 10
SEED = 42
np.random.seed(SEED)

# === Benchmark functions
benchmarks = {
    # "sphere": sphere(),
    # "rastrigin": rastrigin(),
    # "rosenbrock": rosenbrock(),
    # "ackley": ackley(),
    # "griewank": griewank(),
    # "Schwefel": Schwefel(),
    # "breastcancer": BreastCancer(),
    "listsort": ListSort()
}

# === Import algorithm modules
algorithm_modules = ["adam_torch", "mpa", "pso", "lm_impa"]
algorithms = {}

for module_name in algorithm_modules:
    try:
        module = importlib.import_module(f"algorithms.{module_name}")
        run_func = getattr(module, f"run_{module_name}")
        algorithms[module_name] = run_func
    except (ModuleNotFoundError, AttributeError):
        print(f"[SKIPPED] '{module_name}' not found or missing `run_{module_name}()`")

# === Output folders
results_dir = "results/fitness_logs"
weights_dir = "results/weights"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(weights_dir, exist_ok=True)

def run_all():
    for algo_name, algo_func in algorithms.items():
        for bench_name, bench in benchmarks.items():
            print(f"\nRunning {algo_name.upper()} on {bench_name.upper()}")

            if hasattr(bench, 'evaluate'):
                evaluate_func = bench.evaluate
                lb, ub = bench.getBounds()
                dim = bench.getDimensions()
                is_realworld = True
            else:
                evaluate_func = bench["function"]
                lb, ub = bench["bounds"]
                dim = bench["dim"]
                is_realworld = False

            best_fitness = float("inf")
            best_solution = None

            for run in range(1, RUNS + 1):
                start = time.time()

                fitness_log, weights = algo_func(
                    benchmark_func=evaluate_func,
                    dim=dim,
                    pop_size=POP_SIZE,
                    iterations=ITERATIONS,
                    bounds=(lb, ub)
                )

                end = time.time()
                duration = round(end - start, 2)

                log_path = f"{results_dir}/{algo_name}_{bench_name}_run{run}.csv"
                np.savetxt(log_path, fitness_log, delimiter=",")

                final_fitness = fitness_log[-1]
                print(f"Run {run}/{RUNS} complete. Time: {duration}s. Final fitness: {final_fitness:.6f}")

                if bench_name.lower() == "breastcancer" and final_fitness < best_fitness:
                    best_fitness = final_fitness
                    best_solution = weights

            # Save only best solution weights
            if bench_name.lower() == "breastcancer" and best_solution is not None:
                weight_path = f"{weights_dir}/{algo_name}_{bench_name}_BEST.npy"
                np.save(weight_path, best_solution)
                print(f"[SAVED BEST] {algo_name} on {bench_name}: {weight_path}")

                if algo_name != "adam_torch":
                    bench.test_model(best_solution)

def coevolution_sort_demo():
    benchmark = ListSortCoEvo()
    dim = benchmark.getDimensions()
    bounds = benchmark.getBounds()
    
    POP_SIZE = 30
    ITERATIONS = 100
    MUTATION_RATE = 0.2

    # Random initial sorter population
    sorters = [np.random.uniform(bounds[0], bounds[1], dim) for _ in range(POP_SIZE)]
    best_fitnesses = []

    for gen in range(ITERATIONS):
        fitnesses = [benchmark.evaluate_sorter(genome) for genome in sorters]
        sorted_indices = np.argsort(fitnesses)
        sorters = [sorters[i] for i in sorted_indices[:POP_SIZE//2]]  # selection

        # Mutate to produce offspring
        new_sorters = []
        for s in sorters:
            offspring = s.copy()
            mutation_mask = np.random.rand(dim) < MUTATION_RATE
            noise = np.random.uniform(-1, 1, dim) * mutation_mask
            offspring += noise
            new_sorters.append(offspring)

        sorters += new_sorters

        # Evolve the test lists based on current sorters
        benchmark.evolve_lists(sorters)

        best_fitness = fitnesses[sorted_indices[0]]
        best_fitnesses.append(best_fitness)
        print(f"Generation {gen+1}/{ITERATIONS} - Best fitness (avg inversions): {best_fitness:.3f}")

    # Save convergence log
    np.savetxt("results/fitness_logs/coevo_listsort.csv", best_fitnesses, delimiter=",")

if __name__ == "__main__":
    run_all()

    print("\nRunning co-evolution demo...")
    coevolution_sort_demo()

    print("\nGenerating plots...")
    plot_grouped_logs()
    plot_coevolution_listsort()
    print("Plots saved in results/plots/")