import numpy as np

def run_mpa(benchmark_func, dim, pop_size, iterations, bounds):
    lower, upper = bounds
    elite_memory = None

    # Initialize population
    population = np.random.uniform(lower, upper, (pop_size, dim))
    fitness = np.array([benchmark_func(ind) for ind in population])
    fitness_log = []

    for t in range(iterations):
        # Update elite (best solution)
        elite_idx = np.argmin(fitness)
        elite_solution = population[elite_idx].copy()
        elite_fitness = fitness[elite_idx]

        if elite_memory is None or benchmark_func(elite_memory) > elite_fitness:
            elite_memory = elite_solution.copy()

        # Update each agent
        for i in range(pop_size):
            if np.random.rand() < 0.5:
                # Brownian motion (exploration)
                step = np.random.normal(0, 1, dim)
            else:
                # Lévy flight (exploitation)
                step = levy_flight(dim)

            # Move towards elite
            new_position = population[i] + step * (elite_memory - population[i])
            new_position = np.clip(new_position, lower, upper)

            new_fitness = benchmark_func(new_position)
            if new_fitness < fitness[i]:
                population[i] = new_position
                fitness[i] = new_fitness

        fitness_log.append(np.min(fitness))

    return np.array(fitness_log), elite_memory


def levy_flight(dim):
    """Generate Lévy flight step using Mantegna's algorithm (approx)"""
    beta = 1.5
    sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
             (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma, dim)
    v = np.random.normal(0, 1, dim)
    step = u / (np.abs(v) ** (1 / beta))
    return step
