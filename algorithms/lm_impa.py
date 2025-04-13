import numpy as np
from scipy.optimize import minimize

def brownian(dim):
    return np.random.normal(0, 1, dim)

def levy_flight(dim, beta=1.5):
    sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma, dim)
    v = np.random.normal(0, 1, dim)
    return u / (np.abs(v) ** (1 / beta))

def run_lm_impa(benchmark_func, dim, pop_size, iterations, bounds):
    lower, upper = bounds
    population = np.random.uniform(lower, upper, (pop_size, dim))
    fitness = np.array([benchmark_func(ind) for ind in population])

    elite_idx = np.argmin(fitness)
    elite = population[elite_idx].copy()
    elite_fitness = fitness[elite_idx]
    fitness_log = []

    FADs_prob = 0.1
    lm_interval = 10  # Apply LM more frequently
    num_lm_agents = max(1, int(0.2 * pop_size))  # Top 20% of agents
    stagnation_counter = 0

    for t in range(iterations):
        step_size = 1 - t / iterations
        P = 0.3 + 0.7 * (t / iterations)

        for i in range(pop_size):
            r = np.random.rand()
            phase = t / iterations

            if phase < 1/3:
                motion = brownian(dim)
            elif phase < 2/3:
                motion = brownian(dim) if r < P else levy_flight(dim)
            else:
                motion = levy_flight(dim)

            direction = elite - population[i]
            new_position = population[i] + motion * step_size * direction
            new_position = np.clip(new_position, lower, upper)

            if np.random.rand() < FADs_prob:
                rand_step = np.random.uniform(lower, upper, dim)
                new_position += rand_step * step_size
                new_position = np.clip(new_position, lower, upper)

            new_fitness = benchmark_func(new_position)

            if new_fitness < fitness[i]:
                population[i] = new_position
                fitness[i] = new_fitness

        # Apply LM if stagnating or on schedule
        if stagnation_counter >= 5 or t % lm_interval == 0:
            top_indices = np.argsort(fitness)[:num_lm_agents]
            for idx in top_indices:
                initial = population[idx].copy()
                before = fitness[idx]
                result = minimize(benchmark_func, initial, bounds=[(lower, upper)] * dim, method='L-BFGS-B')
                after = result.fun
                if after < before:
                    population[idx] = result.x
                    fitness[idx] = after
                    if after < elite_fitness:
                        elite = result.x
                        elite_fitness = after

        # Update elite & check stagnation
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < elite_fitness:
            elite = population[best_idx].copy()
            elite_fitness = fitness[best_idx]
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        fitness_log.append(elite_fitness)

    return np.array(fitness_log), elite
