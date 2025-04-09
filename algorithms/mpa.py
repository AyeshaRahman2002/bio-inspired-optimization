import numpy as np

def run_mpa(benchmark_func, dim, pop_size, iterations, bounds):
    lower, upper = bounds
    population = np.random.uniform(lower, upper, (pop_size, dim))
    fitness = np.array([benchmark_func(ind) for ind in population])

    elite_idx = np.argmin(fitness)
    elite = population[elite_idx].copy()
    elite_fitness = fitness[elite_idx]
    fitness_log = []

    FADs_prob = 0.1  # Controlled randomness
    for t in range(iterations):
        step_size = 1 - t / iterations
        P = 0.3 + 0.7 * (t / iterations)  # Adaptive P: starts at 0.3 → increases to 1

        for i in range(pop_size):
            r = np.random.rand()
            phase = t / iterations

            # === Phase 1: Brownian | Phase 2: Mixed | Phase 3: Lévy
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

        best_idx = np.argmin(fitness)
        if fitness[best_idx] < elite_fitness:
            elite = population[best_idx].copy()
            elite_fitness = fitness[best_idx]

        fitness_log.append(elite_fitness)

    return np.array(fitness_log), elite


def brownian(dim):
    return np.random.normal(0, 1, dim)

def levy_flight(dim, beta=1.5):
    sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma, dim)
    v = np.random.normal(0, 1, dim)
    return u / (np.abs(v) ** (1 / beta))
