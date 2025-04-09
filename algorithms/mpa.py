import numpy as np

def run_mpa(benchmark_func, dim, pop_size, iterations, bounds):
    lower, upper = bounds

    # === Initialize agents (predators/prey)
    population = np.random.uniform(lower, upper, (pop_size, dim))

    # === Track fitness values
    fitness = np.array([benchmark_func(ind) for ind in population])
    elite_idx = np.argmin(fitness)
    elite = population[elite_idx].copy()
    elite_fitness = fitness[elite_idx]
    fitness_log = []

    # === Algorithm parameters
    FADs_prob = 0.2  # Probability to apply Fish Aggregating Devices (random walks)
    P = 0.5          # Probability switch for Brownian vs Lévy flight (used in Phase 2)

    for t in range(iterations):
        # Linearly decreasing step size (used in motion)
        step_size = 1 - t / iterations

        for i in range(pop_size):
            r = np.random.rand()

            # --- Determine the current phase based on iteration progress
            if t / iterations < 1/3:
                # PHASE 1: Global exploration — Brownian motion
                motion = brownian(dim)
            elif t / iterations < 2/3:
                # PHASE 2: Exploitation and exploration balance
                motion = brownian(dim) if r < P else levy_flight(dim)
            else:
                # PHASE 3: Strong exploitation — Lévy flight
                motion = levy_flight(dim)

            # === Move agent towards elite with adaptive step
            direction = elite - population[i]
            new_position = population[i] + motion * step_size * direction
            new_position = np.clip(new_position, lower, upper)

            # === Apply FADs effect with certain probability to escape local optima
            if np.random.rand() < FADs_prob:
                rand_step = np.random.uniform(lower, upper, dim)
                new_position = new_position + rand_step * step_size
                new_position = np.clip(new_position, lower, upper)

            # === Evaluate new position
            new_fitness = benchmark_func(new_position)

            if new_fitness < fitness[i]:
                population[i] = new_position
                fitness[i] = new_fitness

        # === Update elite
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < elite_fitness:
            elite = population[best_idx].copy()
            elite_fitness = fitness[best_idx]

        fitness_log.append(elite_fitness)

    return np.array(fitness_log), elite


def brownian(dim):
    """Small steps (normal distribution): used for steady movement in known areas"""
    return np.random.normal(0, 1, dim)

def levy_flight(dim, beta=1.5):
    """Lévy flight steps for sudden long jumps: used for escaping local minima"""
    sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
             (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma, dim)
    v = np.random.normal(0, 1, dim)
    return u / (np.abs(v) ** (1 / beta))
