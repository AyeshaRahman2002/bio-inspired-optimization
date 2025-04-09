import numpy as np

def run_mpa(benchmark_func, dim, pop_size, iterations, bounds):
    # Unpack variable bounds
    lower, upper = bounds

    # Initialize a population of agents randomly within bounds
    population = np.random.uniform(lower, upper, (pop_size, dim))

    # Evaluate initial fitness for each agent
    fitness = np.array([benchmark_func(ind) for ind in population])

    # Identify the elite agent (best solution so far)
    elite_idx = np.argmin(fitness)
    elite = population[elite_idx].copy()
    elite_fitness = fitness[elite_idx]
    fitness_log = []  # Track best fitness over time

    # FADs probability controls random perturbation to escape local optima
    FADs_prob = 0.1  

    for t in range(iterations):
        # Step size decays over time to shift from exploration → exploitation
        step_size = 1 - t / iterations

        # Adaptive probability for choosing Brownian vs Lévy (used in Phase 2)
        P = 0.3 + 0.7 * (t / iterations)  # Starts at 0.3 → increases to 1

        for i in range(pop_size):
            r = np.random.rand()
            phase = t / iterations

            # Select motion strategy based on current phase of iteration
            if phase < 1/3:
                # Phase 1: Global exploration using Brownian motion
                motion = brownian(dim)
            elif phase < 2/3:
                # Phase 2: Mixture of Brownian and Lévy motion (adaptive)
                motion = brownian(dim) if r < P else levy_flight(dim)
            else:
                # Phase 3: Full exploitation using Lévy flight
                motion = levy_flight(dim)

            #   Move agent toward elite using scaled motion
            direction = elite - population[i]
            new_position = population[i] + motion * step_size * direction

            #   Keep new position within bounds
            new_position = np.clip(new_position, lower, upper)

            #   Occasionally add a random jump (Fish Aggregating Devices behavior)
            if np.random.rand() < FADs_prob:
                rand_step = np.random.uniform(lower, upper, dim)
                new_position += rand_step * step_size
                new_position = np.clip(new_position, lower, upper)

            #   Evaluate fitness of new position
            new_fitness = benchmark_func(new_position)

            #   Replace current agent if new position is better
            if new_fitness < fitness[i]:
                population[i] = new_position
                fitness[i] = new_fitness

        # Update elite agent if a better one is found
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < elite_fitness:
            elite = population[best_idx].copy()
            elite_fitness = fitness[best_idx]

        # Record best fitness at this iteration
        fitness_log.append(elite_fitness)

    return np.array(fitness_log), elite


def brownian(dim):
    """
    Brownian motion: small random step drawn from normal distribution.
    Useful for fine-grained exploration in continuous spaces.
    """
    return np.random.normal(0, 1, dim)

def levy_flight(dim, beta=1.5):
    """
    Lévy flight: generates long-range jumps based on heavy-tailed distribution.
    Helps agents escape local optima during exploitation phase.
    """
    sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma, dim)
    v = np.random.normal(0, 1, dim)
    return u / (np.abs(v) ** (1 / beta))
