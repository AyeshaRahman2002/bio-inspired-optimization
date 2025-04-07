import numpy as np

def run_pso(benchmark_func, dim, pop_size, iterations, bounds, w=0.5, c1=1, c2=2):
    lower, upper = bounds

    # Initialize position and velocity
    particles = np.random.uniform(lower, upper, (pop_size, dim))
    velocities = velocities = np.zeros((pop_size, dim))

    # Evaluate initial fitness
    best_positions = np.copy(particles)
    best_fitness = np.array([benchmark_func(i) for i in particles])

    global_best_position = best_positions[np.argmin(best_fitness)]
    global_best_fitness = np.min(best_fitness)
    fitness_log = []

    for i in range(iterations):

        # Update velocity
        r1 = np.random.uniform(0, 1, (pop_size, dim))
        r2 = np.random.uniform(0, 1, (pop_size, dim))
        velocities = (
            w * velocities +
            c1 * r1 * (best_positions - particles) +
            c2 * r2 * (global_best_position - particles)
        )

        # Update position
        particles += velocities

        # Evaluate new fitness
        new_fitness = np.array([benchmark_func(p) for p in particles])

        # Update best postions and fitness
        improved_idx = np.where(new_fitness < best_fitness)
        best_positions[improved_idx] = particles[improved_idx]
        best_fitness[improved_idx] = new_fitness[improved_idx]
        if np.min(new_fitness) < global_best_fitness:
            global_best_position = particles[np.argmin(new_fitness)]
            global_best_fitness = np.min(new_fitness)

        fitness_log.append(global_best_fitness)

    return np.array(fitness_log), global_best_position
