import numpy as np

def run_adam(benchmark_func, dim, pop_size, iterations, bounds):
    lower, upper = bounds
    
    # Initialize a population of candidate solutions
    population = np.random.uniform(lower, upper, (pop_size, dim))
    fitness = np.array([benchmark_func(ind) for ind in population])

    # Select best initial individual as elite
    elite_idx = np.argmin(fitness)
    elite = population[elite_idx].copy()
    elite_fitness = fitness[elite_idx]
    fitness_log = []

    # Adam parameters
    alpha = 0.05
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    m = np.zeros((pop_size, dim))  # First moment
    v = np.zeros((pop_size, dim))  # Second moment

    for t in range(1, iterations + 1):
        for i in range(pop_size):
            grad = estimate_gradient(benchmark_func, population[i], h=1e-4)

            m[i] = beta1 * m[i] + (1 - beta1) * grad
            v[i] = beta2 * v[i] + (1 - beta2) * (grad ** 2)

            m_hat = m[i] / (1 - beta1 ** t)
            v_hat = v[i] / (1 - beta2 ** t)

            update = -alpha * m_hat / (np.sqrt(v_hat) + epsilon)
            population[i] += update
            population[i] = np.clip(population[i], lower, upper)

            fitness[i] = benchmark_func(population[i])

        best_idx = np.argmin(fitness)
        if fitness[best_idx] < elite_fitness:
            elite = population[best_idx].copy()
            elite_fitness = fitness[best_idx]

        fitness_log.append(elite_fitness)

    return np.array(fitness_log), elite

def estimate_gradient(func, x, h=1e-4):
    grad = np.zeros_like(x)
    fx = func(x)
    for i in range(len(x)):
        x_step = np.array(x)
        x_step[i] += h
        grad[i] = (func(x_step) - fx) / h
    return grad