import numpy as np

class ListSortCoEvo:
    def __init__(self, list_length=10, n_lists=5):
        self.list_length = list_length
        self.n_lists = n_lists
        self.dim = list_length * list_length  # genome is swap matrix
        self.list_population = self.init_test_lists(n_lists)

    def init_test_lists(self, n):
        return [np.random.permutation(self.list_length) for _ in range(n)]

    def evaluate_sorter(self, sorter_genome, test_lists=None):
        """Evaluate a sorting genome against all test lists."""
        if test_lists is None:
            test_lists = self.list_population

        swap_matrix = sorter_genome.reshape(self.list_length, self.list_length)
        swap_matrix = (swap_matrix > 0).astype(int)

        total_error = 0
        for test_list in test_lists:
            lst = test_list.copy()
            for i in range(self.list_length):
                for j in range(self.list_length - 1):
                    if swap_matrix[lst[j]][lst[j + 1]]:
                        lst[j], lst[j + 1] = lst[j + 1], lst[j]
            total_error += self.count_inversions(lst)

        return total_error / len(test_lists)

    def evaluate_list(self, test_list, sorter_genomes):
        """Evaluate how hard a test list is for a population of sorters."""
        total_inversions = 0
        for genome in sorter_genomes:
            total_inversions += self.evaluate_sorter(genome, [test_list])
        return total_inversions / len(sorter_genomes)

    def count_inversions(self, arr):
        inv = 0
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                if arr[i] > arr[j]:
                    inv += 1
        return inv

    def getBounds(self):
        return [-1] * self.dim, [1] * self.dim

    def getDimensions(self):
        return self.dim

    def evolve_lists(self, sorter_population, mutation_rate=0.2):
        """Evolve the test list population."""
        fitness_scores = [self.evaluate_list(lst, sorter_population) for lst in self.list_population]
        best_indices = np.argsort(fitness_scores)[-len(self.list_population)//2:]  # top half
        selected = [self.list_population[i].copy() for i in best_indices]

        # Generate new population
        new_lists = []
        for lst in selected:
            if np.random.rand() < mutation_rate:
                i, j = np.random.choice(self.list_length, 2, replace=False)
                lst[i], lst[j] = lst[j], lst[i]
            new_lists.append(lst)

        self.list_population = selected + new_lists
