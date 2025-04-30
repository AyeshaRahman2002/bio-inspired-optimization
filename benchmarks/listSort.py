import numpy as np

class ListSort:
    def __init__(self, list_length=10):
        self.list_length = list_length
        self.dim = list_length * list_length  # genome represents swap decisions in a matrix
        self.test_lists = [np.random.permutation(list_length) for _ in range(5)]

    def evaluate(self, genome):
        # Interpret genome as a swap priority matrix
        swap_matrix = genome.reshape(self.list_length, self.list_length)
        swap_matrix = (swap_matrix > 0).astype(int)  # convert to binary: swap or not

        total_error = 0
        for test_list in self.test_lists:
            sorted_list = test_list.copy()

            # Apply sorting strategy (simple bubble-like sort with custom swap logic)
            for i in range(self.list_length):
                for j in range(self.list_length - 1):
                    if swap_matrix[sorted_list[j]][sorted_list[j + 1]]:
                        sorted_list[j], sorted_list[j + 1] = sorted_list[j + 1], sorted_list[j]

            total_error += self.count_inversions(sorted_list)

        # Lower inversion count is better
        return total_error / len(self.test_lists)

    def count_inversions(self, arr):
        # Count number of out-of-order pairs (Bubble Sort distance)
        inv_count = 0
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                if arr[i] > arr[j]:
                    inv_count += 1
        return inv_count

    def getBounds(self):
        # Values > 0 mean perform swap, ≤ 0 means skip swap
        return [-1] * self.dim, [1] * self.dim

    def getDimensions(self):
        return self.dim
