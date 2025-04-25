import numpy as np

class Schwefel:
    def evaluate(self, position):
        """
        Schwefel Function
        Global minimum: f(x) = 0 at x = [420.9687,..., 420.9687]
        Domain: Typically [-500,500]
        Characteristics: Complex landscape with many local minima.
        """
        position = np.clip(position, -500, 500)
        a = 418.9829
        d = len(position)  # Dimensionality

        firstTerm = a * d
        sumTerm = np.sum(position * np.sin(np.sqrt(np.abs(position))))

        return firstTerm - sumTerm

    def getBounds(self):
        return (-500, 500)

    def getDimensions(self):
        return 30  # Default dimensionality