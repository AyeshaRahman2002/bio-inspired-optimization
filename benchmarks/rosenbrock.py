import numpy as np

class rosenbrock:
    
    def evaluate(self, position):
        """
        Rosenbrock Function (a.k.a. Banana Function)
        Global minimum: f(x) = 0 at x = [1, 1, ..., 1]
        Domain: Typically [-5, 10]
        Characteristics: Non-convex, narrow curved valley.
        """
        return np.sum(
            100.0 * (position[1:] - position[:-1]**2.0)**2.0 +
            (1 - position[:-1])**2.0
        )

    def getBounds(self):
        return (-10, 10)

    def getDimensions(self):
        return 30  # Default dimensionality
