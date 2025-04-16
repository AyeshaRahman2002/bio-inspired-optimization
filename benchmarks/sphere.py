import numpy as np

class sphere:
    def evaluate(self, position):
        """
        Sphere Function
        f(x) = sum(x_i^2)
        Global Minimum at x = 0, f(x) = 0
        """
        return np.sum(position ** 2)

    def getBounds(self):
        """
        Returns lower and upper bounds for each dimension.
        Typically used in optimization problems.
        """
        return (-5, 5)

    def getDimensions(self):
        """
        Returns the dimensionality of the problem.
        Default is 30 dimensions.
        """
        return 30
