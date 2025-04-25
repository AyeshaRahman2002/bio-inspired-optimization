import numpy as np
class rastrigin:
    
    def evaluate(position):
        """
        Rastrigin Function
        f(x) = 10n + sum(x_i^2 - 10 * cos(2πx_i))
        Global Minimum at x = 0, f(x) = 0
        """
        n = len(position)
        return 10 * n + np.sum(position**2 - 10 * np.cos(2 * np.pi * position))
    def getBounds():
        return (-5.12, 5.12)
    def getDimensions():
        return 30