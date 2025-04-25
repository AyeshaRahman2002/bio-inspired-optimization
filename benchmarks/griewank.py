import numpy as np

class griewank:
    
    def evaluate(self, position):
        """
        Griewank Function
        Global minimum: f(x) = 0 at x = [0, 0, ..., 0]
        Domain: Typically [-600, 600]
        Characteristics: Many widespread local minima but all regularly spaced.
        Suitable for testing an algorithm's ability to escape local optima.
        
        Formula:
            f(x) = 1 + (1/4000) * sum(x_i^2) - prod(cos(x_i / sqrt(i)))
        """
        sum_term = np.sum(position**2) / 4000
        prod_term = np.prod(np.cos(position / np.sqrt(np.arange(1, len(position) + 1))))
        return sum_term - prod_term + 1

    def getBounds(self):
        return (-600, 600)

    def getDimensions(self):
        return 30  # Default dimensionality
