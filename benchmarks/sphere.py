import numpy as np
class sphere:
    
    def evaluate(position):
        """
        Sphere Function
        f(x) = sum(x_i^2)
        Global Minimum at x = 0, f(x) = 0
        """
        return np.sum(position ** 2)

    def getBounds():
        return (-5, 5)
    
    def getDimensions():
        return 30