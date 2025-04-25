import numpy as np
class rosenbrock:
    
    def evaluate(position):
        """
        Rosenbrock Function (a.k.a. Banana Function)
        Global minimum: f(x) = 0 at x = [1, 1, ..., 1]
        Domain: Typically [-5, 10] for each dimension
        Characteristics: Non-convex, narrow curved valley, used to test optimization algorithms' ability to converge in difficult landscapes
        """
        # Compute sum over all dimensions except the first
        # The function has the form: sum(100 * (x[i+1] - x[i]^2)^2 + (1 - x[i])^2)
        return sum(
            100.0 * (position[1:] - position[:-1]**2.0)**2.0 +  # Measures how far x[i+1] is from x[i]^2
            (1 - position[:-1])**2.0                            # Penalizes distance from 1
        )
    def getBounds():
        return (-10, 10)
    def getDimensions():
        return 30