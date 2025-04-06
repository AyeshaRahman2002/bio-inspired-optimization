import numpy as np

def evaluate(position):
    """
    Sphere Function
    f(x) = sum(x_i^2)
    Global Minimum at x = 0, f(x) = 0
    """
    return np.sum(position ** 2)
