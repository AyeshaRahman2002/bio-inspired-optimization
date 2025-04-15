import numpy as np

class ackley:
    
    def evaluate(position):
        """
        Ackley Function
        Global minimum: f(x) = 0 at x = [0, 0, ..., 0]
        Domain: Typically [-32.768, 32.768]
        Characteristics: Many local minima, with a large nearly flat outer region
        Useful for testing an algorithm's ability to converge to the global minimum despite local traps.
        
        Formula:
            f(x) = -a * exp(-b * sqrt(1/d * sum(x_i^2)))
                - exp(1/d * sum(cos(c * x_i)))
                + a + exp(1)
        Common constants: a=20, b=0.2, c=2π
        """

        a = 20         # Amplitude of the exponential decay
        b = 0.2        # Controls the rate of decay in the first exponential
        c = 2 * np.pi  # Frequency used in the cosine component
        d = len(position)  # Dimensionality

        # First term: exponential decay based on the distance from origin
        sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(position**2) / d))

        # Second term: mean of cosine values, to create oscillations (local minima)
        cos_term = -np.exp(np.sum(np.cos(c * position)) / d)

        # Final Ackley function value
        return sum_sq_term + cos_term + a + np.exp(1)
    def getBounds():
        return (-32.768, 32.768)
    
    def getDimensions():
        return 30
    