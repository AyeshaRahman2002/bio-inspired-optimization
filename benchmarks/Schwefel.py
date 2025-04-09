import numpy as np
class Schwefel:
    def evaluate(position):
        """
        Shwefel problem
        Global minimum: f(x) = 0 at x = [420.9687,..., 420.9687]
        Domain: Typically [-500,500]
        Characteristics: Complex large function with lots of local minima
        Useful for testing an algorithm's ability in tightly clumped local minima 
        
        Formula:
            f(x) = a * d - sum(x* sin(sqrt(abs(x))))
        where a = 418.9829
        """
        position = np.clip(position, -500, 500)
        a = 418.9829
        d = len(position)  # Dimensionality
        
        firstTerm = a*d
        sumTerm = np.sum(position * np.sin(np.sqrt(np.abs(position))))

        return firstTerm - sumTerm

    def getBounds():
        return (-500, 500)