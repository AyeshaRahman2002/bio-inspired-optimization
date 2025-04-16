from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score

class BreastCancer:
    def __init__(self):
        data = load_breast_cancer()
        X, y = data.data, data.target
        X = StandardScaler().fit_transform(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.input_size = X.shape[1]
        self.hidden_size = 5
        self.output_size = 1
        self.dim = (self.input_size + 1) * self.hidden_size + (self.hidden_size + 1) * self.output_size  # weights + biases

    def evaluate(self, weights):
        try:
            hidden_weights = weights[:self.input_size * self.hidden_size].reshape(self.input_size, self.hidden_size)
            hidden_bias = weights[self.input_size * self.hidden_size : self.input_size * self.hidden_size + self.hidden_size]
            output_weights = weights[self.input_size * self.hidden_size + self.hidden_size : -1]
            output_bias = weights[-1]

            # Forward pass with clipped sigmoid input to prevent overflow
            hidden_layer = np.dot(self.X_train, hidden_weights) + hidden_bias
            hidden_layer = 1 / (1 + np.exp(-np.clip(hidden_layer, -100, 100)))

            output = np.dot(hidden_layer, output_weights) + output_bias
            output = np.clip(output, -100, 100)
            predictions = 1 / (1 + np.exp(-output)) > 0.5

            acc = accuracy_score(self.y_train, predictions)
            return 1 - acc  # minimize error
        except Exception:
            return 1  # Return bad fitness on error

    def getBounds(self):
        return [-5] * self.dim, [5] * self.dim

    def getDimensions(self):
        return self.dim

    def test_model(self, weights):
        hidden_weights = weights[:self.input_size * self.hidden_size].reshape(self.input_size, self.hidden_size)
        hidden_bias = weights[self.input_size * self.hidden_size : self.input_size * self.hidden_size + self.hidden_size]
        output_weights = weights[self.input_size * self.hidden_size + self.hidden_size : -1]
        output_bias = weights[-1]

        hidden_layer = np.dot(self.X_test, hidden_weights) + hidden_bias
        hidden_layer = 1 / (1 + np.exp(-np.clip(hidden_layer, -100, 100)))

        output = np.dot(hidden_layer, output_weights) + output_bias
        output = np.clip(output, -100, 100)
        predictions = 1 / (1 + np.exp(-output)) > 0.5

        acc = accuracy_score(self.y_test, predictions)
        print(f"Test Accuracy on Breast Cancer: {acc:.4f}")
