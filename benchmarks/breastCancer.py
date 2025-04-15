import numpy as np
import sklearn.datasets as skdb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class BreastCancer:
    def __init__(self):
        
        data = skdb.load_breast_cancer()
        x, y = data.data, data.target

        # Normalize features
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

        # Train/test split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )

        self.dimensions = x.shape[1] + 1 # Weights and bias optimization
        self.bounds = (-5, 5)  # can be changed 

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def evaluate(self, weights):
        """
        Evaluate logistic regression accuracy with given weights.
        """
        z = np.dot(self.x_train, weights[:-1]) + weights[-1]
        y_pred = self.sigmoid(z) > 0.5
        accuracy = accuracy_score(self.y_train, y_pred)
        return 1 - accuracy

    def getBounds(self):
        return self.bounds
    def getDimensions(self):
        return self.dimensions
    
    def test_model(self, weights):
        z = np.dot(self.x_test, weights[:-1]) + weights[-1]
        y_pred = self.sigmoid(z) > 0.5
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        return accuracy