from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score

class BreastCancerNN:
    def __init__(self):
        data = load_breast_cancer()
        X, y = data.data, data.target
        X = StandardScaler().fit_transform(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.maxDim = (self.input_size + 1) * 20 + (20 + 1) * 20 + (20 + 1) * self.output_size # weights + biases

    def decode_architecture(self, params):
        H1 = int(np.clip(params[0], 2, 20))
        H2 = int(np.clip(params[1], 0, 20))  # 0 = no second layer
        return H1, H2
    
    def getParams(self,H1,H2):
        if H2 > 0:
            return (self.input_size + 1) * H1 + (H1 + 1) * H2 + (H2 + 1) * self.output_size  # weights + biases
        else:
            return (self.input_size + 1) * H1 + (H1 + 1) * self.output_size  # weights + biases
    
    def evaluate(self, genome):

        architecture = genome[:2]
        H1,H2 = self.decode_architecture(architecture)
        parameters = self.getParams(H1,H2)
        if len(genome) < parameters:
            return 1.0  ## para
        weights = genome[2:2 + parameters]

        try:
            pointer = 0
            w1 = weights[pointer:pointer + self.input_size * H1].reshape(self.input_size, H1)
            pointer += self.input_size * H1
            b1 = weights[pointer:pointer + H1]
            pointer += H1
            
            hidden1 = np.dot(self.X_train, w1) + b1
            hidden_output1 = 1 / (1 + np.exp(-np.clip(hidden1, -100, 100)))
            
            if H2 > 0:
                w2 = weights[pointer:pointer + H1 * H2].reshape(H1, H2)
                pointer += H1 * H2
                b2 = weights[pointer:pointer + H2]
                pointer += H2
                hidden2 = np.dot(hidden_output1, w2) + b2
                hidden_output2 = 1 / (1 + np.exp(-np.clip(hidden2, -100, 100)))
                
                w_out = weights[pointer:pointer + H2]
                pointer += H2
                b_out = weights[pointer]
                output = np.dot(hidden_output2, w_out) + b_out
            else:
                w_out = weights[pointer:pointer + H1]
                pointer += H1
                b_out = weights[pointer]
                output = np.dot(hidden_output1, w_out) + b_out
                
            output = np.clip(output, -100, 100)
            predictions = 1 / (1 + np.exp(-output)) > 0.5

            acc = accuracy_score(self.y_train, predictions)
            return 1 - acc  # minimize error
        except Exception:
            return 1  # Return bad fitness on error

    def getBounds(self):
        arch_bounds = [(2, 20), (0, 20)]
        weight_bounds = [(-5, 5)] * self.maxDim
        lower = [b[0] for b in arch_bounds + weight_bounds]
        upper = [b[1] for b in arch_bounds + weight_bounds]
        return lower, upper

    def getDimensions(self):
        return (self.maxDim + 2)

    def test_model(self, genome):
        architecture = genome[:2]
        H1,H2 = self.decode_architecture(architecture)
        parameters = self.getParams(H1,H2)
        weights = genome[2:2 + parameters]

        pointer = 0
        w1 = weights[pointer:pointer + self.input_size * H1].reshape(self.input_size, H1)
        pointer += self.input_size * H1
        b1 = weights[pointer:pointer + H1]
        pointer += H1
        
        hidden1 = np.dot(self.X_test, w1) + b1
        hidden_output1 = 1 / (1 + np.exp(-np.clip(hidden1, -100, 100)))
        
        if H2 > 0:
            w2 = weights[pointer:pointer + H1 * H2].reshape(H1, H2)
            pointer += H1 * H2
            b2 = weights[pointer:pointer + H2]
            pointer += H2
            hidden2 = np.dot(hidden_output1, w2) + b2
            hidden_output2 = 1 / (1 + np.exp(-np.clip(hidden2, -100, 100)))
            
            w_out = weights[pointer:pointer + H2]
            pointer += H2
            b_out = weights[pointer]
            output = np.dot(hidden_output2, w_out) + b_out
        else:
            w_out = weights[pointer:pointer + H1]
            pointer += H1
            b_out = weights[pointer]
            output = np.dot(hidden_output1, w_out) + b_out
                
        output = np.clip(output, -100, 100)
        predictions = 1 / (1 + np.exp(-output)) > 0.5
        acc = accuracy_score(self.y_test, predictions)
        print(f"Test Accuracy on Breast Cancer: {acc:.4f}")
