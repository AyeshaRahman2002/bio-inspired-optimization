# benchmarks/LSTMBreastCancer.py

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class LSTMBreastCancer:
    def __init__(self):
        data = load_breast_cancer()
        X, y = data.data, data.target
        X = StandardScaler().fit_transform(X)

        # Reshape to (samples, timesteps, features_per_step)
        self.timesteps = 5
        self.input_dim = 6
        X = X.reshape((-1, self.timesteps, self.input_dim))

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.hidden_size = 4
        self.output_size = 1

        # LSTM weights:
        # Input Gate: W_i (input_dim x hidden), U_i (hidden x hidden), b_i (hidden)
        # Forget Gate: W_f, U_f, b_f
        # Output Gate: W_o, U_o, b_o
        # Cell Gate: W_c, U_c, b_c
        # Output layer: (hidden -> output), bias

        self.dim = 4 * ((self.input_dim + self.hidden_size) * self.hidden_size + self.hidden_size) + \
                   self.hidden_size * self.output_size + self.output_size

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -100, 100)))

    def tanh(self, x):
        return np.tanh(np.clip(x, -100, 100))

    def evaluate(self, weights):
        try:
            idx = 0
            H, I, O = self.hidden_size, self.input_dim, self.output_size

            def extract():
                nonlocal idx
                end = idx + H * I
                W = weights[idx:end].reshape(I, H)
                idx = end

                end = idx + H * H
                U = weights[idx:end].reshape(H, H)
                idx = end

                b = weights[idx:idx + H]
                idx += H
                return W, U, b

            W_i, U_i, b_i = extract()
            W_f, U_f, b_f = extract()
            W_o, U_o, b_o = extract()
            W_c, U_c, b_c = extract()

            W_y = weights[idx:idx + H * O].reshape(H, O)
            idx += H * O
            b_y = weights[idx:idx + O]

            X = self.X_train
            batch_size = X.shape[0]
            h = np.zeros((batch_size, H))
            c = np.zeros((batch_size, H))

            for t in range(self.timesteps):
                x_t = X[:, t, :]
                i = self.sigmoid(np.dot(x_t, W_i) + np.dot(h, U_i) + b_i)
                f = self.sigmoid(np.dot(x_t, W_f) + np.dot(h, U_f) + b_f)
                o = self.sigmoid(np.dot(x_t, W_o) + np.dot(h, U_o) + b_o)
                g = self.tanh(np.dot(x_t, W_c) + np.dot(h, U_c) + b_c)
                c = f * c + i * g
                h = o * self.tanh(c)

            y = np.dot(h, W_y) + b_y
            pred = self.sigmoid(y).flatten() > 0.5
            acc = accuracy_score(self.y_train, pred)
            return 1 - acc
        except:
            return 1

    def test_model(self, weights):
        idx = 0
        H, I, O = self.hidden_size, self.input_dim, self.output_size

        def extract():
            nonlocal idx
            end = idx + H * I
            W = weights[idx:end].reshape(I, H)
            idx = end

            end = idx + H * H
            U = weights[idx:end].reshape(H, H)
            idx = end

            b = weights[idx:idx + H]
            idx += H
            return W, U, b

        W_i, U_i, b_i = extract()
        W_f, U_f, b_f = extract()
        W_o, U_o, b_o = extract()
        W_c, U_c, b_c = extract()

        W_y = weights[idx:idx + H * O].reshape(H, O)
        idx += H * O
        b_y = weights[idx:idx + O]

        X = self.X_test
        batch_size = X.shape[0]
        h = np.zeros((batch_size, H))
        c = np.zeros((batch_size, H))

        for t in range(self.timesteps):
            x_t = X[:, t, :]
            i = self.sigmoid(np.dot(x_t, W_i) + np.dot(h, U_i) + b_i)
            f = self.sigmoid(np.dot(x_t, W_f) + np.dot(h, U_f) + b_f)
            o = self.sigmoid(np.dot(x_t, W_o) + np.dot(h, U_o) + b_o)
            g = self.tanh(np.dot(x_t, W_c) + np.dot(h, U_c) + b_c)
            c = f * c + i * g
            h = o * self.tanh(c)

        y = np.dot(h, W_y) + b_y
        pred = self.sigmoid(y).flatten() > 0.5
        acc = accuracy_score(self.y_test, pred)
        print(f"Test Accuracy (LSTM): {acc:.4f}")

    def getBounds(self):
        return [-1] * self.dim, [1] * self.dim

    def getDimensions(self):
        return self.dim
