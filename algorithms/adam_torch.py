import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def run_adam_torch(benchmark_func=None, dim=None, pop_size=None, iterations=100, bounds=None):
    # Load dataset
    data = load_breast_cancer()
    X = StandardScaler().fit_transform(data.data)
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    input_size = X_train.shape[1]
    hidden_size = 5
    output_size = 1

    # Define model
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.Sigmoid(),
        nn.Linear(hidden_size, output_size),
        nn.Sigmoid()
    )

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    fitness_log = []

    for epoch in range(iterations):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        fitness_log.append(loss.item())

    # Evaluate on test
    with torch.no_grad():
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
        preds = model(X_test)
        predicted = (preds > 0.5).float()
        acc = accuracy_score(y_test.numpy(), predicted.numpy())
        print(f"Test Accuracy (PyTorch Adam): {acc:.4f}")

    return np.array(fitness_log), model
