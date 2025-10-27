import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
import utils

class PriorEnsemble:
    def __init__(self, in_dim, hidden_dim, out_dim=1, ensemble_size=30, xavier=False):
        self.nets = [
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(), # choice of ReLU is deliberate as it is an unbounded activation
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim)
            )
            for _ in range(ensemble_size)
        ]
        if xavier:
            for net in self.nets:
                net.apply(utils.initize_xavier)

    def expectation(self, x):
        predictions = torch.stack([net(x) for net in self.nets], dim=0)  # shape: [ensemble_size, batch_size, 1]
        # Compute the mean across the ensemble
        m = torch.mean(predictions, dim=0)  # variance along the ensemble dimension
        return m

    def variance(self, x):
        predictions = torch.stack([net(x) for net in self.nets], dim=0)  # shape: [ensemble_size, batch_size, 1]
        # Compute the variance across the ensemble
        v = torch.var(predictions, unbiased=True, dim=0)  # variance along the ensemble dimension
        return v

def generate_uniform(n, lower, upper):
    x = torch.rand((n,2))
    x = (upper - lower) * x + lower
    return x

hidden_dims = [int(2**i) for i in range(12,13)]


lower, upper = -100, 100
grid_size = 10  # You can adjust this for higher resolution

# Visualization for each hidden dimension
for hidden_dim in hidden_dims:
    ensemble = PriorEnsemble(2, hidden_dim, xavier=False) # variance of xavier decreases with hidden width

    # Generate a grid of points in the domain [lower, upper]^2

    x_values = np.linspace(lower, upper, grid_size)
    y_values = np.linspace(lower, upper, grid_size)
    X, Y = np.meshgrid(x_values, y_values)
    grid_points = torch.tensor(np.vstack([X.ravel(), Y.ravel()]).T, dtype=torch.float32)

    # Compute variance for each grid point
    variance_values = ensemble.variance(grid_points).detach().numpy().reshape((grid_size, grid_size))

    # Plotting the variance as a heatmap
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, variance_values, 20, cmap='viridis')
    plt.colorbar(label="Variance")
    plt.title(f"Ensemble variance for a 3 layer ReLU net of hidden dim {hidden_dim}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()