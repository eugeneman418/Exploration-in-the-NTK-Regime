import copy
import torch
import torch.nn as nn
import numpy as np
import torch.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
from utils import ensemble_mean, ensemble_variance
import os
class DeepEnsemble:
    def __init__(self, in_dim, hidden_dim, out_dim, ensemble_size):
        self.hidden_dim = hidden_dim
        self.nets = [nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, out_dim)
        ) for _ in range(ensemble_size)]

    def train(self, X, y, lr, steps, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        lr = lr / np.log10(self.hidden_dim)  # normalize so we are in NTK regime
        steps = int(steps * np.log10(self.hidden_dim))
        ensemble_losses = []
        for i, net in enumerate(self.nets):
            print(f"training net {i+1}/{len(self.nets)}")
            net.train()

            net.to(device)

            X = X.to(device)

            y = y.to(device)

            optimizer = optim.SGD(net.parameters(), lr=lr)

            losses = []
            for i in range(steps):
                # whole training set at once to avoid stochastic GD
                optimizer.zero_grad()
                pred = net(X)
                loss = (pred - y).pow(2).mean()
                loss.backward()
                optimizer.step()
                losses.append(loss.detach().cpu().item())
            net.eval()

            ensemble_losses.append(losses)
        return ensemble_losses

class Rnd(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Rnd, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.init_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.hidden_dim = hidden_dim

    def fit(self, X, lr, steps, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        lr = lr / np.log10(self.hidden_dim) # normalize so we are in NTK regime
        steps = int(steps * np.log10(self.hidden_dim))

        self.net.train()

        self.net.to(device)
        X = X.to(device)

        self.init_net.train()
        self.init_net.to(device)
        y = self.init_net(X).detach()

        optimizer = optim.SGD(self.net.parameters(), lr=lr)


        losses = []
        for i in range(steps):
            # whole training set at once to avoid stochastic GD
            optimizer.zero_grad()
            pred = self.net(X)
            loss = (pred - y).pow(2).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().item())
        self.net.eval()
        self.init_net.eval()
        return losses

    def forward(self, X):
        return 0.5*(self.net(X) - self.init_net(X).detach())**2

class RndEnsemble:
    def __init__(self, in_dim, hidden_dim, out_dim, ensemble_size):
        self.hidden_dim = hidden_dim
        self.nets = [Rnd(in_dim, hidden_dim, out_dim) for _ in range(ensemble_size)]
    def train(self, X, lr, steps, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        ensemble_losses = []
        for i, net in enumerate(self.nets):
            print(f"training net {i + 1}/{len(self.nets)}")
            losses = net.fit(X, lr, steps, device)
            ensemble_losses.append(losses)
        return ensemble_losses

hidden_dims = [8192]
ensemble_size = 30
lr = 1e-3
steps = 100

X_train, X_test, y_train, _ = utils.load_data_to_tensor("data/yacht_hydro.csv", "Rr", random_seed=42)

# out of distribution test
X_test = 2*(torch.rand_like(X_test)-0.5)*500

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for hidden_dim in hidden_dims:
    print(f"\n==============================")
    print(f"Hidden width: {hidden_dim}")
    print(f"==============================")
    print(f"Deep Ensemble")
    print(f"==============================")
    torch.manual_seed(42)
    np.random.seed(42)

    deep_ensemble = DeepEnsemble(6, hidden_dim, 1, ensemble_size)
    ensemble_losses = deep_ensemble.train(X_train, y_train, lr, steps)
    ensemble_mean_loss = np.mean(ensemble_losses, axis=0)
    ensemble_std_loss = np.std(ensemble_mean_loss, axis=0, ddof=1)
    plt.figure(figsize=(10, 6))
    plt.plot(ensemble_mean_loss, label=f'Deep Ensemble Mean Loss', linewidth=2)
    plt.fill_between(
        range(len(ensemble_mean_loss)),
        ensemble_mean_loss - ensemble_std_loss,
        ensemble_mean_loss + ensemble_std_loss,
        alpha=0.3,
        label=f'Deep Ensemble Loss Std'
    )

    plt.title(f"Deep Ensemble Training Loss (Hidden Dim {hidden_dim})")
    plt.xlabel("Training Step")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.show()

    print(f"==============================")
    print(f"RND")
    torch.manual_seed(42)
    np.random.seed(42)

    rnd_ensemble = RndEnsemble(6, hidden_dim, 1, ensemble_size)
    rnd_losses = rnd_ensemble.train(X_train,lr,steps)
    rnd_mean_loss = np.mean(rnd_losses, axis=0)
    rnd_std_loss = np.std(rnd_mean_loss, axis=0, ddof=1)
    plt.figure(figsize=(10, 6))
    plt.plot(rnd_mean_loss, label=f'RND Mean Loss', linewidth=2)
    plt.fill_between(
        range(len(rnd_mean_loss)),
        rnd_mean_loss - rnd_std_loss,
        rnd_mean_loss + rnd_std_loss,
        alpha=0.3,
        label=f'RND Loss Std'
    )
    plt.title(f"RND Training Loss (Hidden Dim {hidden_dim})")
    plt.xlabel("Training Step")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.show()


    ensemble_test_variance = utils.ensemble_variance(deep_ensemble.nets, X_test, device).detach().cpu()
    rnd_test_mean = utils.ensemble_mean(rnd_ensemble.nets, X_test, device).detach().cpu()



    plt.plot(ensemble_test_variance, label="Deep Ensemble Variance")
    plt.plot(rnd_test_mean, label="RND Prediction")
    plt.legend()
    plt.show()
    print((ensemble_test_variance / rnd_test_mean).mean())