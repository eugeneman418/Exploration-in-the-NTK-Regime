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
import os




class DetermineNet:
    def __init__(self, in_dim, hidden_dim, out_dim, determinism=True, xavier=False):
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, out_dim)
        )
        if xavier:
            self.net.apply(utils.initize_xavier)
        self.init_net = copy.deepcopy(self.net)
        self.hidden_dim = hidden_dim
        self.determinism = determinism

    def train(self, X, y, lr, steps, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        lr = lr / np.log10(self.hidden_dim) # normalize so we are in NTK regime
        steps = int(steps * np.log10(self.hidden_dim))

        self.net.train()

        self.net.to(device)
        X = X.to(device)
        y = y.to(device)

        self.init_net.train()
        self.init_net.to(device)
        init_y = self.init_net(X).detach()

        optimizer = optim.SGD(self.net.parameters(), lr=lr)

        criterion = nn.MSELoss()

        losses = []
        for i in range(steps):
            # whole training set at once to avoid stochastic GD
            optimizer.zero_grad()
            pred = self.net(X)
            loss = criterion(pred, init_y + y) if self.determinism else criterion(pred, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().item())
        self.net.eval()
        return losses

    def inference(self, X, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.net.eval()
        self.init_net.eval()
        with torch.no_grad():
            self.net.to(device)
            X = X.to(device)
            pred = self.net(X)

            if self.determinism:
                self.init_net.to(device)
                return pred - self.init_net(X)
            else:
                return pred

def ensemble_mean(ensemble, X, device):
    preds = torch.stack([net.inference(X, device) for net in ensemble], dim=0)
    return preds.mean(dim=0)


def ensemble_variance(ensemble, X, device):
    preds = torch.stack([net.inference(X, device) for net in ensemble], dim=0)
    return preds.var(dim=0, unbiased=True)


hidden_dims = [int(2**i) for i in range(4,12)]
ensemble_size = 30
lr = 1e-3
steps = 100

# log_dir = "log/determine_net"
X_train, X_test, y_train, _ = utils.load_data_to_tensor("data/yacht_hydro.csv", "Rr", random_seed=42)

# out of distribution test
X_test = 2*(torch.rand_like(X_test)-0.5)*500

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

graph_dir = "graphs"
os.makedirs(graph_dir, exist_ok=True)

for hidden_dim in hidden_dims:
    print(f"\n==============================")
    print(f"Hidden width: {hidden_dim}")
    print(f"==============================")

    results = {}
    for determinism in [True, False]:
        det_str = "Deterministic" if determinism else "Non-Deterministic"
        print(f"\n--- Training {det_str} Ensemble ---")

        torch.manual_seed(42)
        np.random.seed(42)

        losses = []
        ensemble = []

        for i in range(ensemble_size):
            print(f"Training ensemble {i + 1}/{ensemble_size}")
            net = DetermineNet(6, hidden_dim, 1, determinism=determinism, xavier=True)
            losses.append(net.train(X_train, y_train, lr, steps, device))
            ensemble.append(net)

        losses_array = np.array(losses)
        mean_loss = np.mean(losses_array, axis=0)
        loss_var = np.var(losses_array, axis=0)

        results[det_str] = {
            "ensemble": ensemble,
            "mean_loss": mean_loss,
            "loss_var": loss_var
        }

        # Plot training losses
        plt.figure(figsize=(10, 6))
        plt.plot(mean_loss, label=f'{det_str} Mean Loss', linewidth=2)
        plt.fill_between(
            range(len(mean_loss)),
            mean_loss - np.sqrt(loss_var),
            mean_loss + np.sqrt(loss_var),
            alpha=0.3,
            label=f'{det_str} Loss Std'
        )
        plt.title(f"{det_str} Training Loss (Hidden Dim {hidden_dim})")
        plt.xlabel("Training Step")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{graph_dir}/{det_str}_training_loss_hidden{hidden_dim}.png")
        plt.close()

    # ======= Ensemble Prediction Comparison =======
    print("\nComputing ensemble predictions...")

    det_mean = ensemble_mean(results["Deterministic"]["ensemble"], X_test, device).cpu().squeeze()
    det_var = ensemble_variance(results["Deterministic"]["ensemble"], X_test, device).cpu().squeeze()

    nondet_mean = ensemble_mean(results["Non-Deterministic"]["ensemble"], X_test, device).cpu().squeeze()
    nondet_var = ensemble_variance(results["Non-Deterministic"]["ensemble"], X_test, device).cpu().squeeze()

    # --- Plot mean comparison ---
    plt.figure(figsize=(10, 6))
    plt.plot(det_mean, label="Deterministic Ensemble Mean", marker='o')
    plt.plot(nondet_mean, label="Non-Deterministic Ensemble Mean", marker='x')
    plt.title(f"Ensemble Mean Predictions Comparison (Hidden Dim {hidden_dim})")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Mean Prediction")
    plt.legend()
    # Mean comparison plot
    plt.tight_layout()
    plt.savefig(f"{graph_dir}/ensemble_mean_comparison_hidden{hidden_dim}.png")
    plt.close()

    # --- Plot variance comparison ---
    plt.figure(figsize=(10, 6))
    plt.plot(det_var, label='Deterministic Ensemble Variance', marker='o')
    plt.plot(nondet_var, label='Non-Deterministic Ensemble Variance', marker='x')
    plt.title(f"Ensemble Variance Comparison (Hidden Dim {hidden_dim})")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Prediction Variance")
    plt.legend()
    # Variance comparison plot
    plt.tight_layout()
    plt.savefig(f"{graph_dir}/ensemble_variance_comparison_hidden{hidden_dim}.png")
    plt.close()

