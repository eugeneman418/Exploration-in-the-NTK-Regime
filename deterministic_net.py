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


class DetermineNet:
    def __init__(self, in_dim, hidden_dim, out_dim, xavier=False):
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, out_dim)
        )
        if xavier:
            self.net.apply(utils.initize_xavier)
        self.init_net = copy.deepcopy(self.net)
        self.hidden_dim = hidden_dim

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
            loss = criterion(pred, init_y - y)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().item())
        self.net.eval()
        return losses

    def inference(self, X, deterministic=True, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.net.eval()
        self.init_net.eval()
        with torch.no_grad():
            self.net.to(device)
            X = X.to(device)
            pred = self.net(X)

            if deterministic:
                self.init_net.to(device)
                return pred - self.init_net(X)
            else:
                return pred


hidden_dims = [int(2**i) for i in range(4,16)]
ensemble_size = 30
lr = 1e-3
steps = 100

log_dir = "log/determine_net"
X_train, X_test, y_train, _ = utils.load_data_to_tensor("data/yacht_hydro.csv", "Rr", random_seed=42)

# out of distribution test
X_test = 2*(torch.rand_like(X_test)-0.5)*500

for hidden_dim in hidden_dims:
    # reset seed
    print(f"Starting ensemble training with hidden width {hidden_dim}")
    torch.manual_seed(42)
    np.random.seed(42)

    losses = []
    nondetermine_out = []
    determine_out = []

    for i in range(ensemble_size):
        print(f"Training ensemble {i+1}/{ensemble_size}")
        net = DetermineNet(6, hidden_dim,1, xavier=True) # with LeCunn I think numerical issues happens for smaller hidden weight already 
        losses.append(net.train(X_train, y_train, lr, steps))
        nondetermine_out.append(net.inference(X_test,False).cpu())
        determine_out.append(net.inference(X_test, True).cpu())

    losses_array = np.array(losses)  # Convert list of losses into a numpy array
    mean_loss = np.mean(losses_array, axis=0)  # Compute mean loss across the ensemble
    loss_variance = np.var(losses_array, axis=0)  # Compute variance of loss across the ensemble

    plt.figure(figsize=(10, 6))
    plt.plot(mean_loss, color='red', linewidth=2, label='Mean loss')
    plt.fill_between(range(len(mean_loss)), mean_loss - np.sqrt(loss_variance), mean_loss + np.sqrt(loss_variance), color='orange', alpha=0.3, label='Loss variance')
    plt.title(f"Training Losses with Variance for Hidden Dim {hidden_dim}")
    plt.xlabel("Step")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.show()

    # 2. Compute variance at test points
    determine_out_tensor = torch.stack(determine_out)  # shape: [ensemble_size, n_test, 1]
    nondet_out_tensor = torch.stack(nondetermine_out)

    determine_var = determine_out_tensor.var(dim=0)

    nondet_var = nondet_out_tensor.var(dim=0)

    # 3. Visualize variance at test points
    plt.figure(figsize=(10, 6))
    plt.plot(determine_var.squeeze(), label='Deterministic ensemble variance', marker='o')
    plt.plot(nondet_var.squeeze(), label='Non-deterministic ensemble variance', marker='x')
    plt.title(f"Ensemble Variance at Test Points (Hidden Dim {hidden_dim})")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Variance")
    plt.legend()
    plt.show()

    losses_df = pd.DataFrame(losses).transpose()  # Transpose so that each column is an ensemble member
    losses_df.to_csv(f"{log_dir}/hidden_{hidden_dim}_losses.csv", index=False)

    # Save predictions from the ensemble
    determine_out_tensor = torch.cat(determine_out, dim=0).cpu().numpy()  # shape: [ensemble_size * n_test, 1]
    nondet_out_tensor = torch.cat(nondetermine_out, dim=0).cpu().numpy()

    # Save deterministic and non-deterministic outputs
    determine_out_df = pd.DataFrame(determine_out_tensor)
    determine_out_df.to_csv(f"{log_dir}/hidden_{hidden_dim}_det_out.csv", index=False)

    nondet_out_df = pd.DataFrame(nondet_out_tensor)
    nondet_out_df.to_csv(f"{log_dir}/hidden_{hidden_dim}_nondet_out.csv", index=False)