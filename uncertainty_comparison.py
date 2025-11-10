import copy
import torch
import torch.nn as nn
import numpy as np
import torch.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import utils

torch.manual_seed(42)
np.random.seed(42)

class Net(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, deterministic=False):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.init_net = copy.deepcopy(self.layers)
        self.deterministic = deterministic
        self.hidden_dim = hidden_dim

    def fit(self, X, y, lr, steps, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        lr = lr / np.log10(self.hidden_dim)  # normalize so we are in NTK regime
        steps = int(steps * np.log10(self.hidden_dim))
        losses = []

        self.layers.train()
        self.to(device)
        X = X.to(device)
        y = y.to(device)
        optimizer = optim.SGD(self.layers.parameters(), lr=lr)

        for i in range(steps):
            # whole training set at once to avoid stochastic GD
            optimizer.zero_grad()
            pred = self.forward(X)
            loss = (pred - y).pow(2).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().item())
        self.layers.eval()

        return losses
    def forward(self, x):
        if self.deterministic:
            return self.layers(x) - self.init_net(x)
        else:
            return self.layers(x)
class ParameterEnsemble:
    def __init__(self, ensemble_size, in_dim, out_dim, hidden_dim=1024):
        self.ensemble_size = ensemble_size
        self.nets = [Net(in_dim, hidden_dim, out_dim) for _ in range(ensemble_size)]

    def train(self, X, y, lr, steps, visualize_loss=True):
        ensemble_losses = []
        for net in self.nets:
            ensemble_losses.append(net.fit(X, y, lr, steps))

        if visualize_loss:
            ensemble_losses = np.array(ensemble_losses)
            utils.plot_ensemble_loss(ensemble_losses)

    def mean(self, X, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        return utils.ensemble_mean(self.nets, X, device)

    def variance(self, X, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        return utils.ensemble_variance(self.nets, X, device)

class BootstrapEnsemble:
    def __init__(self, ensemble_size, in_dim, out_dim, hidden_dim=1024, deterministic_net=True):
        if deterministic_net:
            net = Net(in_dim, hidden_dim, out_dim, deterministic_net)
            self.nets = [copy.deepcopy(net) for _ in range(ensemble_size)]
        else:
            self.nets = [Net(in_dim, hidden_dim, out_dim, deterministic_net) for _ in range(ensemble_size)]
    def train(self, X, y, lr, steps, visualize_loss=True):
        ensemble_losses = []
        n = X.size(0)
        for net in self.nets:
            resamples = torch.randint(0,n,(n,),device=X.device)
            X_resampled = X[resamples]
            y_resampled = y[resamples]
            ensemble_losses.append(net.fit(X_resampled, y_resampled, lr, steps))

        if visualize_loss:
            ensemble_losses = np.array(ensemble_losses)
            utils.plot_ensemble_loss(ensemble_losses)

    def mean(self, X, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        return utils.ensemble_mean(self.nets, X, device)

    def variance(self, X, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        return utils.ensemble_variance(self.nets, X, device)

def step_function(X,r=0.3):
    norms = X.max(dim=1)[0]
    return torch.where(norms > r, 1.0, 0.0)

sample_sizes = [int(10*i) for i in range(1,11)]
ensemble_size = 50

lr = 1e-3
steps = 100

# Define a grid over [0,1]^2
grid_points = 100
x1, x2 = torch.meshgrid(
    torch.linspace(0, 1, grid_points),
    torch.linspace(0, 1, grid_points),
    indexing='ij'
)
X_grid = torch.stack([x1.flatten(), x2.flatten()], dim=1)

for sample_size in sample_sizes:
    print(f"\n=== Sample size: {sample_size} ===")
    X = torch.rand((sample_size, 2))
    y = step_function(X).unsqueeze(1)  # ensure shape (n,1)

    print("Training Parameter Ensemble")
    param_ens = ParameterEnsemble(ensemble_size, in_dim=2, out_dim=1)
    param_ens.train(X, y, lr, steps, visualize_loss=False)
    var_param = param_ens.variance(X_grid)
    print("Training Bootstrap Ensemble")
    boot_ens = BootstrapEnsemble(ensemble_size, in_dim=2, out_dim=1, deterministic_net=False)
    boot_ens.train(X, y, lr, steps, visualize_loss=False)
    var_boot = boot_ens.variance(X_grid)
    print("Training Bootstrap Ensemble with Deterministic Nets")
    boot_det_ens = BootstrapEnsemble(ensemble_size, in_dim=2, out_dim=1, deterministic_net=True)
    boot_det_ens.train(X, y, lr, steps, visualize_loss=False)
    var_boot_det = boot_det_ens.variance(X_grid)

    # Reshape variances for plotting
    var_param = var_param.reshape(grid_points, grid_points).detach().cpu().numpy()
    var_boot = var_boot.reshape(grid_points, grid_points).detach().cpu().numpy()
    var_boot_det = var_boot_det.reshape(grid_points, grid_points).detach().cpu().numpy()

    # Plot comparison
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for ax, var, title in zip(
        axs,
        [var_param, var_boot, var_boot_det],
        ['Parameter Ensemble', 'Bootstrap Ensemble', 'Bootstrap (Deterministic Net)']
    ):
        im = ax.imshow(var, extent=[0,1,0,1], origin='lower', cmap='viridis')
        ax.set_title(f"{title}\nSample size: {sample_size}")
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()
