# %% Imports
import torch
from sbi.inference import NRE_A
import matplotlib.pyplot as plt
from scipy.stats import genpareto

torch.manual_seed(123)
device = "cpu" # or cuda lol

# %% Prior and Simulator
prior = torch.distributions.LogNormal(torch.tensor([1.0]), torch.tensor([1.0]))

def simulator(theta: torch.tensor) -> torch.tensor:
    #here theta is sigma, xi is fixed at 0.5
    xi = 0.5
    u = torch.rand_like(theta)
    return (theta * (u ** (-xi) - 1)) / xi
# %% Training X and Theta

N = 10_000
theta_train = prior.sample((N,)) #(N, 1)
x_train = simulator(theta_train)
# %% The Classifier

inference = NRE_A(prior = prior)
inference.append_simulations(theta_train, x_train)
classifier = inference.train()
# %% The Log-Likelihoods
sig_grid = torch.linspace(0.1, 10, 100)
x_o = torch.tensor([1.0])

ll_train = classifier(sig_grid.unsqueeze(1), x_o.expand(torch.Size((100, 1)))).squeeze(-1).detach().numpy()
ll_true = -torch.log(sig_grid) - ((1/0.5) + 1) * (torch.log(sig_grid + 0.5 * x_o.item()) - torch.log(sig_grid))
# %% The Plots

plt.scatter(ll_train, ll_true)
plt.xlabel("True Log Likelihood")
plt.ylabel("Trained Log Likelihood")
plt.show()
# %%
