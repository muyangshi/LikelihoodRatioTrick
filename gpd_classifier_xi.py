# %% Imports
import torch
from sbi.inference import NRE_A
import matplotlib.pyplot as plt
from scipy.stats import genpareto

torch.manual_seed(123)
device = "cpu" # or cuda lol

# %% Prior and Simulator
prior = torch.distributions.Uniform(torch.tensor([-0.5]), torch.tensor([1.0]))

def simulator(theta: torch.tensor) -> torch.tensor:
    #here theta is xi, sigma is fixed at 1.0
    sigma = 1.0
    u = torch.rand_like(theta)
    return (sigma * (u ** (-theta) - 1)) / theta
# %% Training X and Theta

N = 100_000
theta_train = prior.sample((N,)) #(N, 1)
x_train = simulator(theta_train)
# %% The Classifier

inference = NRE_A(prior = prior)
inference.append_simulations(theta_train, x_train)
classifier = inference.train()
# %% The Log-Likelihoods
xi_grid = torch.linspace(-0.5, 1, 100)
x_o = torch.tensor([5.0])

ll_train = classifier(xi_grid.unsqueeze(1), x_o.expand(torch.Size((100, 1)))).squeeze(-1).detach().numpy()
ll_true = -((1/xi_grid) + 1) * torch.log(1 + xi_grid * x_o.item())
# %% The Plots

plt.scatter(ll_train, ll_true)
plt.xlabel("True Log Likelihood")
plt.ylabel("Trained Log Likelihood")
plt.show()
# %%
