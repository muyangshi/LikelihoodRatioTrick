# %% Imports
import torch
from sbi.inference import NRE_A
from sbi.utils import MultipleIndependent
import matplotlib.pyplot as plt

torch.manual_seed(123)
device = "cuda" # or cuda lol

# %% Prior and Simulator
prior = MultipleIndependent([
    torch.distributions.LogNormal(torch.tensor([1.0]), torch.Tensor([1.0])),
    torch.distributions.Uniform(torch.tensor([-0.5]), torch.tensor([1.0])),
])

def simulator(theta: torch.tensor) -> torch.tensor:
    #sigma is stored in the 0th column of theta, xi is stored in the first
    sigma, xi = theta[:, 0], theta[:, 1]
    u = torch.rand_like(sigma)
    return (sigma * (u ** (-xi) - 1)) / xi
# %% Training X and Theta

N = 1_000_000
theta_train = prior.sample((N,)) #(N, 2)
x_train = simulator(theta_train).unsqueeze(1)
# %% The Classifier

inference = NRE_A(prior = prior)
inference.append_simulations(theta_train, x_train)
classifier = inference.train()
# %% The Log-Likelihoods
sig_grid = torch.linspace(0, 5, 1000)
xi_grid = torch.linspace(-0.5, 1, 1000)
joint_grid = torch.cat((sig_grid.unsqueeze(1), xi_grid.unsqueeze(1)), dim = 1) #(N, 2)
x_o = torch.tensor([0.1])

ll_train = classifier(joint_grid, x_o.expand(torch.Size((1000, 1)))).squeeze(-1).detach().numpy()
ll_true = -torch.log(sig_grid) - ((1/xi_grid) + 1) * (torch.log(sig_grid + xi_grid * x_o.item()) - torch.log(sig_grid))
# %% The Plot
plt.scatter(ll_train, ll_true)
plt.title(f"True LL against Trained LL, observed {x_o.item()}")
plt.xlabel("True Log Likelihood")
plt.ylabel("Trained Log Likelihood")
plt.show()
# %% Saving the Model

torch.save(classifier, "C:/Users/chase/OneDrive/Documents/Colorado State/Summer Research 25/ll_classifier.pt")
# %%
