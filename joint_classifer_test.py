# %% Imports
import numpy as np
import torch
from sbi.neural_nets.ratio_estimators import RatioEstimator
import pandas as pd
import matplotlib as plt
from scipy.stats import genpareto, norm, uniform

torch.manual_seed(1234)

# %% Initialization

n = 1000
y = torch.tensor(genpareto.rvs(c=0, scale=1, loc=0, size=n), dtype=torch.float32).reshape(n, 1)
beta = 0
xi = 0.1
S = 10_000
beta_keep = torch.empty(S)
xi_keep = torch.empty(S)
counter = 0
Sigma = torch.tensor([[1.27e-3, 6.22e-4],
                      [6.22e-4, 2.80e-3]])
vector_prop_test = torch.empty(S)

path = "C:/Users/chase/OneDrive/Documents/Colorado State/Summer Research 25/ll_classifier.pt"
classifier = torch.load(path, weights_only = False)
classifier.eval()

# %% Joint Function
def fun_joint(beta, xi, y):
    scale = np.exp(beta)
    if not np.isfinite(scale) or scale <= 0:
        return -torch.inf
    #Classifier for likelihood + prior
    theta = torch.zeros(n ,2)
    theta[:, 0] += scale
    theta[:, 1] += xi
    likelihood = classifier(theta, y)
    return likelihood

# %% Metropolis-Hastings
for s in range(S):
    vector_prop = torch.distributions.MultivariateNormal(loc = torch.tensor([beta, xi]), scale = Sigma).sample((2, 1))

    likelihood_R = fun_joint(beta = vector_prop[0], xi = vector_prop[1], y = y) - fun_joint(beta, xi, y)
