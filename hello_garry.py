import numpy as np
from scipy.stats import poisson, norm, pareto


np.random.seed(1234)

n = 100
x = np.random.normal(size=n)
y = np.random.poisson(np.exp(x))

# Log posterior function (likelihood + prior)
def Kfun(beta):
    log_likelihood = np.sum(poisson.logpmf(y, np.exp(beta * x)))
    log_prior = norm.logpdf(beta, loc=0, scale=100)
    return log_likelihood + log_prior

# Metropolis-Hastings settings
S = 10000
beta_keep = np.empty(S)
beta = 1.0  # initial value

# Helper for Pareto sampling: scipy uses "b" = shape; scale = scale
shape = 0.5
scale = 0.5


for s in range(S):
    
    # Propose from Pareto
    beta_prop = pareto.rvs(b=shape, scale=scale)

    if beta_prop < -10 or beta_prop > 10:
    # Reject immediately
        beta_keep[s] = beta
    continue

    # Proposal densities (log)
    log_q_forward = pareto.logpdf(beta_prop, b=shape, scale=scale)
    log_q_backward = pareto.logpdf(beta, b=shape, scale=scale)

    # Log Metropolis-Hastings ratio
    logR = Kfun(beta_prop) - Kfun(beta) + log_q_backward - log_q_forward

    # Accept or reject
    if np.log(np.random.rand()) < logR:
        beta = beta_prop

    beta_keep[s] = beta

print(np.mean(beta_keep[1:] != beta_keep[:-1]))
