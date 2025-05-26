import numpy as np
from scipy.stats import genpareto, norm

# Seed for reproducibility
np.random.seed(1234)

# Data generation
n = 100
x = np.random.normal(0, 1, size=n)

# genpareto uses parameterization:
# genpareto.cdf(y, c=shape, loc=loc, scale=scale)
# For shape = 0, it's exponential with scale = 1
y = genpareto.rvs(c=0, loc=0, scale=1, size=n)

# Log-posterior function
def Kfun(beta):
    scale = np.exp(beta)
    if not np.isfinite(scale) or scale <= 0:
        return -np.inf
    log_lik = np.sum(genpareto.logpdf(y, c=0, loc=0, scale=scale))
    log_prior = norm.logpdf(beta, loc=0, scale=100)
    return log_lik + log_prior

# Metropolis-Hastings setup
S = 10000
beta = -1.0
beta_keep = np.empty(S)

for s in range(S):
    beta_prop = np.random.normal(loc=beta, scale=0.5)
    logR = Kfun(beta_prop) - Kfun(beta)
    
    if np.log(np.random.rand()) < logR:
        beta = beta_prop
    beta_keep[s] = beta

