import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import genpareto, norm, uniform

np.random.seed(1234)

# Simulate data
n = 1000
y = genpareto.rvs(c=0, scale=1, loc=0, size=n)

# Define log posterior for beta
def Kfun_beta(beta, xi, y):
    scale = np.exp(beta)
    if not np.isfinite(scale) or scale <= 0:
        return -np.inf
    log_lik = np.sum(genpareto.logpdf(y, c=xi, loc=0, scale=scale))
    log_prior = norm.logpdf(beta, 0, 100)
    return log_lik + log_prior

# Define log posterior for xi
def Kfun_xi(xi, beta, y):
    scale = np.exp(beta)
    if not np.isfinite(scale) or scale <= 0:
        return -np.inf
    if xi <= -0.5 or xi >= 1:
        return -np.inf
    log_lik = np.sum(genpareto.logpdf(y, c=xi, loc=0, scale=scale))
    log_prior = uniform.logpdf(xi, loc=-0.5, scale=1.5)  # support [-0.5, 1]
    return log_lik + log_prior

# Initialization
beta = -1
xi = 0
S = 10000
beta_keep = np.empty(S)
xi_keep = np.empty(S)
beta_counter = 0
xi_counter = 0

# Metropolis-Hastings Sampling
for s in range(S):
    beta_prop = np.random.normal(beta, 0.1)
    xi_prop = np.random.normal(xi, 0.08)

    likelihood_R_beta = Kfun_beta(beta_prop, xi, y) - Kfun_beta(beta, xi, y)
    if np.log(np.random.rand()) < likelihood_R_beta:
        beta = beta_prop
        beta_counter += 1
    beta_keep[s] = beta

    likelihood_R_xi = Kfun_xi(xi_prop, beta, y) - Kfun_xi(xi, beta, y)
    if np.log(np.random.rand()) < likelihood_R_xi:
        xi = xi_prop
        xi_counter += 1
    xi_keep[s] = xi

# Acceptance rates
print("Beta acceptance rate:", beta_counter / S)
print("Xi acceptance rate:", xi_counter / S)

# Burn-in and summarization
burn = 2000
mean_beta = np.mean(np.exp(beta_keep[burn:]))
mean_xi = np.mean(xi_keep[burn:])
print("Posterior mean of exp(beta):", mean_beta)
print("Posterior mean of xi:", mean_xi)

# Data for plotting
df = pd.DataFrame({
    'iteration': np.arange(1, S + 1),
    'beta': beta_keep,
    'xi': xi_keep
})
df_burning = df[df['iteration'] > burn]

# Plotting
sns.set(style="whitegrid")
plt.figure(figsize=(12, 5))
sns.lineplot(data=df_burning, x='iteration', y=np.exp(df_burning['beta']), color="cadetblue")
sns.lineplot(data=df_burning, x='iteration', y=np.exp(df_burning['beta']), color="firebrick", label='Smoothed')
plt.axhline(mean_beta, color='grey', linestyle='--')
plt.title("Trace Plot of the Scale Parameter (exp(beta))")
plt.xlabel("Iteration")
plt.ylabel("exp(beta) values")
plt.legend()
plt.show()

plt.figure(figsize=(12, 5))
sns.lineplot(data=df_burning, x='iteration', y='xi', color="darkseagreen")
sns.lineplot(data=df_burning, x='iteration', y='xi', color="darkorchid", label='Smoothed')
plt.axhline(mean_xi, color='hotpink', linestyle='--')
plt.title("Trace Plot of the Shape Parameter (xi)")
plt.xlabel("Iteration")
plt.ylabel("xi values")
plt.legend()
plt.show()