import torch
from sbi.inference import NPE

# define shifted Gaussian simulator.
def simulator(θ): return θ + torch.randn_like(θ)
# draw parameters from Gaussian prior.
θ = torch.randn(1000, 2)
# simulate data
x = simulator(θ)

# choose sbi method and train
inference = NPE()
inference.append_simulations(θ, x).train()

# do inference given observed data
x_o = torch.ones(2)
posterior = inference.build_posterior()
samples = posterior.sample((1000,), x=x_o)

#My own plotting of above code from sbi
import matplotlib.pyplot as plt

plt.figure(figsize= (9, 3))
plt.subplot(131)
plt.title("Sampled θ from Prior")
plt.hist(θ)
plt.subplot(132)
plt.title("Simulated Data")
plt.hist(x)
plt.subplot(133)
plt.title("Sampled θ from Posterior")
plt.hist(samples)

plt.show()

#sbi pairplot
from sbi.analysis import pairplot
_ = pairplot(samples,)