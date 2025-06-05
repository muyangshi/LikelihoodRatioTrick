import numpy as np
import matplotlib.pyplot as plt
import torch
from sbi.inference import NRE
from sbi.inference import ratio_estimator_based_potential

#class TrickPrior:
#    def sample(self, b):
 #       return torch.zeros(100)
#    def log_prob(self, b):
#        return torch.zeros(100)
    
#trick = TrickPrior()


np.random.seed(1234)

obs = np.random.normal(1.5, 1, 100) #100 generated observations with mean 1.5
const = -(len(obs)/2) * np.log(2*np.pi) 

#Log-likelihood function with fixed sd = 1
def log_likelihood(mu):
    res = 0
    for x in obs:
        res += (x - mu)**2
    return -(res/2) + const

x_tr = np.linspace(0,3,100)

llr = log_likelihood(x_tr) - log_likelihood(np.mean(obs))

lr_true = np.exp(llr)


#trained
torch.manual_seed(1234)

def simulator(theta):
    return theta + torch.randn_like(theta) # returns mean + a normally generated number w/ sd 1

obs_tensor = torch.tensor(obs, dtype=torch.float64).reshape(100,1)



#sbi
prior = torch.distributions.Uniform(torch.tensor([0.0]), torch.tensor([3.0]))

inference = NRE()
mu = prior.sample((1000,)) #samples from uniform 0-5 1000 times
x = simulator(mu)
out = inference.append_simulations(mu, x).train()
train_func, transform = ratio_estimator_based_potential(out, prior = prior, x_o= obs_tensor, enable_transform=False) #f(theta) = log(p(theta, x_o)p(theta))

logy = train_func(torch.linspace(0, 3, 100).reshape(100,1)) #gets output of potential function
lr_train = np.exp(logy[0].detach().numpy()) #potential is stored in a (100 x 100) matrix whose rows are identical, so I'm taking the first one


plt.figure(figsize=(12,3))

plt.subplot(131)
plt.title("True Likelihood")
plt.plot(x_tr, lr_true)
plt.subplot(132)
plt.title("Potential Function")
plt.plot(x_tr, lr_train)
plt.subplot(133)
plt.title("True over Potential Fn")
plt.plot(x_tr, lr_true / lr_train)
plt.show()