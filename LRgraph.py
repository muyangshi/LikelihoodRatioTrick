import numpy as np
import matplotlib.pyplot as plt
import torch
from sbi.inference import NRE_A

np.random.seed(1234)

obs = np.random.normal(1.5, 1, 5)
const = 1/np.sqrt(2*np.pi)

#Normal with fixed sd = 1
def likelihood(mu):
    res = 1
    for x in obs:
        res *= const * np.exp((-(x - mu)**2)/2)
    return res

x_tr = np.linspace(0,5,100)

llr = np.log(likelihood(x_tr)) - np.log(likelihood(1))

lr_true = np.exp(llr)

#trained
torch.manual_seed(1234)

def simulator(theta):
    return torch.distributions.Normal(theta, 1).sample()

x_o_mean = torch.tensor([torch.mean(torch.tensor(obs))], dtype=torch.float32)
theta_train = torch.linspace(0, 5, 100)



#sbi
inference = NRE_A()
mu = torch.linspace(0, 5, 100).reshape(100,1)
x = simulator(mu)
out = inference.append_simulations(mu, x).train()
y = torch.tensor([])
for i in range(100):
    y = torch.cat((y, torch.tensor([out.forward(x = x_o_mean, theta = torch.tensor([theta_train[i]]))])))

lr_train = y.numpy()
lr_train = np.exp(lr_train)

plt.plot(x_tr, lr_true)
plt.plot(x_tr, lr_train)
plt.show()