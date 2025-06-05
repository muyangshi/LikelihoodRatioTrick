import numpy as np
import matplotlib.pyplot as plt
import torch
from sbi.inference import NRE_A

np.random.seed(1234)

obs = np.random.normal(1.5, 1, 5)
const = -(len(obs)/2) * np.log(2*np.pi)

#Normal with fixed sd = 1
def log_likelihood(mu):
    res = 0
    for x in obs:
        res += (x - mu)**2
    return -(res/2) - const

x_tr = np.linspace(0,5,100)

llr = log_likelihood(x_tr) - log_likelihood(1)

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