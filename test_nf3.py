#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 16:32:57 2020

@author: shireen
"""

#%%
import itertools

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
from torch import distributions
from torch.distributions import MultivariateNormal, Uniform, TransformedDistribution, SigmoidTransform
from torch.nn.parameter import Parameter

from nflib.flows import (
    AffineConstantFlow, ActNorm, AffineHalfFlow, 
    SlowMAF, MAF, IAF, Invertible1x1Conv,
    NormalizingFlow, NormalizingFlowModel,
)
from nflib.spline_flows import NSF_AR, NSF_CL

from datasets import *

#%%

def scatter2_in_loop(x, z):
    n = x.shape[0]
    for i in range(n):
        xcur = x[i]
        zcur = z[i]
        plt.figure()
        plt.subplot(121)
        plt.scatter(xcur[:,0], xcur[:,1])
        plt.title('x')
        plt.subplot(122)
        plt.scatter(zcur[:,0], zcur[:,1])
        plt.title('z')
        plt.show
        
def scatter_in_loop(x, title = ''):
    for xcur in x:
        plt.figure()
        plt.scatter(xcur[:,0], xcur[:,1])
        plt.title(title)
        plt.show

def plot_in_loop(x, title = ''):
    for xcur in x:
        plt.figure()
        plt.plot(xcur[:,0], xcur[:,1])
        plt.title(title)
        plt.show
        
def splom(z,dim2show = 5):
    k=0
    if type(dim2show) is list:
        show_dim = dim2show
        dim2show = len(show_dim)
    else:
        zdim = int(np.prod(z.shape[1:]))
        show_dim = np.random.randint(zdim, size=dim2show)
    print(show_dim)    
    for ii,i in enumerate(show_dim):
        for jj,j in enumerate(show_dim):
            k = k +1
            plt.subplot(dim2show,dim2show,k)
            plt.scatter(z[:,i], z[:,j])
            if ii == len(show_dim)-1:
                plt.xlabel('dim-' + str(j))# + ' k='+str(k))
            if jj == 0:
                plt.ylabel('dim-' + str(i))# + ' k='+str(k))
            
            

#%%

config = dict()

config["data_step"]   = 2   # 10
config["nf_type"]     = 'maf' #'realnvp'
config["num_blocks"]  = 8 #4
config["num_hidden"]  = 64 #128 #64 #24 
         
config["use_scheduler"]  = False
config["num_epoches"]    = 5000 
config["batch_size"]     = 100
config["learning_rate"]  = 1e-5 #1e-4 #1e-6 #1e-4
config["weight_decay"]   = 1e-5
config["prior_type"]     = "gauss" #"stdnorm"

#hidden_factor = 2 #4 #0.25 #2 #2,4
#num_hidden = int(xdim * hidden_factor) #*4 #2 #*4

#%%
data = DatasetSuperShapes(4,2500, step = config["data_step"]) 

#%%
nshow = 10
x = data.sample(nshow)
scatter_in_loop(x, title='x')
#plot_in_loop(x, title='x')

#%%


xdim = int(np.prod(data.xdim))

# construct a model
if config["prior_type"] == "stdnorm":
    prior_mean = torch.zeros(xdim) 
    prior_cov  = torch.eye(xdim)
    dim2show   = 5

if config["prior_type"] == "gauss":
    prior_mean = torch.zeros(xdim) 
    prior_cov  = 0.01 * torch.eye(xdim)
    prior_cov[0,0] = 1
    prior_cov[1,1] = 0.5 
    dim2show       = [1, 2, 3, 4]

prior = MultivariateNormal(prior_mean, prior_cov)
#prior = TransformedDistribution(Uniform(torch.zeros(xdim), torch.ones()), SigmoidTransform().inv) # Logistic distribution

# RealNVP
if config["nf_type"] == 'realnvp':
    flows = [AffineHalfFlow(dim=xdim, parity=i%2, nh=config["num_hidden"]) for i in range(2*config["num_blocks"])]


# NICE
# flows = [AffineHalfFlow(dim=2, parity=i%2, scale=False) for i in range(4)]
# flows.append(AffineConstantFlow(dim=2, shift=False))

# SlowMAF (MAF, but without any parameter sharing for each dimension's scale/shift)
# flows = [SlowMAF(dim=2, parity=i%2) for i in range(4)]

# MAF (with MADE net, so we get very fast density estimation)
if config["nf_type"] == 'maf':
    flows = [MAF(dim=xdim, parity=i%2, nh=config["num_hidden"]) for i in range(config["num_blocks"])]

# IAF (with MADE net, so we get very fast sampling)
# flows = [IAF(dim=xdim, parity=i%2) for i in range(3)]

# insert ActNorms to any of the flows above
# norms = [ActNorm(dim=xdim) for _ in flows]
# flows = list(itertools.chain(*zip(norms, flows)))

# Glow paper
# flows = [Invertible1x1Conv(dim=xdim) for i in range(3)]
# norms = [ActNorm(dim=xdim) for _ in flows]
# couplings = [AffineHalfFlow(dim=xdim, parity=i%2, nh=32) for i in range(len(flows))]
# flows = list(itertools.chain(*zip(norms, flows, couplings))) # append a coupling layer after each 1x1

# Neural splines, coupling
# nfs_flow = NSF_CL if True else NSF_AR
# flows = [nfs_flow(dim=xdim, K=8, B=3, hidden_dim=16) for _ in range(3)]
# convs = [Invertible1x1Conv(dim=xdim) for _ in flows]
# norms = [ActNorm(dim=xdim) for _ in flows]
# flows = list(itertools.chain(*zip(norms, convs, flows)))


# construct the model
model = NormalizingFlowModel(prior, flows)
print(model)
#%%

num_iterations = config["num_epoches"] * int(data.num_samples/config["batch_size"]) #00

# optimizer
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]) # todo tune WD
print("number of params: ", sum(p.numel() for p in model.parameters()))

if config["use_scheduler"]:
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iterations, eta_min=1e-8)

#%% training
batch_size = config["batch_size"]

loss_store = []
prior_logprob_store = []
log_det_store       = []

lrs = []

model.train()
for k in range(num_iterations):
    x = data.sample(batch_size)
    x = x.reshape(batch_size, -1).float()
    
    zs, prior_logprob, log_det = model(x)
    logprob = prior_logprob + log_det
    loss = -torch.sum(logprob) # NLL

    model.zero_grad()
    loss.backward()
    
    optimizer.step()
    lrs.append(optimizer.param_groups[0]["lr"])
    
    if config["use_scheduler"]:
        scheduler.step()

    loss_store.append(loss.item())
    prior_logprob_store.append(torch.sum(prior_logprob).item())
    log_det_store.append(torch.sum(log_det).item())

    if k % 100 == 0:
        print("iter #%d of %d" % (k,num_iterations), 'loss = ', loss.item(), ', prior_logprob = ', prior_logprob_store[-1], ', log_det = ', log_det_store[-1], ', lr = ', lrs[-1])

plt.plot(loss_store)
plt.title('loss')
plt.show()

plt.plot(prior_logprob_store)
plt.title('prior logprob')
plt.show()

plt.plot(log_det_store)
plt.title('log det')
plt.show()

plt.plot(lrs)
plt.title('lrs')
plt.show()

#%% testing x -> z

model.eval()

num_samples = 1000
n2show      = 10

x = data.sample(num_samples)
zs, prior_logprob, log_det = model(x.reshape(num_samples, -1).float())
z = zs[-1]

z = z.reshape(-1, data.xdim[0], data.xdim[1])
x = x.detach().numpy()
z = z.detach().numpy()

splom(z.reshape(-1, xdim),dim2show = dim2show)

#scatter_in_loop(x[0:n2show], title='x')
#scatter_in_loop(z[0:n2show], title='z')

scatter2_in_loop(x[0:n2show], z[0:n2show])
#%% testing z -> x

xs, z = model.sample(num_samples)
x = xs[-1]

x = x.detach().numpy()
z = z.detach().numpy()

x = x.reshape(-1, data.xdim[0], data.xdim[1])
z = z.reshape(-1, data.xdim[0], data.xdim[1])

splom(z.reshape(-1, xdim),dim2show = dim2show)

#scatter_in_loop(z[0:n2show], title='z')
#scatter_in_loop(x[0:n2show], title='x')

scatter2_in_loop(x[0:n2show], z[0:n2show])