#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 16:34:08 2020

@author: shireen
"""

import pickle
from sklearn import datasets
import numpy as np
import torch

class DatasetSIGGRAPH:
    """ 
    haha, found from Eric https://blog.evjang.com/2018/01/nf2.html
    https://github.com/ericjang/normalizing-flows-tutorial/blob/master/siggraph.pkl
    """
    def __init__(self):
        with open('siggraph.pkl', 'rb') as f:
            XY = np.array(pickle.load(f), dtype=np.float32)
            XY -= np.mean(XY, axis=0) # center
        self.XY = torch.from_numpy(XY)
    
    def sample(self, n):
        X = self.XY[np.random.randint(self.XY.shape[0], size=n)]
        return X

class DatasetMoons:
    """ two half-moons """
    def sample(self, n):
        moons = datasets.make_moons(n_samples=n, noise=0.05)[0].astype(np.float32)
        return torch.from_numpy(moons)

class DatasetMixture:
    """ 4 mixture of gaussians """
    def sample(self, n):
        assert n%4 == 0
        r = np.r_[np.random.randn(n // 4, 2)*0.5 + np.array([0, -2]),
                  np.random.randn(n // 4, 2)*0.5 + np.array([0, 0]),
                  np.random.randn(n // 4, 2)*0.5 + np.array([2, 2]),
                  np.random.randn(n // 4, 2)*0.5 + np.array([-2, 2])]
        return torch.from_numpy(r.astype(np.float32))


class DatasetSuperShapes:

    def __init__(self, nlobes, nsamples, step=1):
        filename = 'SSData/%d_lobes_%d.npy' % (nlobes, nsamples)
        self.data = np.load(filename)
        xdim_     = self.data.shape[1:]
        self.data = self.data[:,np.arange(0,xdim_[0],step),:]
        self.xdim = self.data.shape[1:]
        self.num_samples = self.data.shape[0]
        self.data = torch.from_numpy(self.data)

    
    def sample(self, n):
        X = self.data[np.random.randint(self.data.shape[0], size=n)]
        return X