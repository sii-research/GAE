import torch
import torch.nn as nn
import torch.nn.functional as F
# import pytorch_lightning as pl
from contextlib import contextmanager
import numpy as np 


class SigmaGaussianDistribution(object):
    def __init__(self, parameters, std: float):
        self.parameters = parameters
        self.mean = parameters
        self.std = std 

    def sample(self):
        batch_size = self.mean.size(0)
        device = self.parameters.device
        value = self.std / 0.8
        std_batch = torch.randn(batch_size, device=device) * value
        while std_batch.dim() < self.mean.dim():
            std_batch = std_batch.unsqueeze(-1)
        x = self.mean + std_batch * torch.randn(self.mean.shape, device=device)
        return x

    def kl(self):
        target = torch.zeros_like(self.mean)
        mse_per_element = F.mse_loss(self.mean, target, reduction='none')
        
        return mse_per_element 

    def mode(self):
        return self.mean