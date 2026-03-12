import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager

import numpy as np 

class Adapter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Linear(in_channels, out_channels,bias=False)

    def forward(self, x):
        # (B, C, H, W)
        b, c, h, w = x.shape
        # (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.model(x)
        # (B, H, W, C_out) -> (B, C_out, H, W)
        x = x.permute(0, 3, 1, 2)
        return x
