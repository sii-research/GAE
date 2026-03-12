

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import torch.nn.functional as F
import math

class DINOv2Encoder(nn.Module):
    def __init__(self, model_name='dinov2_vitl14'):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.out_channels = self.model.embed_dim

    def forward(self, x):
        features = self.model.forward_features(x)
        patch_tokens = features["x_norm_patchtokens"] # [B, L, C]
        b, l, c = patch_tokens.shape
        h = w = int(l**0.5)
        return patch_tokens.permute(0, 2, 1).reshape(b, c, h, w)