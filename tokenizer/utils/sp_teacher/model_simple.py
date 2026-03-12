
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'
    


    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, num_kv_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        
        self.head_dim = dim // num_heads
        kv_dim = self.num_kv_heads * self.head_dim
        
        self.scale = self.head_dim ** -0.5


        self.qkv = nn.Linear(dim, dim + 2 * kv_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=False) 
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6, elementwise_affine=False)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6, elementwise_affine=False)
        

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)

        q, k, v = torch.split(qkv, [self.dim, self.num_kv_heads * self.head_dim, self.num_kv_heads * self.head_dim], dim=-1)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.unsqueeze(2).expand(B, self.num_kv_heads, n_rep, N, self.head_dim).reshape(B, self.num_heads, N, self.head_dim)
            v = v.unsqueeze(2).expand(B, self.num_kv_heads, n_rep, N, self.head_dim).reshape(B, self.num_heads, N, self.head_dim)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
            scale=self.scale 
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BasicTransformerBlock_Simple(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, mlp_ratio=4.0, proj_drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.attn = Attention(
            hidden_size, 
            num_heads=num_heads, 
            num_kv_heads=num_kv_heads, 
            qkv_bias=False, # DiT 默认为 False
            attn_drop=attn_drop, 
            proj_drop=proj_drop
        )
        
    def forward(self, x):
        x = self.attn(self.norm1(x))
        return x