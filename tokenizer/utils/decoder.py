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
    

class SwiGLU(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        drop=0.,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim, bias=False)
        self.gate = nn.Linear(self.embed_dim, ffn_dim, bias=False)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x_shape = x.shape
        x = x.reshape(-1, x.size(-1)) 
        x_proj = self.fc1(x)
        x = F.silu(x_proj) * self.gate(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        output = x.view(x_shape)
        return output
    
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


class BasicTransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, mlp_ratio=4.0, proj_drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.attn = Attention(
            hidden_size, 
            num_heads=num_heads, 
            num_kv_heads=num_kv_heads, 
            qkv_bias=False, 
            attn_drop=attn_drop, 
            proj_drop=proj_drop
        )
        self.norm2 = RMSNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        
        ffn_dim = int((hidden_size * mlp_ratio * 2 / 3))
        ffn_dim = 256 * ((ffn_dim + 256 - 1) // 256) 
        
        self.mlp = SwiGLU(
            embed_dim=hidden_size, 
            ffn_dim=ffn_dim, 
            drop=proj_drop
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTDecoder(nn.Module):
    def __init__(self, *, 
                 input_resolution,

                 out_ch,            
                 resolution,          
                 z_channels,          
                 give_pre_end=False, 
                 tanh_out=False, 
                 
    
                 embed_dim=1024,     
                 num_layers=12,      
                 num_heads=16,       
                 
 
                 num_kv_heads=None,   
                 mlp_ratio=4.0,       
                 dropout=0.0,
                 
  
                 **ignorekwargs):
        
        super().__init__()
        self.out_ch = out_ch
        self.resolution = resolution
        self.z_channels = z_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out


        if num_kv_heads is None:
            num_kv_heads = num_heads

        self.start_res = input_resolution
        self.num_patches = self.start_res * self.start_res 
        self.patch_size = self.resolution // self.start_res 
        
        if self.patch_size * self.start_res != self.resolution:
            raise ValueError("ViTDecoder: Output resolution must be an integer multiple of the input resolution.")
            
        print(f"--- ViTDecoder (Custom ViT Block) Initialized ---")
        print(f"Input:  {self.start_res}x{self.start_res} feature map")
        print(f"Tokens: {self.num_patches}")
        print(f"Output: {self.resolution}x{self.resolution} image")
        print(f"Upsampling 'patch size': {self.patch_size}x{self.patch_size}")
        print(f"Transformer: {num_layers} layers, {num_heads} heads, {num_kv_heads} KV heads, {mlp_ratio} MLP ratio")
        print(f"--------------------------------------------------")

        # To do: let bias = False
        self.z_proj = nn.Linear(self.z_channels, embed_dim, bias=False)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        self.blocks = nn.ModuleList([
            BasicTransformerBlock(
                hidden_size=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                mlp_ratio=mlp_ratio,
                proj_drop=dropout,
                attn_drop=dropout
            ) for _ in range(num_layers)
        ])
        
        self.norm_out = RMSNorm(embed_dim, eps=1e-6, elementwise_affine=False)
        
        final_out_dim = (self.patch_size ** 2) * self.out_ch
        self.linear_out = nn.Linear(embed_dim, final_out_dim, bias=False)

        self.apply(self._init_weights)
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm, RMSNorm)):
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
             if m.bias is not None:
                 nn.init.constant_(m.bias, 0)



    def forward(self, z):
        self.last_z_shape = z.shape
        h = z.flatten(2).permute(0, 2, 1) # (B, H_z*W_z, C_z) -> (B, 256, 256)
        h = self.z_proj(h) # (B, 256, embed_dim)
        h = h + self.pos_embed
        for block in self.blocks:
            h = block(h) # (B, 256, embed_dim)
        
        if self.give_pre_end:
            return h
        h = self.norm_out(h)
        h = self.linear_out(h) # (B, 256, patch_size*patch_size*out_ch)
        h = rearrange(h, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', 
                      h=self.start_res, w=self.start_res, 
                      p1=self.patch_size, p2=self.patch_size, 
                      c=self.out_ch)
        if self.tanh_out:
            h = torch.tanh(h)
            
        return h
    
    
class ViTEncoder(nn.Module):
    def __init__(self, *, 
                 img_size=256,          
                 patch_size=16,         
                 in_ch=3,               
                 out_ch=1024,         
                 
                 embed_dim=1024,      
                 num_layers=24,       
                 num_heads=16,        
                 num_kv_heads=None,   
                 mlp_ratio=4.0,       
                 dropout=0.0,
                 **ignorekwargs):
        
        super().__init__()
        self.img_size = img_size
        self.input_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        


        self.patch_embed = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        

        if num_kv_heads is None:
            num_kv_heads = num_heads

        self.blocks = nn.ModuleList([
            BasicTransformerBlock(
                hidden_size=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                mlp_ratio=mlp_ratio,
                proj_drop=dropout,
                attn_drop=dropout
            ) for _ in range(num_layers)
        ])
        

        self.norm = RMSNorm(embed_dim, eps=1e-6, elementwise_affine=False)
        self.proj_out = nn.Linear(embed_dim, out_ch, bias=False) 

        self.apply(self._init_weights)
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm, RMSNorm)):
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
             if m.bias is not None:
                 nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 1. Patch Embed: (B, 3, 256, 256) -> (B, embed_dim, 16, 16)
        x = self.patch_embed(x)
        
        # 2. Flatten: (B, embed_dim, 16, 16) -> (B, embed_dim, 256) -> (B, 256, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        
        # 3. Add Pos Embed
        x = x + self.pos_embed
        
        # 4. Transformer Layers
        for block in self.blocks:
            x = block(x)
            
        # 5. Final Norm & Projection
        x = self.norm(x)
        x = self.proj_out(x) # (B, 256, out_ch)
        
        # 6. Reshape back to Spatial Feature Map for Adapter
        # (B, H*W, C) -> (B, C, H, W)
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.transpose(1, 2).reshape(B, C, H, W).contiguous()
        
        return x