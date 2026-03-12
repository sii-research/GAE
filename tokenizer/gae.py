import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from contextlib import contextmanager
from tokenizer.utils.decoder import ViTDecoder
from tokenizer.utils.util import instantiate_from_config
 
from tokenizer.utils.sample.ldm_vae import DiagonalGaussianDistribution
from tokenizer.utils.sample.sigma_vae import SigmaGaussianDistribution
from tokenizer.utils.sample.rms_norm import RMSNorm

import os 
from omegaconf import DictConfig

class FoundationEncoderModel(nn.Module):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 encoder_config,
                 adapter_config,
                 frozen_encoder_config_w=None,
                 stage="stage1",
                 learning_rate: float = 1e-4,
                 semantic_loss_weight=1.0,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 lr_g_factor=1.0,
                 sigma_std: float = 0.1,
                 use_sigma_vae: bool = False,
                 latent_dim: int = 32,
                 encoder_channels: int = 1024,
                 use_norm: bool = False,
                 use_vf_loss: bool = False,
                 use_adaptive_sp_weight: bool = False,
                 align_type: str = "pre",
                 recon_ckpt_path: str = None,
                 weight_decay: float = 1e-2,
                 ):

        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.image_key = image_key
        self.stage = stage
        self.lr_g_factor = lr_g_factor
        self.semantic_loss_weight = semantic_loss_weight
        self.use_sigma_vae = use_sigma_vae
        self.sigma_std = sigma_std
        self.latent_dim = latent_dim
        self.encoder_channels = encoder_channels
        self.use_norm = use_norm
        self.norm = RMSNorm(latent_dim, eps=1e-6, elementwise_affine=False)
        self.use_vf_loss = use_vf_loss
        self.use_adaptive_sp_weight = use_adaptive_sp_weight
        self.align_type = align_type
       
       
        self.foundation_encoder = instantiate_from_config(encoder_config)
        self.adapter = instantiate_from_config(adapter_config)
        self.decoder = ViTDecoder(**ddconfig)
       

        self.frozen_encoder = None
        self.linear_proj = None

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0: print(f"Missing Keys: {missing}")
        if len(unexpected) > 0: print(f"Unexpected Keys: {unexpected}")
        

    def encode(self, x):
        target_size = self.foundation_encoder.input_size
       
        if x.shape[2:] != (target_size, target_size):
            x_scaled = F.interpolate(x, size=(target_size, target_size), mode="bicubic", align_corners=False)
        else:
            x_scaled = x
           
        h = self.foundation_encoder(x_scaled) 
        moments = self.adapter(h) 
        if self.align_type == "post" or self.use_norm:
            moments_permuted = moments.permute(0, 2, 3, 1).contiguous()
            moments_normed = self.norm(moments_permuted)
            moments = moments_normed.permute(0, 3, 1, 2).contiguous()
        else:
            pass
        
        if self.use_sigma_vae:
            posterior = SigmaGaussianDistribution(moments, std=self.sigma_std)
        else:
            posterior = DiagonalGaussianDistribution(moments)
        return posterior, h

    def decode(self, z):
        dec = self.decoder(z)
        return dec
    

    @classmethod

    def load_from_checkpoint(cls, ckpt_path: str, **kwargs):
        model = cls(**kwargs)
        print(f"Loading FoundationEncoderModel from checkpoint: {ckpt_path}")
        ignore_keys = kwargs.get("ignore_keys", [])
        model.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        
        return model