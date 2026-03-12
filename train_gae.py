import os
import sys
import types

os.environ['XFORMERS_DISABLED'] = '1'
os.environ['DISABLE_XFORMERS'] = '1'



import argparse
from train import do_train, load_config
from accelerate import Accelerator
from omegaconf import OmegaConf
import torch.distributed as dist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_gae_dit_xl.yaml", required=False)
    parser.add_argument("--vae_config", type=str, default="configs/vae_configs/gae.yaml",required=False, help="gae yaml config")
    args = parser.parse_args()
        
    accelerator = Accelerator()
        
    train_config = load_config(args.config)

    train_config['vae']['model_name'] = 'gae'


    gae_config = OmegaConf.load(args.vae_config)
    train_config['vae']['latent_dim'] = gae_config.model.params.get("latent_dim", 32)
    downsample_ratio = gae_config.model.params.encoder_config.params.get("patch_size", 16)
    train_config['vae']['downsample_ratio'] = downsample_ratio
    if accelerator.process_index == 0:
        print(f"==========================================")
        print(f"Switching to gae (Offline Latents)")
        print(f"VAE Config: {args.vae_config}")
        print(f"Downsample Ratio: {downsample_ratio}")
        print(f"Data Path: {train_config['data']['data_path']}")
        print(f"==========================================")
    do_train(train_config, accelerator)


if __name__ == "__main__":
    main()
