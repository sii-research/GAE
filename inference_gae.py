import os
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_TIMEOUT"] = "72000"
import os
import sys
import importlib.util
import argparse
import logging
from datetime import datetime

from omegaconf import OmegaConf


import torch
# from inference import do_sample, load_config # random sample
from inference_sample import do_sample, load_config # category balance
from accelerate import Accelerator
from models.lightningdit_we import LightningDiT_models
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from PIL import Image

from tokenizer.utils.util import instantiate_from_config

def create_npz_from_sample_folder(sample_dir, num=50000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

def setup_logger(log_dir, rank):
    """Setup logger that outputs to both console and file (rank 0 only)."""
    logger = logging.getLogger('inference')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler only for rank 0
    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'inference_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f'Log file: {log_file}')

    return logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./", required=False)
    parser.add_argument("--vae_config", type=str, default="./",required=False, help="gae yaml config")
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--use_ema', type=lambda x: (str(x).lower() == 'true'), default=True, 
                        help="Whether to use EMA weights (True/False)")
    args = parser.parse_args()

    accelerator = Accelerator()
    train_config = load_config(args.config)

    # Setup logger - log to sample output directory
    exp_name = train_config['train']['exp_name']
    output_dir = train_config['train']['output_dir']
    log_dir = os.path.join(output_dir, exp_name, 'logs')
    logger = setup_logger(log_dir, accelerator.process_index)

    # Log sampling config
    if accelerator.process_index == 0:
        sample_cfg = train_config.get('sample', {})
        logger.info(f"Sampling config: "
            f"method={sample_cfg.get('sampling_method')}, "
            f"steps={sample_cfg.get('num_sampling_steps')}, "
            f"shift={sample_cfg.get('timestep_shift')}, "
            f"cfg={sample_cfg.get('cfg_scale')}, "
            f"fid_num={sample_cfg.get('fid_num')}, "
            f"seed={train_config.get('train', {}).get('global_seed')}"
        )

    train_config['vae']['model_name'] = 'gae'
    
    gae_config = OmegaConf.load(args.vae_config)
    patch_size = OmegaConf.select(gae_config, "model.params.encoder_config.params.patch_size", default=16)
    in_chans = OmegaConf.select(gae_config, "model.params.latent_dim", default=32)
    train_config['vae']['downsample_ratio'] = patch_size
    train_config['vae']['in_chans'] = in_chans

    hf_model_path = train_config['vae'].get('hf_model_path')
    if hf_model_path is None:
        raise ValueError("vae.hf_model_path (checkpoint) must be specified in config")

    ckpt_path = train_config.get('ckpt_path')
    if ckpt_path is None:
        raise ValueError("ckpt_path must be specified in config")
    latent_size = train_config['data']['image_size'] // patch_size
    
    

    if accelerator.process_index == 0:
        logger.info(f'Using ckpt: {ckpt_path}')
    
    
    model = LightningDiT_models[train_config['model']['model_type']](
        input_size=latent_size,
        num_classes=train_config['data']['num_classes'],
        use_qknorm=train_config['model']['use_qknorm'],
        use_swiglu=train_config['model'].get('use_swiglu', False),
        use_rope=train_config['model'].get('use_rope', False),
        use_rmsnorm=train_config['model'].get('use_rmsnorm', False),
        wo_shift=train_config['model'].get('wo_shift', False),
        in_channels=train_config['model'].get('in_chans', in_chans),
        learn_sigma=train_config['model'].get('learn_sigma', False),
        use_abs_pos=train_config['model'].get('use_abs_pos', True),
    )


    if accelerator.process_index == 0:
        logger.info(f"Loading VAE weights from {hf_model_path}")
    

    
    vae = instantiate_from_config(gae_config.model)
    ckpt_vae = hf_model_path
    vae.init_from_ckpt(ckpt_vae)
    vae.to(accelerator.device)
    vae.eval()
    
    sample_folder_dir = do_sample(train_config, accelerator, ckpt_path=ckpt_path, model=model, vae=vae, demo_sample_mode=args.demo,use_ema=args.use_ema)
    
    dist.barrier()

    create_npz_from_sample_folder(sample_folder_dir, 50000)

if __name__ == "__main__":
    main()

