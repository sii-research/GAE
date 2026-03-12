import os
import math, json, pickle, logging, argparse, yaml, torch, numpy as np
from time import time, strftime
from glob import glob
from copy import deepcopy
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm
import torch.distributed as dist
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import torchvision
# local imports
from models.lightningdit_we import LightningDiT_models
from transport import create_transport, Sampler
from datasets.img_latent_dataset import ImgLatentDataset



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

# sample function
def do_sample(train_config, accelerator, ckpt_path=None, cfg_scale=None, model=None, vae=None, demo_sample_mode=False):
    """
    Run sampling.
    """

    folder_name = f"{train_config['model']['model_type'].replace('/', '-')}-ckpt-{ckpt_path.split('/')[-1].split('.')[0]}-{train_config['sample']['sampling_method']}-{train_config['sample']['num_sampling_steps']}".lower()
    if cfg_scale is None:
        cfg_scale = train_config['sample']['cfg_scale']
    cfg_interval_start = train_config['sample']['cfg_interval_start'] if 'cfg_interval_start' in train_config['sample'] else 0
    timestep_shift = train_config['sample']['timestep_shift'] if 'timestep_shift' in train_config['sample'] else 0
    if cfg_scale > 1.0:
        folder_name += f"-interval{cfg_interval_start:.2f}"+f"-cfg{cfg_scale:.2f}"
        folder_name += f"-shift{timestep_shift:.2f}"

    if demo_sample_mode:
        cfg_interval_start = 0
        timestep_shift = 0
        cfg_scale = 9.0

    sample_folder_dir = os.path.join(train_config['train']['output_dir'], train_config['train']['exp_name'], folder_name)
    if accelerator.process_index == 0:
        if not demo_sample_mode:
            print_with_prefix('Sample_folder_dir=', sample_folder_dir)
        print_with_prefix('ckpt_path=', ckpt_path)
        print_with_prefix('cfg_scale=', cfg_scale)
        print_with_prefix('cfg_interval_start=', cfg_interval_start)
        print_with_prefix('timestep_shift=', timestep_shift)

    if not os.path.exists(sample_folder_dir):
        if accelerator.process_index == 0:
            os.makedirs(sample_folder_dir, exist_ok=True) 
    else:
        png_files = [f for f in os.listdir(sample_folder_dir) if f.endswith('.png')]
        png_count = len(png_files)
        if png_count > train_config['sample']['fid_num']:
            if accelerator.process_index == 0:
                print_with_prefix(f"Found {png_count} PNG files in {sample_folder_dir}, skip sampling.")
            return sample_folder_dir

    torch.backends.cuda.matmul.allow_tf32 = True  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup accelerator:
    device = accelerator.device

    # Setup DDP:
    device = accelerator.device
    seed = train_config['train']['global_seed'] * accelerator.num_processes + accelerator.process_index
    torch.manual_seed(seed)
    # torch.cuda.set_device(device)
    print_with_prefix(f"Starting rank={accelerator.local_process_index}, seed={seed}, world_size={accelerator.num_processes}.")
    rank = accelerator.local_process_index

    # Load model:
    if 'downsample_ratio' in train_config['vae']:
        downsample_ratio = train_config['vae']['downsample_ratio']
    else:
        downsample_ratio = 16
    latent_size = train_config['data']['image_size'] // downsample_ratio

    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["ema"]
    model.load_state_dict(checkpoint)
    model.eval()  # important!
    model.to(device)

    transport = create_transport(
        train_config['transport']['path_type'],
        train_config['transport']['prediction'],
        train_config['transport']['loss_weight'],
        train_config['transport']['train_eps'],
        train_config['transport']['sample_eps'],
        use_cosine_loss = train_config['transport']['use_cosine_loss'] if 'use_cosine_loss' in train_config['transport'] else False,
        use_lognorm = train_config['transport']['use_lognorm'] if 'use_lognorm' in train_config['transport'] else False,
    )  # default: velocity;
    sampler = Sampler(transport)
    mode = train_config['sample']['mode']
    if mode == "ODE":
        sample_fn = sampler.sample_ode(
            sampling_method=train_config['sample']['sampling_method'],
            num_steps=train_config['sample']['num_sampling_steps'],
            atol=train_config['sample']['atol'],
            rtol=train_config['sample']['rtol'],
            reverse=train_config['sample']['reverse'],
            timestep_shift=timestep_shift,
        )
    elif mode == "SDE":
        sample_fn = sampler.sample_sde(
            sampling_method=train_config['sample']['sampling_method'], 
            num_steps=train_config['sample']['num_sampling_steps'],    
            diffusion_form=train_config['sample'].get('diffusion_form', "SBDM"),
            last_step=train_config['sample'].get('last_step', "Mean"),
            timestep_shift=timestep_shift,
            last_step_size=0.04,
            diffusion_norm=1.0,
        )
    else:
        raise NotImplementedError(f"Sampling mode {mode} is not supported.")
    
    


    using_cfg = cfg_scale > 1.0
    if using_cfg:
        if accelerator.process_index == 0:
            print_with_prefix('Using cfg:', using_cfg)

    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        if accelerator.process_index == 0 and not demo_sample_mode:
            print_with_prefix(f"Saving .png samples at {sample_folder_dir}")
    accelerator.wait_for_everyone()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = train_config['sample']['per_proc_batch_size']
    global_batch_size = n * accelerator.num_processes
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    num_samples = len([name for name in os.listdir(sample_folder_dir) if (os.path.isfile(os.path.join(sample_folder_dir, name)) and ".png" in name)])
    total_samples = int(math.ceil(train_config['sample']['fid_num'] / global_batch_size) * global_batch_size)
    if rank == 0:
        if accelerator.process_index == 0:
            print_with_prefix(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % accelerator.num_processes == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // accelerator.num_processes)
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    done_iterations = int( int(num_samples // accelerator.num_processes) // n)
    pbar = range(iterations)
    if not demo_sample_mode:
        pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    
    if accelerator.process_index == 0:
        print_with_prefix("Using latent normalization")
    dataset = ImgLatentDataset(
        data_dir=train_config['data']['data_path'],
        latent_norm=train_config['data']['latent_norm'] if 'latent_norm' in train_config['data'] else False,
        latent_multiplier=train_config['data']['latent_multiplier'] if 'latent_multiplier' in train_config['data'] else 0.18215,
    )
    latent_mean, latent_std = dataset.get_latent_stats()
    latent_multiplier = train_config['data']['latent_multiplier'] if 'latent_multiplier' in train_config['data'] else 0.18215
    # move to device
    latent_mean = latent_mean.clone().detach().to(device)
    latent_std = latent_std.clone().detach().to(device)


    if demo_sample_mode:
        if accelerator.process_index == 0:
            images = []
            for label in tqdm([975, 3, 207, 387, 388, 88, 979, 279], desc="Generating Demo Samples"):
                z = torch.randn(1, model.in_channels, latent_size, latent_size, device=device)
                y = torch.tensor([label], device=device)
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([1000] * 1, device=device)
                y = torch.cat([y, y_null], 0)
                model_kwargs = dict(y=y, cfg_scale=cfg_scale, cfg_interval=False, cfg_interval_start=cfg_interval_start)
                model_fn = model.forward_with_cfg
                samples = sample_fn(z, model_fn, **model_kwargs)[-1]
                samples = (samples * latent_std) / latent_multiplier + latent_mean
                samples = vae.decode_to_images(samples)
                images.append(samples)
            # Combine 8 images into a 2x4 grid
            os.makedirs('demo_images', exist_ok=True)
            # Stack all images into a large numpy array
            all_images = np.stack([img[0] for img in images])  # Take first image from each batch            
            # Rearrange into 2x4 grid
            h, w = all_images.shape[1:3]
            grid = np.zeros((2 * h, 4 * w, 3), dtype=np.uint8)
            for idx, image in enumerate(all_images):
                i, j = divmod(idx, 4)  # Calculate position in 2x4 grid
                grid[i*h:(i+1)*h, j*w:(j+1)*w] = image
                
            # Save the combined image
            Image.fromarray(grid).save('demo_images/demo_samples.png')

            return None
    else:
        for i in pbar:
            # Sample inputs:
            z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
            y = torch.randint(0, train_config['data']['num_classes'], (n,), device=device)
            
            # Setup classifier-free guidance:
            if using_cfg:
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([1000] * n, device=device)
                y = torch.cat([y, y_null], 0)
                model_kwargs = dict(y=y, cfg_scale=cfg_scale, cfg_interval=True, cfg_interval_start=cfg_interval_start)
                model_fn = model.forward_with_cfg
            else:
                model_kwargs = dict(y=y)
                model_fn = model.forward

            samples = sample_fn(z, model_fn, **model_kwargs)[-1]
            if using_cfg:
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

            samples = (samples * latent_std) / latent_multiplier + latent_mean
            # samples = vae.decode_to_images(samples)
            samples = vae.decode(samples)
            samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            # Save samples to disk as individual .png files
            for i, sample in enumerate(samples):
                index = i * accelerator.num_processes + accelerator.process_index + total
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
            total += global_batch_size
            accelerator.wait_for_everyone()

    return sample_folder_dir

# some utils
def print_with_prefix(*messages):
    prefix = f"\033[34m[LightningDiT-Sampling {strftime('%Y-%m-%d %H:%M:%S')}]\033[0m"
    combined_message = ' '.join(map(str, messages))
    print(f"{prefix}: {combined_message}")

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config
