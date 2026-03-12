import os
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_TIMEOUT"] = "72000"
import sys
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import argparse
from safetensors.torch import save_file
from datetime import datetime
from omegaconf import OmegaConf


from datasets.img_latent_dataset import ImgLatentDataset

from PIL import Image
import numpy as np

from datetime import timedelta



from tokenizer.utils.util import instantiate_from_config


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126

    Args:
        pil_image: Input PIL Image
        image_size: Target size for both dimensions

    Returns:
        Center-cropped PIL Image of size (image_size, image_size)
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size),
            resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size),
        resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])



def main(args):
    assert torch.cuda.is_available(), "Requires at least one GPU"

    try:
        # dist.init_process_group("nccl")
        dist.init_process_group(
        "nccl", 
        timeout=timedelta(hours=5)
    )
        rank, world_size = dist.get_rank(), dist.get_world_size()
        device = rank % torch.cuda.device_count()
        seed = args.seed + rank
        if rank == 0:
            print(f"rank={rank}, seed={seed}, world_size={world_size}")
    except:
        rank, device, world_size, seed = 0, 0, 1, args.seed

    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Determine output directory based on model name
    model_name = os.path.basename(args.hf_model_path.rstrip('/'))
    output_dir = os.path.join(args.output_path, 'latents', model_name, f'imgnet{args.image_size}_norm{args.normalize_type}')
    if rank == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        else:
            print(f"dir exist: {output_dir}")
    if dist.is_initialized():
        dist.barrier()
    
    if rank == 0:
        print(f"Loading GAE model from: {args.hf_model_path}")
    gae_config = OmegaConf.load(args.vae_config_path)
    vae = instantiate_from_config(gae_config.model)
    ckpt_path = args.hf_model_path
    vae.init_from_ckpt(ckpt_path)
    vae.to(device)
    vae.eval()
    # if args.fp16:
    #     vae.half()
    
    def img_transform(p_hflip=0, img_size=None):
        img_size = img_size if img_size is not None else 256
        return transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, img_size)),
            transforms.RandomHorizontalFlip(p=p_hflip),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        
    datasets = [
        ImageFolder(args.data_path, transform=img_transform(p_hflip=p)) for p in [0.0, 1.0]
    ]
    samplers = [
        DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False, seed=args.seed)
        for ds in datasets
    ]
    loaders = [
        DataLoader(ds, batch_size=args.batch_size, shuffle=False, sampler=s,
                   num_workers=args.num_workers, pin_memory=True, drop_last=False)
        for ds, s in zip(datasets, samplers)
    ]

    if rank == 0:
        print(f"Total data: {len(loaders[0].dataset)}")

    run_images = saved_files = 0
    latents, latents_flip, labels = [], [], []

    for batch_idx, batch_data in enumerate(zip(*loaders)):
        run_images += batch_data[0][0].shape[0]
        if run_images % 100 == 0 and rank == 0:
            print(f'{datetime.now()} processing {run_images}/{len(loaders[0].dataset)}')

        for loader_idx, (x, y) in enumerate(batch_data):
            x = x.to(device)
            # if args.fp16:
            #     x = x.half()
            with torch.no_grad():
                posterior = vae.encode(x)[0]
                # z = posterior.sample().detach().cpu()
                z = posterior.mode().detach().cpu()
                
            if batch_idx == 0 and rank == 0:
                print('latent shape', z.shape, 'dtype', z.dtype)
            (latents if loader_idx == 0 else latents_flip).append(z)
            if loader_idx == 0:
                labels.append(y)

        if len(latents) == 10000 // args.batch_size:
            save_dict = {
                'latents': torch.cat(latents, dim=0).contiguous(),
                'latents_flip': torch.cat(latents_flip, dim=0).contiguous(),
                'labels': torch.cat(labels, dim=0).contiguous()
            }
            if rank == 0:
                for k, v in save_dict.items():
                    print(k, v.shape)
            save_file(save_dict, os.path.join(output_dir, f'latents_rank{rank:02d}_shard{saved_files:03d}.safetensors'),
                     metadata={'total_size': f'{save_dict["latents"].shape[0]}',
                              'dtype': f'{save_dict["latents"].dtype}',
                              'device': f'{save_dict["latents"].device}'})
            if rank == 0:
                print(f'Saved shard {saved_files}')
            latents, latents_flip, labels = [], [], []
            saved_files += 1

    if len(latents) > 0:
        save_dict = {
            'latents': torch.cat(latents, dim=0).contiguous(),
            'latents_flip': torch.cat(latents_flip, dim=0).contiguous(),
            'labels': torch.cat(labels, dim=0).contiguous()
        }
        if rank == 0:
            for k, v in save_dict.items():
                print(k, v.shape)
        save_file(save_dict, os.path.join(output_dir, f'latents_rank{rank:02d}_shard{saved_files:03d}.safetensors'),
                 metadata={'total_size': f'{save_dict["latents"].shape[0]}',
                          'dtype': f'{save_dict["latents"].dtype}',
                          'device': f'{save_dict["latents"].device}'})
        if rank == 0:
            print(f'Saved shard {saved_files}')

    dist.barrier()
    if rank == 0:
        ImgLatentDataset(output_dir, latent_norm=True)
        print(f"Latent stats saved to {output_dir}/latents_stats.pt")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_gae_dit_xl.yaml", required=False)
    # parser.add_argument("--fp16", type=str, action="store_true")
    parser.add_argument("--vae_config", type=str, default="configs/vae_configs/gae.yaml",required=False, help="gae yaml config")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    args.vae_config_path = args.vae_config
    args.data_path = config.data.raw_image_path
    args.output_path = config.train.output_dir
    args.image_size = config.data.image_size
    args.batch_size = config.vae.per_proc_batch_size
    args.seed = 42
    args.num_workers = 8

    args.hf_model_path = config.vae.hf_model_path
    args.normalize_type = config.vae.get('normalize_type', 'imagenet')

    main(args)
