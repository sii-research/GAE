import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
import wandb
import argparse
import math
from torch.optim.lr_scheduler import LambdaLR

from tokenizer.utils.sp_teacher.model_simple import BasicTransformerBlock_Simple
from tokenizer.utils.sp_teacher.model import BasicTransformerBlock
from tokenizer.utils.sp_teacher.dino import DINOv2Encoder




def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1, min_lr_ratio=0.1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        original_cosine_val = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * original_cosine_val
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def separate_weight_decay(modules, default_decay):
    decay = []
    no_decay = []
    
    if isinstance(modules, torch.nn.Module):
        modules_iter = modules.named_parameters()
    elif isinstance(modules, (list, torch.nn.ModuleList)):
        def yield_params(mod_list):
            for m in mod_list:
                if m is not None:
                    for n, p in m.named_parameters():
                        yield n, p
        modules_iter = yield_params(modules)
    else:
            modules_iter = modules.named_parameters()

    for name, param in modules_iter:
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    
    return [
        {'params': decay, 'weight_decay': default_decay},
        {'params': no_decay, 'weight_decay': 0.0}
    ]
    
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
    
class DinoReconstructor4D(nn.Module):
    def __init__(self, base_model, latent_dim=8, num_decoder_layers=1, num_patches=256, window_size=4):
        super().__init__()
        
        if isinstance(base_model, dict) or (hasattr(base_model, "__contains__") and "target" in base_model):
            print("Instantiating base_model inside DinoReconstructor4D...")
            from tokenizer.utils.util import instantiate_from_config
            base_model = instantiate_from_config(base_model)
            
        self.base_model = base_model
        self.dino_dim = base_model.out_channels
        self.num_patches = num_patches
        self.window_size = window_size
        self.H = self.W = int(math.sqrt(num_patches))
        self.num_patches_in_window = self.window_size ** 2
        
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()
        
        self.norm_target = RMSNorm(self.dino_dim, elementwise_affine=False) 
        self.norm_recon = RMSNorm(self.dino_dim, elementwise_affine=False)
        self.norm_neck1 = RMSNorm(self.dino_dim, elementwise_affine=False) 
        self.norm_neck2 = RMSNorm(latent_dim, elementwise_affine=False)

        

        # --- Encoder / Neck ---
        self.pos_embed_pro = nn.Parameter(torch.zeros(1, self.num_patches, self.dino_dim))
        self.projector_block = BasicTransformerBlock_Simple(self.dino_dim, self.dino_dim // 64, self.dino_dim // 64)
        self.to_latent = nn.Linear(
            self.num_patches_in_window * self.dino_dim, 
            self.num_patches_in_window * latent_dim, 
            bias=False
        )

        # Decoder
        self.pos_embed_dec = nn.Parameter(torch.zeros(1, self.num_patches, self.dino_dim))
        self.decoder_layers = nn.ModuleList([
            BasicTransformerBlock(self.dino_dim, self.dino_dim // 64, self.dino_dim // 64)
            for _ in range(num_decoder_layers)
        ])
        self.recon_head = nn.Linear(latent_dim, self.dino_dim,bias=False)
        
        self.projector_block.apply(self._init_weights)
        self.to_latent.apply(self._init_weights)
        self.decoder_layers.apply(self._init_weights)
        self.recon_head.apply(self._init_weights)
        nn.init.trunc_normal_(self.pos_embed_pro, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_dec, std=0.02)
    
        
    def _init_weights(self, m):
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm, RMSNorm)):
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

                 
    def train(self, mode=True):
        super().train(mode)
        self.base_model.eval()
        return self

    def forward(self, x):
        
        with torch.no_grad():
            target_feat_4d = self.base_model(x) # (B, 1024, 16, 16)
            b, c, h, w = target_feat_4d.shape
            x_seq = target_feat_4d.reshape(b, c, h * w).permute(0, 2, 1) # (B, L, 1024)
            target_seq = self.norm_target(x_seq)
        
        # 2. Neck 
        x_neck = target_seq + self.pos_embed_pro
        x_neck = self.projector_block(x_neck)
        x_neck = self.norm_neck1(x_neck)
        

        x_neck_2d = x_neck.view(b, h, w, -1)
        x_win = x_neck_2d.view(
            b, h // self.window_size, self.window_size, w // self.window_size, self.window_size, c
        )
        x_win = x_win.permute(0, 1, 3, 2, 4, 5).contiguous()
        x_win_flatten = x_win.view(b, -1, self.num_patches_in_window * c)
        continuous_tokens_win = self.to_latent(x_win_flatten)
        continuous_tokens_2d = continuous_tokens_win.view(
            b, h // self.window_size, w // self.window_size, self.window_size, self.window_size, -1
        )
        continuous_tokens_2d = continuous_tokens_2d.permute(0, 1, 3, 2, 4, 5).contiguous()
        continuous_tokens_2d = continuous_tokens_2d.view(b, h, w, -1)
        continuous_tokens = continuous_tokens_2d.view(b, h * w, -1)
        continuous_tokens = self.norm_neck2(continuous_tokens)
        
        # 3. Decoder 
        x_recon = self.recon_head(continuous_tokens)
        x_recon = x_recon + self.pos_embed_dec
        for layer in self.decoder_layers:
            x_recon = layer(x_recon)
    
        x_recon = self.norm_recon(x_recon)
        
        recon_feat_4d = x_recon.permute(0, 2, 1).reshape(b, c, h, w)
        target_feat_4d_norm = target_seq.permute(0, 2, 1).reshape(b, c, h, w)
        
        return recon_feat_4d, target_feat_4d_norm



def validate(model, loader, device, amp_dtype):
    model.eval()
    total_mse, total_cos, total_elements, count = 0.0, 0.0, 0, 0
    use_amp = amp_dtype is not None
    
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                recon_4d, target_4d = model(imgs)
                mse = F.mse_loss(recon_4d, target_4d, reduction='sum')
                cos = F.cosine_similarity(recon_4d, target_4d, dim=1).sum()
            
            total_mse += mse.item()
            total_cos += cos.item()
            count += imgs.size(0)
            total_elements += target_4d.numel()

    stats = torch.tensor([total_mse, total_cos, float(count), float(total_elements)], device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    
    global_avg_mse = stats[0] / stats[3] 
    _, _, h, w = recon_4d.shape
    global_avg_cos = stats[1] / (stats[2] * h * w) 
    
    return global_avg_mse.item(), global_avg_cos.item()



def main(args):
    dist.init_process_group(backend='nccl')
    global_rank = dist.get_rank() 
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    seed_everything(args.seed + global_rank)

    amp_dtype = None
    use_scaler = False
    if args.precision == "fp16":
        amp_dtype = torch.float16
        use_scaler = True
    elif args.precision == "bf16":
        amp_dtype = torch.bfloat16
        use_scaler = False 
    
    if global_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        wandb.init(
            dir=args.output_dir,  
            config=args,
            mode=args.wandb_mode,
            project=os.environ.get("WANDB_PROJECT", "Dino-4D-Recon"),
            name=os.environ.get("WANDB_NAME","Dino-4D-Recon")
        )
    
        
    dino = DINOv2Encoder().to(device)
    model = DinoReconstructor4D(dino, latent_dim=args.latent_dim, num_decoder_layers=args.dec_layers,window_size=args.window_size).to(device)
    model = DDP(model, device_ids=[local_rank])

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(root=args.train_path, transform=transform)
    train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=global_rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, sampler=train_sampler, pin_memory=True)

    val_dataset = ImageFolder(root=args.val_path, transform=transform)
    val_sampler = DistributedSampler(val_dataset, num_replicas=dist.get_world_size(), rank=global_rank, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, sampler=val_sampler, pin_memory=True)


    optim_groups = separate_weight_decay(model.module, args.weight_decay)
    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.98), fused=True)
    
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.05) 
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    
    start_epoch = 0
    resume_path = os.path.join(args.output_dir, "latest.pth")
    
    if os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if use_scaler and 'scaler_state_dict' in checkpoint: scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    steps_per_epoch = len(train_loader)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loader.sampler.set_epoch(epoch)
        for i, (imgs, _) in enumerate(train_loader):
            imgs = imgs.to(device)
            
            with torch.cuda.amp.autocast(enabled=(amp_dtype is not None), dtype=amp_dtype):
                recon_4d, target_4d = model(imgs)
                loss = F.mse_loss(recon_4d, target_4d)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step() 

            if global_rank == 0 and i % 20 == 0:
                global_step = epoch * steps_per_epoch + i
                with torch.cuda.amp.autocast(enabled=(amp_dtype is not None), dtype=amp_dtype):
                    batch_cos_sim = F.cosine_similarity(recon_4d, target_4d, dim=1).mean()
                wandb.log({
                    "train/global_step": global_step, 
                    "train/loss_4d": loss.item(),
                    "train/cos_sim": batch_cos_sim.item(),
                    "train/lr": scheduler.get_last_lr()[0]
                })

        val_mse, val_cos = validate(model, val_loader, device, amp_dtype)
        dist.barrier()
        if global_rank == 0:
            print(f"Epoch {epoch} | Val MSE: {val_mse:.8f} | CosSim: {val_cos:.4f}")
            val_step = (epoch + 1) * steps_per_epoch - 1
            wandb.log({
                "train/global_step": val_step,
                "val/mse": val_mse,
                "val/cos_sim": val_cos
            })
    
 
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if use_scaler else None,
                'scheduler_state_dict': scheduler.state_dict(),
            }
            
  
            torch.save(checkpoint, os.path.join(args.output_dir, "latest.pth"))
            

            if (epoch + 1) % 10 == 0:
                torch.save(checkpoint, os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pth"))
                
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--dec_layers", type=int, default=1)
    parser.add_argument("--window_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--wandb_mode", type=str, default="offline")
    args = parser.parse_args()
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    main(args)