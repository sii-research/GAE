"""
ImageNet Latent Dataset with safetensors.
"""

import os
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from safetensors import safe_open


class ImgLatentDataset(Dataset):
    def __init__(self, data_dir, latent_norm=True, latent_multiplier=1.0):
        self.data_dir = data_dir
        self.latent_norm = latent_norm
        self.latent_multiplier = latent_multiplier

        self.files = sorted(glob(os.path.join(data_dir, "*.safetensors")))
        self.img_to_file_map = self.get_img_to_safefile_map()
        
        if latent_norm:
            self._latent_mean, self._latent_std = self.get_latent_stats()

    def get_img_to_safefile_map(self):
        img_to_file = {}
        for safe_file in self.files:
            with safe_open(safe_file, framework="pt", device="cpu") as f:
                labels = f.get_slice('labels')
                labels_shape = labels.get_shape()
                num_imgs = labels_shape[0]
                cur_len = len(img_to_file)
                for i in range(num_imgs):
                    img_to_file[cur_len+i] = {
                        'safe_file': safe_file,
                        'idx_in_file': i
                    }
        return img_to_file

    def get_latent_stats(self):
        latent_stats_cache_file = os.path.join(self.data_dir, "latents_stats.pt")
        if not os.path.exists(latent_stats_cache_file):
            latent_stats = self.compute_latent_stats()
            torch.save(latent_stats, latent_stats_cache_file)
        else:
            latent_stats = torch.load(latent_stats_cache_file)
        return latent_stats['mean'], latent_stats['std']
    
    def compute_latent_stats(self):
        num_samples = min(10000, len(self.img_to_file_map))
        random_indices = np.random.choice(len(self.img_to_file_map), num_samples, replace=False)
        latents = []
        for idx in tqdm(random_indices):
            img_info = self.img_to_file_map[idx]
            safe_file, img_idx = img_info['safe_file'], img_info['idx_in_file']
            with safe_open(safe_file, framework="pt", device="cpu") as f:
                features = f.get_slice('latents')
                feature = features[img_idx:img_idx+1]
                latents.append(feature)
        latents = torch.cat(latents, dim=0)
        mean = latents.mean(dim=[0, 2, 3], keepdim=True)
        std = latents.std(dim=[0, 2, 3], keepdim=True)
        latent_stats = {'mean': mean, 'std': std}
        print(latent_stats)
        return latent_stats

    def __len__(self):
        return len(self.img_to_file_map.keys())

    def __getitem__(self, idx):
        img_info = self.img_to_file_map[idx]
        safe_file, img_idx = img_info['safe_file'], img_info['idx_in_file']
        with safe_open(safe_file, framework="pt", device="cpu") as f:
            tensor_key = "latents" if np.random.uniform(0, 1) > 0.5 else "latents_flip"
            features = f.get_slice(tensor_key)
            labels = f.get_slice('labels')
            feature = features[img_idx:img_idx+1]
            label = labels[img_idx:img_idx+1]

        if self.latent_norm:
            feature = (feature - self._latent_mean) / self._latent_std
        feature = feature * self.latent_multiplier
        
        # remove the first batch dimension (=1) kept by get_slice()
        feature = feature.squeeze(0)
        label = label.squeeze(0)
        return feature, label
    

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')   
def run_analysis(dir_path, name, num_samples=2000):
    print(f"\n--- 开始分析 {name} ---")
    ds = ImgLatentDataset(dir_path)
    
    total_len = len(ds)
    if total_len == 0:
        return None

    actual_samples = min(num_samples, total_len)
    indices = np.random.choice(total_len, actual_samples, replace=False)
    
    collected = []
    for i in tqdm(indices, desc=f"Reading {name}"):
        try:
            # --- 修改处：处理返回的元组 ---
            result = ds[i]
            if isinstance(result, tuple):
                latent = result[0]  # 取出 feature 部分
            else:
                latent = result
                
            collected.append(latent.unsqueeze(0)) 
            # ---------------------------
        except Exception as e:
            print(f"读取索引 {i} 失败: {e}")
            continue
    
    if not collected:
        return None

    data = torch.cat(collected, dim=0).to(torch.float32)
    N, C, H, W = data.shape
    
    # 后续计算逻辑保持不变...
    data_flat = data.view(N, C, -1)
    ch_means = data_flat.mean(dim=[0, 2]).numpy()
    ch_stds = data_flat.std(dim=[0, 2]).numpy()
    ch_mins = data_flat.min(dim=2)[0].min(dim=0)[0].numpy()
    ch_maxs = data_flat.max(dim=2)[0].max(dim=0)[0].numpy()
    
    raw_flat = data.flatten()
    sample_size_for_hist = min(len(raw_flat), 100000)
    raw_sample = np.random.choice(raw_flat.numpy(), sample_size_for_hist, replace=False)

    return {
        "name": name, "means": ch_means, "stds": ch_stds, 
        "mins": ch_mins, "maxs": ch_maxs, "raw_sample": raw_sample,
        "global_mean": raw_flat.mean().item(),
        "global_std": raw_flat.std().item(),
        "global_max_abs": torch.max(torch.abs(raw_flat)).item()
    }

# ==========================================
# 3. Main 函数入口
# ==========================================
def main():
    # --- 【重要】请在此处修改你的文件夹路径 ---
    # 确保这些路径包含你用 VAE 生成的 .safetensors 文件
    VAE_1_DIR = "/inspire/hdd/global_user/liuhangyu-253108120189/AIGC/LightningDiT/output/sigma/latents/last.ckpt/imgnet256_normnormalize_type" 
    VAE_2_DIR = "/inspire/hdd/global_user/liuhangyu-253108120189/AIGC/VTP/outputs/generation/aligntok/latents/last.ckpt/imgnet256_normnormalize_type"
    
    # 输出图片文件名
    OUTPUT_IMAGE_NAME = "/inspire/hdd/global_user/liuhangyu-253108120189/c/vae_comparison_result2.png"

    # 执行分析 (增加采样数以提高准确性)
    res1 = run_analysis(VAE_1_DIR, "VAE_Model_A", num_samples=5000)
    res2 = run_analysis(VAE_2_DIR, "VAE_Model_B", num_samples=5000)

    if res1 is None or res2 is None:
        print("\n分析中断，因为至少有一个数据集未能成功读取。")
        return

    print(f"\n开始绘制对比图并保存至 {OUTPUT_IMAGE_NAME}...")

    # --- 绘图对比 ---
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("VAE Latent Space Comparison for Diffusion Training", fontsize=16, y=0.99)
    
    # 子图1: 全局直方图 (判断分布形态)
    ax1 = plt.subplot(2, 2, 1)
    # 设置范围以聚焦核心区域，忽略极端值对直方图的拉伸
    combined_data = np.concatenate([res1['raw_sample'], res2['raw_sample']])
    q1, q3 = np.percentile(combined_data, [1, 99])
    plot_range = (q1 * 1.2, q3 * 1.2)
    
    ax1.hist(res1['raw_sample'], bins=150, range=plot_range, alpha=0.6, label=f"{res1['name']}", density=True, color='blue')
    ax1.hist(res2['raw_sample'], bins=150, range=plot_range, alpha=0.6, label=f"{res2['name']}", density=True, color='orange')
    ax1.set_title(f"Global Value Distribution (PDF)\nShowing range: [{plot_range[0]:.2f}, {plot_range[1]:.2f}]")
    ax1.set_xlabel("Latent Value")
    ax1.set_ylabel("Density")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 子图2: 通道均值 (判断中心偏置)
    ax2 = plt.subplot(2, 2, 2)
    x = range(len(res1['means']))
    ax2.plot(x, res1['means'], 'o-', markersize=4, label=res1['name'], alpha=0.7, color='blue')
    ax2.plot(x, res2['means'], 's-', markersize=4, label=res2['name'], alpha=0.7, color='orange')
    ax2.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Ideal Mean (0)')
    ax2.set_title("Mean per Channel")
    ax2.set_xlabel("Channel Index")
    ax2.set_ylabel("Mean Value")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 子图3: 通道标准差 (判断缩放尺度)
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(x, res1['stds'], 'o-', markersize=4, label=res1['name'], alpha=0.7, color='blue')
    ax3.plot(x, res2['stds'], 's-', markersize=4, label=res2['name'], alpha=0.7, color='orange')
    # 绘制两条参考线，SDXL通常在0.6-1.2之间
    ax3.axhline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Ideal Std (1.0)')
    ax3.set_title("Standard Deviation per Channel")
    ax3.set_xlabel("Channel Index")
    ax3.set_ylabel("Std Value")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 子图4: 通道数值范围 (判断离群值和各向异性)
    ax4 = plt.subplot(2, 2, 4)
    # 使用填充区域表示范围
    ax4.fill_between(x, res1['mins'], res1['maxs'], color='blue', alpha=0.2, label=f"{res1['name']} Range")
    ax4.plot(x, res1['mins'], color='blue', alpha=0.4, linewidth=1)
    ax4.plot(x, res1['maxs'], color='blue', alpha=0.4, linewidth=1)

    ax4.fill_between(x, res2['mins'], res2['maxs'], color='orange', alpha=0.2, label=f"{res2['name']} Range")
    ax4.plot(x, res2['mins'], color='orange', alpha=0.4, linewidth=1)
    ax4.plot(x, res2['maxs'], color='orange', alpha=0.4, linewidth=1)
    
    ax4.set_title("Min/Max Range per Channel")
    ax4.set_xlabel("Channel Index")
    ax4.set_ylabel("Value Range")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # 保存图片
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局以为标题留出空间
    plt.savefig(OUTPUT_IMAGE_NAME, dpi=150)
    plt.close(fig) # 关闭图形以释放内存

    print(f"成功保存对比图表至: {os.path.abspath(OUTPUT_IMAGE_NAME)}")

    # --- 打印量化评估报告 ---
    print("\n" + "="*55)
    print("       VAE LATENT STATS SUMMARY REPORT")
    print("="*55)
    # 格式化字符串，确保对齐
    header = f"{'Metric':<20} | {res1['name']:<15} | {res2['name']:<15}"
    print(header)
    print("-" * len(header))
    
    # 定义要打印的指标
    metrics = [
        ("Global Mean (-> 0)", res1['global_mean'], res2['global_mean']),
        ("Global Std (-> 1)", res1['global_std'], res2['global_std']),
        ("Max Abs Value", res1['global_max_abs'], res2['global_max_abs']),
        # 计算通道间均值的方差，越小说明通道越一致
        ("Channel Mean Var.", np.var(res1['means']), np.var(res2['means'])),
        # 计算通道间标准差的方差，越小说明通道缩放越一致
        ("Channel Std Var.", np.var(res1['stds']), np.var(res2['stds'])),
    ]

    for title, v1, v2 in metrics:
        print(f"{title:<20} | {v1:15.4f} | {v2:15.4f}")
    print("="*55)
    print("提示: ")
    print("1. Global Mean 越接近 0 越好。")
    print("2. Global Std 越接近 1 越好（否则需要 Scaling Factor）。")
    print("3. Max Abs Value 过大（如 >50）可能导致训练不稳定。")
    print("4. Channel Var 越小，说明各个通道的分布越一致（各向同性好）。")
    print("请查看生成的图片进行详细可视化分析。")

if __name__ == "__main__":
    # 检查是否安装了必要的库
    try:
        import safetensors
        import matplotlib
    except ImportError as e:
        print(f"错误: 缺少必要的库。请确保安装了 safetensors 和 matplotlib。\n错误信息: {e}")
        exit(1)
        
    main()