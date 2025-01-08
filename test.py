import os
import torch
import numpy as np
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from src.data.dataset import DehazeDataset
from src.models.improved_model import PhysicalDehazeNet

# 设置设备为CPU
device = torch.device("cpu")

# 数据集路径
test_hazy_dir = "F:/Final_Project/data/RESIDE/test/hazy"
test_clear_dir = "F:/Final_Project/data/RESIDE/test/clear"

# 数据预处理
transform = Compose([
    Resize((256, 256)),  # 调整图像大小
    ToTensor(),          # 转为张量
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 归一化
])

# 加载测试数据集，每次处理一个图像对
test_dataset = DehazeDataset(hazy_dir=test_hazy_dir, clear_dir=test_clear_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 初始化模型并加载权重
model = PhysicalDehazeNet().to(device)
model_path = "F:/Final_Project/result/models/best_model.pth"  # 选择最佳模型权重路径
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # 设置模型为评估模式

# 保存结果
output_visual_path = "F:/Final_Project/result/visualization_500/"
os.makedirs(output_visual_path, exist_ok=True)

# 评估函数
# 可视化展示 PSNR 和 SSIM
def evaluate_model_with_visualization(model, test_loader):
    psnr_scores = []
    ssim_scores = []
    top_k = 5
    bottom_k = 5

    for idx, (hazy_image, clear_image) in enumerate(test_loader):
        hazy_image = hazy_image.to(device)
        clear_image = clear_image.to(device)

        with torch.no_grad():
            # 模型预测三个输出
            dehazed_image, transmission, atmosphere = model(hazy_image)

        # 反归一化
        hazy_image = hazy_image.squeeze(0).cpu() * 0.5 + 0.5
        dehazed_image = dehazed_image.squeeze(0).cpu() * 0.5 + 0.5
        clear_image = clear_image.squeeze(0).cpu() * 0.5 + 0.5

        # 动态设置 SSIM win_size
        min_dim = min(dehazed_image.shape[1:])  # 图像的最小边
        win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)

        # 计算 PSNR 和 SSIM
        psnr_value = psnr(dehazed_image.numpy(), clear_image.numpy(), data_range=1)
        ssim_value = ssim(
            dehazed_image.permute(1, 2, 0).numpy(),
            clear_image.permute(1, 2, 0).numpy(),
            win_size=win_size,
            channel_axis=-1,
            data_range=1
        )

        psnr_scores.append(psnr_value)
        ssim_scores.append(ssim_value)

        print(f"Image {idx + 1}/{len(test_loader)}: PSNR = {psnr_value:.4f}, SSIM = {ssim_value:.4f}")

    # 计算平均 PSNR 和 SSIM
    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")

    # PSNR 和 SSIM 趋势图
    plt.figure(figsize=(12, 6))
    plt.plot(psnr_scores, label="PSNR", color="blue")
    plt.xlabel("Image Index")
    plt.ylabel("PSNR")
    plt.title("PSNR Trend Across Images")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_visual_path}psnr_trend.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(ssim_scores, label="SSIM", color="orange")
    plt.xlabel("Image Index")
    plt.ylabel("SSIM")
    plt.title("SSIM Trend Across Images")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_visual_path}ssim_trend.png")
    plt.close()

    # PSNR 和 SSIM 分布直方图
    plt.figure(figsize=(12, 6))
    plt.hist(psnr_scores, bins=20, color='blue', alpha=0.7, label='PSNR')
    plt.xlabel("PSNR Score")
    plt.ylabel("Frequency")
    plt.title("PSNR Distribution")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_visual_path}psnr_distribution.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.hist(ssim_scores, bins=20, color='orange', alpha=0.7, label='SSIM')
    plt.xlabel("SSIM Score")
    plt.ylabel("Frequency")
    plt.title("SSIM Distribution")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_visual_path}ssim_distribution.png")
    plt.close()

    # Top-k 和 Bottom-k可视化
    sorted_indices = np.argsort(psnr_scores)
    top_indices = sorted_indices[-top_k:]
    bottom_indices = sorted_indices[:bottom_k]

    # 保存前5和后5的去雾效果图片
    def save_top_bottom_images(indices, label, path_prefix):
        for idx in indices:
            hazy_image, clear_image = test_dataset[idx]
            with torch.no_grad():
                dehazed_image, _, _ = model(hazy_image.unsqueeze(0).to(device))
            hazy_image = hazy_image * 0.5 + 0.5
            dehazed_image = dehazed_image.squeeze(0).cpu() * 0.5 + 0.5
            clear_image = clear_image * 0.5 + 0.5
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.title("Hazy")
            plt.imshow(hazy_image.permute(1, 2, 0).numpy())
            plt.subplot(1, 3, 2)
            plt.title("Dehazed")
            plt.imshow(dehazed_image.permute(1, 2, 0).numpy())
            plt.subplot(1, 3, 3)
            plt.title("Clear")
            plt.imshow(clear_image.permute(1, 2, 0).numpy())
            plt.savefig(f"{path_prefix}_{label}_{idx + 1}.png")
            plt.close()

    save_top_bottom_images(top_indices, "top", f"{output_visual_path}top")
    save_top_bottom_images(bottom_indices, "bottom", f"{output_visual_path}bottom")

# 开始评估
if __name__ == "__main__":
    evaluate_model_with_visualization(model, test_loader)