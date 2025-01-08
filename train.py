import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, ColorJitter, GaussianBlur
from torchvision.models import vgg16
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.data.dataset import DehazeDataset
from src.models.improved_model import PhysicalDehazeNet

# 经查询设备配置无可用的GPU，所以这里使用CPU进行实验
device = torch.device("cpu")

# 数据集路径
train_hazy_dir = "F:/Final_Project/data/RESIDE/train/hazy"
train_clear_dir = "F:/Final_Project/data/RESIDE/train/clear"

# 数据增强与预处理
# 训练集数据增强
train_transform = Compose([
    Resize((256, 256)),  # 调整图像大小
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 模拟光照变化
    GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  # 模拟非均匀雾气
    ToTensor(),  # 转为张量
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到 [-1, 1]
])

# 验证集数据增强
val_transform = Compose([
    Resize((256, 256)),  # 调整图像大小
    ToTensor(),  # 转为张量
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到 [-1, 1]
])

# 加载完整训练数据集
full_dataset = DehazeDataset(hazy_dir=train_hazy_dir, clear_dir=train_clear_dir, transform=None)

# 划分训练集和验证集（90%训练，10%验证）
val_size = int(0.1 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# 训练集和验证集分别应用不同的transform
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

# 创建DataLoader，分批次加载每批次加载8张图像，训练集进行随机打乱，并设置4个子进程
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

# 初始化模型
model = PhysicalDehazeNet().to(device)

# 感知损失辅助模块
class PerceptualLoss(nn.Module):
    def __init__(self, feature_extractor):
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = feature_extractor.eval()  # 设置为评估模式
        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # 冻结VGG参数，特征提取器的参数在训练过程中不更新

    # 前向传播
    def forward(self, outputs, targets):
        # 将图像转换成[0, 1],方便后续的输入
        outputs = (outputs + 1) / 2
        targets = (targets + 1) / 2
        # 将生成图像和目标图像输入到预训练的 VGG16 提取器中，提取高层次特征
        output_features = self.feature_extractor(outputs)
        target_features = self.feature_extractor(targets)
        return nn.functional.mse_loss(output_features, target_features)

# 加载预训练的 VGG16 模型，并提取提取 VGG16 的前 16 层
vgg = vgg16(pretrained=True).features[:16].to(device)
perceptual_loss = PerceptualLoss(vgg)

# 自定义损失函数
class CombinedLoss(nn.Module):
    # MSE用于重建损失
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.mse = nn.MSELoss()

    # 计算四种损失重建、感知、透射率约束和大气光约束
    def forward(self, outputs, targets, transmission, atmosphere):
        reconstruction_loss = self.mse(outputs, targets)
        perceptual_loss_value = perceptual_loss(outputs, targets)
        # 透射率约束损失，要将透射率限制在合理的物理范围[0.1, 1]，并对所有的像素约束损失求平均
        transmission_loss = torch.mean(
            torch.clamp(transmission - 1, min=0) + torch.clamp(0.1 - transmission, min=0)
        )
        # 大气光约束损失，对大气光值控制在合理物理范围内[0, 1]，并对所有像素约束损失求平均
        atmosphere_loss = torch.mean(
            torch.clamp(atmosphere - 1, min=0) + torch.clamp(0 - atmosphere, min=0)
        )

        # 总损失，以重建损失为主（保证图像的整体清晰度和还原度），其他三种损失权重设置成0.1辅助重建损失
        return (
            reconstruction_loss +
            0.1 * perceptual_loss_value +
            0.1 * transmission_loss +
            0.1 * atmosphere_loss
        )

criterion = CombinedLoss()

# 使用 Adam 优化器更新模型参数，初始学习率为 0.001
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# 学习率调度器，余弦退火调度器动态调整学习率，在训练后期减小学习率
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# 验证函数
def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for hazy_images, clear_images in val_loader:
            hazy_images, clear_images = hazy_images.to(device), clear_images.to(device)
            outputs, transmission, atmosphere = model(hazy_images)
            loss = criterion(outputs, clear_images, transmission, atmosphere)
            total_loss += loss.item()
    return total_loss / len(val_loader)

# 训练函数
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print(f"\nStarting Epoch {epoch + 1}/{num_epochs}...")
        model.train()
        total_train_loss = 0

        # tqdm显示训练进度
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}")

        for batch_idx, (hazy_images, clear_images) in progress_bar:
            hazy_images, clear_images = hazy_images.to(device), clear_images.to(device)

            optimizer.zero_grad() # 清零梯度
            outputs, transmission, atmosphere = model(hazy_images)
            loss = criterion(outputs, clear_images, transmission, atmosphere)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪，限制梯度的大小
            optimizer.step()
            total_train_loss += loss.item()
            progress_bar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})  # 更新进度条显示当前batch损失

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        avg_val_loss = validate(model, val_loader, criterion)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # 计算损失，更新最新模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "F:/Final_Project/result/models/best_model.pth")

        # 更新学习率
        scheduler.step()

    # 绘制损失曲线
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.grid()
    plt.savefig("F:/Final_Project/result/logs/loss_curve.png")
    plt.show()

# 开始训练
if __name__ == "__main__":
    train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10)