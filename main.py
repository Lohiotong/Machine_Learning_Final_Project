import os
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from src.data.dataset import DehazeDataset

import torch
from src.models.improved_model import PhysicalDehazeNet

# 数据路径
train_hazy_dir = "F:/Final_Project/data/RESIDE/train/hazy"
train_clear_dir = "F:/Final_Project/data/RESIDE/train/clear"
test_hazy_dir = "F:/Final_Project/data/RESIDE/test/hazy"
test_clear_dir = "F:/Final_Project/data/RESIDE/test/clear"

# 检查路径的有效
assert os.path.exists(train_hazy_dir), f"Path does not exist : {train_hazy_dir}"
assert os.path.exists(train_clear_dir), f"Path does not exist : {train_clear_dir}"
assert os.path.exists(test_hazy_dir), f"Path does not exist : {test_hazy_dir}"
assert os.path.exists(test_clear_dir), f"Path does not exist : {test_clear_dir}"
print("All dataset paths are valid!")

# 数据预处理
transform = Compose([
    Resize((256, 256)),              # 调整图像大小
    ToTensor(),                      # 转为张量
    Normalize(mean=[0.5, 0.5, 0.5], # 归一化
              std=[0.5, 0.5, 0.5]),
])

# 加载数据集
train_dataset = DehazeDataset(hazy_dir=train_hazy_dir, clear_dir=train_clear_dir, transform=transform)
test_dataset = DehazeDataset(hazy_dir=test_hazy_dir, clear_dir=test_clear_dir, transform=transform)

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 检查加载结果
for hazy, clear in train_loader:
    print(f"Hazy image shape: {hazy.shape}, Clear image shape: {clear.shape}")
    break

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PhysicalDehazeNet().to(device)
print(f"Model initialized on {device}")

# 创建测试输入（模拟16张256x256的RGB 图像）
test_input = torch.randn(16, 3, 256, 256)

# 测试模型
model.eval()
with torch.no_grad():
    outputs, transmission, atmosphere = model(test_input)
    print(f"Model output shape: {outputs.shape}")
    print(f"Transmission shape: {transmission.shape}")
    print(f"Atmosphere shape: {atmosphere.shape}")