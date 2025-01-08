import torch
import torch.nn as nn


# 通道注意力模块（SE）
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        # 全局平均池化，将每个通道的特征图压缩为单一的全局平均值，获取全局语义信息
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        # 两层全连接层
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    # 前向传播，特征加权
    def forward(self, x):
        batch, channels, _, _ = x.size()
        y = self.global_avg_pool(x).view(batch, channels)
        y = self.fc(y).view(batch, channels, 1, 1)
        return x * y

# 残差模块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        # 两层卷积
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
    # 残差连接：输入结果于卷积结果相加
    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))

# 多尺度卷积块，提取多尺度特征
class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleBlock, self).__init__()
        # 三种尺度卷积核
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    # 三种卷积结果逐元素相加，整合多尺度信息
    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x3 = self.relu(self.conv3(x))
        x5 = self.relu(self.conv5(x))
        return x1 + x3 + x5

# 改进的去雾网络：结合物理模型、多尺度特征提取和注意力机制。
class PhysicalDehazeNet(nn.Module):
    def __init__(self):
        super(PhysicalDehazeNet, self).__init__()

        # 预测传输率t(x)
        self.transmission = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            MultiScaleBlock(64, 64),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # 确保 t(x) 在 [0, 1] 范围内
        )

        # 预测大气光A
        self.atmosphere = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(3, 1, kernel_size=1),
            nn.Sigmoid()  # 确保 A 在 [0, 1] 范围内
        )

        # 主网络部分
        self.initial = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.residual_blocks = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )
        self.se_block = SEBlock(64)
        self.refinement = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

        # Dropout减少过拟合
        self.dropout = nn.Dropout(p=0.3)

    # 前向传播
    def forward(self, x):
        # 预测传输率 t(x) 和大气光 A
        t = self.transmission(x)
        A = self.atmosphere(x)
        A = A.view(-1, 1, 1, 1)

        # 初步去雾图像 J(x)
        I_minus_A = x - A
        J = I_minus_A / (t + 1e-6) + A  # 避免除以 0

        # 对 J(x) 进一步优化
        features = self.initial(J)
        features = self.residual_blocks(features)
        features = self.se_block(features)
        features = self.dropout(features)
        refined_J = self.refinement(features)

        return refined_J, t, A