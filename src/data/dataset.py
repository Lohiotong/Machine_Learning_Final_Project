import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, Normalize

class DehazeDataset(Dataset):
    # 初始化导入文件，并且按照字母顺序排列，不进行数据转换在train.py单独分开转换
    def __init__(self, hazy_dir, clear_dir, transform=None):
        self.hazy_dir = hazy_dir
        self.clear_dir = clear_dir
        self.hazy_images = sorted(os.listdir(hazy_dir))
        self.clear_images = sorted(os.listdir(clear_dir))
        self.transform = transform

    # 获取样本数量，即有雾图片数量
    def __len__(self):
        return len(self.hazy_images)

    # 获取数据集样本
    def __getitem__(self, idx):
        # 路径拼接
        hazy_path = os.path.join(self.hazy_dir, self.hazy_images[idx])
        clear_path = os.path.join(self.clear_dir, self.clear_images[idx])

        # 打开图像，并且转换成RGB格式
        hazy_image = Image.open(hazy_path).convert("RGB")
        clear_image = Image.open(clear_path).convert("RGB")

        # 数据增强和预处理
        if self.transform:
            hazy_image = self.transform(hazy_image)
            clear_image = self.transform(clear_image)

        return hazy_image, clear_image