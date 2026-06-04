import cv2
import os
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 流程图：
# 文件名列表 imgs
#      ↓
# 读取 image、推导出 mask_name
#      ↓
# OpenCV 读取 RGB 图像 + 灰度标签
#      ↓
# Albumentations 同步增强处理
#      ↓
# 转换为 Tensor 格式
#      ↓
# 返回 image, mask（一个训练样本）


# 这是一个自定义的数据集类，用于图像语义分割任务。它继承自 torch.utils.data.Dataset，必须实现两个函数：__len__ 和 __getitem__。
class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, max_images):

        # img_dir:图像文件夹路径；mask_dir:标签（掩码）文件夹路径； max_images:最多读取的图片数量
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        # 获取 img_dir 目录下所有的图像文件（后缀是 .jpg 或 .png）
        self.imgs = [img for img in os.listdir(img_dir) if img.endswith(('.jpg', '.png'))]

        # 只选取前max_images张图片
        self.imgs = self.imgs[:max_images]

        # 使用裁剪替代缩放压缩，提升分割质量
        # 这是一个 图像增强流水线（transform pipeline）：
        # HorizontalFlip：以 50% 的概率水平翻转图像和掩码（增加数据多样性）；
        # ShiftScaleRotate：以 50% 的概率平移、缩放、旋转图像；
        # 加入ElasticTransform进行边界增强
        # Normalize()：归一化图像通道（使用 ImageNet 的均值方差）；
        # ToTensorV2()：把图像/掩码从 numpy 数组转换为 PyTorch tensor 格式。
        self.transform = A.Compose([
            A.Resize(768, 768),
            # A.HorizontalFlip(p=0.5),
            # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.ElasticTransform(),  # 增强边界变化感知
            A.Normalize(),
            ToTensorV2()
        ])

    # 返回数据集的长度（图片数量），用于 DataLoader 迭代。
    def __len__(self):
        return len(self.imgs)

    # 用于获取第 idx 个图像-标签对
    def __getitem__(self, idx):
        # 获取图像文件名和对应的完整路径
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # 从图像名推导出标签名，比如 0001.jpg → 0001_lab.png，是自定义数据集中图片和标签的一一对应规则。
        if img_name.endswith('.jpg'):
            mask_name = img_name.replace('.jpg', '_lab.png')
        elif img_name.endswith('.png'):
            mask_name = img_name.replace('.png', '_lab.png')
        else:
            raise ValueError(f"Unsupported file format: {img_name}")

        mask_path = os.path.join(self.mask_dir, mask_name)

        # 使用 OpenCV 读取图像和掩码；
        # 图像从 BGR → RGB；
        # 掩码用灰度图读入（每个像素点的值就是分类 ID）。
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 如果图像或标签没读成功，抛出异常。
        if image is None:
            raise ValueError(f"Failed to read image {img_path}.")
        if mask is None:
            raise ValueError(f"Failed to read mask {mask_path}.")

        # image 会变成 float32 Tensor (3, 256, 256)；
        # mask 会变成 int64 Tensor (256, 256)，代表类别索引。
        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask'].long()

        return image, mask
