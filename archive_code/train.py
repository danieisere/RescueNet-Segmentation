import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


# ========== 1. Focal Loss ==========
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, target_classes=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.target_classes = target_classes

    def forward(self, input, target):
        N, C, H, W = input.shape
        input = input.permute(0, 2, 3, 1).reshape(-1, C)  # [N*H*W, C]
        target = target.view(-1)  # [N*H*W]
        # print("input shape:", input.shape)  # 调试
        # print("target shape:", target.shape)  # 调试

        if self.target_classes is not None:
            # 只保留目标标签
            valid_mask = torch.zeros_like(target, dtype=torch.bool)
            for cls in self.target_classes:
                valid_mask |= (target == cls)
            # print("valid_mask shape:", valid_mask.shape)  # 调试

            # print(valid_mask.shape)  # 应该是 [B*H*W]
            input = input[valid_mask]
            target = target[valid_mask]
            # print("input shape after mask:", input.shape)  # 调试
            # print("target shape after mask:", target.shape)  # 调试

        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = logpt.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, target.unsqueeze(1)).squeeze(1)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.weight is not None:
            wt = self.weight.gather(0, target)
            loss *= wt
        return loss.mean()


# ========== 2. Dice Loss ==========
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, target_classes=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.target_classes = target_classes

    def forward(self, input, target):
        input = F.softmax(input, dim=1)  # [B, C, H, W]
        num_classes = input.size(1)
        target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

        if self.target_classes is not None:
            # 只保留目标类的通道
            input = input[:, self.target_classes, :, :]
            target_one_hot = target_one_hot[:, self.target_classes, :, :]

        dims = (0, 2, 3)
        intersection = torch.sum(input * target_one_hot, dims)
        union = torch.sum(input + target_one_hot, dims)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


# ========== 3. Tversky Loss ==========

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.6, beta=0.4, smooth=1e-6, target_classes=None):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.target_classes = target_classes

    def forward(self, input, target):
        input = F.softmax(input, dim=1)  # [B, C, H, W]
        num_classes = input.shape[1]

        target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

        if self.target_classes is not None:
            input = input[:, self.target_classes, :, :]
            target_one_hot = target_one_hot[:, self.target_classes, :, :]

        TP = (input * target_one_hot).sum(dim=(2, 3))
        FP = ((1 - target_one_hot) * input).sum(dim=(2, 3))
        FN = (target_one_hot * (1 - input)).sum(dim=(2, 3))

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return 1 - tversky.mean()


# ========== 3. 权重计算 ==========
def get_class_weights():
    pixel_counts = np.array([
        2.26798943e+10, 3.50981388e+09, 1.14351218e+09, 1.13662843e+09,
        7.26309268e+08, 6.22711602e+08, 1.43126533e+08, 2.99246963e+09,
        6.85492293e+08, 9.52374313e+09, 2.45137680e+07
    ])
    pixel_counts = np.maximum(pixel_counts, 1)
    class_weights = 1.0 / np.log(pixel_counts + 1)  # 对数平滑
    class_weights = class_weights / class_weights.max() * 10
    return torch.tensor(class_weights, dtype=torch.float32)


# ========== 4. 完整训练流程 ==========
def train(model, dataloader, device, epochs, save_path=None, model_name=""):
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"🧠 使用 {torch.cuda.device_count()} 块GPU训练中")
        model = nn.DataParallel(model)
    model.train()

    class_weights = get_class_weights().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

    target_classes = [2, 3, 4, 5]  # 只计算这四类的 loss

    focal_loss = FocalLoss(gamma=2.0, weight=class_weights, target_classes=target_classes)
    tversky_loss = TverskyLoss(alpha=0.6, beta=0.4, target_classes=target_classes)

    # 新增：记录损失
    epoch_losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        for imgs, labels in progress_bar:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)

            loss = 0.5 * focal_loss(preds, labels) + 0.5 * tversky_loss(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_loss)  # 记录当前epoch的平均损失

        scheduler.step(avg_loss)
        print(f"✅ {model_name} Epoch {epoch + 1}/{epochs} - 平均 Loss: {avg_loss:.4f}")

        if save_path:
            torch.save(model.state_dict(), f"{save_path}/{model_name}_epoch_{epoch + 1}.pth")

    return {
        'epoch_losses': epoch_losses,
        'epochs': epochs,

            }  # 返回所有epoch的损失列表
