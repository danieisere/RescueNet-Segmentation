import torch
import torch.nn as nn
from tqdm import tqdm
from modules.loss import FocalLoss, TverskyLoss, get_class_weights


def train(model, dataloader, device, epochs, save_path=None, model_name="", 
          target_classes=[2, 3, 4, 5], lr=1e-4):
    """
    完整的训练流程
    
    Args:
        model: 模型对象
        dataloader: 数据加载器
        device: 训练设备（cuda/cpu）
        epochs: 训练轮数
        save_path: 模型保存路径
        model_name: 模型名称，用于保存文件命名
        target_classes: 计算loss时关注的类别
        lr: 学习率
        
    Returns:
        dict: 包含训练历史信息的字典
    """
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"🧠 使用 {torch.cuda.device_count()} 块GPU训练中")
        model = nn.DataParallel(model)
    model.train()

    class_weights = get_class_weights().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5, verbose=True
    )

    focal_loss = FocalLoss(gamma=2.0, weight=class_weights, target_classes=target_classes)
    tversky_loss = TverskyLoss(alpha=0.6, beta=0.4, target_classes=target_classes)

    # 记录损失
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
    }
