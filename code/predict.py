import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import matplotlib.pyplot as plt
import albumentations as A
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2

from Detection_Model.train import FocalLoss, DiceLoss, get_class_weights, TverskyLoss

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体

# # 分类名和颜色
# CLASS_NAMES = {
#     2: "建筑物无损",
#     3: "轻微损坏",
#     4: "严重损坏",
#     5: "完全毁坏"
# }
#
# CLASS_COLORS = {
#     2: (150, 150, 150),
#     3: (255, 255, 0),
#     4: (255, 165, 0),
#     5: (255, 0, 0)
# }

VISUALIZATION_PALETTE = {
    0: [0, 0, 0],  # 背景 - 黑色
    1: [0, 0, 255],  # 水 - 纯蓝色
    2: [150, 150, 150],  # 建筑物无损 - 浅灰
    3: [255, 255, 0],  # 建筑物轻微损坏 - 黄色
    4: [255, 165, 0],  # 建筑物严重损坏 - 橙色
    5: [255, 0, 0],  # 建筑物完全毁坏 - 红色
    6: [0, 255, 255],  # 车辆 - 青色
    7: [128, 64, 128],  # 道路畅通 - 紫灰色
    8: [64, 0, 128],  # 道路阻塞 - 深紫色
    9: [0, 128, 0],  # 树木 - 绿色
    10: [0, 191, 255],  # 水池 - 天蓝色
}


def get_chinese_font(size=100):
    font_path = "C:/Windows/Fonts/msyh.ttc"
    return ImageFont.truetype(font_path, size)


def pixel_accuracy_filtered(pred, label, include_classes=[2, 3, 4, 5]):
    """
    仅在指定类别上计算像素准确率。

    Args:
        pred: numpy array, shape [H, W]
        label: numpy array, shape [H, W]
        include_classes: list[int]，只统计这些类别的像素

    Returns:
        float: 过滤后的像素准确率
    """
    mask = np.isin(label, include_classes)  # 只看指定类别的区域
    correct = (pred[mask] == label[mask]).sum()
    total = mask.sum()
    return correct / total if total > 0 else 0.0


def compute_miou(pred, label, num_classes=11, include_classes=None):
    """
    计算 mean IoU，可选指定参与计算的类别。
    Args:
        pred : 预测的标签图，shape = [H, W]
        label : 真实标签图，shape = [H, W]
        num_classes : 数据集中总类别数（默认11）
        include_classes : 指定计算哪些类别的 IoU，例如 [2, 3, 4, 5] 表示只计算建筑物类
    Returns:
        float: 平均 IoU
    """
    ious = []
    for cls in range(num_classes):
        if include_classes is not None and cls not in include_classes:
            continue  # 跳过不感兴趣的类别

        pred_inds = (pred == cls)
        label_inds = (label == cls)
        intersection = np.logical_and(pred_inds, label_inds).sum()
        union = np.logical_or(pred_inds, label_inds).sum()

        if union == 0:
            continue  # 如果没有这个类的像素，就跳过

        iou = intersection / union
        ious.append(iou)

    return np.mean(ious) if ious else 0.0


def unnormalize(img_tensor, mean, std):
    """
    img_tensor: torch.Tensor shape [3, H, W]
    mean, std: list or tuple of length 3
    """
    for t, m, s in zip(img_tensor, mean, std):
        t.mul_(s).add_(m)
    return img_tensor


def predict_and_draw(model, image_path, device, model_name, label_path, save_name, color_mask_name):
    target_classes = [2, 3, 4, 5]  # 只计算这四类的 loss
    class_weights = get_class_weights().to(device)
    focal_loss_fn = FocalLoss(gamma=2.0, weight=class_weights, target_classes=target_classes)
    dice_loss_fn = DiceLoss(target_classes=target_classes)

    model.eval()

    # === 1. 加载图像并 Resize 到固定尺寸 ===
    raw_image = cv2.imread(image_path)
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

    # fixed_size = ()
    transform = A.Compose([
        A.Resize(768, 768),
        A.Normalize(),
        ToTensorV2()
    ])
    augmented = transform(image=raw_image)
    img_tensor = augmented['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # # === 2. 绘制预测框并检测建筑物 ===
    # # === 反标准化图像以便正确显示 ===
    # mean = (0.485, 0.456, 0.406)
    # std = (0.229, 0.224, 0.225)
    # unnorm_img = unnormalize(augmented['image'].clone(), mean, std)
    # image_np = (unnorm_img.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    # # image_np = (augmented['image'].permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    # image_pil = Image.fromarray(image_np).convert("RGB")
    # draw = ImageDraw.Draw(image_pil)
    # font = get_chinese_font(size=30)
    # detected_building = False
    # min_area = 1000  # 最小框面积
    #
    # detected_classes = set()
    #
    # for cls_id in CLASS_NAMES.keys():
    #     mask = (pred == cls_id).astype(np.uint8)
    #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     for cnt in contours:
    #         x, y, w, h = cv2.boundingRect(cnt)
    #         area = w * h
    #         if area < min_area:  # ✅ 同时满足面积范围才绘制
    #             continue
    #         draw.rectangle([x, y, x + w, y + h], outline=CLASS_COLORS[cls_id], width=2)
    #         draw.text((x, y - 10), CLASS_NAMES[cls_id], fill=CLASS_COLORS[cls_id], font=font)
    #         detected_building = True
    #         detected_classes.add(cls_id)
    #
    # print(f"✅ 检测到 {len(detected_classes)} 种类别: {', '.join([CLASS_NAMES[cls] for cls in detected_classes])}")
    # print("✅ 检测到建筑物！" if detected_building else "⚠️ 未检测到建筑物！")

    # # === 3. 保存绘图结果 ===
    # output_dir = "./output"
    # os.makedirs(output_dir, exist_ok=True)
    # base_name = os.path.splitext(os.path.basename(image_path))[0]
    # save_name = save_name
    # image_pil.save(os.path.join(output_dir, save_name))
    # image_pil.show()

    # === 4. 输出预测的伪彩图 ===
    color_label = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for class_id, color in VISUALIZATION_PALETTE.items():
        color_label[pred == class_id] = color

    plt.figure(figsize=(6, 6))
    plt.imshow(color_label)
    plt.title("预测结果的彩色标签图")
    plt.axis("off")
    plt.show()
    cv2.imwrite(os.path.join("./output", color_mask_name),
                cv2.cvtColor(color_label, cv2.COLOR_RGB2BGR))

    # === 5. 加载并 Resize 标签图用于 mIoU、准确率和 loss 计算 ===
    label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    resized_label = A.Resize(768, 768)(image=label_img)['image']

    # 转成 tensor 并与 output 对齐
    label_tensor = torch.from_numpy(resized_label).long().unsqueeze(0).to(device)

    # 计算两类loss（注意 output 是 [B, C, H, W]，label 是 [B, H, W]）
    focal_loss = focal_loss_fn(output, label_tensor)
    dice_loss = dice_loss_fn(output, label_tensor)

    # === 输出性能指标 ===
    acc = pixel_accuracy_filtered(pred, resized_label)
    miou = compute_miou(pred, resized_label, num_classes=11)

    print(f"Pixel Accuracy: {acc:.4f}")
    print(f"平均IoU（mIoU）: {miou:.4f}")
    print(f"Focal Loss: {focal_loss.item():.4f}")
    print(f"Dice Loss: {dice_loss.item():.4f}")

    return {
        'accuracy': acc,
        'miou': miou,
        'Focal Loss': focal_loss.item(),
        'Dice Loss': dice_loss.item(),
    }
