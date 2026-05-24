from matplotlib import pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from model import get_model
from dataset import SegmentationDataset
from train import train
from predict import predict_and_draw
from tabulate import tabulate
import os

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体


def train_main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_dir = "../RescueNet/train/train-org-img"
    mask_dir = "../RescueNet/train/train-label-img"

    train_dataset = SegmentationDataset(img_dir, mask_dir, max_images=100)
    dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=6, pin_memory=True)

    # 模型定义（支持多个模型）
    models = {
        "UNet": get_model('unet').to(device),
        "DeepLabV3": get_model('deeplabv3').to(device),
        "PSPNet": get_model('pspnet').to(device)
    }

    # 存储各模型的训练损失
    all_losses = {}
    output_dir = "./runs"
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        losses = train(model, dataloader, device, epochs=10)['epoch_losses']
        torch.save(model.state_dict(), os.path.join(output_dir, f"{model_name}_model_14.pth"))
        print(f"{model_name} training completed and model saved!")
        all_losses[model_name] = losses

    # Epochs from 1 to 10
    epochs = list(range(1, 11))
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, all_losses["unet"], 'o-', color='green', label='U-Net')
    plt.plot(epochs, all_losses["deeplabv3"], 'o-', color='blue', label='DeepLabV3')
    plt.plot(epochs, all_losses["pspnet"], 'o-', color='red', label='PSPNet')

    plt.title('Training Loss Curve Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)

    save_curve = '/Loss_value_curve'
    # 保存图像
    if save_curve:
        plt.savefig(f"{save_curve}/损失函数曲线.png")
    plt.show()


def predict_main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型定义（支持多个模型）
    models = {
        "UNet": get_model('unet').to(device),
        "DeepLabV3": get_model('deeplabv3').to(device),
        "PSPNet": get_model('pspnet').to(device)
    }

    # 加载多个模型的权重
    model_weights = {
        "UNet": "./runs/UNet_model_12.pth",
        "DeepLabV3": "./runs/DeepLabV3_model_10.pth",
        "PSPNet": "./runs/PSPNet_model_10.pth"
    }

    # 存储各个模型的评估结果
    evaluation_results = []
    # 进行预测并计算性能评估
    for model_name, model in models.items():
        # 加载模型权重
        model.load_state_dict(torch.load(model_weights[model_name], map_location=device))
        model.eval()

        # 预测与评估
        image_path = "../RescueNet/test/test-org-img/10782.jpg"  # 可以根据需求更换
        label_path = '../RescueNet/test/test-label-img/10782_lab.png'
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_name = f"{model_name}_output_box_{base_name}_10.png"
        color_mask_name = f"{model_name}_predict_output_mask_{base_name}_10.png"

        pred_result = predict_and_draw(model, image_path, device, model_name, label_path, save_name, color_mask_name)

        # 评估模型的性能（准确率、mIoU、损失等）
        accuracy = pred_result['accuracy']
        miou = pred_result['miou']
        focal_loss = pred_result['Focal Loss']
        dice_loss = pred_result['Dice Loss']

        # 记录每个模型的评估结果
        evaluation_results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "mIoU": miou,
            "Focal Loss": focal_loss,
            "Dice Loss": dice_loss
        })

    # 打印表格形式的评估结果
    headers = ["Model", "Accuracy", "mIoU", "Focal Loss", "Dice Loss"]
    table = [[r["Model"], r["Accuracy"], r["mIoU"], r["Focal Loss"], r["Dice Loss"]] for r in evaluation_results]
    print(tabulate(table, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    # 选择调用哪个主函数
    train_main()
    # predict_main()
