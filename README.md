# 图像语义分割研究项目

基于RescueNet数据集的建筑物损坏程度语义分割研究，支持UNet、PSPNet和DeepLabV3三种模型。

## 项目结构

```
.
├── data/                    # 数据目录
│   └── raw/                # 原始数据集
│       ├── train/          # 训练集
│       └── test/           # 测试集
├── models/                 # 模型定义
│   ├── __init__.py        # 模型接口
│   ├── unet.py            # UNet模型
│   ├── pspnet.py          # PSPNet模型
│   ├── deeplabv3.py       # DeepLabV3模型
│   └── layers/            # 自定义网络层
│       ├── conv_blocks.py # 基础卷积模块
│       ├── attention.py   # 注意力机制
│       ├── pyramid.py     # 金字塔模块
│       └── aspp.py        # ASPP模块
├── modules/               # 核心功能模块
│   ├── datasets.py        # 数据集类
│   ├── dataloaders.py     # 数据加载器
│   ├── trainer.py         # 训练逻辑
│   ├── evaluator.py       # 评估逻辑
│   ├── metrics.py         # 评估指标
│   ├── loss.py            # 损失函数
│   ├── visualization.py   # 可视化
│   └── utils.py           # 工具函数
├── configs/               # 配置文件
│   ├── dataset_config.py  # 数据集配置
│   ├── model_config.py    # 模型配置
│   └── training_config.py # 训练配置
├── runs/                  # 训练输出（模型权重）
├── output/                # 预测结果输出
├── train.py              # 训练入口
├── test.py               # 测试入口
├── inference.py          # 推理入口
└── config.py             # 全局配置
```

## 环境要求

- Python 3.7+
- PyTorch 1.8+
- torchvision
- OpenCV
- albumentations
- matplotlib
- tqdm
- tabulate

安装依赖：
```bash
pip install torch torchvision opencv-python albumentations matplotlib tqdm tabulate
```

## 数据集准备

本项目使用RescueNet数据集，请按以下结构组织数据：

```
data/raw/
├── train/
│   ├── train-org-img/     # 训练图像
│   └── train-label-img/   # 训练标签
└── test/
    ├── test-org-img/      # 测试图像
    └── test-label-img/    # 测试标签
```

**注意**: RescueNet数据集的命名规则为 `xxx.jpg` → `xxx_lab.png`

## 使用方法

### 1. 训练模型

```bash
python train.py
```

这将同时训练UNet、PSPNet和DeepLabV3三个模型，并绘制损失曲线。

### 2. 测试评估

```bash
python test.py
```

对所有已训练的模型进行评估，输出对比表格。

### 3. 单张图像推理

```bash
python inference.py --model unet --weights ./runs/unet_final.pth --image path/to/image.jpg [--label path/to/label.png]
```

参数说明：
- `--model`: 模型类型 (unet/pspnet/deeplabv3)
- `--weights`: 模型权重文件路径
- `--image`: 输入图像路径
- `--label`: 标签图像路径（可选，用于计算评估指标）
- `--output`: 输出文件名（默认: output.png）

### 4. 自定义配置

修改 `configs/` 目录下的配置文件：

- `dataset_config.py`: 数据集路径、批大小等
- `model_config.py`: 模型超参数
- `training_config.py`: 学习率、epoch数等

## 支持的模型

### UNet
- Backbone: ResNet18
- 特色: Custom Pyramid Module + SE Attention
- 适用场景: 需要精确边界定位的任务

### PSPNet
- Backbone: ResNet18
- 特色: Pyramid Scene Parsing Module
- 适用场景: 需要全局上下文信息的任务

### DeepLabV3
- Backbone: ResNet18
- 特色: ASPP + SE Attention
- 适用场景: 多尺度目标检测

## 损失函数

采用组合损失函数：
- **Focal Loss**: 处理类别不平衡
- **Tversky Loss**: 优化分割边界

权重配置可在 `configs/training_config.py` 中调整。

## 评估指标

- **Pixel Accuracy**: 像素级准确率（仅统计建筑物相关类别）
- **mIoU**: 平均交并比
- **Focal Loss**: Focal损失值
- **Dice Loss**: Dice损失值

## 扩展指南

### 添加新模型

1. 在 `models/layers/` 中实现所需的自定义层
2. 在 `models/` 下创建新模型文件（如 `new_model.py`）
3. 在 `models/__init__.py` 中注册新模型
4. 在 `configs/model_config.py` 中添加配置

### 添加新数据集

1. 在 `modules/datasets.py` 中创建新的Dataset类
2. 在 `configs/dataset_config.py` 中添加数据路径
3. 修改数据加载逻辑以适配新数据集的命名规则

### 添加新损失函数

1. 在 `modules/loss.py` 中实现新的Loss类
2. 在 `modules/trainer.py` 中集成新损失
3. 在 `configs/training_config.py` 中配置损失权重

## 项目特点

✅ **模块化设计**: 模型、训练、评估完全解耦  
✅ **易于扩展**: 清晰的接口设计，便于添加新组件  
✅ **配置管理**: 集中化的配置系统  
✅ **多模型支持**: 内置三种主流分割模型  
✅ **完整流程**: 涵盖训练、测试、推理全流程  

## 许可证

本项目仅供学术研究使用。

## 联系方式

如有问题，请提交Issue或联系作者。
