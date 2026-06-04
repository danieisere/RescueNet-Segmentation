<div align="center">

# 🏗️ RescueNet-Segmentation

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-red?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Academic-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research-yellow)]()

[📖 中文文档](README_cn.md) | [🌍 English](README.md)

</div>

---

## 📋 项目概述

**RescueNet-Segmentation** 是一个基于 RescueNet 数据集的建筑物损坏程度语义分割研究项目。本项目实现并对比了三种先进的深度学习模型：**UNet**、**PSPNet** 和 **DeepLabV3**，用于灾害场景理解。

### 🎯 核心特性

- ✅ **多模型对比**: UNet、PSPNet、DeepLabV3，均使用 ResNet18 骨干网络
- ✅ **模块化架构**: 模型、训练、评估清晰分离
- ✅ **高级损失函数**: Focal Loss + Tversky Loss 处理类别不平衡
- ✅ **完整流程**: 训练 → 测试 → 推理 → 可视化一体化
- ✅ **配置驱动**: 集中化配置系统，易于定制
- ✅ **生产就绪**: 文档完善、可扩展的代码库

### 🏛️ 应用场景

- 灾后建筑物损坏评估
- 应急响应规划
- 城市韧性分析
- 遥感图像解译

---

## 📁 项目结构

```
RescueNet-Segmentation/
├── 📊 RescueNet/                 # 数据集
│   ├── train/
│   │   ├── train-org-img/        # 100张训练图像
│   │   └── train-label-img/      # 100张训练标签
│   └── test/
│       ├── test-org-img/         # 1张测试图像
│       └── test-label-img/       # 1张测试标签
│
├── 🧠 models/                    # 模型定义
│   ├── unet.py                   # 带自定义金字塔模块的UNet
│   ├── pspnet.py                 # 带场景解析的PSPNet
│   ├── deeplabv3.py              # 带ASPP的DeepLabV3
│   └── layers/                   # 自定义网络层
│       ├── conv_blocks.py        # 基础卷积块
│       ├── attention.py          # SE注意力机制
│       ├── pyramid.py            # 金字塔模块
│       └── aspp.py               # 空洞空间金字塔池化
│
├── ⚙️ modules/                   # 核心功能模块
│   ├── datasets.py               # 数据集加载器
│   ├── dataloaders.py            # DataLoader工厂
│   ├── trainer.py                # 训练引擎
│   ├── evaluator.py              # 评估逻辑
│   ├── metrics.py                # 性能指标
│   ├── loss.py                   # 损失函数
│   ├── visualization.py          # 结果可视化
│   └── utils.py                  # 工具函数
│
├── 📝 configs/                   # 配置文件
│   ├── dataset_config.py         # 数据集路径和参数
│   ├── model_config.py           # 模型超参数
│   └── training_config.py        # 训练设置
│
├── 🚀 入口脚本
│   ├── train.py                  # 训练所有模型
│   ├── test.py                   # 评估模型
│   └── inference.py              # 单张图像预测
│
├── 📦 runs/                      # 训练好的模型权重
├── 🖼️ output/                    # 预测结果
└── 📚 文档
    ├── README.md                 # 英文文档
    ├── README_cn.md              # 中文文档
    ├── QUICKSTART.md             # 快速入门指南
    ├── PROJECT_STRUCTURE.md      # 架构详解
    └── MIGRATION_GUIDE.md        # 迁移指南
```

---

## 🛠️ 安装

### 前置要求

- Python 3.7 或更高版本
- PyTorch 1.8 或更高版本
- CUDA 兼容 GPU（训练时推荐）

### 快速安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/RescueNet-Segmentation.git
cd RescueNet-Segmentation

# 安装依赖
pip install torch torchvision opencv-python albumentations matplotlib tqdm tabulate Pillow
```

### 验证安装

```bash
# 验证数据集配置
python verify_dataset.py

# 运行综合测试
python verify_refactoring.py
```

---

## 📊 数据集

本项目使用 **RescueNet** 数据集进行建筑物损坏分割。

### 数据集结构

```
RescueNet/
├── train/
│   ├── train-org-img/     # 原始图像 (.jpg)
│   └── train-label-img/   # 分割掩码 (_lab.png)
└── test/
    ├── test-org-img/      # 测试图像
    └── test-label-img/    # 测试掩码
```

### 类别分布

| ID | 类别名称 | 颜色 | 描述 |
|----|----------|------|------|
| 0 | 背景 | ⚫ 黑色 | 非建筑物区域 |
| 1 | 水体 | 🔵 蓝色 | 水域 |
| 2 | 建筑物-无损 | ⚪ 浅灰 | 完好建筑物 |
| 3 | 建筑物-轻微损坏 | 🟡 黄色 | 轻度结构损坏 |
| 4 | 建筑物-严重损坏 | 🟠 橙色 | 重度结构损坏 |
| 5 | 建筑物-完全毁坏 | 🔴 红色 | 完全倒塌 |
| 6 | 车辆 | 🔷 青色 | 汽车和卡车 |
| 7 | 道路-畅通 | 🟣 紫灰 | 可通行道路 |
| 8 | 道路-阻塞 | 🟪 深紫 | 不可通行道路 |
| 9 | 树木 | 🟢 绿色 | 植被 |
| 10 | 水池 | 💙 天蓝 | 游泳池 |

**注意**: 文件命名规则：`xxx.jpg` → `xxx_lab.png`

---

## 🚀 快速开始

### 1️⃣ 训练模型

同时训练三个模型：

```bash
python train.py
```

**功能说明：**
- 训练 UNet、PSPNet 和 DeepLabV3
- 保存模型权重到 `runs/` 目录
- 绘制并保存训练损失曲线
- 实时显示训练进度

### 2️⃣ 测试评估

在测试集上评估训练好的模型：

```bash
python test.py
```

**输出内容：**
- 性能对比表格
- 指标：像素准确率、mIoU、Focal Loss、Dice Loss
- 重点关注建筑物损坏类别（2, 3, 4, 5）

### 3️⃣ 推理预测

对单张图像进行预测：

```bash
python inference.py --model unet --weights ./runs/unet_final.pth --image path/to/image.jpg
```

**可选参数：**
```bash
# 带真实标签以计算评估指标
python inference.py \
    --model pspnet \
    --weights ./runs/pspnet_final.pth \
    --image test.jpg \
    --label test_lab.png \
    --output result.png
```

**参数说明：**
- `--model`: 模型类型（`unet` / `pspnet` / `deeplabv3`）
- `--weights`: 模型权重文件路径
- `--image`: 输入图像路径
- `--label`: 真实标签（可选）
- `--output`: 输出文件名（默认：`output.png`）

---

## 🏆 支持的模型

### UNet 🏗️
- **骨干网络**: ResNet18 编码器
- **核心特性**: 
  - 自定义金字塔模块提取多尺度特征
  - SE 注意力机制进行通道级优化
  - 跳跃连接实现精确定位
- **适用场景**: 需要精确边界检测的任务
- **参数量**: ~1930万

### PSPNet 🌆
- **骨干网络**: ResNet18 编码器
- **核心特性**:
  - 金字塔场景解析模块
  - 全局上下文聚合
  - 多层级特征融合
- **适用场景**: 需要全局理解的场景
- **参数量**: ~2520万

### DeepLabV3 🔬
- **骨干网络**: ResNet18 编码器
- **核心特性**:
  - 空洞空间金字塔池化（ASPP）
  - SE 注意力机制
  - 多尺度感受野
- **适用场景**: 多尺度目标分割
- **参数量**: ~1590万

---

## 📈 性能指标

### 评估指标

- **Pixel Accuracy**: 整体分类准确率（过滤建筑物类别）
- **mIoU**: 所有类别的平均交并比
- **Focal Loss**: 训练时处理类别不平衡
- **Dice Loss**: 优化分割质量

### 目标类别

主要关注建筑物损坏评估：
- 类别 2: 建筑物-无损
- 类别 3: 建筑物-轻微损坏
- 类别 4: 建筑物-严重损坏
- 类别 5: 建筑物-完全毁坏

---

## ⚙️ 配置说明

所有配置集中在 `configs/` 目录下。

### 数据集配置（`configs/dataset_config.py`）

```python
DATASET_CONFIG = {
    'train_img_dir': '.../RescueNet/train/train-org-img',
    'train_mask_dir': '.../RescueNet/train/train-label-img',
    'test_img_dir': '.../RescueNet/test/test-org-img',
    'test_mask_dir': '.../RescueNet/test/test-label-img',
}

DATALOADER_CONFIG = {
    'batch_size': 2,
    'num_workers': 6,
    'max_images': 100,  # None表示使用全部图像
}
```

### 训练配置（`configs/training_config.py`）

```python
TRAINING_CONFIG = {
    'epochs': 10,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
}

LOSS_CONFIG = {
    'focal_weight': 1.0,
    'tversky_weight': 1.0,
    'focal_gamma': 2.0,
}
```

### 模型配置（`configs/model_config.py`）

```python
MODEL_CONFIG = {
    'num_classes': 11,
    'backbone': 'resnet18',
    'pretrained': True,
}
```

---

## 🔧 高级用法

### 添加新模型

1. 如需自定义层，在 `models/layers/` 中创建
2. 在 `models/new_model.py` 中实现模型
3. 在 `models/__init__.py` 中注册：
   ```python
   from models.new_model import NewModel
   
   def get_model(model_name, num_classes=11):
       if model_name == 'new_model':
           return NewModel(num_classes=num_classes)
       # ... 现有模型
   ```
4. 在 `configs/model_config.py` 中添加配置

### 添加新数据集

1. 在 `modules/datasets.py` 中创建数据集类
2. 在 `configs/dataset_config.py` 中更新路径
3. 在 `__getitem__()` 中适配文件命名规则

### 自定义损失函数

1. 在 `modules/loss.py` 中实现新损失
2. 在 `modules/trainer.py` 中集成
3. 在 `configs/training_config.py` 中配置权重

---

## 📚 文档

- **[QUICKSTART.md](QUICKSTART.md)**: 5分钟快速入门指南
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)**: 详细架构说明
- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)**: 旧代码迁移指南
- **[REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)**: 重构总结
- **[CHECKLIST.md](CHECKLIST.md)**: 项目完成清单

---

## 🤝 贡献指南

欢迎贡献！请随时提交 Pull Request。

1. Fork 本仓库
2. 创建特性分支（`git checkout -b feature/AmazingFeature`）
3. 提交更改（`git commit -m 'Add some AmazingFeature'`）
4. 推送到分支（`git push origin feature/AmazingFeature`）
5. 开启 Pull Request

---

## 📄 许可证

本项目采用**学术研究许可证**。完整条款请查看 [LICENSE](LICENSE) 文件。

### 快速概览

**✅ 您可以：**
- 将本软件用于学术研究和教育目的
- 根据研究需求修改代码
- 与其他研究人员分享修改版本
- 在出版物中引用本作品

**❌ 您不可以：**
- 将本软件用于商业目的
- 基于此代码申请专利
- 将其用于军事或国防应用
- 移除版权声明或许可条款

**📝 您必须：**
- 在出版物中提供适当的署名
- 在任何分发中包含此许可证
- 适当引用本项目（见下方引用格式）

### 引用格式

如果您在研究中使用了本软件，请引用：

``bibtex
@software{rescuenet_segmentation,
  title = {RescueNet-Segmentation: Building Damage Assessment via Semantic Segmentation},
  author = {{RescueNet-Segmentation Contributors}},
  year = {2026},
  url = {https://github.com/yourusername/RescueNet-Segmentation},
  note = {Academic Research License}
}
```

或者使用纯文本格式：
> 本工作使用了 RescueNet-Segmentation 项目的代码（https://github.com/yourusername/RescueNet-Segmentation），采用学术研究许可证。

---

## 📧 联系方式

如有问题、建议或合作意向：

- 📬 提交 [Issue](https://github.com/yourusername/RescueNet-Segmentation/issues)
- 📧 直接联系作者

---

## 🙏 致谢

- **RescueNet 数据集**: 提供高质量的灾害影像数据
- **PyTorch 团队**: 提供优秀的深度学习框架
- **开源社区**: 为本项目使用的各种库和工具

---

<div align="center">

**如果这个项目对您有帮助，请考虑给个 ⭐ star！**

为灾害救援研究而制作 ❤️

</div>
