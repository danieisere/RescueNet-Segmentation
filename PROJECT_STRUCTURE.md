# 项目架构详解

## 📁 目录结构总览

```
毕业设计代码/
│
├── 📂 data/                      # 数据目录
│   └── 📂 raw/                   # 原始数据
│       ├── 📂 train/             # 训练集（需手动放入数据）
│       └── 📂 test/              # 测试集（需手动放入数据）
│
├── 📂 models/                    # 模型定义模块 ⭐
│   ├── __init__.py              # 模型接口和get_model函数
│   ├── unet.py                  # UNet模型实现
│   ├── pspnet.py                # PSPNet模型实现
│   ├── deeplabv3.py             # DeepLabV3模型实现
│   │
│   └── 📂 layers/               # 自定义网络层
│       ├── __init__.py          # 导出所有层
│       ├── conv_blocks.py       # conv_block, up_conv
│       ├── attention.py         # SEBlock注意力机制
│       ├── pyramid.py           # CustomPyramidModule, PSPModule
│       └── aspp.py              # ASPP模块
│
├── 📂 modules/                   # 核心功能模块 ⭐⭐
│   ├── __init__.py              # 导出所有模块
│   ├── datasets.py              # SegmentationDataset数据集类
│   ├── dataloaders.py           # 数据加载器工厂函数
│   ├── trainer.py               # 训练循环逻辑
│   ├── evaluator.py             # 模型评估逻辑
│   ├── metrics.py               # 评估指标计算
│   ├── loss.py                  # 损失函数（Focal/Dice/Tversky）
│   ├── visualization.py         # 预测结果可视化
│   └── utils.py                 # 工具函数
│
├── 📂 configs/                   # 配置管理 ⭐
│   ├── __init__.py
│   ├── dataset_config.py        # 数据集配置（路径、batch size等）
│   ├── model_config.py          # 模型配置（超参数）
│   └── training_config.py       # 训练配置（学习率、epoch等）
│
├── 📂 packages/                  # 辅助工具包
│   ├── __init__.py
│   └── argument_parser.py       # 命令行参数解析
│
├── 📂 runs/                      # 训练输出（自动生成）
│   └── *.pth                    # 保存的模型权重
│
├── 📂 output/                    # 推理输出（自动生成）
│   └── *.png                    # 预测结果图像
│
├── 📄 train.py                   # 🚀 训练入口脚本
├── 📄 test.py                    # 🚀 测试入口脚本
├── 📄 inference.py               # 🚀 推理入口脚本
├── 📄 config.py                  # 全局配置（向后兼容）
│
├── 📄 requirements.txt           # Python依赖
├── 📄 .gitignore                 # Git忽略文件
│
└── 📄 README.md                  # 📖 项目说明文档
📄 QUICKSTART.md                  # 📖 快速开始指南
📄 MIGRATION_GUIDE.md             # 📖 迁移指南
📄 PROJECT_STRUCTURE.md           # 📖 本文件
│
└── 📂 code/                      # 旧代码（已废弃，可删除）
    ├── dataset.py
    ├── model.py
    ├── train.py
    ├── predict.py
    └── main_detection.py
```

## 🎯 核心设计理念

### 1. 职责分离原则

| 模块 | 职责 | 示例 |
|------|------|------|
| `models/` | **只负责**网络架构定义 | UNet、PSPNet的前向传播 |
| `modules/trainer.py` | **只负责**训练流程 | 优化器、反向传播、损失记录 |
| `modules/evaluator.py` | **只负责**评估逻辑 | 计算mIoU、Accuracy |
| `modules/datasets.py` | **只负责**数据加载 | 读取图像、数据增强 |
| `configs/` | **只负责**参数配置 | 学习率、batch size |

**优势**: 
- ✅ 修改模型不影响训练逻辑
- ✅ 调整超参数只需改配置文件
- ✅ 添加新数据集只需修改datasets.py

### 2. 配置驱动设计

所有可变参数都集中在 `configs/` 目录：

```python
# configs/dataset_config.py
DATASET_CONFIG = {
    'train_img_dir': '../RescueNet/train/train-org-img',
    'batch_size': 2,
}

# configs/training_config.py
TRAINING_CONFIG = {
    'epochs': 10,
    'learning_rate': 1e-4,
}
```

**使用方式**:
```python
from configs.dataset_config import DATASET_CONFIG

# 直接使用，无需硬编码
batch_size = DATASET_CONFIG['batch_size']
```

### 3. 统一接口设计

#### 模型加载接口
```python
from models import get_model

# 一行代码加载任意模型
model = get_model('unet')      # UNet
model = get_model('pspnet')    # PSPNet
model = get_model('deeplabv3') # DeepLabV3
```

#### 训练接口
```python
from modules.trainer import train

# 统一的训练接口
history = train(
    model=model,
    dataloader=dataloader,
    device=device,
    epochs=10
)
```

#### 评估接口
```python
from modules.evaluator import evaluate_model

# 统一的评估接口
metrics = evaluate_model(model, dataloader, device)
```

## 🔧 模块详细说明

### models/ - 模型定义

#### 层次结构
```
models/
├── 顶层: 完整模型（UNet, PSPNet, DeepLabV3）
└── layers/: 可复用的基础组件
    ├── conv_blocks.py: 卷积块
    ├── attention.py: 注意力机制
    ├── pyramid.py: 金字塔池化
    └── aspp.py: 空洞空间金字塔
```

#### 设计模式
- **组合模式**: 模型由多个layer组合而成
- **工厂模式**: `get_model()` 根据名称创建模型
- **继承模式**: 所有模型继承自 `nn.Module`

#### 扩展示例
```python
# 添加新模型: models/new_model.py
import torch.nn as nn
from models.layers.conv_blocks import conv_block

class NewModel(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        # 定义网络结构
    
    def forward(self, x):
        # 定义前向传播
        return output

# 在 models/__init__.py 中注册
from models.new_model import NewModel

def get_model(model_name, num_classes=11):
    if model_name == 'new_model':
        return NewModel(num_classes=num_classes)
    # ... 其他模型
```

### modules/ - 功能模块

#### datasets.py - 数据加载
```python
class SegmentationDataset(Dataset):
    """
    职责:
    1. 读取图像和标签
    2. 应用数据增强
    3. 转换为Tensor格式
    
    特点:
    - 支持RescueNet命名规则 (xxx.jpg → xxx_lab.png)
    - 可自定义transform
    - 支持限制数据量 (max_images)
    """
```

#### trainer.py - 训练逻辑
```python
def train(model, dataloader, device, epochs, ...):
    """
    职责:
    1. 初始化优化器和调度器
    2. 执行训练循环
    3. 记录损失历史
    4. 保存模型权重
    
    特点:
    - 支持多GPU (DataParallel)
    - 自动学习率调整
    - 进度条显示 (tqdm)
    """
```

#### loss.py - 损失函数
```python
class FocalLoss(nn.Module):
    """处理类别不平衡"""
    
class DiceLoss(nn.Module):
    """优化分割边界"""
    
class TverskyLoss(nn.Module):
    """Focal + Dice的泛化"""

def get_class_weights():
    """基于像素统计计算类别权重"""
```

#### metrics.py - 评估指标
```python
def pixel_accuracy_filtered(pred, label, include_classes):
    """计算指定类别的像素准确率"""
    
def compute_miou(pred, label, num_classes, include_classes):
    """计算平均交并比"""
```

### configs/ - 配置管理

#### 三层配置体系

1. **dataset_config.py**: 数据相关
   - 数据路径
   - batch size
   - 类别定义

2. **model_config.py**: 模型相关
   -  backbone选择
   -  金字塔池化参数
   -  ASPP参数

3. **training_config.py**: 训练相关
   - 学习率
   - epoch数
   - 损失权重

#### 配置优先级
```
命令行参数 > 配置文件 > 代码默认值
```

## 📊 数据流向

### 训练流程
```
1. 数据准备
   data/raw/train/
   ├── train-org-img/*.jpg
   └── train-label-img/*_lab.png
   ↓
2. 数据加载 (modules/datasets.py)
   SegmentationDataset.__getitem__()
   - 读取图像
   - 数据增强
   - 转Tensor
   ↓
3. 批次打包 (torch.utils.data.DataLoader)
   DataLoader → batch of (images, labels)
   ↓
4. 模型前向传播 (models/unet.py)
   images → model → predictions
   ↓
5. 损失计算 (modules/loss.py)
   predictions + labels → loss
   ↓
6. 反向传播 (modules/trainer.py)
   loss.backward() → optimizer.step()
   ↓
7. 保存模型 (runs/*.pth)
```

### 推理流程
```
1. 加载模型
   torch.load('runs/unet_final.pth')
   ↓
2. 预处理图像 (albumentations)
   image → resize → normalize → tensor
   ↓
3. 模型推理
   tensor → model → predictions
   ↓
4. 后处理
   predictions → argmax → class map
   ↓
5. 可视化 (modules/visualization.py)
   class map → color palette → image
   ↓
6. 保存结果 (output/*.png)
```

## 🚀 扩展指南

### 场景1: 添加新模型

**步骤**:
1. 在 `models/layers/` 中实现所需的新层（如果需要）
2. 创建 `models/my_model.py`
3. 在 `models/__init__.py` 中注册
4. 在 `configs/model_config.py` 中添加配置

**代码量**: ~50-100行

### 场景2: 添加新数据集

**步骤**:
1. 在 `modules/datasets.py` 中创建新的Dataset类
2. 在 `configs/dataset_config.py` 中添加路径
3. 修改数据加载逻辑适配新命名规则

**代码量**: ~30-50行

### 场景3: 添加新损失函数

**步骤**:
1. 在 `modules/loss.py` 中实现新Loss类
2. 在 `modules/trainer.py` 中集成
3. 在 `configs/training_config.py` 中配置权重

**代码量**: ~20-30行

### 场景4: 添加新评估指标

**步骤**:
1. 在 `modules/metrics.py` 中实现新指标函数
2. 在 `modules/evaluator.py` 中调用
3. 更新输出格式

**代码量**: ~10-20行

## 💡 最佳实践

### 1. 配置管理
```python
# ✅ 推荐: 从配置文件读取
from configs.training_config import TRAINING_CONFIG
lr = TRAINING_CONFIG['learning_rate']

# ❌ 避免: 硬编码
lr = 1e-4
```

### 2. 模型使用
```python
# ✅ 推荐: 使用统一接口
from models import get_model
model = get_model('unet')

# ❌ 避免: 直接导入具体类
from models.unet import UNet
model = UNet()
```

### 3. 数据处理
```python
# ✅ 推荐: 使用DataLoader工厂
from modules.dataloaders import create_dataloader
loader = create_dataloader(img_dir, mask_dir)

# ❌ 避免: 手动创建DataLoader
dataset = SegmentationDataset(...)
loader = DataLoader(dataset, ...)
```

### 4. 错误处理
```python
# ✅ 推荐: 检查文件存在性
if not os.path.exists(weight_path):
    raise FileNotFoundError(f"权重文件不存在: {weight_path}")

# ❌ 避免: 直接加载可能失败的文件
model.load_state_dict(torch.load(weight_path))
```

## 🔍 调试技巧

### 1. 验证数据加载
```python
from modules.datasets import SegmentationDataset

dataset = SegmentationDataset(img_dir, mask_dir)
image, mask = dataset[0]
print(f"Image shape: {image.shape}")  # 应为 (3, 768, 768)
print(f"Mask shape: {mask.shape}")    # 应为 (768, 768)
print(f"Unique classes: {torch.unique(mask)}")
```

### 2. 验证模型输出
```python
from models import get_model
import torch

model = get_model('unet')
x = torch.randn(1, 3, 768, 768)
y = model(x)
print(f"Output shape: {y.shape}")  # 应为 (1, 11, 768, 768)
```

### 3. 监控训练过程
```python
# 在 trainer.py 中添加打印
print(f"Batch {i}: Loss = {loss.item():.4f}")
print(f"LR: {optimizer.param_groups[0]['lr']}")
```

## 📈 性能优化建议

### 1. 数据加载优化
```python
# 增加num_workers
DATALOADER_CONFIG = {
    'num_workers': 8,  # 根据CPU核心数调整
    'pin_memory': True, # 加速GPU传输
}
```

### 2. 混合精度训练
```python
# 使用AMP (Automatic Mixed Precision)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    preds = model(images)
    loss = criterion(preds, labels)
scaler.scale(loss).backward()
```

### 3. 梯度累积
```python
# 模拟更大的batch size
accumulation_steps = 4
for i, (images, labels) in enumerate(dataloader):
    loss = compute_loss(images, labels)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 🎓 学习路线

### 初级使用者
1. 阅读 `QUICKSTART.md`
2. 运行 `python train.py`
3. 修改 `configs/` 中的参数

### 中级使用者
1. 阅读 `README.md` 了解架构
2. 查看 `models/` 理解模型实现
3. 尝试添加简单的自定义层

### 高级使用者
1. 深入 `modules/` 理解训练流程
2. 实现新的损失函数或评估指标
3. 贡献新的模型架构

---

**提示**: 建议按以下顺序阅读代码：
1. `configs/*.py` - 了解配置
2. `models/layers/*.py` - 了解基础组件
3. `models/*.py` - 了解完整模型
4. `modules/datasets.py` - 了解数据流
5. `modules/trainer.py` - 了解训练逻辑
6. `train.py` - 了解整体流程
