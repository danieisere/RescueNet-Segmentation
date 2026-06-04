# 项目重构迁移指南

本文档说明如何将旧的 `archive_code/` 目录下的代码迁移到新的模块化架构。

## 文件映射关系

### 原文件 → 新位置

| 原文件 | 新位置 | 说明 |
|--------|--------|------|
| `archive_code/model.py` | `models/layers/conv_blocks.py` | conv_block, up_conv |
| `archive_code/model.py` | `models/layers/attention.py` | SEBlock |
| `archive_code/model.py` | `models/layers/pyramid.py` | CustomPyramidModule, PSPModule |
| `archive_code/model.py` | `models/layers/aspp.py` | ASPP |
| `archive_code/model.py` | `models/unet.py` | UNet模型 |
| `archive_code/model.py` | `models/pspnet.py` | PSPNet模型 |
| `archive_code/model.py` | `models/deeplabv3.py` | DeepLabV3模型 |
| `archive_code/dataset.py` | `modules/datasets.py` | SegmentationDataset |
| `archive_code/train.py` (loss部分) | `modules/loss.py` | FocalLoss, DiceLoss, TverskyLoss |
| `archive_code/train.py` (train函数) | `modules/trainer.py` | train函数 |
| `archive_code/predict.py` (metrics部分) | `modules/metrics.py` | pixel_accuracy_filtered, compute_miou |
| `archive_code/predict.py` (visualization部分) | `modules/visualization.py` | predict_and_draw |
| `archive_code/main_detection.py` | `train.py` + `test.py` | 训练和测试主入口 |

## 主要改进

### 1. 模块化设计

**之前**: 所有代码集中在5个文件中，职责不清  
**现在**: 
- 模型定义 (`models/`)
- 训练逻辑 (`modules/trainer.py`)
- 评估逻辑 (`modules/evaluator.py`)
- 数据处理 (`modules/datasets.py`)
- 配置管理 (`configs/`)

### 2. 配置集中化

**之前**: 超参数硬编码在各个函数中  
**现在**: 所有配置集中在 `configs/` 目录下
- `dataset_config.py`: 数据路径、批大小等
- `model_config.py`: 模型参数
- `training_config.py`: 学习率、epoch数等

### 3. 接口标准化

**之前**: 各函数参数不统一  
**现在**: 
- 统一的模型加载接口: `get_model(model_name)`
- 统一的数据加载接口: `create_dataloader(...)`
- 统一的训练接口: `train(model, dataloader, ...)`
- 统一的评估接口: `evaluate_model(model, dataloader, ...)`

### 4. 可扩展性提升

**添加新模型**:
```python
# 1. 在 models/ 下创建 new_model.py
# 2. 在 models/__init__.py 中注册
from models.new_model import NewModel

def get_model(model_name, num_classes=11):
    if model_name == 'new_model':
        return NewModel(num_classes=num_classes)
    # ... 其他模型
```

**添加新数据集**:
```python
# 在 modules/datasets.py 中添加新的Dataset类
class NewDataset(Dataset):
    def __init__(self, img_dir, mask_dir, ...):
        # 实现新的数据加载逻辑
```

## 使用示例对比

### 训练模型

**旧方式**:
```python
from archive_code.model import get_model
from archive_code.dataset import SegmentationDataset
from archive_code.train import train

# 需要手动配置所有参数
model = get_model('unet')
dataset = SegmentationDataset(img_dir, mask_dir, max_images=100)
# ... 手动创建DataLoader
train(model, dataloader, device, epochs=10)
```

**新方式**:
```python
from models import get_model
from modules.datasets import SegmentationDataset
from modules.trainer import train
from configs.dataset_config import DATASET_CONFIG, DATALOADER_CONFIG

# 使用配置，更简洁
model = get_model('unet')
dataset = SegmentationDataset(
    DATASET_CONFIG['train_img_dir'],
    DATASET_CONFIG['train_mask_dir'],
    DATALOADER_CONFIG['max_images']
)
train(model, dataloader, device, epochs=TRAINING_CONFIG['epochs'])
```

或者直接运行:
```bash
python train.py
```

### 推理预测

**旧方式**:
```python
from code.predict import predict_and_draw
# 需要了解所有参数细节
predict_and_draw(model, image_path, device, ...)
```

**新方式**:
```bash
python inference.py --model unet --weights ./runs/unet_final.pth --image test.jpg
```

## 兼容性说明

为了保持向后兼容，保留了以下文件：
- `config.py`: 从configs导入所有配置
- `archive_code/` 目录保持不变（可以删除或归档）

如果需要完全迁移，可以：
1. 删除 `archive_code/` 目录
2. 更新所有import语句
3. 使用新的配置文件

## 常见问题

### Q1: 如何修改数据路径？
A: 编辑 `configs/dataset_config.py` 中的 `DATASET_CONFIG` 字典

### Q2: 如何调整学习率？
A: 编辑 `configs/training_config.py` 中的 `TRAINING_CONFIG['learning_rate']`

### Q3: 如何添加自定义数据增强？
A: 在 `modules/datasets.py` 的 `SegmentationDataset.__init__()` 中修改 `self.transform`

### Q4: 如何使用不同的损失函数组合？
A: 修改 `modules/trainer.py` 中的损失计算逻辑，并在 `configs/training_config.py` 中配置权重

## 下一步计划

1. ✅ 完成基础架构重构
2. ⏳ 添加TensorBoard日志支持
3. ⏳ 添加模型导出功能（ONNX格式）
4. ⏳ 添加数据可视化预览工具
5. ⏳ 添加自动超参数调优

## 技术支持

如遇到迁移问题，请检查：
1. Python版本是否 >= 3.7
2. 依赖包是否正确安装 (`pip install -r requirements.txt`)
3. 数据路径配置是否正确
4. 模型权重文件是否存在

---

**注意**: 建议在迁移前先备份原有代码，确保可以回滚。
