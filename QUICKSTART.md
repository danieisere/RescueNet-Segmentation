# 快速开始指南

## 5分钟上手

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

确保RescueNet数据集按以下结构放置：

```
data/raw/
├── train/
│   ├── train-org-img/     # 放入训练图像 (.jpg)
│   └── train-label-img/   # 放入训练标签 (_lab.png)
└── test/
    ├── test-org-img/      # 放入测试图像
    └── test-label-img/    # 放入测试标签
```

**重要**: 标签文件命名必须是 `xxx_lab.png` 格式

### 3. 配置路径

编辑 `configs/dataset_config.py`，确认数据路径正确：

```python
DATASET_CONFIG = {
    'train_img_dir': '../RescueNet/train/train-org-img',  # 修改为你的实际路径
    'train_mask_dir': '../RescueNet/train/train-label-img',
    'test_img_dir': '../RescueNet/test/test-org-img',
    'test_mask_dir': '../RescueNet/test/test-label-img',
}
```

### 4. 开始训练

```bash
python train.py
```

这将：
- 训练UNet、PSPNet、DeepLabV3三个模型
- 每个模型训练10个epoch（可配置）
- 自动保存模型权重到 `runs/` 目录
- 绘制并保存损失曲线到 `output/损失函数曲线.png`

### 5. 评估模型

```bash
python test.py
```

输出类似：
```
======================================================================
模型评估结果汇总
======================================================================
+-----------+----------+--------+------------+-----------+
| Model     | Accuracy | mIoU   | Focal Loss | Dice Loss |
+===========+==========+========+============+===========+
| UNet      | 0.8523   | 0.6234 | 0.3421     | 0.2156    |
+-----------+----------+--------+------------+-----------+
| DeepLabV3 | 0.8612   | 0.6412 | 0.3298     | 0.2087    |
+-----------+----------+--------+------------+-----------+
| PSPNet    | 0.8489   | 0.6189 | 0.3456     | 0.2178    |
+-----------+----------+--------+------------+-----------+
```

### 6. 单张图像推理

```bash
python inference.py --model deeplabv3 --weights ./runs/deeplabv3_final.pth --image path/to/test.jpg
```

可选参数：
```bash
# 如果有标签，可以计算评估指标
python inference.py --model unet --weights ./runs/unet_final.pth \
    --image test.jpg --label test_lab.png --output result.png
```

## 自定义训练

### 只训练特定模型

修改 `train.py` 中的 `models_dict`：

```python
models_dict = {
    "UNet": get_model('unet').to(device),
    # 注释掉不需要的模型
    # "DeepLabV3": get_model('deeplabv3').to(device),
    # "PSPNet": get_model('pspnet').to(device)
}
```

### 调整超参数

编辑 `configs/training_config.py`：

```python
TRAINING_CONFIG = {
    'epochs': 20,              # 改为20个epoch
    'learning_rate': 5e-5,     # 调整学习率
    'target_classes': [2, 3, 4, 5],
}
```

### 使用更多训练数据

编辑 `configs/dataset_config.py`：

```python
DATALOADER_CONFIG = {
    'batch_size': 4,           # 增大batch size（如果显存允许）
    'max_images': None,        # None表示使用全部数据
}
```

## 常见问题排查

### 问题1: CUDA out of memory

**解决方案**:
```python
# 在 configs/dataset_config.py 中减小batch size
DATALOADER_CONFIG = {
    'batch_size': 1,  # 从2减到1
}
```

### 问题2: 找不到数据文件

**检查清单**:
- [ ] 数据路径在 `configs/dataset_config.py` 中配置正确
- [ ] 图像文件是 `.jpg` 或 `.png` 格式
- [ ] 标签文件命名为 `xxx_lab.png`
- [ ] 路径使用正斜杠 `/` 或双反斜杠 `\\`

### 问题3: 导入错误

**解决方案**:
```bash
# 确保在项目根目录运行
cd "d:\Desktop\Personal Project\...\毕业设计代码"
python train.py
```

### 问题4: 模型加载失败

**检查**:
```python
# 确认权重文件存在
import os
print(os.path.exists('./runs/unet_final.pth'))

# 如果不存在，先运行训练
python train.py
```

## 进阶使用

### 查看项目结构

```bash
# Windows PowerShell
Get-ChildItem -Recurse -Directory | Select-Object FullName

# 或使用tree命令
tree /F
```

### 监控训练过程

训练时会显示进度条：
```
Epoch 1/10: 100%|████████████| 50/50 [05:23<00:00, loss=0.4521]
✅ UNet Epoch 1/10 - 平均 Loss: 0.4521
```

### 自定义评估类别

默认只评估建筑物相关类别 [2, 3, 4, 5]，如需评估所有类别：

```python
# 在 configs/dataset_config.py 中修改
TARGET_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

## 下一步

- 📖 阅读 `README.md` 了解完整功能
- 🔧 查看 `MIGRATION_GUIDE.md` 了解重构细节
- 💡 探索 `models/` 和 `modules/` 目录了解实现细节
- 🎯 尝试添加自己的模型或数据集

## 获取帮助

遇到问题？
1. 检查 `README.md` 的常见问题部分
2. 查看 `MIGRATION_GUIDE.md` 的故障排查
3. 检查控制台错误信息
4. 联系项目维护者

---

祝研究顺利！🚀
