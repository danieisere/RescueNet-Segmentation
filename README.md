<div align="center">

# 🏗️ RescueNet-Segmentation

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-red?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Academic-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research-yellow)]()

[📖 中文文档](README_cn.md) | [🌍 English](README.md)

</div>

---

## 📋 Overview

**RescueNet-Segmentation** is a semantic segmentation research project for building damage assessment based on the RescueNet dataset. This project implements and compares three state-of-the-art deep learning models: **UNet**, **PSPNet**, and **DeepLabV3** for disaster scene understanding.

### 🎯 Key Features

- ✅ **Multi-Model Comparison**: UNet, PSPNet, DeepLabV3 with ResNet18 backbone
- ✅ **Modular Architecture**: Clean separation of models, training, and evaluation
- ✅ **Advanced Loss Functions**: Focal Loss + Tversky Loss for class imbalance
- ✅ **Complete Pipeline**: Training → Testing → Inference → Visualization
- ✅ **Configuration-Driven**: Centralized config system for easy customization
- ✅ **Production-Ready**: Well-documented and extensible codebase

### 🏛️ Application Scenarios

- Post-disaster building damage assessment
- Emergency response planning
- Urban resilience analysis
- Remote sensing image interpretation

---

## 📁 Project Structure

```
RescueNet-Segmentation/
├── 📊 RescueNet/                 # Dataset
│   ├── train/
│   │   ├── train-org-img/        # 100 training images
│   │   └── train-label-img/      # 100 training labels
│   └── test/
│       ├── test-org-img/         # 1 test image
│       └── test-label-img/       # 1 test label
│
├── 🧠 models/                    # Model definitions
│   ├── unet.py                   # UNet with Custom Pyramid Module
│   ├── pspnet.py                 # PSPNet with Scene Parsing
│   ├── deeplabv3.py              # DeepLabV3 with ASPP
│   └── layers/                   # Custom network layers
│       ├── conv_blocks.py        # Basic convolution blocks
│       ├── attention.py          # SE Attention mechanism
│       ├── pyramid.py            # Pyramid modules
│       └── aspp.py               # Atrous Spatial Pyramid Pooling
│
├── ⚙️ modules/                   # Core functionality
│   ├── datasets.py               # Dataset loader
│   ├── dataloaders.py            # DataLoader factory
│   ├── trainer.py                # Training engine
│   ├── evaluator.py              # Evaluation logic
│   ├── metrics.py                # Performance metrics
│   ├── loss.py                   # Loss functions
│   ├── visualization.py          # Result visualization
│   └── utils.py                  # Utility functions
│
├── 📝 configs/                   # Configuration files
│   ├── dataset_config.py         # Dataset paths & params
│   ├── model_config.py           # Model hyperparameters
│   └── training_config.py        # Training settings
│
├── 🚀 Entry Points
│   ├── train.py                  # Train all models
│   ├── test.py                   # Evaluate models
│   └── inference.py              # Single image prediction
│
├── 📦 runs/                      # Trained model weights
├── 🖼️ output/                    # Prediction results
└── 📚 Documentation
    ├── README.md                 # English documentation
    ├── README_cn.md              # Chinese documentation
    ├── QUICKSTART.md             # Quick start guide
    ├── PROJECT_STRUCTURE.md      # Architecture details
    └── MIGRATION_GUIDE.md        # Migration guide
```

---

## 🛠️ Installation

### Prerequisites

- Python 3.7 or higher
- PyTorch 1.8 or higher
- CUDA-compatible GPU (recommended for training)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/danieisere/RescueNet-Segmentation.git
cd RescueNet-Segmentation

# Install dependencies
pip install torch torchvision opencv-python albumentations matplotlib tqdm tabulate Pillow
```

### Verify Installation

```bash
# Verify dataset configuration
python verify_dataset.py

# Run comprehensive tests
python verify_refactoring.py
```

---

## 📊 Dataset

This project uses the **RescueNet** dataset for building damage segmentation.

### Dataset Structure

```
RescueNet/
├── train/
│   ├── train-org-img/     # Original images (.jpg)
│   └── train-label-img/   # Segmentation masks (_lab.png)
└── test/
    ├── test-org-img/      # Test images
    └── test-label-img/    # Test masks
```

### Class Distribution

| ID | Class Name | Color | Description |
|----|------------|-------|-------------|
| 0 | Background | ⚫ Black | Non-building areas |
| 1 | Water | 🔵 Blue | Water bodies |
| 2 | Building - No Damage | ⚪ Light Gray | Intact buildings |
| 3 | Building - Minor Damage | 🟡 Yellow | Slight structural damage |
| 4 | Building - Major Damage | 🟠 Orange | Severe structural damage |
| 5 | Building - Destroyed | 🔴 Red | Completely collapsed |
| 6 | Vehicle | 🔷 Cyan | Cars and trucks |
| 7 | Road - Clear | 🟣 Purple-Gray | Passable roads |
| 8 | Road - Blocked | 🟪 Dark Purple | Impassable roads |
| 9 | Tree | 🟢 Green | Vegetation |
| 10 | Pool | 💙 Sky Blue | Swimming pools |

**Note**: File naming convention: `xxx.jpg` → `xxx_lab.png`

---

## 🚀 Quick Start

### 1️⃣ Training

Train all three models simultaneously:

```bash
python train.py
```

**What it does:**
- Trains UNet, PSPNet, and DeepLabV3
- Saves model weights to `runs/` directory
- Plots and saves training loss curves
- Displays real-time progress

### 2️⃣ Testing & Evaluation

Evaluate trained models on test set:

```bash
python test.py
```

**Output:**
- Performance comparison table
- Metrics: Pixel Accuracy, mIoU, Focal Loss, Dice Loss
- Focus on building damage classes (2, 3, 4, 5)

### 3️⃣ Inference

Predict on a single image:

```bash
python inference.py --model unet --weights ./runs/unet_final.pth --image path/to/image.jpg
```

**Optional parameters:**
```bash
# With ground truth label for metric calculation
python inference.py \
    --model pspnet \
    --weights ./runs/pspnet_final.pth \
    --image test.jpg \
    --label test_lab.png \
    --output result.png
```

**Arguments:**
- `--model`: Model type (`unet` / `pspnet` / `deeplabv3`)
- `--weights`: Path to model weights
- `--image`: Input image path
- `--label`: Ground truth label (optional)
- `--output`: Output filename (default: `output.png`)

---

## 🏆 Supported Models

### UNet 🏗️
- **Backbone**: ResNet18 encoder
- **Key Features**: 
  - Custom Pyramid Module for multi-scale features
  - SE Attention for channel-wise refinement
  - Skip connections for precise localization
- **Best For**: Tasks requiring accurate boundary detection

### PSPNet 🌆
- **Backbone**: ResNet18 encoder
- **Key Features**:
  - Pyramid Scene Parsing Module
  - Global context aggregation
  - Multi-level feature fusion
- **Best For**: Scenes requiring global understanding

### DeepLabV3 🔬
- **Backbone**: ResNet18 encoder
- **Key Features**:
  - Atrous Spatial Pyramid Pooling (ASPP)
  - SE Attention mechanism
  - Multi-scale receptive fields
- **Best For**: Multi-scale object segmentation

---

## 📈 Performance Metrics

### Evaluation Metrics

- **Pixel Accuracy**: Overall classification accuracy (filtered for building classes)
- **mIoU**: Mean Intersection over Union across all classes
- **Focal Loss**: Handles class imbalance during training
- **Dice Loss**: Optimizes segmentation quality

### Target Classes

Primary focus on building damage assessment:
- Class 2: Building - No Damage
- Class 3: Building - Minor Damage
- Class 4: Building - Major Damage
- Class 5: Building - Destroyed

---

## ⚙️ Configuration

All configurations are centralized in the `configs/` directory.

### Dataset Configuration (`configs/dataset_config.py`)

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
    'max_images': 100,  # None for all images
}
```

### Training Configuration (`configs/training_config.py`)

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

### Model Configuration (`configs/model_config.py`)

```python
MODEL_CONFIG = {
    'num_classes': 11,
    'backbone': 'resnet18',
    'pretrained': True,
}
```

---

## 🔧 Advanced Usage

### Adding a New Model

1. Create custom layers in `models/layers/` if needed
2. Implement model in `models/new_model.py`
3. Register in `models/__init__.py`:
   ```python
   from models.new_model import NewModel
   
   def get_model(model_name, num_classes=11):
       if model_name == 'new_model':
           return NewModel(num_classes=num_classes)
       # ... existing models
   ```
4. Add configuration in `configs/model_config.py`

### Adding a New Dataset

1. Create dataset class in `modules/datasets.py`
2. Update paths in `configs/dataset_config.py`
3. Adapt file naming convention in `__getitem__()`

### Customizing Loss Functions

1. Implement new loss in `modules/loss.py`
2. Integrate in `modules/trainer.py`
3. Configure weights in `configs/training_config.py`

---

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)**: 5-minute quick start guide
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)**: Detailed architecture explanation
- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)**: Guide for migrating from old code
- **[REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)**: Refactoring overview
- **[CHECKLIST.md](CHECKLIST.md)**: Project completion checklist

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **Academic Research License**. See the [LICENSE](LICENSE) file for full details.

### Quick Summary

**✅ You CAN:**
- Use this software for academic research and education
- Modify the code for your research needs
- Share modified versions with other researchers
- Cite this work in your publications

**❌ You CANNOT:**
- Use this software for commercial purposes
- Apply for patents based on this code
- Use it in military or defense applications
- Remove copyright notices or license terms

**📝 You MUST:**
- Provide proper attribution in publications
- Include this license with any distribution
- Cite the project appropriately (see citation format below)

### Citation

If you use this software in your research, please cite:

``bibtex
@software{rescuenet_segmentation,
  title = {RescueNet-Segmentation: Building Damage Assessment via Semantic Segmentation},
  author = {{RescueNet-Segmentation Contributors}},
  year = {2026},
  url = {https://github.com/danieisere/RescueNet-Segmentation},
  note = {Academic Research License}
}
```

---

## 📧 Contact

For questions, suggestions, or collaborations:

- 📬 Submit an [Issue](https://github.com/danieisere/RescueNet-Segmentation/issues)
- 📧 Contact the author directly

---

## 🙏 Acknowledgments

- **RescueNet Dataset**: For providing high-quality disaster imagery data
- **PyTorch Team**: For the excellent deep learning framework
- **Open Source Community**: For various libraries and tools used in this project

---

<div align="center">

**If you find this project helpful, please consider giving it a ⭐ star!**

Made with ❤️ for disaster relief research

</div>
