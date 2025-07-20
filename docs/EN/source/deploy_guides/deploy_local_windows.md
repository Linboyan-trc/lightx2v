# Windows Local Deployment Guide

## 📖 Overview

This document provides detailed instructions for deploying LightX2V locally on Windows environments, including batch file inference, Gradio Web interface inference, and other usage methods.

## 🚀 Quick Start

### Environment Requirements

#### Hardware Requirements
- **GPU**: NVIDIA GPU, recommended 8GB+ VRAM
- **Memory**: Recommended 16GB+ RAM
- **Storage**: Strongly recommended to use SSD solid-state drives, mechanical hard drives will cause slow model loading

#### Software Requirements
- **Operating System**: Windows 10/11
- **Python**: 3.12 or higher version
- **CUDA**: 12.4 or higher version
- **Dependencies**: Refer to LightX2V project's requirements_win.txt

### Installation Steps

1. **Clone Project**
```cmd
git clone https://github.com/ModelTC/LightX2V.git
cd LightX2V
```

2. **Install Dependencies**
```cmd
pip install -r requirements_win.txt
```

3. **Download Models**
Refer to [Model Download Guide](../getting_started/quickstart.md) to download required models

## 🎯 Usage Methods

### Method 1: Using Batch File Inference

Refer to [Quick Start Guide](../getting_started/quickstart.md) to install environment, and use [batch files](https://github.com/ModelTC/LightX2V/tree/main/scripts/win) to run.

### Method 2: Using Gradio Web Interface Inference

#### Manual Gradio Configuration

Refer to [Quick Start Guide](../getting_started/quickstart.md) to install environment, refer to [Gradio Deployment Guide](./deploy_gradio.md)

#### One-Click Gradio Startup (Recommended)

**📦 Download Software Package**
- [Baidu Cloud](https://pan.baidu.com/s/1ef3hEXyIuO0z6z9MoXe4nQ?pwd=7g4f)
- [Quark Cloud](https://pan.quark.cn/s/36a0cdbde7d9)

**📁 Directory Structure**
After extraction, ensure the directory structure is as follows:

```
├── env/                        # LightX2V environment directory
├── LightX2V/                   # LightX2V project directory
├── start_lightx2v.bat          # One-click startup script
├── lightx2v_config.txt         # Configuration file
├── LightX2V使用说明.txt         # LightX2V usage instructions
└── models/                     # Model storage directory
    ├── 说明.txt                       # Model documentation
    ├── Wan2.1-I2V-14B-480P-Lightx2v/  # Image-to-video model (480P)
    ├── Wan2.1-I2V-14B-720P-Lightx2v/  # Image-to-video model (720P)
    ├── Wan2.1-I2V-14B-480P-StepDistill-CfgDistil-Lightx2v/  # Image-to-video model (4-step distillation, 480P)
    ├── Wan2.1-I2V-14B-720P-StepDistill-CfgDistil-Lightx2v/  # Image-to-video model (4-step distillation, 720P)
    ├── Wan2.1-T2V-1.3B-Lightx2v/      # Text-to-video model (1.3B parameters)
    ├── Wan2.1-T2V-14B-Lightx2v/       # Text-to-video model (14B parameters)
    └── Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v/      # Text-to-video model (4-step distillation)
```

**📋 Configuration Parameters**

Edit the `lightx2v_config.txt` file and modify the following parameters as needed:

```ini
# Task type (i2v: image-to-video, t2v: text-to-video)
task=i2v

# Interface language (zh: Chinese, en: English)
lang=en

# Server port
port=8032

# GPU device ID (0, 1, 2...)
gpu=0

# Model size (14b: 14B parameter model, 1.3b: 1.3B parameter model)
model_size=14b

# Model class (wan2.1: standard model, wan2.1_distill: distilled model)
model_cls=wan2.1
```

**⚠️ Important Note**: If using distilled models (model names containing StepDistill-CfgDistil field), please set `model_cls` to `wan2.1_distill`

**🚀 Start Service**

Double-click to run the `start_lightx2v.bat` file, the script will:
1. Automatically read configuration file
2. Verify model paths and file integrity
3. Start Gradio Web interface
4. Automatically open browser to access service

**💡 Usage Suggestion**: After opening the Gradio Web page, it's recommended to check "Auto-configure Inference Options", the system will automatically select appropriate optimization configurations for your machine. When reselecting resolution, you also need to re-check "Auto-configure Inference Options".

**⚠️ Important Note**: On first run, the system will automatically extract the environment file `env.zip`, which may take several minutes. Please be patient. Subsequent launches will skip this step. You can also manually extract the `env.zip` file to the current directory to save time on first startup.

### Method 3: Using ComfyUI Inference

TODO - To be added ComfyUI integration guide
