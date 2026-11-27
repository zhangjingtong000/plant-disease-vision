# -*- coding: utf-8 -*-
"""
config.py — 全局配置
"""
import os
import json
import io

# ---------------- 路径 ----------------
# 用文件位置推导项目根目录，保证整个文件夹拷到任何地方都能运行
# src/config.py -> 上一级即项目根
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
SPLITS_JSON = os.path.join(DATA_DIR, "splits.json")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
# 发布包里模型放在 models/，开发目录里放在 outputs/checkpoints/，自动适配
_MODELS_DIR = os.path.join(BASE_DIR, "models")
_has_ckpt = os.path.isdir(_MODELS_DIR) and any(
    f.endswith("_best.pth") for f in os.listdir(_MODELS_DIR)
)
CKPT_DIR = _MODELS_DIR if _has_ckpt else os.path.join(OUTPUT_DIR, "checkpoints")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
METRIC_DIR = os.path.join(OUTPUT_DIR, "metrics")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

for _d in (CKPT_DIR, FIG_DIR, METRIC_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------- 训练超参数 ----------------
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 25
LR = 1e-4
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
PATIENCE = 5          # 早停耐心
SEED = 42

# ImageNet 归一化参数
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# 支持的骨干网络
BACKBONES = ["resnet50", "efficientnet_b0", "mobilenet_v3_large"]


def load_splits(fname="splits.json"):
    """读取划分文件，返回 dict。fname 可切换为不均衡实验用的清单。"""
    path = os.path.join(DATA_DIR, fname)
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)


def get_classes():
    """返回 38 个类别名。"""
    return load_splits()["classes"]
