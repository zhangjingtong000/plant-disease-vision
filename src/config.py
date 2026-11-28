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


# 38 类名称，索引与 PlantVillage parquet 的 label 编码一致。
# 与 scripts/prepare_data.py 的 CLASS_NAMES 同源；这里保留一份是为了在没有
# data/splits.json 时，界面仍能渲染知识库与页面（只读演示模式），
# 训练 / 评估脚本走 get_classes() 的严格路径，缺文件直接报错。
PLANTVILLAGE_CLASSES = [
    "Apple__Apple_scab", "Apple__Black_rot", "Apple__Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy", "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot",
    "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
]

# data/splits.json 是否存在；缺它时界面进只读演示模式
HAS_SPLITS = os.path.exists(SPLITS_JSON)


def load_splits(fname="splits.json"):
    """读取划分文件，返回 dict。fname 可切换为不均衡实验用的清单。"""
    path = os.path.join(DATA_DIR, fname)
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)


def get_classes(fallback=False):
    """返回 38 个类别名。

    fallback=True 时，缺 data/splits.json 就返回 PLANTVILLAGE_CLASSES，
    供界面以只读演示模式启动；其余调用方保持严格，缺文件直接抛错。
    """
    try:
        return load_splits()["classes"]
    except (OSError, ValueError, KeyError):
        if not fallback:
            raise
        return list(PLANTVILLAGE_CLASSES)
