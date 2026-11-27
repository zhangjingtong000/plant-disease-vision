# -*- coding: utf-8 -*-
"""
tta_eval.py — 测试时增强（TTA）评估

不重新训练，只在推理时对同一张图做多种变换，把概率平均。
常见变换：多尺度中心裁剪 + 水平翻转。

用法：
    python src/tta_eval.py --ckpt outputs/checkpoints/resnet50_best.pth
"""
import os
import sys
import json
import io
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from models import build_model

CLS_MAP = {
    "Apple Scab Leaf": "Apple__Apple_scab", "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple__Cedar_apple_rust",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Bell_pepper leaf": "Pepper,_bell___healthy", "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_", "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight", "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato leaf": "Tomato___healthy", "Tomato mold leaf": "Tomato___Leaf_Mold",
    "grape leaf black rot": "Grape___Black_rot", "grape leaf": "Grape___healthy",
}


def _open(p):
    if os.name == "nt" and not p.startswith("\\\\?\\"):
        p = "\\\\?\\" + os.path.abspath(p)
    return Image.open(p).convert("RGB")


def find_split():
    for c in [os.path.join(config.DATA_DIR, "plantdoc_extract", "PlantDoc-Dataset-master", "test"),
              os.path.join(config.DATA_DIR, "plantdoc", "test")]:
        if os.path.isdir(c):
            return c
    return None


def make_views(pil, size):
    """多尺度中心裁剪 + 水平翻转。"""
    from torchvision import transforms
    views = []
    for scale in (1.0, 1.14, 1.28):
        base = int(size * scale)
        for flip in (False, True):
            tf = transforms.Compose([
                transforms.Resize(base),
                transforms.CenterCrop(size),
                transforms.Lambda(lambda im, f=flip: im.transpose(Image.FLIP_LEFT_RIGHT) if f else im),
                transforms.ToTensor(),
                transforms.Normalize(config.MEAN, config.STD),
            ])
            views.append(tf(pil))
    return views


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="outputs/checkpoints/resnet50_best.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = config.get_classes()
    root = find_split()
    if not root:
        print("未找到 PlantDoc test")
        return

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], 38, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    targets = [c for c in dict.fromkeys(CLS_MAP.values()) if c in classes]
    tgt_index = {c: i for i, c in enumerate(targets)}
    fold = np.zeros((38, len(targets)), dtype=np.float32)
    for d, c in CLS_MAP.items():
        if c in tgt_index:
            fold[classes.index(c), tgt_index[c]] = 1.0

    items = []
    for d in sorted(os.listdir(root)):
        sd = os.path.join(root, d)
        if not os.path.isdir(sd) or d not in CLS_MAP or CLS_MAP[d] not in tgt_index:
            continue
        for f in sorted(os.listdir(sd)):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                items.append((os.path.join(sd, f), tgt_index[CLS_MAP[d]]))

    plain_correct = tta_correct = 0
    plain_top3 = tta_top3 = 0
    n = len(items)

    for p, y in items:
        pil = _open(p)
        views = make_views(pil, config.IMG_SIZE)
        x = torch.stack(views).to(device)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(x)
        probs = F.softmax(out.float(), 1).cpu().numpy()
        p_plain = probs[0]
        p_tta = probs.mean(0)

        f_plain = p_plain @ fold
        f_tta = p_tta @ fold
        if int(f_plain.argmax()) == y:
            plain_correct += 1
        if int(f_tta.argmax()) == y:
            tta_correct += 1
        if y in np.argsort(f_plain)[::-1][:3]:
            plain_top3 += 1
        if y in np.argsort(f_tta)[::-1][:3]:
            tta_top3 += 1

    res = {
        "ckpt": os.path.basename(args.ckpt), "n": n,
        "plain_top1": plain_correct / n, "tta_top1": tta_correct / n,
        "plain_top3": plain_top3 / n, "tta_top3": tta_top3 / n,
    }
    print(f"\n样本 {n} 张")
    print(f"原始   Top-1 {res['plain_top1']*100:.2f}%   Top-3 {res['plain_top3']*100:.2f}%")
    print(f"TTA    Top-1 {res['tta_top1']*100:.2f}%   Top-3 {res['tta_top3']*100:.2f}%")
    print(f"变化   Top-1 {(res['tta_top1']-res['plain_top1'])*100:+.2f} 个点   "
          f"Top-3 {(res['tta_top3']-res['plain_top3'])*100:+.2f} 个点")

    tag = os.path.basename(args.ckpt).replace("_best.pth", "")
    with io.open(os.path.join(config.METRIC_DIR, f"tta_{tag}.json"), "w",
                 encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
