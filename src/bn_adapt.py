# -*- coding: utf-8 -*-
"""
bn_adapt.py — BN 统计量重估（AdaBN）

原理：模型在实验室白底数据上训练时，BatchNorm 的 running_mean / running_var
被固化成了「白底域」的统计量。换到真实场景照片，这些统计量和输入分布不匹配，
是跨域掉点的重要原因之一。

做法：不更新任何权重，只把 BN 层切到 train 模式，用目标域的无标签图像
前向若干轮，让 running stats 重新统计；再切回 eval 测目标域。

这属于「无监督域适配」，报告里必须标注使用了目标域无标签数据。

用法：
    python src/bn_adapt.py --tag resnet50
"""
import os
import sys
import json
import io
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from models import build_model

TF = transforms.Compose([
    transforms.Resize(int(config.IMG_SIZE * 1.14)),
    transforms.CenterCrop(config.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(config.MEAN, config.STD),
])

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


def find_split(which):
    for c in [
        os.path.join(config.DATA_DIR, "plantdoc_extract", "PlantDoc-Dataset-master", which),
        os.path.join(config.DATA_DIR, "plantdoc", which),
    ]:
        if os.path.isdir(c):
            return c
    return None



def _open_img(path):
    """Windows 长路径安全打开。"""
    if os.name == "nt" and not path.startswith("\\\\?\\"):
        path = "\\\\?\\" + os.path.abspath(path)
    return Image.open(path).convert("RGB")

class UnlabeledDS(Dataset):
    def __init__(self, root):
        self.files = []
        for d in sorted(os.listdir(root)):
            sd = os.path.join(root, d)
            if not os.path.isdir(sd):
                continue
            for f in sorted(os.listdir(sd)):
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    self.files.append(os.path.join(sd, f))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return TF(_open_img(self.files[i]))


class LabeledDS(Dataset):
    def __init__(self, root, classes):
        self.items = []
        for d in sorted(os.listdir(root)):
            sd = os.path.join(root, d)
            if not os.path.isdir(sd) or d not in CLS_MAP:
                continue
            cls = CLS_MAP[d]
            if cls not in classes:
                continue
            for f in sorted(os.listdir(sd)):
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    self.items.append((os.path.join(sd, f), classes.index(cls)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return TF(_open_img(p)), y


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(x)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="resnet50")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(config.CKPT_DIR, f"{args.tag}_best.pth")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    classes = config.get_classes()

    model = build_model(ckpt["model_name"], num_classes=len(classes), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    test_root = find_split("test")
    train_root = find_split("train")
    if not test_root or not train_root:
        print("未找到 PlantDoc 数据")
        return

    test_loader = DataLoader(LabeledDS(test_root, classes), batch_size=32,
                             shuffle=False, num_workers=0)
    adapt_loader = DataLoader(UnlabeledDS(train_root), batch_size=32,
                              shuffle=True, num_workers=0)

    before = evaluate(model, test_loader, device)
    print(f"[BN 重估前] PlantDoc 准确率: {before*100:.2f}%")

    # 只把 BN 层切到 train 模式，其余保持 eval；权重全部冻结
    for p in model.parameters():
        p.requires_grad = False
    bn_layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    print(f"重估 {len(bn_layers)} 个 BN 层，用 {len(adapt_loader.dataset)} 张无标签图像，{args.epochs} 轮")

    with torch.no_grad():
        for ep in range(args.epochs):
            for m in model.modules():
                m.eval()
            for m in bn_layers:
                m.train()
                m.momentum = 0.1
            n = 0
            for x in adapt_loader:
                x = x.to(device)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    model(x)
                n += x.size(0)
            print(f"  第 {ep+1}/{args.epochs} 轮完成，累计 {n} 张")

    model.eval()
    after = evaluate(model, test_loader, device)
    print(f"[BN 重估后] PlantDoc 准确率: {after*100:.2f}%  (变化 {(after-before)*100:+.2f} 个点)")

    # 保存适配后的权重
    out_ckpt = os.path.join(config.CKPT_DIR, f"{args.tag}_bnadapt.pth")
    torch.save({
        "model_name": ckpt["model_name"],
        "state_dict": model.state_dict(),
        "classes": classes,
        "note": "BN statistics re-estimated on PlantDoc train (unlabeled)",
    }, out_ckpt)

    result = {
        "tag": args.tag, "method": "AdaBN",
        "before": before, "after": after, "delta": after - before,
        "adapt_samples": len(adapt_loader.dataset), "epochs": args.epochs,
    }
    with io.open(os.path.join(config.METRIC_DIR, f"bn_adapt_{args.tag}.json"), "w",
                 encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print("已保存:", out_ckpt)
    print("指标已保存: metrics/bn_adapt_%s.json" % args.tag)


if __name__ == "__main__":
    main()
