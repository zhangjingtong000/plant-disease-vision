# -*- coding: utf-8 -*-
"""
mixed_finetune.py — 混合域微调（PlantVillage + PlantDoc）

这是「用了目标域标注数据」的对照实验，量化的是**适配能力**，不是泛化能力。
报告里必须与「零样本跨域」区分开写。

数据划分：
  PlantDoc train 2345 张 -> 80% 微调 / 20% 验证
  PlantDoc test  236 张  -> 只在最后评估一次

用法：
    python src/mixed_finetune.py --init outputs/checkpoints/resnet50_dr_best.pth --tag resnet50_mix
"""
import os
import sys
import json
import io
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from models import build_model, count_parameters
from utils import set_seed, Logger

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

TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(config.MEAN, config.STD),
])
EVAL_TF = transforms.Compose([
    transforms.Resize(int(config.IMG_SIZE * 1.14)),
    transforms.CenterCrop(config.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(config.MEAN, config.STD),
])


def _open(p):
    if os.name == "nt" and not p.startswith("\\\\?\\"):
        p = "\\\\?\\" + os.path.abspath(p)
    return Image.open(p).convert("RGB")


class PD(Dataset):
    def __init__(self, items, tf):
        self.items = items
        self.tf = tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(_open(p)), y


def find_split(which):
    for c in [os.path.join(config.DATA_DIR, "plantdoc_extract", "PlantDoc-Dataset-master", which),
              os.path.join(config.DATA_DIR, "plantdoc", which)]:
        if os.path.isdir(c):
            return c
    return None


def collect(root, classes, want_idx=True):
    items = []
    for d in sorted(os.listdir(root)):
        sd = os.path.join(root, d)
        if not os.path.isdir(sd) or d not in CLS_MAP:
            continue
        cls = CLS_MAP[d]
        if cls not in classes:
            continue
        y = classes.index(cls)
        for f in sorted(os.listdir(sd)):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                items.append((os.path.join(sd, f), y))
    return items


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", default="outputs/checkpoints/resnet50_best.pth")
    parser.add_argument("--tag", default="resnet50_mix")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--mix-a", type=int, default=1, help="每个 batch 中域 A 的数量占比")
    args = parser.parse_args()

    log = Logger(os.path.join(config.LOG_DIR, f"train_{args.tag}.log"))
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = config.get_classes()

    tr_root, te_root = find_split("train"), find_split("test")
    if not tr_root or not te_root:
        print("未找到 PlantDoc")
        return

    b_items = collect(tr_root, classes)
    random.Random(0).shuffle(b_items)
    cut = int(len(b_items) * 0.8)
    b_train, b_val = b_items[:cut], b_items[cut:]
    test_items = collect(te_root, classes)
    log(f"域 B 微调 {len(b_train)} | 域 B 验证 {len(b_val)} | 域 B 测试 {len(test_items)}")

    # 域 A 数据（保持同分布能力）
    sp = config.load_splits()
    a_items = [(os.path.join(config.DATA_DIR, r["path"]), r["label"]) for r in sp["train"]]
    log(f"域 A 训练 {len(a_items)}")

    train_loader = DataLoader(PD(a_items + b_train, TRAIN_TF), batch_size=args.batch,
                              shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(PD(b_val, EVAL_TF), batch_size=args.batch, shuffle=False,
                            num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(PD(test_items, EVAL_TF), batch_size=args.batch, shuffle=False,
                             num_workers=config.NUM_WORKERS)

    ckpt = torch.load(args.init, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"],
                    len(sp["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    log(f"从 {os.path.basename(args.init)} 初始化")

    before = evaluate(model, test_loader, device)
    log(f"微调前 域 B 测试准确率 {before*100:.2f}%")

    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    sched = CosineAnnealingLR(optim, T_max=args.epochs, eta_min=args.lr * 0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best, best_ep = 0.0, -1
    out_ckpt = os.path.join(config.CKPT_DIR, f"{args.tag}_best.pth")
    t0 = time.time()
    history = []

    for ep in range(1, args.epochs + 1):
        model.train()
        ls, cor, tot = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = model(x)
                loss = crit(out, y)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            ls += loss.item() * x.size(0)
            cor += (out.argmax(1) == y).sum().item()
            tot += x.size(0)
        sched.step()
        vacc = evaluate(model, val_loader, device)
        history.append({"epoch": ep, "train_acc": cor / tot, "val_b_acc": vacc})
        log(f"Epoch {ep:02d}/{args.epochs} train {ls/tot:.4f}/{cor/tot:.4f} 域B验证 {vacc:.4f}")
        if vacc > best:
            best, best_ep = vacc, ep
            torch.save({"model_name": ckpt["model_name"], "state_dict": model.state_dict(),
                        "classes": classes, "val_b_acc": vacc,
                        "note": "mixed-domain finetune (A + B_train)", "author": "【待填写】"},
                       out_ckpt)
            log(f"  保存最佳 {vacc:.4f}")

    after = evaluate(model, test_loader, device)
    log(f"微调后 域 B 测试准确率 {after*100:.2f}%（{before*100:.2f}% -> {after*100:.2f}%）")
    log(f"耗时 {(time.time()-t0)/60:.1f} 分钟")

    with io.open(os.path.join(config.METRIC_DIR, f"mixed_{args.tag}.json"), "w",
                 encoding="utf-8") as f:
        json.dump({"tag": args.tag, "init": os.path.basename(args.init),
                   "before": before, "after": after, "delta": after - before,
                   "best_val_b": best, "history": history,
                   "n_finetune": len(b_train), "n_val": len(b_val), "n_test": len(test_items),
                   "note": "uses target-domain labeled data; measures adaptation, not generalization"},
                  f, ensure_ascii=False, indent=1)
    log.close()


if __name__ == "__main__":
    main()
