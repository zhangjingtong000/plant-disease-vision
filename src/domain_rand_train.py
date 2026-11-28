# -*- coding: utf-8 -*-
"""
domain_rand_train.py — 域随机化训练

依据偏移谱的结论：跨域掉点的主因是目标尺度（叶片在画面中占比过小），
其次是模糊压缩，背景和光照影响很小。因此本脚本的增强重点放在尺度上。

增强策略（对应偏移谱的 S1~S4）：
  - RandomResizedCrop scale=(0.35, 1.0)：主力，模拟小目标
  - 背景替换 p=0.6：白底换复杂背景
  - 光照 / 色温 p=0.7
  - 模糊 / 压缩 / 噪声 p=0.5
  - 反转 / 旋转 / 透视 p=0.5

用法：
    python src/domain_rand_train.py --model resnet50 --epochs 25 --tag resnet50_dr
"""
import os
import sys
import json
import io
import time
import argparse
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from models import build_model, count_parameters
from utils import set_seed, Logger


def _open(path):
    if os.name == "nt" and not path.startswith("\\\\?\\"):
        path = "\\\\?\\" + os.path.abspath(path)
    return Image.open(path).convert("RGB")


# ---------------- 增强 ----------------
def leaf_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    bg = ((hsv[..., 1] < 30) & (hsv[..., 2] > 200)).astype(np.uint8)
    leaf = 1 - bg
    k = np.ones((5, 5), np.uint8)
    leaf = cv2.morphologyEx(leaf, cv2.MORPH_OPEN, k)
    leaf = cv2.morphologyEx(leaf, cv2.MORPH_CLOSE, k)
    return cv2.GaussianBlur(leaf.astype(np.float32), (7, 7), 3)[..., None]


def synth_bg(h, w, rng):
    base = rng.uniform(40, 190, (3,)).astype(np.float32)
    bg = np.zeros((h, w, 3), np.float32) + base
    grad = np.linspace(0, 1, h)[:, None, None] * rng.uniform(-60, 60, (3,))
    bg += grad + rng.normal(0, 12, (h, w, 3)).astype(np.float32)
    return np.clip(cv2.GaussianBlur(bg, (9, 9), 0), 0, 255)


class StrongAug:
    """针对「实验室白底 → 真实场景」的增强。

    mild=True 时采用温和版参数（尺度下限更高、其余概率减半），用于对照实验。
    """

    def __init__(self, size=224, mild=False):
        self.size = size
        self.mild = mild
        self.scale_lo = 0.55 if mild else 0.35
        self.p_crop = 0.9
        self.p_bg = 0.40 if mild else 0.60
        self.p_light = 0.60 if mild else 0.70
        self.p_blur = 0.35 if mild else 0.50
        self.p_geo = 0.40 if mild else 0.50

    def __call__(self, pil):
        rng = np.random.RandomState(random.randint(0, 2 ** 31 - 1))
        im = np.array(pil.resize((256, 256)))

        # 1) 随机裁剪缩放：主力，模拟小目标
        if rng.rand() < 0.9:
            scale = rng.uniform(self.scale_lo, 1.0)
            side = int(256 * np.sqrt(scale))
            side = max(64, side)
            y0 = rng.randint(0, 256 - side + 1)
            x0 = rng.randint(0, 256 - side + 1)
            im = im[y0:y0 + side, x0:x0 + side]
        im = cv2.resize(im, (self.size, self.size), interpolation=cv2.INTER_LINEAR)

        # 2) 背景替换
        if rng.rand() < self.p_bg:
            m = leaf_mask(im.astype(np.uint8))
            bg = synth_bg(self.size, self.size, rng)
            im = (m * im.astype(np.float32) + (1 - m) * bg)

        # 3) 光照
        if rng.rand() < self.p_light:
            f = rng.uniform(0.8, 1.2)
            b = rng.uniform(-22, 22)
            im = np.clip(im * f + b, 0, 255)
            hsv = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] + rng.uniform(-8, 8)) % 180
            hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.85, 1.15), 0, 255)
            im = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

        # 4) 模糊 / 压缩 / 噪声
        if rng.rand() < self.p_blur:
            if rng.rand() < 0.6:
                k = int(rng.choice([3, 5]))
                im = cv2.GaussianBlur(im, (k, k), 0)
            if rng.rand() < 0.5:
                ok, enc = cv2.imencode(".jpg", im.astype(np.uint8),
                                       [int(cv2.IMWRITE_JPEG_QUALITY), int(rng.uniform(45, 80))])
                if ok:
                    im = cv2.imdecode(enc, cv2.IMREAD_COLOR).astype(np.float32)
            im = np.clip(im + rng.normal(0, rng.uniform(2, 8), im.shape), 0, 255)

        # 5) 几何
        if rng.rand() < self.p_geo:
            if rng.rand() < 0.5:
                im = im[:, ::-1].copy()
            ang = rng.uniform(-20, 20)
            M = cv2.getRotationMatrix2D((self.size / 2, self.size / 2), ang,
                                        rng.uniform(0.9, 1.1))
            im = cv2.warpAffine(im.astype(np.uint8), M, (self.size, self.size),
                                borderMode=cv2.BORDER_REPLICATE).astype(np.float32)

        im = np.clip(im, 0, 255).astype(np.uint8)
        x = torch.from_numpy(im).permute(2, 0, 1).float() / 255.0
        x = (x - torch.tensor(config.MEAN)[:, None, None]) / torch.tensor(config.STD)[:, None, None]
        return x


def plain_tf(pil):
    from torchvision import transforms
    t = transforms.Compose([
        transforms.Resize(int(config.IMG_SIZE * 1.14)),
        transforms.CenterCrop(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD),
    ])
    return t(pil)


class DS(Dataset):
    def __init__(self, records, root, tf):
        self.records = records
        self.root = root
        self.tf = tf

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        p = os.path.join(self.root, r["path"])
        return self.tf(_open(p)), r["label"]


def evaluate(model, loader, device):
    model.eval()
    loss_sum = correct = total = 0
    crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = model(x)
                loss = crit(out, y)
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--tag", default="resnet50_dr")
    parser.add_argument("--mild", action="store_true")
    args = parser.parse_args()

    log = Logger(os.path.join(config.LOG_DIR, f"train_{args.tag}.log"))
    log(f"域随机化训练 参数: {vars(args)}")
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sp = config.load_splits()
    train_ds = DS(sp["train"], config.DATA_DIR, StrongAug(mild=args.mild))
    val_ds = DS(sp["val"], config.DATA_DIR, plain_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=True)
    log(f"训练 {len(train_ds)} | 验证 {len(val_ds)}")

    model = build_model(args.model, 38, pretrained=True).to(device)
    tp, _ = count_parameters(model)
    log(f"{args.model} 参数量 {tp:.2f}M")

    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    sched = CosineAnnealingLR(optim, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best, best_ep, bad = 0.0, -1, 0
    ckpt_path = os.path.join(config.CKPT_DIR, f"{args.tag}_best.pth")
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

        vloss, vacc = evaluate(model, val_loader, device)
        sched.step()
        history.append({"epoch": ep, "train_loss": ls / tot, "train_acc": cor / tot,
                        "val_loss": vloss, "val_acc": vacc})
        log(f"Epoch {ep:02d}/{args.epochs} train {ls/tot:.4f}/{cor/tot:.4f} "
            f"val {vloss:.4f}/{vacc:.4f}")

        if vacc > best:
            best, best_ep, bad = vacc, ep, 0
            torch.save({"model_name": args.model, "state_dict": model.state_dict(),
                        "classes": config.get_classes(), "val_acc": vacc, "epoch": ep},
                       ckpt_path)
            log(f"  保存最佳 val_acc={vacc:.4f}")
        else:
            bad += 1
            if bad >= config.PATIENCE:
                log("早停")
                break

    log(f"完成 最佳 {best:.4f}（第 {best_ep} 轮）耗时 {(time.time()-t0)/60:.1f} 分钟")
    with io.open(os.path.join(config.METRIC_DIR, f"train_{args.tag}.json"), "w",
                 encoding="utf-8") as f:
        json.dump({"model": args.model, "tag": args.tag, "epochs_run": len(history),
                   "best_val_acc": best, "best_epoch": best_ep,
                   "params_m": round(tp, 2), "history": history,
                   "augmentation": "strong (scale 0.35-1.0 + bg replace + lighting + blur)",
                   "env": {"gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
                           "torch": torch.__version__}}, f, ensure_ascii=False, indent=1)
    log.close()


if __name__ == "__main__":
    main()
