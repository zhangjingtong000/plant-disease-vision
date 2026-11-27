# -*- coding: utf-8 -*-
"""
degenerate.py — 图像退化鲁棒性实验

对测试集施加不同程度的退化，观察模型精度变化，画「退化等级 - 精度」曲线。
全部程序化生成，不需要任何真实拍摄。

退化类型：
  - 高斯噪声（σ = 0.02 / 0.05 / 0.10）
  - 运动模糊（核 5 / 9 / 15）
  - 降分辨率再放大（128 / 96 / 64）

用法：
    python src/degenerate.py --ckpt outputs/checkpoints/resnet50_best.pth --tag resnet50
"""
import os
import sys
import json
import io
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from models import build_model
from dataset import build_loaders

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def add_gaussian_noise(img_tensor, sigma):
    """img_tensor: [C,H,W]，取值在标准化后的分布上，用像素域加噪更合理。"""
    noise = torch.randn_like(img_tensor) * sigma
    return img_tensor + noise


def motion_blur(img_tensor, k):
    """简单水平方向运动模糊（逐通道卷积）。"""
    c = img_tensor.shape[0]
    kernel = torch.zeros(c, 1, k, k, dtype=img_tensor.dtype, device=img_tensor.device)
    kernel[:, 0, k // 2, :] = 1.0 / k
    x = img_tensor.unsqueeze(0)
    x = F.conv2d(x, kernel, padding=k // 2, groups=c)
    return x.squeeze(0)


def downscale(img_tensor, size):
    """降分辨率后再双线性放大回原尺寸。"""
    x = img_tensor.unsqueeze(0)
    small = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    back = F.interpolate(small, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return back.squeeze(0)


@torch.no_grad()
def evaluate_variant(model, loader, device, variant_fn):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if variant_fn is not None:
            x = torch.stack([variant_fn(t) for t in x])
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(x)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    tag = args.tag or os.path.basename(args.ckpt).replace("_best.pth", "")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], num_classes=len(config.get_classes()), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    _, _, test_loader = build_loaders(batch_size=64, num_workers=config.NUM_WORKERS)

    # 基线
    base_acc = evaluate_variant(model, test_loader, device, None)
    print(f"基线准确率: {base_acc:.4f}")

    experiments = {
        "高斯噪声": [("σ=0.02", lambda t: add_gaussian_noise(t, 0.02)),
                     ("σ=0.05", lambda t: add_gaussian_noise(t, 0.05)),
                     ("σ=0.10", lambda t: add_gaussian_noise(t, 0.10))],
        "运动模糊": [("核=5", lambda t: motion_blur(t, 5)),
                     ("核=9", lambda t: motion_blur(t, 9)),
                     ("核=15", lambda t: motion_blur(t, 15))],
        "降分辨率": [("128px", lambda t: downscale(t, 128)),
                     ("96px", lambda t: downscale(t, 96)),
                     ("64px", lambda t: downscale(t, 64))],
    }

    results = {"tag": tag, "baseline": base_acc, "groups": {}}
    for gname, variants in experiments.items():
        vals = []
        for label, fn in variants:
            acc = evaluate_variant(model, test_loader, device, fn)
            vals.append({"level": label, "acc": round(acc, 4)})
            print(f"  {gname} {label}: {acc:.4f}")
        results["groups"][gname] = vals

    with io.open(os.path.join(config.METRIC_DIR, f"degenerate_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    # 画图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (gname, vals) in zip(axes, results["groups"].items()):
        xs = [v["level"] for v in vals]
        ys = [v["acc"] * 100 for v in vals]
        ax.plot(["-"] + xs, [base_acc * 100] + ys, "o-", color="#c0392b", lw=2)
        ax.axhline(base_acc * 100, ls="--", color="gray", lw=1)
        ax.set_title(f"{gname} 对精度的影响")
        ax.set_ylabel("测试准确率 (%)")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        for x, y in zip(["-"] + xs, [base_acc * 100] + ys):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=9)
    plt.tight_layout()
    out = os.path.join(config.FIG_DIR, f"07_退化鲁棒性_{tag}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("图已保存:", out)


if __name__ == "__main__":
    main()
