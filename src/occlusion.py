# -*- coding: utf-8 -*-
"""
occlusion.py — 遮挡实验

对测试集样本做两组遮挡：
  A. 遮挡 CAM 高响应区（模型认为的病斑）
  B. 遮挡同面积随机区域
比较置信度下降幅度，证明模型确实在看病斑而不是背景。

产出：
    metrics/occlusion_<tag>.json
    figures/10_遮挡实验_<tag>.png
"""
import os
import sys
import json
import io
import random
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from models import build_model
from gradcam import GradCAM, leaf_mask, overlay_cam_on_image, occlusion_test

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

TF = transforms.Compose([
    transforms.Resize(int(config.IMG_SIZE * 1.14)),
    transforms.CenterCrop(config.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(config.MEAN, config.STD),
])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--n", type=int, default=40, help="抽样数量")
    args = parser.parse_args()

    tag = args.tag or os.path.basename(args.ckpt).replace("_best.pth", "")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], num_classes=len(config.get_classes()), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    sp = config.load_splits()
    rng = random.Random(0)
    samples = rng.sample(sp["test"], min(args.n, len(sp["test"])))

    gc = GradCAM(model, ckpt["model_name"])
    drops_lesion, drops_random, bases = [], [], []
    examples = []

    for rec in samples:
        p = os.path.join(config.DATA_DIR, rec["path"])
        pil = Image.open(p).convert("RGB")
        x = TF(pil).unsqueeze(0).to(device)

        cam, _ = gc(x)
        rgb = np.array(pil.resize((config.IMG_SIZE, config.IMG_SIZE)))
        mask = leaf_mask(rgb)
        st = occlusion_test(model, x, cam, mask, device)

        drops_lesion.append(st["drop_lesion"])
        drops_random.append(st["drop_random"])
        bases.append(st["base_conf"])

        if len(examples) < 4:
            examples.append({
                "img": rgb,
                "cam": cam,
                "mask": mask,
                "stat": st,
                "true": rec["class_name"],
            })

    gc.close()

    mean_lesion = float(np.mean(drops_lesion))
    mean_random = float(np.mean(drops_random))
    result = {
        "tag": tag, "n": len(samples),
        "mean_base_conf": float(np.mean(bases)),
        "mean_drop_occlude_lesion": mean_lesion,
        "mean_drop_occlude_random": mean_random,
        "ratio": mean_lesion / max(mean_random, 1e-6),
        "samples": [{"drop_lesion": float(a), "drop_random": float(b)}
                    for a, b in zip(drops_lesion, drops_random)],
    }
    with io.open(os.path.join(config.METRIC_DIR, f"occlusion_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    print(f"遮挡病斑平均置信度下降: {mean_lesion*100:.2f} 个百分点")
    print(f"遮挡随机区域平均下降:   {mean_random*100:.2f} 个百分点")
    print(f"倍数: {result['ratio']:.1f}x")

    # 画图
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, ex in enumerate(examples[:4]):
        cam_overlay = overlay_cam_on_image(ex["img"], ex["cam"], ex["mask"])
        axes[0, i].imshow(ex["img"])
        axes[0, i].set_title("原图", fontsize=10)
        axes[0, i].axis("off")
        axes[1, i].imshow(cam_overlay)
        axes[1, i].set_title(
            f"遮挡病斑: -{ex['stat']['drop_lesion']*100:.1f}%\n遮挡随机: -{ex['stat']['drop_random']*100:.1f}%",
            fontsize=9)
        axes[1, i].axis("off")
    plt.suptitle(f"遮挡实验：遮住病斑区 vs 遮住随机区（{len(samples)} 张平均）", fontsize=13)
    plt.tight_layout()
    out = os.path.join(config.FIG_DIR, f"10_遮挡实验_{tag}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("图已保存:", out)


if __name__ == "__main__":
    main()
