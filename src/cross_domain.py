# -*- coding: utf-8 -*-
"""
cross_domain.py — 外部数据集跨域测试

用 PlantVillage 训练的模型，去测 PlantDoc 的真实场景照片，看看泛化能力。

PlantDoc 的图片来自互联网，背景杂乱、光照不一、角度随意，
与 PlantVillage 的白底实验室照片分布差异很大，是用来检验"换个数据集还灵不灵"的理想素材。

用法：
    python src/cross_domain.py --ckpt outputs/checkpoints/resnet50_best.pth --tag resnet50

产出：
    outputs/metrics/cross_domain_<tag>.json
    outputs/figures/11_跨域对比_<tag>.png
    outputs/figures/12_跨域错分样例_<tag>.png
    outputs/cross_domain_<tag>.csv
"""
import os
import sys
import json
import io
import csv
import argparse
import random

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

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# PlantDoc 目录名 -> PlantVillage 类别名
CLS_MAP = {
    "Apple Scab Leaf": "Apple__Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple__Cedar_apple_rust",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato leaf": "Tomato___healthy",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "grape leaf black rot": "Grape___Black_rot",
    "grape leaf": "Grape___healthy",
}

TF = transforms.Compose([
    transforms.Resize(int(config.IMG_SIZE * 1.14)),
    transforms.CenterCrop(config.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(config.MEAN, config.STD),
])


def find_plantdoc_root():
    """自动定位 PlantDoc test 目录。"""
    candidates = [
        os.path.join(config.DATA_DIR, "plantdoc_extract", "PlantDoc-Dataset-master", "test"),
        os.path.join(config.DATA_DIR, "plantdoc", "test"),
        os.path.join(config.DATA_DIR, "plantdoc_extract", "test"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    tag = args.tag or os.path.basename(args.ckpt).replace("_best.pth", "")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = find_plantdoc_root()
    if not root:
        print("未找到 PlantDoc test 目录")
        return
    print("外部数据集:", root)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], num_classes=len(config.get_classes()),
                        pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    classes = config.get_classes()

    rows = []
    for dirname in sorted(os.listdir(root)):
        src_dir = os.path.join(root, dirname)
        if not os.path.isdir(src_dir):
            continue
        cls = CLS_MAP.get(dirname)
        if cls is None or cls not in classes:
            print("跳过未映射类别:", dirname)
            continue
        true_idx = classes.index(cls)
        for fn in sorted(os.listdir(src_dir)):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                continue
            p = os.path.join(src_dir, fn)
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                continue
            x = TF(img).unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = model(x)
            probs = torch.softmax(out.float(), 1)[0].cpu().numpy()
            pred = int(probs.argmax())
            rows.append({
                "file": p, "dir": dirname,
                "true": cls, "pred": classes[pred],
                "conf": float(probs[pred]),
                "correct": pred == true_idx,
            })

    if not rows:
        print("没有可评估的样本")
        return

    total = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    acc = correct / total
    print(f"外部数据样本 {total} 张，准确率 {acc*100:.2f}%")

    # 每类
    per_cls = {}
    for r in rows:
        d = per_cls.setdefault(r["true"], {"n": 0, "correct": 0})
        d["n"] += 1
        d["correct"] += int(r["correct"])

    # 同分布基线
    base = None
    ev = os.path.join(config.METRIC_DIR, f"eval_{tag}.json")
    if os.path.exists(ev):
        with io.open(ev, encoding="utf-8") as f:
            base = json.load(f)["accuracy"]

    result = {
        "tag": tag, "external_dataset": "PlantDoc test",
        "n_samples": total, "accuracy": acc,
        "in_domain_accuracy": base,
        "per_class": {k: {"n": v["n"], "acc": v["correct"] / v["n"]}
                      for k, v in sorted(per_cls.items())},
        "correct": correct,
    }
    with io.open(os.path.join(config.METRIC_DIR, f"cross_domain_{tag}.json"), "w",
                 encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    # CSV
    with io.open(os.path.join(config.OUTPUT_DIR, f"cross_domain_{tag}.csv"), "w",
                 encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["文件", "真实类别", "预测类别", "置信度", "是否正确"])
        for r in rows:
            w.writerow([r["file"], r["true"], r["pred"], f"{r['conf']:.4f}",
                        "是" if r["correct"] else "否"])

    # ---------- 对比图 ----------
    fig, ax = plt.subplots(figsize=(7, 4.8))
    labels = ["PlantVillage 测试集\n（同分布）", "PlantDoc 测试集\n（真实场景）"]
    vals = [base * 100 if base else 0, acc * 100]
    bars = ax.bar(labels, vals, color=["#17603A", "#C0392B"], width=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%",
                ha="center", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.set_ylabel("准确率 (%)")
    ax.set_title("跨域对比：实验室数据 vs 真实场景照片")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    p1 = os.path.join(config.FIG_DIR, f"11_跨域对比_{tag}.png")
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()

    # ---------- 错分样例 ----------
    wrong = [r for r in rows if not r["correct"]]
    random.Random(0).shuffle(wrong)
    wrong = wrong[:6]
    if wrong:
        fig, axes = plt.subplots(2, 3, figsize=(13, 8))
        for ax, r in zip(axes.ravel(), wrong):
            try:
                im = Image.open(r["file"]).convert("RGB")
                ax.imshow(im)
            except Exception:
                pass
            ax.set_title(f"真实：{r['true'].split('___')[-1][:16]}\n"
                         f"预测：{r['pred'].split('___')[-1][:16]} ({r['conf']*100:.0f}%)",
                         fontsize=9, color="#C0392B")
            ax.axis("off")
        plt.suptitle("跨域错分样例（真实场景照片）", fontsize=13)
        plt.tight_layout()
        p2 = os.path.join(config.FIG_DIR, f"12_跨域错分样例_{tag}.png")
        plt.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close()

    # ---------- 每类准确率 ----------
    items = sorted(per_cls.items(), key=lambda kv: kv[1]["correct"] / kv[1]["n"])
    names = [k.replace("___", " / ").replace("__", " / ").replace("_", " ")[:26] for k, _ in items]
    accs = [v["correct"] / v["n"] * 100 for _, v in items]
    fig, ax = plt.subplots(figsize=(10, max(5, len(items) * 0.34)))
    colors = ["#C0392B" if a < 50 else ("#E67E22" if a < 80 else "#17603A") for a in accs]
    ax.barh(names, accs, color=colors)
    ax.set_xlim(0, 105)
    ax.set_xlabel("准确率 (%)")
    ax.set_title(f"PlantDoc 各类别准确率（n={total}）")
    ax.axvline(acc * 100, ls="--", color="#555", lw=1)
    for i, a in enumerate(accs):
        ax.text(a + 1, i, f"{a:.0f}%", va="center", fontsize=8)
    plt.tight_layout()
    p3 = os.path.join(config.FIG_DIR, f"13_跨域各类准确率_{tag}.png")
    plt.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close()

    print("图已保存:", p1)
    if wrong:
        print("错分样例:", os.path.join(config.FIG_DIR, f"12_跨域错分样例_{tag}.png"))
    print("各类准确率:", p3)


if __name__ == "__main__":
    main()
