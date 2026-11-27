# -*- coding: utf-8 -*-
"""
evaluate.py — 模型评估与图表生成

用法：
    python src/evaluate.py --ckpt outputs/checkpoints/resnet50_best.pth --tag resnet50

产出（outputs/metrics/ 与 outputs/figures/）：
    eval_<tag>.json          全部指标
    02_训练曲线_<tag>.png     train/val 的 loss 与 accuracy
    03_混淆矩阵_<tag>.png     38×38 归一化
    04_混淆矩阵_番茄_<tag>.png 番茄 10 类子矩阵
    05_各类F1_<tag>.png       每类 F1 条形图
    06_ROC_病害vs健康_<tag>.png 病害/健康二分类 ROC
"""
import os
import sys
import json
import io
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score,
    precision_recall_fscore_support, roc_curve, auc, top_k_accuracy_score,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from dataset import build_loaders
from models import build_model

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    name = ckpt.get("model_name", "resnet50")
    classes = ckpt.get("classes", config.get_classes())
    model = build_model(name, num_classes=len(classes), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, name, classes


@torch.no_grad()
def predict_all(model, loader, device):
    """返回 (预测概率矩阵, 真实标签, 预测标签)。"""
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(x)
        probs.append(torch.softmax(out.float(), dim=1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def plot_curves(tag):
    """从 train_<tag>.json 读取历史，画 loss 与 accuracy 曲线。"""
    p = os.path.join(config.METRIC_DIR, f"train_{tag}.json")
    if not os.path.exists(p):
        return None
    with io.open(p, encoding="utf-8") as f:
        hist = json.load(f)["history"]

    ep = [h["epoch"] for h in hist]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(ep, [h["train_loss"] for h in hist], "o-", label="训练损失", color="#c0392b")
    axes[0].plot(ep, [h["val_loss"] for h in hist], "s-", label="验证损失", color="#2980b9")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("损失曲线"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(ep, [h["train_acc"] for h in hist], "o-", label="训练准确率", color="#27ae60")
    axes[1].plot(ep, [h["val_acc"] for h in hist], "s-", label="验证准确率", color="#8e44ad")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("准确率曲线"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(config.FIG_DIR, f"02_训练曲线_{tag}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def plot_confusion(cm, classes, tag, title_suffix="", fname_suffix="", short=True):
    n = len(classes)
    figsize = (16, 14) if n > 20 else (9, 8)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(f"混淆矩阵（归一化）{title_suffix}", fontsize=13)
    ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别")

    names = [c.replace("___", " / ").replace("__", " / ").replace("_", " ")[:26] if short else c
             for c in classes]
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticklabels(names, fontsize=7)

    if n <= 12:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                        color="white" if cm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out = os.path.join(config.FIG_DIR, f"03_混淆矩阵{fname_suffix}_{tag}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def plot_per_class_f1(report, classes, tag):
    f1s = [report[c]["f1-score"] for c in classes]
    names = [c.replace("___", " / ").replace("__", " / ").replace("_", " ")[:30] for c in classes]

    order = np.argsort(f1s)
    fig, ax = plt.subplots(figsize=(11, 11))
    colors = ["#c0392b" if f1s[i] < 0.9 else "#27ae60" for i in order]
    ax.barh(range(len(order)), [f1s[i] for i in order], color=colors)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[i] for i in order], fontsize=8)
    ax.set_xlabel("F1-score")
    ax.set_title("各类别 F1-score（红色为低于 0.90 的类别）", fontsize=13)
    ax.axvline(0.9, ls="--", color="gray", lw=1)
    ax.set_xlim(0, 1.02)
    plt.tight_layout()
    out = os.path.join(config.FIG_DIR, f"05_各类F1_{tag}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def plot_healthy_roc(probs, labels, classes, tag):
    """病害 vs 健康 二分类 ROC。类名含 healthy 的视为健康。"""
    is_healthy = np.array([1 if "healthy" in c.lower() else 0 for c in classes])
    y_true = is_healthy[labels]
    # P(健康) = 该类概率之和
    p_healthy = probs[:, np.where(is_healthy == 1)[0]].sum(axis=1)

    fpr, tpr, _ = roc_curve(y_true, p_healthy)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(fpr, tpr, color="#c0392b", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("假正率 FPR"); ax.set_ylabel("真正率 TPR")
    ax.set_title("病害 vs 健康 二分类 ROC")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(config.FIG_DIR, f"06_ROC_病害vs健康_{tag}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out, float(roc_auc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    tag = args.tag or os.path.basename(args.ckpt).replace("_best.pth", "")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, name, classes = load_model(args.ckpt, device)
    _, _, test_loader = build_loaders(num_workers=config.NUM_WORKERS)

    print(f"[{tag}] 模型 {name}，测试集 {len(test_loader.dataset)} 张")
    probs, labels = predict_all(model, test_loader, device)
    preds = probs.argmax(1)

    # ---------- 指标 ----------
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    top3 = float(top_k_accuracy_score(labels, probs, k=3, labels=list(range(len(classes)))))

    report = classification_report(labels, preds, target_names=classes,
                                   output_dict=True, zero_division=0)
    report_txt = classification_report(labels, preds, target_names=classes, zero_division=0)

    # 混淆矩阵（归一化）
    cm = confusion_matrix(labels, preds, labels=list(range(len(classes))))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    tomato_idx = [i for i, c in enumerate(classes) if c.lower().startswith("tomato")]
    tomato_cm = cm_norm[np.ix_(tomato_idx, tomato_idx)]

    fig_curves = plot_curves(tag)
    fig_cm = plot_confusion(cm_norm, classes, tag)
    fig_cm_tomato = plot_confusion(tomato_cm, [classes[i] for i in tomato_idx], tag,
                                   title_suffix="（番茄 10 类）", fname_suffix="_番茄")
    fig_f1 = plot_per_class_f1(report, classes, tag)
    fig_roc, roc_auc = plot_healthy_roc(probs, labels, classes, tag)

    result = {
        "tag": tag, "model": name, "total_test": int(len(labels)),
        "accuracy": float(acc), "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1), "top3_accuracy": top3,
        "healthy_vs_disease_auc": roc_auc,
        "per_class": {c: {k: float(v) for k, v in report[c].items()
                          if k in ("precision", "recall", "f1-score", "support")}
                      for c in classes},
        "macro_avg": {k: float(v) for k, v in report["macro avg"].items()},
        "figures": {
            "curves": fig_curves, "confusion": fig_cm,
            "confusion_tomato": fig_cm_tomato, "per_class_f1": fig_f1,
            "roc_healthy": fig_roc,
        },
    }
    with io.open(os.path.join(config.METRIC_DIR, f"eval_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    with io.open(os.path.join(config.METRIC_DIR, f"report_{tag}.txt"), "w", encoding="utf-8") as f:
        f.write(report_txt)

    print(f"[{tag}] Accuracy={acc:.4f}  macro-F1={macro_f1:.4f}  Top-3={top3:.4f}  AUC={roc_auc:.4f}")
    print(f"[{tag}] 指标已保存到 metrics/eval_{tag}.json")


if __name__ == "__main__":
    main()
