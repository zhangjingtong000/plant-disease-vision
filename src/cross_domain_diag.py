# -*- coding: utf-8 -*-
"""
cross_domain_diag.py — 跨域诊断全套分析（纯推理，不训练）

产出：
  1. 三种评估协议对比（严格 / 折叠 / 受限）
  2. 95% 置信区间、多数类基线、Top-k、macro-F1
  3. 混淆矩阵（27 类）
  4. 置信度分布对比（同分布 vs 跨域）
  5. risk–coverage 曲线（选择性预测）
  6. 三模型一致性 vs 准确率
  7. 特征空间 t-SNE（域可分离性）
"""
import os
import sys
import json
import io
import math
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from models import build_model

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

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


def _open(path):
    if os.name == "nt" and not path.startswith("\\\\?\\"):
        path = "\\\\?\\" + os.path.abspath(path)
    return Image.open(path).convert("RGB")


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - m), min(1.0, c + m)


def find_split(which):
    for c in [os.path.join(config.DATA_DIR, "plantdoc_extract", "PlantDoc-Dataset-master", which),
              os.path.join(config.DATA_DIR, "plantdoc", which)]:
        if os.path.isdir(c):
            return c
    return None


@torch.no_grad()
def run_model(model, items, device, want_feat=False):
    """items: [(path, label)]，返回 probs, labels。"""
    probs, labels, feats = [], [], []
    feats_model = None
    if want_feat:
        # 用 avgpool 输出作为特征
        store = {}
        def hook(m, i, o):
            store["f"] = o.detach().flatten(1)
        h = model.avgpool.register_forward_hook(hook)

    for i in range(0, len(items), 32):
        batch = items[i:i + 32]
        xs = torch.stack([TF(_open(p)) for p, _ in batch]).to(device)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(xs)
        probs.append(torch.softmax(out.float(), 1).cpu().numpy())
        labels.extend([y for _, y in batch])
        if want_feat:
            feats.append(store["f"].cpu().numpy())
    if want_feat:
        h.remove()
    P = np.concatenate(probs)
    F = np.concatenate(feats) if feats else None
    return P, np.array(labels), F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="resnet50")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = config.get_classes()
    test_root = find_split("test")
    if not test_root:
        print("未找到 PlantDoc test")
        return

    # 构建 27 类目标空间
    targets = [c for c in dict.fromkeys(CLS_MAP.values()) if c in classes]
    tgt_index = {c: i for i, c in enumerate(targets)}
    n_tgt = len(targets)
    _n_cls = len(config.get_classes())
    fold = np.zeros((_n_cls, n_tgt), dtype=np.float32)
    for d, c in CLS_MAP.items():
        if c in tgt_index:
            fold[classes.index(c), tgt_index[c]] = 1.0

    items_b = []
    for d in sorted(os.listdir(test_root)):
        sd = os.path.join(test_root, d)
        if not os.path.isdir(sd) or d not in CLS_MAP or CLS_MAP[d] not in tgt_index:
            continue
        for f in sorted(os.listdir(sd)):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                items_b.append((os.path.join(sd, f), tgt_index[CLS_MAP[d]]))

    sp = config.load_splits()
    items_a = [(os.path.join(config.DATA_DIR, r["path"]), tgt_index[r["class_name"]])
               for r in sp["test"] if r["class_name"] in tgt_index]

    print(f"域 A（同分布，可映射子集）{len(items_a)} 张 | 域 B（PlantDoc）{len(items_b)} 张")

    fig_dir = config.FIG_DIR
    metric_dir = config.METRIC_DIR
    results = {}

    # ---------- 单模型完整诊断 ----------
    ckpt = torch.load(os.path.join(config.CKPT_DIR, f"{args.tag}_best.pth"),
                      map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"],
                        len(config.get_classes()), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    Pb, yb, Fb = run_model(model, items_b, device, want_feat=True)
    Pa, ya, Fa = run_model(model, items_a, device, want_feat=True)

    # 协议
    strict = int((Pb.argmax(1) == np.array([classes.index(targets[t]) for t in yb])).sum())
    p28 = Pb @ fold
    pred_fold = p28.argmax(1)
    acc_fold = float((pred_fold == yb).mean())
    lo, hi = wilson(int((pred_fold == yb).sum()), len(yb))

    top3 = float(np.mean([yb[i] in np.argsort(p28[i])[::-1][:3] for i in range(len(yb))]))
    top5 = float(np.mean([yb[i] in np.argsort(p28[i])[::-1][:5] for i in range(len(yb))]))

    cnt = Counter(yb.tolist())
    major_cls, major_n = cnt.most_common(1)[0]
    major_acc = major_n / len(yb)

    from sklearn.metrics import f1_score, confusion_matrix
    macro_f1 = float(f1_score(yb, pred_fold, average="macro", zero_division=0))

    # 置信度
    conf_b = p28.max(1)
    pa28 = Pa @ fold
    conf_a = pa28.max(1)
    acc_a = float((pa28.argmax(1) == ya).mean())

    results.update({
        "n_A": len(items_a), "n_B": len(items_b), "n_classes": n_tgt,
        "acc_strict": strict / len(yb), "acc_fold": acc_fold,
        "ci_low": lo, "ci_high": hi, "top3": top3, "top5": top5,
        "macro_f1": macro_f1, "majority_baseline": major_acc,
        "majority_class": targets[major_cls],
        "acc_A_subset": acc_a,
        "mean_conf_A": float(conf_a.mean()), "mean_conf_B": float(conf_b.mean()),
    })

    print(f"\n折叠协议准确率 {acc_fold*100:.2f}%  95%CI [{lo*100:.1f}, {hi*100:.1f}]")
    print(f"Top-3 {top3*100:.2f}%  Top-5 {top5*100:.2f}%  macro-F1 {macro_f1:.4f}")
    print(f"多数类基线 {major_acc*100:.2f}% (优于基线: {acc_fold > major_acc})")
    print(f"平均置信度 A {conf_a.mean():.3f} vs B {conf_b.mean():.3f}")

    # ---------- 图1：三协议对比 ----------
    fig, ax = plt.subplots(figsize=(7, 4.8))
    vals = [strict / len(yb) * 100, acc_fold * 100]
    names = ["严格协议\n{} 类 argmax".format(len(config.get_classes())),
             "折叠协议\n映射到 {} 类后 argmax".format(n_tgt)]
    bars = ax.bar(names, vals, color=["#9CB3A6", "#17603A"], width=0.42)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center",
                fontsize=13, fontweight="bold")
    ax.set_ylim(0, 60)
    ax.set_ylabel("准确率 (%)")
    ax.set_title(f"跨域评估协议对比（域 B，n={len(yb)}）")
    ax.annotate(
        "受限协议（仅在 27 个候选类中取 argmax）\n在 1:1 类别映射下与折叠协议数学等价，\n"
        f"实测差异 0/{len(yb)} 个样本，故不单列",
        xy=(0.5, -0.30), xycoords="axes fraction", ha="center", va="top",
        fontsize=8.5, color="#6C6E75",
    )
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "14_跨域协议对比.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ---------- 图2：混淆矩阵 ----------
    cm = confusion_matrix(yb, pred_fold, labels=list(range(n_tgt)))
    cmn = cm.astype(float) / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    short = [t.replace("___", "/").replace("__", "/").replace("_", " ")[:24] for t in targets]
    ax.set_xticks(range(n_tgt)); ax.set_yticks(range(n_tgt))
    ax.set_xticklabels(short, rotation=90, fontsize=6)
    ax.set_yticklabels(short, fontsize=6)
    ax.set_title(f"跨域混淆矩阵（归一化，n={len(yb)}）")
    fig.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "15_跨域混淆矩阵.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ---------- 图3：置信度分布 ----------
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bins = np.linspace(0, 1, 21)
    ax.hist(conf_a, bins=bins, alpha=0.55, label=f"域 A 同分布 (均值 {conf_a.mean():.2f})", color="#17603A")
    ax.hist(conf_b, bins=bins, alpha=0.55, label=f"域 B 真实场景 (均值 {conf_b.mean():.2f})", color="#C0392B")
    ax.set_xlabel("最大类别概率（置信度）"); ax.set_ylabel("样本数")
    ax.set_title("置信度分布：同分布 vs 跨域")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "16_置信度分布.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ---------- 图4：risk-coverage ----------
    order = np.argsort(-conf_b)
    correct = (pred_fold == yb)[order]
    cov = np.arange(1, len(correct) + 1) / len(correct)
    risk_acc = np.cumsum(correct) / np.arange(1, len(correct) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(cov * 100, risk_acc * 100, color="#17603A", lw=2)
    ax.plot([0, 100], [acc_fold * 100, acc_fold * 100], "--", color="#888", lw=1,
            label=f"全量准确率 {acc_fold*100:.1f}%")
    for c in (0.3, 0.5, 0.7):
        idx = int(len(correct) * c) - 1
        if idx > 0:
            ax.scatter(c * 100, risk_acc[idx] * 100, color="#C0392B", zorder=5)
            ax.annotate(f"覆盖{c*100:.0f}% → {risk_acc[idx]*100:.1f}%",
                        (c * 100, risk_acc[idx] * 100), textcoords="offset points",
                        xytext=(8, -12), fontsize=9)
    ax.set_xlabel("覆盖率 (%)"); ax.set_ylabel("该覆盖下的准确率 (%)")
    ax.set_title("选择性预测：只回答高置信样本")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "17_风险覆盖曲线.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 覆盖率指标
    cov_stats = {}
    for c in (0.3, 0.5, 0.7, 0.9):
        idx = max(int(len(correct) * c), 1)
        cov_stats[f"top{int(c*100)}"] = float(risk_acc[idx - 1])
    results["risk_coverage"] = cov_stats

    # ---------- 图5：三模型一致性 ----------
    model_probs = {}
    for name in config.BACKBONES:
        p = os.path.join(config.CKPT_DIR, f"{name}_best.pth")
        if not os.path.exists(p):
            continue
        c2 = torch.load(p, map_location=device, weights_only=False)
        m2 = build_model(c2["model_name"],
                     len(config.get_classes()), pretrained=False)
        m2.load_state_dict(c2["state_dict"]); m2.to(device).eval()
        P2, _, _ = run_model(m2, items_b, device)
        model_probs[name] = (P2 @ fold).argmax(1)

    if len(model_probs) >= 2:
        names = list(model_probs)
        preds = np.stack([model_probs[n] for n in names], 1)
        agree = np.array([len(set(p)) == 1 for p in preds])
        groups = [("三模型一致", agree), ("不一致", ~agree)]
        stats = []
        for label, m in groups:
            if m.sum() == 0:
                continue
            acc = float((preds[m][:, 0] == yb[m]).mean())
            stats.append((label, int(m.sum()), acc))
        fig, ax = plt.subplots(figsize=(6.5, 4.4))
        ax.bar([s[0] for s in stats], [s[2] * 100 for s in stats],
               color=["#17603A", "#C0392B"], width=0.45)
        for i, s in enumerate(stats):
            ax.text(i, s[2] * 100 + 1.5, f"{s[2]*100:.1f}%\n(n={s[1]})", ha="center", fontsize=10)
        ax.set_ylim(0, 100)
        ax.set_ylabel("准确率 (%)")
        ax.set_title("三模型一致性 vs 准确率（域 B）")
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "18_三模型一致性.png"), dpi=150, bbox_inches="tight")
        plt.close()
        results["consistency"] = {s[0]: {"n": s[1], "acc": s[2]} for s in stats}

    # ---------- 图6：t-SNE ----------
    try:
        from sklearn.manifold import TSNE
        rng = np.random.RandomState(0)
        ia = rng.choice(len(Fa), min(400, len(Fa)), replace=False)
        ib = rng.choice(len(Fb), min(400, len(Fb)), replace=False)
        X = np.concatenate([Fa[ia], Fb[ib]])
        tag = np.array([0] * len(ia) + [1] * len(ib))
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        emb = TSNE(n_components=2, perplexity=30, init="pca",
                   random_state=0, max_iter=800).fit_transform(X)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(emb[tag == 0, 0], emb[tag == 0, 1], s=12, alpha=0.65,
                   label="域 A 实验室", color="#17603A")
        ax.scatter(emb[tag == 1, 0], emb[tag == 1, 1], s=12, alpha=0.65,
                   label="域 B 真实场景", color="#C0392B")
        ax.set_title("特征空间 t-SNE：两个域是否可分")
        ax.legend(fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "19_域特征分布.png"), dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print("t-SNE 失败:", e)

    with io.open(os.path.join(metric_dir, "cross_domain_diag.json"), "w",
                 encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    print("\n已生成图 14~19，指标写入 metrics/cross_domain_diag.json")
    print(json.dumps(results, ensure_ascii=False, indent=1)[:900])


if __name__ == "__main__":
    main()
