# -*- coding: utf-8 -*-
"""
make_figures.py — 汇总三模型对比图与表格

产出：
    08_模型对比_精度.png
    09_精度速度权衡.png
    model_comparison.csv / .md
"""
import os
import sys
import json
import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

DISPLAY = {"resnet50": "ResNet50", "efficientnet_b0": "EfficientNet-B0",
           "mobilenet_v3_large": "MobileNetV3-Large"}


def load_all():
    rows = []
    for name in config.BACKBONES:
        ep = os.path.join(config.METRIC_DIR, f"eval_{name}.json")
        tp = os.path.join(config.METRIC_DIR, f"train_{name}.json")
        bp = os.path.join(config.METRIC_DIR, "benchmark.json")
        if not os.path.exists(ep):
            continue
        with io.open(ep, encoding="utf-8") as f:
            m = json.load(f)
        params = None
        if os.path.exists(tp):
            with io.open(tp, encoding="utf-8") as f:
                params = json.load(f)["params_m"]
        bench = {}
        if os.path.exists(bp):
            with io.open(bp, encoding="utf-8") as f:
                bench = json.load(f).get("models", {}).get(name, {})
        rows.append({
            "name": name,
            "display": DISPLAY.get(name, name),
            "acc": m["accuracy"], "macro_f1": m["macro_f1"],
            "top3": m["top3_accuracy"], "auc": m["healthy_vs_disease_auc"],
            "params": params,
            "flops": bench.get("flops_g"),
            "latency": bench.get("latency_ms"),
            "throughput": bench.get("throughput_img_s"),
            "size": bench.get("model_size_mb"),
        })
    return rows


def main():
    rows = load_all()
    if not rows:
        print("还没有评估结果，先运行 evaluate.py")
        return

    names = [r["display"] for r in rows]
    x = np.arange(len(rows))

    # ---------- 精度对比 ----------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    w = 0.28

    axes[0].bar(x - w, [r["acc"] * 100 for r in rows], w, label="Accuracy", color="#27ae60")
    axes[0].bar(x, [r["macro_f1"] * 100 for r in rows], w, label="macro-F1", color="#2980b9")
    axes[0].bar(x + w, [r["top3"] * 100 for r in rows], w, label="Top-3", color="#8e44ad")
    axes[0].set_xticks(x); axes[0].set_xticklabels(names)
    axes[0].set_ylim(80, 100); axes[0].set_ylabel("百分比 (%)")
    axes[0].set_title("三模型精度对比"); axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)

    # ---------- 参数量 ----------
    ps = [r["params"] or 0 for r in rows]
    axes[1].bar(x, ps, color="#e67e22")
    axes[1].set_xticks(x); axes[1].set_xticklabels(names)
    axes[1].set_ylabel("参数量 (M)")
    axes[1].set_title("模型体积对比")
    for i, p in enumerate(ps):
        axes[1].text(i, p, f"{p:.1f}M", ha="center", va="bottom", fontsize=9)

    # ---------- 延迟 ----------
    lat = [r["latency"] or 0 for r in rows]
    axes[2].bar(x, lat, color="#c0392b")
    axes[2].set_xticks(x); axes[2].set_xticklabels(names)
    axes[2].set_ylabel("单张延迟 (ms)")
    axes[2].set_title("推理速度对比（GPU）")
    for i, v in enumerate(lat):
        axes[2].text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out1 = os.path.join(config.FIG_DIR, "08_模型对比_精度.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()

    # ---------- 精度-速度权衡 ----------
    if any(r["latency"] for r in rows):
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for r in rows:
            if not r["latency"]:
                continue
            ax.scatter(r["latency"], r["acc"] * 100, s=(r["params"] or 1) * 12,
                       color="#2980b9", alpha=0.7)
            ax.annotate(r["display"], (r["latency"], r["acc"] * 100),
                        textcoords="offset points", xytext=(8, 6), fontsize=10)
        ax.set_xlabel("单张推理延迟 (ms)")
        ax.set_ylabel("测试准确率 (%)")
        ax.set_title("精度-速度权衡（点大小正比于参数量）")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        out2 = os.path.join(config.FIG_DIR, "09_精度速度权衡.png")
        plt.savefig(out2, dpi=150, bbox_inches="tight")
        plt.close()

    # ---------- 表格 ----------
    lines = ["| 模型 | Accuracy | macro-F1 | Top-3 | AUC | 参数量 | FLOPs | 单张延迟 | 模型体积 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| {d} | {a:.2f}% | {f:.2f}% | {t:.2f}% | {u:.4f} | {p} | {g} | {l} | {s} |".format(
            d=r["display"], a=r["acc"] * 100, f=r["macro_f1"] * 100, t=r["top3"] * 100,
            u=r["auc"],
            p=f"{r['params']:.2f}M" if r["params"] else "-",
            g=f"{r['flops']:.2f}G" if r["flops"] else "-",
            l=f"{r['latency']:.1f}ms" if r["latency"] else "-",
            s=f"{r['size']:.1f}MB" if r["size"] else "-",
        ))
    md = "\n".join(lines)
    with io.open(os.path.join(config.METRIC_DIR, "model_comparison.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    with io.open(os.path.join(config.METRIC_DIR, "model_comparison.csv"), "w", encoding="utf-8-sig") as f:
        f.write("模型,Accuracy,macro-F1,Top-3,AUC,参数量M,FLOPsG,延迟ms,体积MB\n")
        for r in rows:
            f.write(f"{r['display']},{r['acc']:.4f},{r['macro_f1']:.4f},{r['top3']:.4f},"
                    f"{r['auc']:.4f},{r['params'] or ''},{r['flops'] or ''},"
                    f"{r['latency'] or ''},{r['size'] or ''}\n")

    print(md)
    print("\n已保存:", out1)


if __name__ == "__main__":
    main()
