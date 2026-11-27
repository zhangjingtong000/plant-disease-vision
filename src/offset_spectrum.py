# -*- coding: utf-8 -*-
"""
offset_spectrum.py — 受控域偏移谱

思路：不依赖第二个数据集，从我们自己有标签的测试集出发，
逐级施加合成偏移，画「准确率 vs 偏移强度」曲线。
再把 PlantDoc 的真实跨域结果标在曲线上，看它等价于第几级偏移。

这比"两个数据集对比两个点"强得多：它展示的是整个偏移谱上的连续行为。

偏移阶梯：
  S0  原始（实验室白底）
  S1  背景替换为自然纹理/随机色
  S2  + 光照与色温扰动
  S3  + 模糊 / JPEG 压缩 / 噪声
  S4  + 旋转、透视、尺度变化
"""
import os
import sys
import json
import io
import argparse

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from models import build_model

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def _open(path):
    if os.name == "nt" and not path.startswith("\\\\?\\"):
        path = "\\\\?\\" + os.path.abspath(path)
    return Image.open(path).convert("RGB")


def leaf_mask(rgb):
    """白底分割，返回 0~1 的羽化掩码。"""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    bg = ((hsv[..., 1] < 30) & (hsv[..., 2] > 200)).astype(np.uint8)
    leaf = 1 - bg
    k = np.ones((5, 5), np.uint8)
    leaf = cv2.morphologyEx(leaf, cv2.MORPH_OPEN, k)
    leaf = cv2.morphologyEx(leaf, cv2.MORPH_CLOSE, k)
    m = cv2.GaussianBlur(leaf.astype(np.float32), (7, 7), 3)
    return m[..., None]


def synth_bg(h, w, seed):
    rng = np.random.RandomState(seed)
    base = rng.uniform(40, 190, (3,)).astype(np.float32)
    bg = np.zeros((h, w, 3), np.float32) + base
    grad = np.linspace(0, 1, h)[:, None, None] * rng.uniform(-60, 60, (3,))
    bg += grad
    bg += rng.normal(0, 12, bg.shape).astype(np.float32)
    bg = cv2.GaussianBlur(bg, (9, 9), 0)
    return np.clip(bg, 0, 255)


def replace_bg(rgb, seed):
    m = leaf_mask(rgb)
    bg = synth_bg(rgb.shape[0], rgb.shape[1], seed)
    out = m * rgb.astype(np.float32) + (1 - m) * bg
    return np.clip(out, 0, 255).astype(np.uint8)


def add_light(rgb, seed):
    rng = np.random.RandomState(seed + 1)
    h, w = rgb.shape[:2]
    x = np.linspace(-1, 1, w)[None, :]
    y = np.linspace(-1, 1, h)[:, None]
    ang = rng.uniform(0, 2 * np.pi)
    grad = (np.cos(ang) * x + np.sin(ang) * y) * rng.uniform(0.15, 0.35)
    f = (1.0 + grad)[..., None]
    out = rgb.astype(np.float32) * f
    out = out * rng.uniform(0.85, 1.15) + rng.uniform(-20, 20)
    return np.clip(out, 0, 255).astype(np.uint8)


def add_blur_noise(rgb, seed):
    rng = np.random.RandomState(seed + 2)
    out = rgb
    if rng.rand() < 0.7:
        k = int(rng.choice([3, 5, 7]))
        out = cv2.GaussianBlur(out, (k, k), 0)
    noise = rng.normal(0, rng.uniform(4, 12), out.shape)
    out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), int(rng.uniform(40, 75))])
    if ok:
        out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out


def add_geometric(rgb, seed):
    rng = np.random.RandomState(seed + 3)
    h, w = rgb.shape[:2]
    ang = rng.uniform(-25, 25)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, rng.uniform(0.8, 1.15))
    tx, ty = rng.uniform(-0.08, 0.08, 2) * np.array([w, h])
    M[:, 2] += [tx, ty]
    out = cv2.warpAffine(rgb, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + rng.uniform(-0.05, 0.05, src.shape).astype(np.float32) * [w, h]
    P = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    out = cv2.warpPerspective(out, P, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return out


def shrink_target(rgb, seed, scale=0.42):
    """把叶片缩小后贴到复杂背景上，模拟真实拍摄中目标只占画面一小块。"""
    rng = np.random.RandomState(seed + 4)
    h, w = rgb.shape[:2]
    m = leaf_mask(rgb)
    # 叶片外接框
    ys, xs = np.where(m[..., 0] > 0.5)
    if len(ys) < 10:
        return rgb
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    crop = rgb[y0:y1 + 1, x0:x1 + 1]
    cm = m[y0:y1 + 1, x0:x1 + 1]
    ch, cw = crop.shape[:2]
    nh, nw = max(8, int(ch * scale)), max(8, int(cw * scale))
    crop_r = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    cm_r = cv2.resize(cm[..., 0], (nw, nh), interpolation=cv2.INTER_AREA)[..., None]

    canvas = synth_bg(h, w, seed + 90).astype(np.float32)
    # 再叠一层纹理，避免纯色
    tex = rng.normal(0, 14, (h, w, 3)).astype(np.float32)
    canvas = np.clip(canvas + cv2.GaussianBlur(tex, (11, 11), 0), 0, 255)

    py = rng.randint(0, max(1, h - nh))
    px = rng.randint(0, max(1, w - nw))
    region = canvas[py:py + nh, px:px + nw]
    canvas[py:py + nh, px:px + nw] = cm_r * crop_r.astype(np.float32) + (1 - cm_r) * region
    return np.clip(canvas, 0, 255).astype(np.uint8)


LEVELS = {
    "S0 原图": lambda im, s: im,
    "S1 换背景": lambda im, s: replace_bg(im, s),
    "S2 +光照": lambda im, s: add_light(replace_bg(im, s), s),
    "S3 +模糊压缩": lambda im, s: add_blur_noise(add_light(replace_bg(im, s), s), s),
    "S4 小目标": lambda im, s: shrink_target(add_blur_noise(add_light(replace_bg(im, s), s), s), s),
    "S5 +视角": lambda im, s: add_geometric(shrink_target(add_blur_noise(add_light(replace_bg(im, s), s), s), s), s),
}


@torch.no_grad()
def evaluate_level(model, items, level_fn, device, seed0=0):
    correct = 0
    for i, (p, y) in enumerate(items):
        im = np.array(_open(p).resize((config.IMG_SIZE, config.IMG_SIZE)))
        im = level_fn(im, seed0 + i)
        x = torch.from_numpy(im).permute(2, 0, 1).float() / 255.0
        x = (x - torch.tensor(config.MEAN)[:, None, None]) / torch.tensor(config.STD)[:, None, None]
        x = x.unsqueeze(0).to(device)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(x)
        correct += int(out.argmax(1).item() == y)
    return correct / max(len(items), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="resnet50")
    parser.add_argument("--n", type=int, default=400)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = config.get_classes()
    ckpt = torch.load(os.path.join(config.CKPT_DIR, f"{args.tag}_best.pth"),
                      map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], len(config.load_splits()["classes"]), pretrained=False)
    # 类别数从 splits 读取，避免写死 38 导致 127 类模型加载失败
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    sp = config.load_splits()
    import random
    rng = random.Random(0)
    subset = rng.sample(sp["test"], min(args.n, len(sp["test"])))
    items = [(os.path.join(config.DATA_DIR, r["path"]), r["label"]) for r in subset]
    print(f"抽样 {len(items)} 张，逐级施加合成偏移")

    results = {}
    SEEDS = (0, 101, 202)
    for name, fn in LEVELS.items():
        accs = []
        for sd in SEEDS:
            accs.append(evaluate_level(model, items, fn, device, seed0=sd))
        m = float(np.mean(accs))
        s = float(np.std(accs))
        results[name] = {"mean": m, "std": s, "all": accs}
        print(f"  {name}: {m*100:.2f}% ± {s*100:.2f}  {[round(a*100,2) for a in accs]}")

    # 真实跨域结果
    real = None
    p = os.path.join(config.METRIC_DIR, "cross_domain_diag.json")
    if os.path.exists(p):
        with io.open(p, encoding="utf-8") as f:
            real = json.load(f).get("acc_fold")

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    xs = list(results.keys())
    ys = [results[k]["mean"] * 100 for k in xs]
    es = [results[k]["std"] * 100 for k in xs]
    ax.errorbar(xs, ys, yerr=es, fmt="o-", color="#17603A", lw=2.4, ms=9,
                capsize=4, ecolor="#8FB3A6", label="合成偏移（3 个随机种子，误差棒为标准差）")
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 13),
                    ha="center", fontsize=10, fontweight="bold")
    if real:
        ax.axhline(real * 100, ls="--", color="#C0392B", lw=1.6,
                   label=f"PlantDoc 真实场景 {real*100:.1f}%")
    ax.set_ylim(0, 105)
    ax.set_ylabel("准确率 (%)")
    ax.set_title("受控域偏移谱：合成偏移下的性能衰减")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    plt.xticks(rotation=12)
    plt.tight_layout()
    out = os.path.join(config.FIG_DIR, "20_域偏移谱.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    with io.open(os.path.join(config.METRIC_DIR, "offset_spectrum.json"), "w",
                 encoding="utf-8") as f:
        json.dump({"levels": results, "plantdoc_acc": real}, f, ensure_ascii=False, indent=1)
    print("图已保存:", out)


if __name__ == "__main__":
    main()
