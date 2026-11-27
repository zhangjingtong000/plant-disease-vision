# -*- coding: utf-8 -*-
"""
cross_domain_protocol.py — 跨域评估协议修正

问题：模型输出全类，PlantDoc 只有 28 个目录、可映射 27 类。
直接拿全类 argmax 去比标签，会系统性低估。

三种协议：
  strict   严格：全类 argmax，必须完全命中，否则算错（下界）
  fold     折叠：把全类概率按映射求和折叠到 27 类，再 argmax（推荐）
  restrict 受限：只允许在 27 个候选类里选（与 fold 在 1:1 映射下等价）

同时给出：95% 置信区间、多数类基线、top-k、macro-F1
"""
import os
import sys
import json
import io
import math
import argparse

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from models import build_model

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


def find_root():
    for c in [
        os.path.join(config.DATA_DIR, "plantdoc_extract", "PlantDoc-Dataset-master", "test"),
        os.path.join(config.DATA_DIR, "plantdoc", "test"),
    ]:
        if os.path.isdir(c):
            return c
    return None


def wilson_ci(k, n, z=1.96):
    """Wilson 置信区间，小样本比正态近似稳。"""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="outputs/checkpoints/resnet50_best.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = find_root()
    if not root:
        print("未找到 PlantDoc test")
        return

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"],
                    num_classes=len(config.load_splits()["classes"]),
                    pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    classes = config.get_classes()          # 当前类别数（127）
    n_cls = len(classes)
    # 目标类（按 PlantDoc 目录顺序）
    targets = [c for c in CLS_MAP.values() if c in classes]
    tgt_index = {c: i for i, c in enumerate(targets)}
    n_tgt = len(targets)

    # 源类 -> 目标类的折叠矩阵
    fold = np.zeros((n_cls, n_tgt), dtype=np.float32)
    for dirname, pv_cls in CLS_MAP.items():
        if pv_cls in tgt_index:
            fold[classes.index(pv_cls), tgt_index[pv_cls]] = 1.0
    covered = [i for i in range(len(classes)) if fold[i].sum() > 0]
    print(f"{n_cls} 类中有 {len(covered)} 类能映射到 PlantDoc 的 {n_tgt} 类")
    print(f"无对应的类：{[classes[i] for i in range(len(classes)) if fold[i].sum() == 0]}")

    dir_to_tgt = {d: tgt_index[c] for d, c in CLS_MAP.items() if c in tgt_index}

    # 每个 38 类索引对应的目标类索引（无可映射则为 -1）
    fold_row_to_tgt = {}
    for _i in range(len(classes)):
        _row = fold[_i]
        if _row.sum() > 0:
            fold_row_to_tgt[_i] = int(_row.argmax())

    rows = []
    for dirname in sorted(os.listdir(root)):
        sd = os.path.join(root, dirname)
        if not os.path.isdir(sd) or dirname not in dir_to_tgt:
            continue
        ti = dir_to_tgt[dirname]
        for fn in sorted(os.listdir(sd)):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                continue
            try:
                img = Image.open(os.path.join(sd, fn)).convert("RGB")
            except Exception:
                continue
            x = TF(img).unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = model(x)
            probs38 = torch.softmax(out.float(), 1)[0].cpu().numpy()
            rows.append((probs38, ti, dirname))

    n = len(rows)
    print(f"\n评估样本：{n} 张")

    # ---------- 三种协议 ----------
    strict = 0
    fold_pred = np.zeros(n, dtype=int)
    restrict_pred = np.zeros(n, dtype=int)
    label_tgt = np.zeros(n, dtype=int)
    confs = np.zeros(n)

    for i, (p38, ti, _) in enumerate(rows):
        label_tgt[i] = ti
        # strict：38 类 argmax 是否正好落在该类
        pred38 = int(p38.argmax())
        if pred38 == classes.index(targets[ti]):
            strict += 1
        # fold：折叠到 28 类
        p28 = p38 @ fold
        fold_pred[i] = int(p28.argmax())
        confs[i] = float(p28.max())
        # restrict：只在可映射的原始类中取概率最大者，不做概率质量合并
        best_i = max(covered, key=lambda k: p38[k])
        restrict_pred[i] = int(fold_row_to_tgt[best_i])

    acc_strict = strict / n
    acc_fold = float((fold_pred == label_tgt).mean())
    acc_restrict = float((restrict_pred == label_tgt).mean())

    top3 = float(np.mean([label_tgt[i] in np.argsort(rows[i][0] @ fold)[::-1][:3]
                          for i in range(n)]))
    top5 = float(np.mean([label_tgt[i] in np.argsort(rows[i][0] @ fold)[::-1][:5]
                          for i in range(n)]))

    # 多数类基线
    from collections import Counter
    cnt = Counter(label_tgt.tolist())
    major_cls, major_n = cnt.most_common(1)[0]
    major_acc = major_n / n

    # macro F1
    try:
        from sklearn.metrics import f1_score
        macro_f1 = float(f1_score(label_tgt, fold_pred, average="macro", zero_division=0))
    except Exception:
        macro_f1 = 0.0

    lo, hi = wilson_ci(int((fold_pred == label_tgt).sum()), n)

    print("\n===== 修正后的跨域结果 =====")
    print(f"严格协议（38 类 argmax）      : {acc_strict*100:.2f}%")
    print(f"折叠协议（推荐）             : {acc_fold*100:.2f}%   95% CI [{lo*100:.1f}, {hi*100:.1f}]")
    print(f"受限协议（仅 28 类候选）      : {acc_restrict*100:.2f}%")
    print(f"Top-3                       : {top3*100:.2f}%")
    print(f"Top-5                       : {top5*100:.2f}%")
    print(f"macro-F1                    : {macro_f1:.4f}")
    print(f"多数类基线（始终猜最多那类）  : {major_acc*100:.2f}%  ({targets[major_cls]}, {major_n} 张)")
    print(f"模型是否优于多数类基线        : {'是' if acc_fold > major_acc else '否'}")

    result = {
        "n": n, "n_classes": n_tgt,
        "acc_strict": acc_strict, "acc_fold": acc_fold, "acc_restrict": acc_restrict,
        "ci_low": lo, "ci_high": hi,
        "top3": top3, "top5": top5, "macro_f1": macro_f1,
        "majority_baseline": major_acc, "majority_class": targets[major_cls],
        "majority_count": major_n,
        "beats_majority": bool(acc_fold > major_acc),
        "class_distribution": {targets[k]: v for k, v in cnt.items()},
    }
    os.makedirs(config.METRIC_DIR, exist_ok=True)
    with io.open(os.path.join(config.METRIC_DIR, "cross_domain_protocol.json"), "w",
                 encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print("\n已保存 metrics/cross_domain_protocol.json")


if __name__ == "__main__":
    main()
