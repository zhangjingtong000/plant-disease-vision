# -*- coding: utf-8 -*-
"""
prepare_imbalance.py — 构造「不均衡对照组」数据集

用途：验证"配平是否有必要"。
做法：从原始 54381 张里按原始比例抽 7552 张（保持 5507:152 的悬殊比例），
      生成 splits_imbalance.json，用于训练一个对照组模型。

用法：
    python scripts/prepare_imbalance.py
    python src/train.py --model mobilenet_v3_large --epochs 15 --tag imbalance --splits imbalance
"""
import os
import io
import json
import random
import collections

import pyarrow.parquet as pq
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_DIR = os.path.join(BASE, "data", "parquet")
OUT_JSON = os.path.join(BASE, "data", "splits_imbalance.json")
IMG_DIR = os.path.join(BASE, "data", "images_imbalance")

TARGET_TOTAL = 7552
SEED = 42


def main():
    rng = random.Random(SEED)
    os.makedirs(IMG_DIR, exist_ok=True)

    with io.open(os.path.join(BASE, "data", "splits.json"), encoding="utf-8") as f:
        base = json.load(f)
    classes = base["classes"]
    cls2crop = base["cls2crop"]
    crops = base["crops"]

    # 读取原始 label 分布
    by_label = collections.defaultdict(list)
    for fn in ["train-00000.parquet", "train-00001.parquet", "test-00000.parquet"]:
        table = pq.read_table(os.path.join(PARQUET_DIR, fn), columns=["image", "label"])
        labels = table.column("label").to_pylist()
        imgs = table.column("image").to_pylist()
        for lab, img in zip(labels, imgs):
            by_label[lab].append(img["bytes"])

    total = sum(len(v) for v in by_label.values())
    print("原始总数:", total)

    records = {"classes": classes, "crops": crops, "cls2crop": cls2crop,
               "train": [], "val": [], "test": []}

    for lab in range(38):
        cls = classes[lab]
        items = by_label[lab][:]
        rng.shuffle(items)

        # 按原始比例分配样本量
        n_keep = max(1, int(round(len(items) / total * TARGET_TOTAL)))
        items = items[:n_keep]

        n_tr = int(n_keep * 0.7)
        n_val = int(n_keep * 0.1)
        picked = {"train": items[:n_tr],
                  "val": items[n_tr:n_tr + n_val],
                  "test": items[n_tr + n_val:]}

        out_dir = os.path.join(IMG_DIR, cls)
        os.makedirs(out_dir, exist_ok=True)
        for split, arr in picked.items():
            for i, raw in enumerate(arr):
                fname = f"{split}_{i:05d}.jpg"
                dst = os.path.join(out_dir, fname)
                if not os.path.exists(dst):
                    Image.open(io.BytesIO(raw)).convert("RGB").save(dst, "JPEG", quality=95)
                records[split].append({
                    "path": os.path.join("images_imbalance", cls, fname),
                    "label": lab, "class_name": cls,
                    "crop": cls2crop[cls], "crop_id": crops.index(cls2crop[cls]),
                })
        print(f"[{lab:02d}] {cls:<42s} 保留 {n_keep:5d} 张")

    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=1)

    print("\n不均衡对照组:")
    for k in ("train", "val", "test"):
        print(f"  {k}: {len(records[k])}")
    print("已保存:", OUT_JSON)


if __name__ == "__main__":
    main()
