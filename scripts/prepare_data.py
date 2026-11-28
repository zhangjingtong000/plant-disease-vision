# -*- coding: utf-8 -*-
"""
prepare_data.py
从 PlantVillage parquet 文件构建配平数据集。

流程：
1. 读取 data/parquet 下的全部分片（官方 train 43,503 / test 10,878）
2. 每个类别按 70/10/20 抽取：训练 140、验证 20、测试 40（土豆健康仅 152，按可用量取）
3. 图片落盘到 data/images/<类别>/<split>_<序号>.jpg
4. 生成 data/splits.json（含 38 类名、14 作物映射、样本清单）
5. 生成配平前后分布对比图
"""
import os
import io
import json
import random
import collections

import pyarrow.parquet as pq
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- 配置 ----------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_DIR = os.path.join(BASE, "data", "parquet")
OUT_IMAGES = os.path.join(BASE, "data", "images")
OUT_JSON = os.path.join(BASE, "data", "splits.json")
FIG_DIR = os.path.join(BASE, "outputs", "figures")
os.makedirs(OUT_IMAGES, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

RNG_SEED = 42
N_TRAIN = 140   # 每类训练
N_VAL = 20      # 每类验证
N_TEST = 40     # 每类测试

# 38 类名称，顺序与 parquet 的 label 编码一致
CLASS_NAMES = [
    "Apple__Apple_scab", "Apple__Black_rot", "Apple__Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy", "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot",
    "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
]

# 作物名规范化（把 Cherry_(including_sour) 之类的括号去掉）
CROP_DISPLAY = {
    "Apple": "苹果", "Blueberry": "蓝莓", "Cherry": "樱桃", "Corn": "玉米",
    "Grape": "葡萄", "Orange": "橙", "Peach": "桃", "Pepper,_bell": "甜椒",
    "Potato": "土豆", "Raspberry": "树莓", "Soybean": "大豆", "Squash": "南瓜",
    "Strawberry": "草莓", "Tomato": "番茄",
}


def split_crop(cls: str) -> str:
    """从类名提取作物名。兼容 __ 与 ___ 两种分隔符。"""
    head = cls.split("___")[0] if "___" in cls else cls.split("__")[0]
    head = head.replace("_", " ").strip()
    head = head.split("(")[0].strip()
    head = head.replace(" ", "_") if head in ("Pepper,_bell",) else head
    return head


def build_crop_map():
    crops = []
    cls2crop = {}
    for cls in CLASS_NAMES:
        head = cls.split("___")[0] if "___" in cls else cls.split("__")[0]
        # Cherry_(including_sour) -> Cherry；Corn_(maize) -> Corn
        crop_key = head.split("(")[0].strip().rstrip("_,")
        cls2crop[cls] = crop_key
        if crop_key not in crops:
            crops.append(crop_key)
    return crops, cls2crop


def list_parquet():
    """data/parquet 下的所有 parquet 分片，按文件名分组。

    文件名含 `test` 的算测试集，其余算训练集；HuggingFace 下载下来的
    `train-00000-of-00002.parquet` 这类带分片后缀的名字也能直接用，
    不要求改名为固定的 train-00000.parquet。
    """
    if not os.path.isdir(PARQUET_DIR):
        raise SystemExit(
            "未找到 {0}。请先从 HuggingFace 下载 PlantVillage parquet，"
            "见 README「PlantVillage parquet 从哪来」。".format(PARQUET_DIR))
    files = sorted(f for f in os.listdir(PARQUET_DIR)
                   if f.lower().endswith(".parquet"))
    if not files:
        raise SystemExit(
            "{0} 里没有 .parquet 文件。请先从 HuggingFace 下载 PlantVillage "
            "parquet，见 README「PlantVillage parquet 从哪来」。".format(PARQUET_DIR))
    test_files = [f for f in files if "test" in f.lower()]
    train_files = [f for f in files if f not in test_files]
    return train_files, test_files


def load_parquet_group(files):
    """读取若干 parquet，返回 {label: [(image_bytes, path), ...]}"""
    by_label = collections.defaultdict(list)
    for fn in files:
        p = os.path.join(PARQUET_DIR, fn)
        table = pq.read_table(p, columns=["image", "label"])
        labels = table.column("label").to_pylist()
        imgs = table.column("image").to_pylist()
        for lab, img in zip(labels, imgs):
            by_label[lab].append((img["bytes"], img["path"]))
    return by_label


def save_jpg(raw: bytes, dst: str):
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img.save(dst, "JPEG", quality=95)


def main():
    rng = random.Random(RNG_SEED)

    train_files, test_files = list_parquet()
    print("读取 parquet：训练分片 {0} 个、测试分片 {1} 个 ...".format(
        len(train_files), len(test_files)))
    train_pool = load_parquet_group(train_files)
    test_pool = load_parquet_group(test_files) if test_files else {}
    raw_counts = {i: len(train_pool.get(i, [])) + len(test_pool.get(i, [])) for i in range(38)}
    print("原始总数:", sum(raw_counts.values()))

    crops, cls2crop = build_crop_map()
    print(f"作物数: {len(crops)} -> {crops}")

    records = {"classes": CLASS_NAMES, "crops": crops, "cls2crop": cls2crop,
               "train": [], "val": [], "test": []}
    balanced_counts = {}

    for lab in range(38):
        cls = CLASS_NAMES[lab]
        tr = train_pool.get(lab, [])[:]
        te = test_pool.get(lab, [])[:]
        rng.shuffle(tr)
        rng.shuffle(te)

        # 按 70/10/20 的比例分配；若总量不足 200，则按比例等比缩减
        total = len(tr) + len(te)
        if total >= 200:
            n_tr, n_val, n_te = N_TRAIN, N_VAL, N_TEST
        else:
            n_tr = int(round(total * 0.7))
            n_val = int(round(total * 0.1))
            n_te = total - n_tr - n_val

        picked_test = te[:n_te]
        need = n_te - len(picked_test)

        n_tr = min(n_tr, len(tr))
        picked_train = tr[:n_tr]
        remain = tr[n_tr:]
        n_val = min(n_val, len(remain))
        picked_val = remain[:n_val]
        spill = remain[n_val:]
        if need > 0 and spill:
            picked_test.extend(spill[:need])

        out_dir = os.path.join(OUT_IMAGES, cls)
        os.makedirs(out_dir, exist_ok=True)

        for split, items in (("train", picked_train), ("val", picked_val), ("test", picked_test)):
            for i, (raw, _orig) in enumerate(items):
                fname = f"{split}_{i:04d}.jpg"
                dst = os.path.join(out_dir, fname)
                if not os.path.exists(dst):
                    save_jpg(raw, dst)
                records[split].append({
                    "path": os.path.join("images", cls, fname),
                    "label": lab,
                    "class_name": cls,
                    "crop": cls2crop[cls],
                    "crop_id": crops.index(cls2crop[cls]),
                })

        balanced_counts[lab] = len(picked_train) + len(picked_val) + len(picked_test)
        print(f"[{lab:02d}] {cls:<48s} train={len(picked_train):3d} val={len(picked_val):3d} "
              f"test={len(picked_test):3d}  total={balanced_counts[lab]:3d}")

    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=1)

    print("\n汇总:")
    print("  train:", len(records["train"]))
    print("  val  :", len(records["val"]))
    print("  test :", len(records["test"]))
    print("  合计 :", sum(len(records[k]) for k in ("train", "val", "test")))

    # ---------- 分布对比图 ----------
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    labels_short = [c.replace("___", " / ").replace("__", " / ").replace("_", " ")[:34] for c in CLASS_NAMES]
    raw_vals = [raw_counts[i] for i in range(38)]
    bal_vals = [balanced_counts[i] for i in range(38)]

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    y = range(38)
    axes[0].barh(list(y), raw_vals, color="#c0392b")
    axes[0].set_yticks(list(y))
    axes[0].set_yticklabels(labels_short, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_title(f"配平前：38 类原始样本分布（共 {sum(raw_vals)} 张，最大 {max(raw_vals)} / 最小 {min(raw_vals)}）", fontsize=13)
    axes[0].set_xlabel("样本数")

    axes[1].barh(list(y), bal_vals, color="#27ae60")
    axes[1].set_yticks(list(y))
    axes[1].set_yticklabels(labels_short, fontsize=7)
    axes[1].invert_yaxis()
    axes[1].set_title(f"配平后：每类约 200 张（共 {sum(bal_vals)} 张）", fontsize=13)
    axes[1].set_xlabel("样本数")

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "01_类别分布对比.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("分布图已保存:", fig_path)


if __name__ == "__main__":
    main()
