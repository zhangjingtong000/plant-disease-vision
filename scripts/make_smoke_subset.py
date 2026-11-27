# -*- coding: utf-8 -*-
"""
make_smoke_subset.py — 从完整 PlantVillage parquet 里抽一个小样本子集。

用途：仓库不附带数据，而完整 PlantVillage 有 16GB。要让「准备数据 → 训练 →
推理」这条链路能被真正跑通一次，只需要一个几十 MB 的子集。

本脚本不改变数据格式：输出仍是 `prepare_data.py` 认识的三个 parquet
（train-00000 / train-00001 / test-00000，image 为 struct<bytes,path>，
label 为 38 类里的真实索引），所以后续步骤不需要任何特殊分支。

用法：
    python scripts/make_smoke_subset.py --src-parquet E:/test/plant/data/parquet
    python scripts/make_smoke_subset.py --src-parquet <dir> --per-class 40 --classes Apple__Apple_scab,Tomato___healthy

默认抽 8 类 × 40 张 = 320 张（约 3MB），覆盖 assets/samples/ 里 4 张示例图
对应的类别，方便推理结果对照。
"""
import os
import json
import argparse
import random
import collections

import pyarrow as pa
import pyarrow.parquet as pq

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_FILES = ["train-00000.parquet", "train-00001.parquet", "test-00000.parquet"]
DST_FILES = {"train-00000.parquet": "train",
             "train-00001.parquet": "train",
             "test-00000.parquet": "test"}

# 默认子集：assets/samples/ 的 4 张示例图都能落在其中
# （苹果黑星 0、玉米锈病 8、土豆健康 22、番茄晚疫 30），另加 4 个对照组
DEFAULT_CLASSES = [
    "Apple__Apple_scab",            # 0
    "Apple___healthy",              # 3
    "Corn_(maize)___Common_rust_",  # 8
    "Corn_(maize)___healthy",       # 10
    "Potato___healthy",             # 22
    "Tomato___Early_blight",        # 29
    "Tomato___Late_blight",         # 30
    "Tomato___healthy",             # 37
]

RNG_SEED = 42


def class_names():
    """复用 prepare_data.py 的 38 类名，保证 label 索引口径一致。"""
    import prepare_data
    return list(prepare_data.CLASS_NAMES)


def collect(src_dir, wanted, per_class):
    """按 label 逐批扫描源 parquet，每类收到 per_class 张就停止。"""
    buckets = collections.defaultdict(list)
    for fn in SRC_FILES:
        p = os.path.join(src_dir, fn)
        if not os.path.exists(p):
            raise SystemExit(f"源 parquet 不存在：{p}")
        part = "test" if fn.startswith("test") else "train"
        for batch in pq.ParquetFile(p).iter_batches(batch_size=512,
                                                   columns=["image", "label"]):
            labels = batch.column("label").to_pylist()
            imgs = batch.column("image").to_pylist()
            for lab, img in zip(labels, imgs):
                if lab not in wanted:
                    continue
                if len(buckets[lab]) >= per_class:
                    continue
                buckets[lab].append((part, img))
            if all(len(buckets[lab]) >= per_class for lab in wanted):
                break
        if all(len(buckets[lab]) >= per_class for lab in wanted):
            break
    return buckets


def main():
    ap = argparse.ArgumentParser(description="从完整 PlantVillage parquet 抽小子集")
    ap.add_argument("--src-parquet", required=True, help="完整 parquet 所在目录")
    ap.add_argument("--dest-parquet", default=os.path.join(BASE, "data", "parquet"))
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--classes", default="",
                    help="逗号分隔的类别名；留空用默认 8 类")
    args = ap.parse_args()

    names = class_names()
    wanted_names = ([c for c in args.classes.split(",") if c.strip()]
                    if args.classes else DEFAULT_CLASSES)
    unknown = [c for c in wanted_names if c not in names]
    if unknown:
        raise SystemExit(f"未知类别：{unknown}")
    wanted = {names.index(c): c for c in wanted_names}

    print(f"源目录：{args.src_parquet}")
    print(f"目标目录：{args.dest_parquet}")
    print(f"类别 {len(wanted)} 个，每类最多 {args.per_class} 张，种子 {RNG_SEED}")

    buckets = collect(args.src_parquet, wanted, args.per_class)
    rng = random.Random(RNG_SEED)

    # 按源 split 的 train/test 分流，写出与 prepare_data.py 同名的三个文件
    out = collections.defaultdict(list)
    for lab in sorted(wanted):
        items = buckets.get(lab, [])[:]
        rng.shuffle(items)
        tr = [it for it in items if it[0] == "train"]
        te = [it for it in items if it[0] == "test"]
        n_te = min(len(te), max(1, round(len(items) * 0.25)))
        head = tr + te[:n_te]
        tail = te[n_te:]
        half = len(head) // 2
        out["train-00000.parquet"] += [(it[1], lab) for it in head[:half]]
        out["train-00001.parquet"] += [(it[1], lab) for it in head[half:]]
        out["test-00000.parquet"] += [(it[1], lab) for it in tail]
        print(f"  [{lab:02d}] {wanted[lab]:<46s} {len(items):3d} 张")

    os.makedirs(args.dest_parquet, exist_ok=True)
    empty = pa.table({"image": pa.array([], type=pa.struct(
        [pa.field("bytes", pa.binary()), pa.field("path", pa.string())])),
        "label": pa.array([], type=pa.int64())})
    image_type = empty.schema.field("image").type

    total = 0
    for fn in DST_FILES:
        rows = out[fn]
        pq.write_table(pa.table({
            "image": pa.array([r[0] for r in rows], type=image_type),
            "label": pa.array([r[1] for r in rows], type=pa.int64()),
        }), os.path.join(args.dest_parquet, fn))
        total += len(rows)
        print(f"写出 {fn}: {len(rows)} 行")

    size = sum(os.path.getsize(os.path.join(args.dest_parquet, f)) for f in DST_FILES)
    print(f"\n合计 {total} 张，{size/1024/1024:.1f} MB -> {args.dest_parquet}")
    print("覆盖类别：", json.dumps({str(k): v for k, v in sorted(wanted.items())},
                              ensure_ascii=False))


if __name__ == "__main__":
    main()
