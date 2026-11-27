# -*- coding: utf-8 -*-
"""
predict.py — 命令行推理入口

走的是界面（app/web.py）的同一条推理路径：`web.infer()` → 预处理 → 前向 →
softmax → Top-5，所以这里打印的结果就是界面上会看到的结果，不是另写一套。

用法：
    python src/predict.py                      # 推理 assets/samples/ 的示例图
    python src/predict.py path/to/a.jpg ...    # 推理指定图片
    python src/predict.py --split test         # 在 splits.json 的测试集上算准确率
    python src/predict.py --model efficientnet_b0
"""
import os
import sys
import glob
import json
import time
import argparse

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))
sys.path.insert(0, os.path.join(BASE, "app"))

import config  # noqa: E402
import web     # noqa: E402  （app/web.py，界面用的同一个推理实现）


def short(cls: str) -> str:
    """把 'Tomato___Late_blight' 压成 'Tomato / Late blight' 方便看。"""
    return cls.replace("___", " / ").replace("__", " / ").replace("_", " ").strip()


def load_model(name):
    model, model_name = web.get_model(name)       # 缺权重会抛 FileNotFoundError
    return model, model_name


def predict_one(model_name, path):
    from PIL import Image
    img = Image.open(path).convert("RGB")
    t0 = time.perf_counter()
    top5, probs, pred = web.infer(img, model_name)
    dt = (time.perf_counter() - t0) * 1000
    return top5, pred, dt, img


def bench(model_name, path, runs=20):
    """预热后重复推理，返回平均毫秒（用于和 results/benchmark.json 的口径对齐）。"""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    for _ in range(3):
        web.infer(img, model_name)
    t0 = time.perf_counter()
    for _ in range(runs):
        web.infer(img, model_name)
    return (time.perf_counter() - t0) * 1000 / runs


def eval_split(model_name, split, img_dir=None):
    """在 splits.json 的某个划分上算 top-1 / top-3 准确率，同样走 web.infer。"""
    from PIL import Image
    records = config.load_splits()[split]
    root = img_dir or config.DATA_DIR
    n = top1 = top3 = 0
    t0 = time.perf_counter()
    for rec in records:
        p = os.path.join(root, rec["path"])
        if not os.path.exists(p):
            continue
        img = Image.open(p).convert("RGB")
        top5, _probs, pred = web.infer(img, model_name)
        n += 1
        top1 += int(pred == rec["class_name"])
        top3 += int(rec["class_name"] in [c for c, _ in top5[:3]])
    dt = time.perf_counter() - t0
    return {"split": split, "n": n, "top1": top1 / max(n, 1),
            "top3": top3 / max(n, 1), "seconds": round(dt, 1)}


def main():
    ap = argparse.ArgumentParser(description="叶片病害推理（复用 app/web.py 的推理路径）")
    ap.add_argument("files", nargs="*", help="图片路径；留空则用 assets/samples/*")
    ap.add_argument("--model", default="resnet50", choices=config.BACKBONES)
    ap.add_argument("--split", default="", help="改成在 splits.json 的 val/test 上评估")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--bench", action="store_true", help="额外出一次预热后的平均延迟")
    args = ap.parse_args()

    try:
        _model, model_name = load_model(args.model)
    except FileNotFoundError as e:
        print(f"[缺少权重] {e}", file=sys.stderr)
        print("先训练一次：python src/train.py --model %s --epochs 3" % args.model,
              file=sys.stderr)
        return 2

    classes = config.get_classes()
    print(f"checkpoint: outputs/checkpoints/{args.model}_best.pth "
          f"（{model_name}，{len(classes)} 类）")
    print(f"设备: {'cuda' if web.DEVICE.type == 'cuda' else 'cpu'}\n")

    if args.split:
        res = eval_split(args.model, args.split)
        print(json.dumps(res, ensure_ascii=False))
        return 0

    paths = args.files or sorted(glob.glob(os.path.join(BASE, "assets", "samples", "*.jpg")))
    if not paths:
        print("没有找到图片。", file=sys.stderr)
        return 2

    for p in paths:
        top5, pred, dt, img = predict_one(args.model, p)
        leaf_ok, feats = web.is_leaf_like(img)
        print(f"{os.path.basename(p)}  →  {short(pred)}  ({top5[0][1]*100:.1f}%)  "
              f"{dt:.0f} ms  叶片检查={'通过' if leaf_ok else '未通过'}")
        for c, pr in top5[:args.topk]:
            print(f"    {pr*100:6.2f}%  {short(c)}")
        print()

    if args.bench:
        ms = bench(args.model, paths[0])
        print(f"预热后平均延迟（batch=1，{os.path.basename(paths[0])}）：{ms:.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
