# -*- coding: utf-8 -*-
"""训练意图分类器，产出 `intent_clf.pkl`（供 `src/ai_qa.py` 的 `_load_intent_clf` 加载）。

用法：
    python src/intent_model/train_intent_clf.py \
        --model assets/embedding/bge-small-zh-v1.5 \
        --out src/intent_model/intent_clf.pkl

分类器输入是 bge-small-zh 的**句向量**（与 `ai_qa._detect_intent_by_model` 完全一致：
取 last_hidden_state 按 attention_mask 做均值，再 L2 归一化）。
分类器本体是 LogisticRegression：450 条样本、4 个类别，线性模型足够，
而且 `predict_proba` 给出的置信度可以直接用于「低于 0.30 就交给关键词规则」的兜底。

为什么不用微调：样本只有 450 条，微调 bge 需要 GPU 且容易过拟合；
句向量 + 线性分类器在交叉验证下已经够用，训练 10 秒内完成，也不需要额外存权重。

产物是一个 dict：{"clf": 分类器, "classes": 类别名列表, "meta": {...}}。
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

# 训练集与本脚本同目录
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from cases_intent_train import CASES  # noqa: E402


def load_encoder(model_dir: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModel.from_pretrained(model_dir).eval()
    return tok, mdl


def encode(texts, tok, mdl, max_length: int = 64, batch_size: int = 32):
    """句向量编码：last_hidden_state 按 mask 求均值 + L2 归一化。

    与 `ai_qa._detect_intent_by_model` 的实现逐行对应——训练与推理必须用同一套
    向量化方式，否则分类器在线上会凭空掉点。
    """
    import numpy as np
    import torch

    vecs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        with torch.no_grad():
            enc = tok(chunk, padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt")
            h = mdl(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
            v = torch.nn.functional.normalize(v, dim=-1).numpy()
        vecs.append(v)
    return np.vstack(vecs)


def main():
    ap = argparse.ArgumentParser(description="训练意图分类器")
    ap.add_argument("--model", default="assets/embedding/bge-small-zh-v1.5",
                    help="bge-small-zh 模型目录（或 HF 名字）")
    ap.add_argument("--out", default=os.path.join(HERE, "intent_clf.pkl"))
    ap.add_argument("--C", type=float, default=4.0, help="逻辑回归正则强度倒数")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    texts = [c[0] for c in CASES]
    labels = [c[1] for c in CASES]
    classes = sorted(set(labels))
    print(f"样本 {len(texts)} 条，类别 {len(classes)} 个：{classes}")

    tok, mdl = load_encoder(args.model)
    X = encode(texts, tok, mdl)
    print(f"句向量 {X.shape[0]}×{X.shape[1]}")

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=args.C, max_iter=2000, random_state=args.seed),
    )

    # 5 折交叉验证：留一折不参与训练，报告的是「新问题」上的准确率
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    scores = cross_val_score(clf, X, labels, cv=skf, scoring="accuracy")
    print(f"5 折交叉验证准确率：{scores.mean():.4f} ± {scores.std():.4f}  "
          f"({', '.join(f'{s:.3f}' for s in scores)})")

    # 交叉验证之后再用全量数据拟合，产出实际使用的模型
    clf.fit(X, labels)
    train_acc = clf.score(X, labels)
    print(f"全量训练集准确率：{train_acc:.4f}")

    payload = {
        "clf": clf,
        "classes": classes,
        "meta": {
            "n_samples": len(texts),
            "cv_accuracy_mean": round(float(scores.mean()), 4),
            "cv_accuracy_std": round(float(scores.std()), 4),
            "train_accuracy": round(float(train_acc), 4),
            "encoder": os.path.basename(args.model.rstrip("/\\")) or args.model,
            "vectorizer": "bge-small-zh-v1.5 mean-pooled + L2 normalize",
            "classifier": f"StandardScaler + LogisticRegression(C={args.C})",
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(payload, f)
    size = os.path.getsize(args.out)
    print(f"已写出 {args.out}（{size} 字节）")
    print(json.dumps(payload["meta"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
