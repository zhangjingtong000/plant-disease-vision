# -*- coding: utf-8 -*-
"""意图分类器的回归测试。

只验证**接口与产物**，不依赖 bge 模型：`intent_clf.pkl` 里是一个 sklearn 分类器，
喂它任意 512 维向量都能 `predict_proba`；真正的端到端（文本 → 句向量 → 意图）
需要 `assets/embedding/bge-small-zh-v1.5`，那部分在 `test_intent_e2e` 里按需跳过。
"""
import os
import pickle
import sys

import numpy as np
import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKL = os.path.join(BASE, "src", "intent_model", "intent_clf.pkl")
MODEL_DIR = os.path.join(BASE, "assets", "embedding", "bge-small-zh-v1.5")

EXPECTED_CLASSES = {"compare", "control", "occurrence", "symptom", "system"}


@pytest.fixture(scope="module")
def payload():
    if not os.path.isfile(PKL):
        pytest.skip("没有 intent_clf.pkl —— 先跑 src/intent_model/train_intent_clf.py")
    with open(PKL, "rb") as f:
        return pickle.load(f)


def test_payload_shape(payload):
    """产物必须是 {clf, classes, meta} 三键，且类别与训练集一致。"""
    assert set(payload) >= {"clf", "classes", "meta"}
    assert set(payload["classes"]) == EXPECTED_CLASSES


def test_meta_reports_cross_validation(payload):
    """meta 里要留下交叉验证结果，且不能是「训练集准确率」冒充的。"""
    meta = payload["meta"]
    assert meta["n_samples"] == 450
    assert 0.85 <= meta["cv_accuracy_mean"] <= 1.0, meta
    assert "classifier" in meta and "vectorizer" in meta


def test_predict_proba_on_512d_vectors(payload):
    """喂 512 维向量能出概率，形状与类别数一致。"""
    clf = payload["clf"]
    x = np.random.default_rng(0).normal(size=(3, 512))
    proba = clf.predict_proba(x)
    assert proba.shape == (3, len(payload["classes"]))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.skipif(not os.path.isdir(MODEL_DIR),
                    reason="需要 assets/embedding/bge-small-zh-v1.5 才能做端到端")
def test_end_to_end_with_encoder():
    """真实文本 → 句向量 → 意图，5 条典型问句全部命中。"""
    import torch
    from transformers import AutoModel, AutoTokenizer

    sys.path.insert(0, os.path.join(BASE, "src"))
    import ai_qa

    clf = ai_qa._load_intent_clf()
    assert clf is not None, "有 pkl 和 bge 模型却加载失败"

    cases = [
        ("这个病什么症状", "symptom"),
        ("怎么防治", "control"),
        ("为什么会得", "occurrence"),
        ("两种病怎么区分", "compare"),
        ("你是谁", "system"),
    ]
    for q, want in cases:
        got = ai_qa._detect_intent_by_model(q)
        assert got == want, f"{q!r} 预测成 {got!r}，期望 {want!r}"
