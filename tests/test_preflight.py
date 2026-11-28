# -*- coding: utf-8 -*-
"""启动前置检查的单元测试：缺数据进只读演示模式、缺权重只告警，都不伪造文件。

运行：python -m pytest tests/test_preflight.py -q
"""
import os
import sys
import tempfile

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))

import preflight  # noqa: E402

BACKBONES = ["resnet50", "efficientnet_b0", "mobilenet_v3_large"]


def _write_splits(data_dir):
    with open(os.path.join(data_dir, "splits.json"), "w", encoding="utf-8") as f:
        f.write('{"classes": [], "records": []}')


def test_missing_data_enters_read_only_mode():
    with tempfile.TemporaryDirectory() as tmp:
        read_only, messages = preflight.check(tmp, os.path.join(tmp, "kein"), BACKBONES)

    # 不再是 blocking=True 直接退出：缺数据时以只读演示模式启动
    assert read_only is True
    text = "\n".join(messages)
    assert "splits.json" in text
    assert "prepare_data.py" in text
    assert "只读演示模式" in text
    assert "需数据" in text


def test_missing_weights_warn_but_allow_start():
    with tempfile.TemporaryDirectory() as tmp:
        _write_splits(tmp)
        blocking, messages = preflight.check(tmp, os.path.join(tmp, "ckpt"), BACKBONES)

    assert blocking is False
    text = "\n".join(messages)
    assert "resnet50_best.pth" in text
    assert "efficientnet_b0_best.pth" in text
    # 只提示，不生成权重
    assert not os.path.exists(os.path.join(tmp, "ckpt"))


def test_all_weights_present_is_silent():
    with tempfile.TemporaryDirectory() as tmp:
        _write_splits(tmp)
        ckpt = os.path.join(tmp, "ckpt")
        os.makedirs(ckpt)
        for name in BACKBONES:
            open(os.path.join(ckpt, name + "_best.pth"), "wb").close()

        blocking, messages = preflight.check(tmp, ckpt, BACKBONES)

    assert blocking is False
    assert messages == []


def test_partial_weights_report_only_missing():
    with tempfile.TemporaryDirectory() as tmp:
        _write_splits(tmp)
        ckpt = os.path.join(tmp, "ckpt")
        os.makedirs(ckpt)
        open(os.path.join(ckpt, "resnet50_best.pth"), "wb").close()

        _, messages = preflight.check(tmp, ckpt, BACKBONES)

    text = "\n".join(messages)
    assert "resnet50_best.pth" not in text
    assert "mobilenet_v3_large_best.pth" in text
