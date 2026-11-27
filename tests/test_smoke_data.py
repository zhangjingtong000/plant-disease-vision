# -*- coding: utf-8 -*-
"""第 6 层：数据接入冒烟（小子集 → splits.json → checkpoint → 推理）。

与 `test_rules.py` 的分工：那边验证的是**发布级完整数据集**的完整性（每类都有
图、每类都有知识条目）；这里验证的是**小样本子集**也能把链路走通，重点是
`prepare_data.py` 在小数据下走的另一条配额分支（总量 < 200 时按 70/10/20
等比缩减），这是完整数据集永远走不到的分支。

没有 `data/splits.json` 时整个文件跳过（`conftest.py` 会处理，这里再兜一层）。

运行：python -m pytest tests/test_smoke_data.py -v
"""
import os
import sys
import json

import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))
sys.path.insert(0, os.path.join(BASE, "app"))

import config  # noqa: E402
import conftest as _env  # noqa: E402

pytestmark = pytest.mark.skipif(not _env.HAS_DATA,
                                reason="缺少 data/splits.json，先按 README 生成数据")


def _classes_with_images():
    out = {}
    for cls in config.get_classes():
        d = os.path.join(config.IMAGES_DIR, cls)
        if not os.path.isdir(d):
            continue
        files = [f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if files:
            out[cls] = len(files)
    return out


class TestSplitsStructure:
    def test_required_keys(self):
        """splits.json 必须带类别、作物映射与三个划分清单。"""
        sp = config.load_splits()
        for k in ("classes", "crops", "cls2crop", "train", "val", "test"):
            assert k in sp, f"splits.json 缺字段 {k}"
        assert sp["classes"], "类别清单为空"
        assert len(sp["crops"]) == len(set(sp["crops"])), "作物清单有重复"

    def test_records_point_to_real_files(self):
        """每条记录都要指向真实图片，且 label 在类别范围内。"""
        sp = config.load_splits()
        n = len(sp["classes"])
        bad, oob = [], []
        for split in ("train", "val", "test"):
            for r in sp[split]:
                if not os.path.exists(os.path.join(config.DATA_DIR, r["path"])):
                    bad.append(r["path"])
                if not 0 <= r["label"] < n:
                    oob.append((split, r["label"]))
        assert not bad, f"记录指向的文件不存在: {bad[:3]}"
        assert not oob, f"越界 label: {oob[:3]}"

    def test_no_record_in_two_splits(self):
        """同一条记录不能同时出现在 train / val / test（数据泄漏）。"""
        sp = config.load_splits()
        seen = {}
        for split in ("train", "val", "test"):
            for r in sp[split]:
                assert r["path"] not in seen, f"{r['path']} 同时出现在 {seen.get(r['path'])} 与 {split}"
                seen[r["path"]] = split

    def test_small_subset_quota_ratio(self):
        """小数据分支：每类总量不足 200 时按 70/10/20 等比缩减。"""
        sp = config.load_splits()
        by_cls = {}
        for split in ("train", "val", "test"):
            for r in sp[split]:
                by_cls.setdefault(r["class_name"], {}).setdefault(split, 0)
                by_cls[r["class_name"]][split] += 1

        checked = 0
        for cls, cnt in by_cls.items():
            total = sum(cnt.values())
            if total == 0 or total >= 200:
                continue        # 只覆盖小数据分支
            checked += 1
            expect_tr = int(round(total * 0.7))
            expect_val = int(round(total * 0.1))
            assert cnt.get("train", 0) == expect_tr, f"{cls}: train {cnt.get('train', 0)} != {expect_tr}"
            assert cnt.get("val", 0) == expect_val, f"{cls}: val {cnt.get('val', 0)} != {expect_val}"
            assert cnt.get("test", 0) == total - expect_tr - expect_val
        if checked == 0:
            pytest.skip("当前数据集每类都 >= 200 张，不覆盖小数据分支")


class TestInferencePath:
    def test_checkpoint_infer_returns_valid_top5(self):
        """有 checkpoint 时，界面同一条推理路径必须给出合法 Top-5。"""
        ckpt = os.path.join(config.CKPT_DIR, f"{config.BACKBONES[0]}_best.pth")
        if not os.path.exists(ckpt):
            pytest.skip(f"缺 checkpoint：{ckpt}")

        import web
        from PIL import Image

        classes, n_imgs = _classes_with_images(), None
        assert classes, "没有任何类别目录里有图片"
        cls = sorted(classes)[0]
        d = os.path.join(config.IMAGES_DIR, cls)
        f = sorted(x for x in os.listdir(d) if x.lower().endswith(".jpg"))[0]
        img = Image.open(os.path.join(d, f)).convert("RGB")

        top5, probs, pred = web.infer(img, config.BACKBONES[0])
        assert len(top5) == 5, f"Top-5 长度异常: {len(top5)}"
        assert pred in config.get_classes(), f"预测类别不在类别空间内: {pred}"
        assert all(c in config.get_classes() for c, _ in top5), "Top-5 里有未知类别"
        assert sum(float(p) for p in probs) == pytest.approx(1.0, abs=1e-3), "softmax 概率和不为 1"
        assert 0.0 <= top5[0][1] <= 1.0
        n_imgs = len(classes)
        print(f"  推理路径可用：{cls} → {pred}（{top5[0][1]*100:.1f}%），有图类别 {n_imgs} 个")
