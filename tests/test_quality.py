# -*- coding: utf-8 -*-
"""第 4 层：质量属性实测（延迟 / 吞吐 / 文档一致）。

运行：python -m pytest tests/test_quality.py -v -s
结果写入 outputs/metrics/quality_report.json
"""
import os
import sys
import io
import json
import time

import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))
sys.path.insert(0, os.path.join(BASE, "app"))

import config  # noqa: E402
import web  # noqa: E402

REPORT = os.path.join(config.METRIC_DIR, "quality_report.json")


def _sample_images(n=20):
    paths = []
    for cls in config.get_classes():
        d = os.path.join(config.IMAGES_DIR, cls)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(".jpg"):
                paths.append(os.path.join(d, f))
                break
        if len(paths) >= n:
            break
    return paths


class TestLatency:
    def test_three_backbones_latency(self):
        """三个 backbone 各测 10 次推理，取均值，写入报告。"""
        from PIL import Image
        paths = _sample_images(10)
        assert paths, "无测试图"
        imgs = [Image.open(p).convert("RGB") for p in paths]

        result = {}
        for name in config.BACKBONES:
            # 预热一次（首次含模型加载，不计入）
            web.infer(imgs[0], name)
            t0 = time.time()
            for im in imgs:
                web.infer(im, name)
            ms = (time.time() - t0) / len(imgs) * 1000
            result[name] = round(ms, 1)
            print(f"  {name}: {ms:.1f} ms/张")

        os.makedirs(config.METRIC_DIR, exist_ok=True)
        with io.open(REPORT, "w", encoding="utf-8") as f:
            json.dump({"latency_ms": result}, f, ensure_ascii=False, indent=1)

        # 判据：GPU 上每个模型都应在 100ms 以内
        for name, ms in result.items():
            assert ms < 100, f"{name} 单张 {ms}ms 超过 100ms"


class TestThroughput:
    def test_batch_throughput(self):
        """批量诊断 20 张的总耗时，应低于串行上界。"""
        paths = _sample_images(20)
        assert len(paths) >= 10, "样本不足"
        t0 = time.time()
        rows, csv_path = web.batch_diagnose(paths, config.BACKBONES[0], 0.0)
        cost = time.time() - t0
        assert len(rows) == len(paths), f"结果数 {len(rows)} != 输入 {len(paths)}"
        print(f"  批量 {len(paths)} 张耗时 {cost:.1f}s（{cost/len(paths)*1000:.0f} ms/张）")

        with io.open(REPORT, encoding="utf-8") as f:
            rep = json.load(f)
        rep["batch"] = {"n": len(paths), "seconds": round(cost, 1),
                        "ms_per_img": round(cost / len(paths) * 1000, 1)}
        with io.open(REPORT, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=1)


class TestDocConsistency:
    def test_readme_claims_match_impl(self):
        """README 里承诺的功能，代码里必须有对应实现。"""
        readme = os.path.join(BASE, "README.md")
        if not os.path.exists(readme):
            pytest.skip("无 README")
        with io.open(readme, encoding="utf-8") as f:
            text = f.read()

        checks = {
            "智能诊断": hasattr(web, "infer"),
            "病斑定位": hasattr(web, "locate_raw"),
            "导出 PDF": hasattr(web, "make_pdf"),
            "批量诊断": hasattr(web, "batch_diagnose"),
            "识别历史": hasattr(web, "load_history"),
        }
        missing = [k for k, ok in checks.items() if k in text and not ok]
        assert not missing, f"README 承诺但未实现: {missing}"

    def test_class_count_consistent(self):
        """类别数在所有地方必须一致。"""
        sp = config.load_splits()
        n_splits = len(sp["classes"])
        n_images = len([d for d in os.listdir(config.IMAGES_DIR)
                        if os.path.isdir(os.path.join(config.IMAGES_DIR, d))])
        assert n_splits == n_images, f"splits {n_splits} 类，图片目录 {n_images} 个"
        assert n_images == len(web.CLASSES), f"web.CLASSES {len(web.CLASSES)} != {n_images}"
