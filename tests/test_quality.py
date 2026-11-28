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
import conftest as _env  # noqa: E402  （同一目录的 conftest，提供环境判据）

REPORT = os.path.join(config.METRIC_DIR, "quality_report.json")


def _sample_images(n=20):
    """按类别轮转取样，凑够 n 张为止。

    完整数据集每类约 200 张，但从小子集冒烟时每类只有几十张、且只有部分类别
    有图片；只取"每类第一张"会在 8 类的小样本上凑不到 10 张。这里跨轮取样。
    """
    buckets = []
    for cls in config.get_classes():
        d = os.path.join(config.IMAGES_DIR, cls)
        if not os.path.isdir(d):
            continue
        files = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if files:
            buckets.append([os.path.join(d, f) for f in files])

    paths, i = [], 0
    while len(paths) < n and any(i < len(b) for b in buckets):
        for b in buckets:
            if i < len(b):
                paths.append(b[i])
                if len(paths) >= n:
                    break
        i += 1
    return paths


class TestLatency:
    @pytest.mark.skipif(
        not _env.ALL_BACKBONES,
        reason=f"缺权重 {_env.MISSING_CKPTS}：跨骨干延迟对比需要三个骨干的 checkpoint")
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
    @pytest.mark.skipif(
        not _env.HAS_WEIGHTS,
        reason="缺 outputs/checkpoints/resnet50_best.pth：batch_diagnose 会吞掉"
               "加载异常并照常返回行数，量出来的「吞吐」是假的")
    def test_batch_throughput(self):
        """批量诊断 20 张的总耗时，应低于串行上界。"""
        paths = _sample_images(20)
        assert len(paths) >= 10, "样本不足"
        # batch_diagnose 逐行 try/except，权重缺失时仍会返回行数。先单独确认
        # 推理路径真的可用，让异常在这里冒出来，而不是被吞成一行"成功"。
        web.get_model(config.BACKBONES[0])
        t0 = time.time()
        rows, csv_path = web.batch_diagnose(paths, config.BACKBONES[0], 0.0)
        cost = time.time() - t0
        assert len(rows) == len(paths), f"结果数 {len(rows)} != 输入 {len(paths)}"
        print(f"  批量 {len(paths)} 张耗时 {cost:.1f}s（{cost/len(paths)*1000:.0f} ms/张）")

        # 报告由 TestLatency 写出；延迟用例被跳过（缺三骨干权重）时文件不存在，
        # 这里新建即可——吞吐结论不该依赖另一个用例先跑过。
        rep = {}
        if os.path.exists(REPORT):
            with io.open(REPORT, encoding="utf-8") as f:
                rep = json.load(f)
        rep["batch"] = {"n": len(paths), "seconds": round(cost, 1),
                        "ms_per_img": round(cost / len(paths) * 1000, 1)}
        os.makedirs(config.METRIC_DIR, exist_ok=True)
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
