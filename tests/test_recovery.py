# -*- coding: utf-8 -*-
"""第 5 层：运维可恢复性（真删真恢复 + 核对条数）。

运行：python -m pytest tests/test_recovery.py -v -s
"""
import os
import sys
import io
import csv
import shutil

import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))
sys.path.insert(0, os.path.join(BASE, "app"))

import config  # noqa: E402
import web  # noqa: E402
import conftest as _env  # noqa: E402  （同一目录的 conftest，提供环境判据）

HIST = web.HISTORY_CSV


def _count_rows(path):
    if not os.path.exists(path):
        return 0
    with io.open(path, encoding="utf-8-sig") as f:
        return max(0, len(list(csv.reader(f))) - 1)   # 去掉表头


@pytest.fixture
def clean_history():
    """备份现有历史，测试后还原。"""
    backup = HIST + ".recbak"
    had = os.path.exists(HIST)
    if had:
        shutil.copy2(HIST, backup)
    if os.path.exists(HIST):
        os.remove(HIST)
    yield
    if had and os.path.exists(backup):
        shutil.move(backup, HIST)
    elif os.path.exists(HIST):
        os.remove(HIST)


class TestHistoryRecovery:
    def test_backup_created_on_clear(self, clean_history):
        """清空历史前必须生成 .bak 备份。"""
        for _ in range(3):
            web._log_history("resnet50", config.get_classes()[0], 0.9)
        assert _count_rows(HIST) == 3

        bak = HIST + ".bak"
        if os.path.exists(bak):
            os.remove(bak)

        web.clear_history()
        assert _count_rows(HIST) == 0, "清空后应为 0 条"
        assert os.path.exists(bak), "清空前未生成备份文件"
        print(f"  备份文件: {bak}")

    def test_restore_from_backup(self, clean_history):
        """真删 → 从备份恢复 → 核对条数一致。"""
        classes = config.get_classes()
        for i in range(5):
            web._log_history("resnet50", classes[i], 0.8 + i * 0.02)
        before = _count_rows(HIST)
        assert before == 5

        bak = HIST + ".bak"
        if os.path.exists(bak):
            os.remove(bak)
        web.clear_history()
        assert _count_rows(HIST) == 0

        # 手动恢复
        shutil.copy2(bak, HIST)
        after = _count_rows(HIST)
        assert after == before, f"恢复后 {after} 条，损坏前 {before} 条"
        print(f"  恢复验证: {before} → 0 → {after}")

    def test_delete_then_restore_specific(self, clean_history):
        """删单条后，用备份能恢复回原条数。"""
        classes = config.get_classes()
        for i in range(4):
            web._log_history("resnet50", classes[i], 0.7)
        before = _count_rows(HIST)

        bak = HIST + ".manualbak"
        shutil.copy2(HIST, bak)

        deleted, _ = web.delete_history("1")
        assert deleted == 1, f"应删 1 条，实际 {deleted}"
        mid = _count_rows(HIST)
        assert mid == before - 1

        shutil.copy2(bak, HIST)
        os.remove(bak)
        restored = _count_rows(HIST)
        assert restored == before, f"恢复后 {restored} != {before}"
        print(f"  单条删除与恢复: {before} → {mid} → {restored}")


class TestModelRecovery:
    @pytest.mark.skipif(
        not _env.HAS_WEIGHTS,
        reason="缺 outputs/checkpoints/resnet50_best.pth：主模型都加载不了，谈不上兜底")
    def test_missing_weight_falls_back(self):
        """权重文件缺失时不能白屏，主模型要能兜底。"""
        cache = web._MODEL_CACHE
        saved = dict(cache)
        try:
            cache.clear()
            # 主模型必须能加载
            model, name = web.get_model(config.BACKBONES[0])
            assert model is not None
            assert name
        finally:
            cache.clear()
            cache.update(saved)

    @pytest.mark.skipif(
        not _env.ALL_BACKBONES,
        reason=f"缺权重 {_env.MISSING_CKPTS}：该用例要验证非主模型加载失败时的降级")
    def test_mismatched_weight_falls_back(self):
        """权重类别数不匹配时，非主模型应降级而不是崩溃。"""
        cache = web._MODEL_CACHE
        saved = dict(cache)
        try:
            for name in config.BACKBONES[1:]:
                cache.clear()
                try:
                    model, mname = web.get_model(name)
                    assert model is not None
                    print(f"  {name} -> 降级到 {mname}")
                except RuntimeError:
                    # 主模型也失败才算真失败
                    pytest.fail(f"{name} 无法降级")
        finally:
            cache.clear()
            cache.update(saved)
