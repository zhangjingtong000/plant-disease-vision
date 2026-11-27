# -*- coding: utf-8 -*-
"""公开版测试配置。

本仓库不附带数据集与模型权重。没有 `data/splits.json` 时，依赖真实数据 /
Gradio 界面的测试无法成立，这里在收集阶段跳过它们；把数据准备好（见 README
「快速开始」）后，跳过会自动失效，不需要改任何代码。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_HAS_DATA = (ROOT / "data" / "splits.json").exists()

collect_ignore: list[str] = []

if not _HAS_DATA:
    collect_ignore += [
        "test_rules.py",     # 规则用例都要读示例图（叶片/PDF/历史记录）
        "test_quality.py",   # 需要真实图片与模型做延迟/吞吐实测
        "test_recovery.py",  # 需要真实图片走恢复性用例
        "test_e2e.py",       # 需要起 Gradio 界面并驱动浏览器
    ]


def pytest_report_header(config) -> list[str]:
    if collect_ignore:
        return [
            "公开版：缺少 data/splits.json，已跳过 "
            f"{len(collect_ignore)} 个依赖数据集的测试（{', '.join(collect_ignore)}）。"
            "按 README「快速开始」生成数据后会自动恢复。"
        ]
    return []
