# -*- coding: utf-8 -*-
"""公开版测试配置。

本仓库不附带数据集与模型权重。测试对环境的依赖分三档，这里在收集阶段按实测
环境决定收集哪些文件，避免"环境不满足"被误报成"代码有 bug"：

1. **有没有数据**：没有 `data/splits.json` 时，所有读真实图片 / 起界面的测试
   都不成立，直接不收集。
2. **数据够不够完整**：`scripts/prepare_data.py` 的完整输出是「38 类每类约
   200 张」。用小子集冒烟（见 README「最小复现」）时只有部分类别有图片，
   断言"每类都有图"或"三个骨干都有权重"的用例应当跳过，而不是失败——它们
   验证的是发布级配平数据集，不是冒烟子集。
3. **界面能不能起**：`tests/test_e2e.py` 需要服务已在 7860 运行、且系统装了
   Chrome（它用 playwright 驱动系统 Chrome，不下载浏览器）。服务没起时它
   只会报连接错误，这里直接不收集。

把数据准备好（见 README「快速开始」）后，跳过会自动失效，不需要改任何代码。
"""
import json
import socket
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SPLITS = DATA / "splits.json"
IMAGES = DATA / "images"
CKPT = ROOT / "outputs" / "checkpoints"

BACKBONES = ("resnet50", "efficientnet_b0", "mobilenet_v3_large")
E2E_URL_HOST, E2E_URL_PORT = "127.0.0.1", 7860
CHROME_CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
)


def _classes() -> list[str]:
    try:
        return json.loads(SPLITS.read_text(encoding="utf-8"))["classes"]
    except Exception:
        return []


def _class_has_images(cls: str) -> bool:
    d = IMAGES / cls
    if not d.is_dir():
        return False
    return any(f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")
               for f in d.iterdir())


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


CLASSES = _classes()
HAS_DATA = SPLITS.exists() and bool(CLASSES)

# 完整配平数据集：每个类别都有图片（prepare_data.py 在完整 PlantVillage 上的输出）
FULL_DATASET = HAS_DATA and all(_class_has_images(c) for c in CLASSES)

# 三个骨干的权重是否都在（跨骨干的延迟 / 降级用例需要）
MISSING_CKPTS = [n for n in BACKBONES if not (CKPT / f"{n}_best.pth").exists()]
ALL_BACKBONES = not MISSING_CKPTS

# 端到端：需要完整数据集 + 服务已启动 + 系统 Chrome
HAS_E2E = FULL_DATASET and _port_open(E2E_URL_HOST, E2E_URL_PORT) and any(
    Path(p).exists() for p in CHROME_CANDIDATES)

collect_ignore: list[str] = []

if not HAS_DATA:
    collect_ignore += [
        "test_rules.py",       # 规则用例都要读示例图（叶片/PDF/历史记录）
        "test_quality.py",     # 需要真实图片与模型做延迟/吞吐实测
        "test_recovery.py",    # 需要真实图片走恢复性用例
        "test_smoke_data.py",  # 数据接入冒烟：没有 splits.json 就无从谈起
    ]
if not HAS_E2E:
    collect_ignore.append("test_e2e.py")   # 需要起 Gradio 界面并驱动系统 Chrome


def pytest_report_header(config) -> list[str]:
    msgs = []
    if not HAS_DATA:
        msgs.append("公开版：缺少 data/splits.json，已跳过 "
                    f"{len(collect_ignore)} 个依赖数据集/界面的测试。"
                    "按 README「快速开始」生成数据后会自动恢复。")
    elif not FULL_DATASET:
        msgs.append("检测到数据集是小子集（不是每类都有图片）：依赖完整配平数据集"
                    "与三骨干权重的个别用例会被跳过，其余照常运行。")
    if not HAS_E2E:
        msgs.append("test_e2e.py 未收集：需要 data 完整 + 7860 上的服务 + 系统 Chrome。")
    return msgs
