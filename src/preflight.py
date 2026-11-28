# -*- coding: utf-8 -*-
"""preflight.py — 启动前置检查（数据清单 / 模型权重）。

只检查、只报告，不生成也不伪造任何数据或权重：
  · 缺 data/splits.json：不阻断启动，进入只读演示模式
    知识问答、图片库、评估面板照常可用；诊断 / 批量诊断 / 病斑定位显示「需数据」
  · 缺模型权重：界面照常启动，只告警
"""
import os

DATA_FILENAME = "splits.json"


def data_path(data_dir):
    return os.path.join(data_dir, DATA_FILENAME)


def weights_path(ckpt_dir, backbone):
    return os.path.join(ckpt_dir, "{0}_best.pth".format(backbone))


def missing_weights(ckpt_dir, backbones):
    """返回缺失的权重文件路径列表（按骨干顺序，去重保序）。"""
    missing = []
    for b in backbones:
        p = weights_path(ckpt_dir, b)
        if not os.path.exists(p) and p not in missing:
            missing.append(p)
    return missing


def check(data_dir, ckpt_dir, backbones):
    """返回 (read_only, messages)。

    read_only=True 表示缺数据清单，界面以只读演示模式启动（不阻断）；
    messages 是给用户看的中文明细，可直接打印。
    """
    messages = []

    splits = data_path(data_dir)
    if not os.path.exists(splits):
        messages.append(
            "[启动检查] 缺少数据清单：{0}\n"
            "  以「只读演示模式」启动：知识问答、图片库、评估面板可用；\n"
            "  智能诊断 / 批量诊断 / 病斑定位显示「需数据」。\n"
            "  本仓库不附带数据集与模型权重。需要完整功能时，按 README\n"
            "  「PlantVillage parquet 从哪来」下载 parquet 并运行\n"
            "  `python scripts/prepare_data.py`，生成 data/splits.json 与\n"
            "  data/images/<类别>/ 后重启。".format(splits)
        )
        return True, messages

    missing = missing_weights(ckpt_dir, backbones)
    if missing:
        messages.append(
            "[启动检查] 未找到模型权重（{0} 个）：{1}\n"
            "  放置目录：{2}\n"
            "  界面仍可启动：检索问答、知识库、评估页可用；\n"
            "  单张诊断 / 病斑定位 / 批量诊断会在推理时提示缺少权重。\n"
            "  训练命令：python src/train.py --model {3} --epochs 25".format(
                len(missing),
                ", ".join(os.path.basename(p) for p in missing),
                ckpt_dir,
                backbones[0] if backbones else "resnet50",
            )
        )
    return False, messages
