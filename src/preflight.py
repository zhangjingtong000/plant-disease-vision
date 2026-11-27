# -*- coding: utf-8 -*-
"""preflight.py — 启动前置检查（数据清单 / 模型权重）。

只检查、只报告，不生成也不伪造任何数据或权重：
  · 缺 data/splits.json：没有类别表就构不成界面，直接给出放置说明并退出
  · 缺模型权重：界面照常启动（检索问答、知识库、评估页可用），只告警
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
    """返回 (blocking, messages)。

    blocking=True 时调用方应终止启动（缺数据，无法解析类别表）；
    messages 是给用户看的中文明细，可直接打印。
    """
    messages = []

    splits = data_path(data_dir)
    if not os.path.exists(splits):
        messages.append(
            "[启动检查] 缺少数据清单：{0}\n"
            "  本仓库不附带数据集与模型权重。请先按 README「快速开始」准备\n"
            "  PlantVillage parquet 并运行 `python scripts/prepare_data.py`，\n"
            "  生成 data/splits.json 与 data/images/<类别>/ 后再启动。".format(splits)
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
