# -*- coding: utf-8 -*-
"""1-step CPU smoke test：证明训练链路能跑起来。

完整训练要 GPU + PlantVillage/PlantDoc 数据集 + 权重，无法进单测；这里只走
一遍「构建模型结构 → 前置增强拼 batch → forward → backward → optimizer.step」，
用 CPU + 随机初始化的权重（pretrained=False，不下载 ImageNet 权重、不读任何
checkpoint），断言 loss 是有限数、参数拿到非零梯度、权重确实被更新。

运行：python -m pytest tests/test_train_smoke.py -q
"""
import os
import sys

import torch
import torch.nn as nn
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))

import config  # noqa: E402
from dataset import build_transforms  # noqa: E402
from models import build_model  # noqa: E402

SAMPLES_DIR = os.path.join(BASE, "assets", "samples")
NUM_CLASSES = 3          # 小分类头即可验证一步训练，不必是真实类别数
STEPS = 1


def _sample_files(n: int):
    names = sorted(f for f in os.listdir(SAMPLES_DIR)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    assert len(names) >= n, f"assets/samples/ 里只有 {len(names)} 张图，至少需要 {n} 张"
    return [os.path.join(SAMPLES_DIR, name) for name in names[:n]]


def _sample_batch(n: int = 3):
    """用仓库自带的示例图 + 训练期增强拼一个 batch（不依赖 data/）。"""
    tf = build_transforms(train=True)      # 复用真实训练增强
    images, labels = [], []
    for i, path in enumerate(_sample_files(n)):
        img = Image.open(path).convert("RGB")
        images.append(tf(img))
        labels.append(i % NUM_CLASSES)
    x = torch.stack(images)
    y = torch.tensor(labels, dtype=torch.long)
    return x, y


def test_one_step_cpu():
    """ResNet50 在 CPU 上跑满 1 个训练 step，loss 有限、梯度存在、权重被更新。"""
    torch.manual_seed(config.SEED)
    model = build_model("resnet50", num_classes=NUM_CLASSES, pretrained=False)
    model.train()
    device = torch.device("cpu")

    x, y = _sample_batch(3)
    assert x.shape[1:] == (3, config.IMG_SIZE, config.IMG_SIZE), x.shape

    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    optim = torch.optim.AdamW(model.parameters(), lr=config.LR,
                              weight_decay=config.WEIGHT_DECAY)

    before = model.fc.weight.detach().clone()

    for _ in range(STEPS):
        optim.zero_grad(set_to_none=True)
        logits = model(x.to(device))
        assert logits.shape == (x.size(0), NUM_CLASSES), logits.shape

        loss = criterion(logits, y)
        assert torch.isfinite(loss), f"loss 不是有限数: {loss.item()}"

        loss.backward()
        optim.step()

    # 1) 每个可训练参数都拿到了梯度，且都是有限值
    params = list(model.parameters())
    grads = [p for p in params if p.grad is not None]
    assert len(grads) == len(params), f"只有 {len(grads)}/{len(params)} 个参数有梯度"
    assert all(torch.isfinite(p.grad).all() for p in params), "梯度里有 NaN/Inf"

    # 2) 分类头梯度非零（backward 真的穿到了头）
    assert model.fc.weight.grad.abs().sum().item() > 0, "分类头梯度全零"

    # 3) optimizer.step() 真的改了权重
    assert not torch.equal(before, model.fc.weight.detach()), "optimizer.step() 没有更新权重"


def test_backbones_build_without_weights():
    """三种骨干都能在无权重、无网络的情况下构建，并跑一次前向。"""
    x = torch.randn(1, 3, 96, 96)
    for name in config.BACKBONES:
        model = build_model(name, num_classes=NUM_CLASSES, pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, NUM_CLASSES), f"{name} 输出维度异常: {tuple(out.shape)}"
