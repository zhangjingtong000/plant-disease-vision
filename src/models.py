# -*- coding: utf-8 -*-
"""
models.py — 三种骨干网络定义
"""
import torch.nn as nn
from torchvision import models


def _freeze_backbone(model):
    """可选：冻结除分类头以外的参数。当前训练策略为全量微调，未启用。"""
    for name, p in model.named_parameters():
        if "fc" in name or "classifier" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False
    return model
def build_model(name: str, num_classes: int = 38, pretrained: bool = True):
    """
    构建骨干网络并替换分类头。

    参数
    ----
    name : resnet50 / efficientnet_b0 / mobilenet_v3_large
    num_classes : 输出类别数
    pretrained : 是否加载 ImageNet 预训练权重
    """
    name = name.lower()

    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"不支持的骨干网络: {name}，可选 {['resnet50','efficientnet_b0','mobilenet_v3_large']}")

    return model


def count_parameters(model):
    """统计参数量（百万）。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total / 1e6, trainable / 1e6
