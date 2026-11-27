# -*- coding: utf-8 -*-
"""
dataset.py — 数据集与数据增强
"""
import os

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

import config


class LeafDataset(Dataset):
    """读取 splits.json 中的样本清单，返回 (图像张量, 类别标签)。"""

    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        path = os.path.join(config.DATA_DIR, rec["path"])
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, rec["label"]


def build_transforms(train: bool):
    """训练集用强增强，验证/测试集只做缩放裁剪。"""
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.25, contrast=0.25,
                                   saturation=0.25, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(config.MEAN, config.STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(config.IMG_SIZE * 1.14)),
        transforms.CenterCrop(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD),
    ])
def build_loaders(batch_size=None, num_workers=None, splits_file="splits.json"):
    """构造 train / val / test 三个 DataLoader。"""
    batch_size = batch_size or config.BATCH_SIZE
    num_workers = config.NUM_WORKERS if num_workers is None else num_workers

    sp = config.load_splits(splits_file)
    train_ds = LeafDataset(sp["train"], build_transforms(True))
    val_ds = LeafDataset(sp["val"], build_transforms(False))
    test_ds = LeafDataset(sp["test"], build_transforms(False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
