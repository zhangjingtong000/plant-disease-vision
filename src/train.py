# -*- coding: utf-8 -*-
"""
train.py — 训练主程序

用法：
    python src/train.py --model resnet50 --epochs 25
    python src/train.py --model efficientnet_b0 --epochs 25
"""
import os
import sys
import json
import time
import argparse
import io

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from dataset import build_loaders
from models import build_model, count_parameters
from utils import set_seed, Logger

BANNER = """
================================================================
  植物叶片病害智能诊断系统 —— 模型训练
================================================================
"""


def evaluate(model, loader, criterion, device):
    """在给定 DataLoader 上计算平均损失与准确率。"""
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = model(x)
                loss = criterion(out, y)
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total


def train_one_epoch(model, loader, criterion, optim, scaler, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optim.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="植物叶片病害分类训练")
    parser.add_argument("--model", type=str, default="resnet50", choices=config.BACKBONES)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--splits", type=str, default="splits.json",
                        help="划分文件（不均衡对照组用 splits_imbalance.json）")
    parser.add_argument("--tag", type=str, default="", help="实验标签，用于文件名")
    args = parser.parse_args()

    print(BANNER)
    tag = args.tag or args.model
    log = Logger(os.path.join(config.LOG_DIR, f"train_{tag}.log"))
    log(f"参数: {vars(args)}")

    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"设备: {device}  显存: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    # ---------------- 数据 ----------------
    train_loader, val_loader, test_loader = build_loaders(args.batch, args.workers, args.splits)
    log(f"训练集 {len(train_loader.dataset)} | 验证集 {len(val_loader.dataset)} | 测试集 {len(test_loader.dataset)}")

    # ---------------- 模型 ----------------
    model = build_model(args.model, num_classes=len(config.get_classes())).to(device)
    total_p, trainable_p = count_parameters(model)
    log(f"模型 {args.model}：参数量 {total_p:.2f}M（可训练 {trainable_p:.2f}M）")
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optim, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ---------------- 训练循环 ----------------
    history = []
    best_acc, best_epoch, bad_epochs = 0.0, -1, 0
    ckpt_path = os.path.join(config.CKPT_DIR, f"{tag}_best.pth")
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optim, scaler, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        dt = time.time() - t0

        history.append({
            "epoch": epoch, "train_loss": round(tr_loss, 6), "train_acc": round(tr_acc, 6),
            "val_loss": round(val_loss, 6), "val_acc": round(val_acc, 6),
            "lr": round(optim.param_groups[0]["lr"], 8), "seconds": round(dt, 1),
        })
        log(f"Epoch {epoch:02d}/{args.epochs} | train_loss {tr_loss:.4f} acc {tr_acc:.4f} "
            f"| val_loss {val_loss:.4f} acc {val_acc:.4f} | {dt:.0f}s")

        if val_acc > best_acc:
            best_acc, best_epoch, bad_epochs = val_acc, epoch, 0
            torch.save({
                "model_name": args.model,
                "state_dict": model.state_dict(),
                "classes": config.get_classes(),
                "val_acc": val_acc,
                "epoch": epoch,
            }, ckpt_path)
            log(f"  -> 保存最佳模型（val_acc={val_acc:.4f}）")
        else:
            bad_epochs += 1
            if bad_epochs >= config.PATIENCE:
                log(f"验证集连续 {config.PATIENCE} 轮未提升，提前停止")
                break

    elapsed = time.time() - t_start
    log(f"训练结束：最佳验证准确率 {best_acc:.4f}（第 {best_epoch} 轮），总耗时 {elapsed/60:.1f} 分钟")

    # ---------------- 记录 ----------------
    result = {
        "model": args.model, "tag": tag, "epochs_run": len(history),
        "best_val_acc": best_acc, "best_epoch": best_epoch,
        "total_seconds": round(elapsed, 1),
        "params_m": round(total_p, 2),
        "history": history,
        "env": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "torch": torch.__version__,
        },
    }
    with io.open(os.path.join(config.METRIC_DIR, f"train_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    log(f"训练曲线数据已保存: metrics/train_{tag}.json")
    log.close()


if __name__ == "__main__":
    main()
