# -*- coding: utf-8 -*-
"""
benchmark.py — 模型性能基准测试

输出：参数量、FLOPs、模型体积、单张延迟、批量吞吐。
结果写入 outputs/metrics/benchmark.json，可在任意机器上复跑。

用法：
    python src/benchmark.py
    python src/benchmark.py --device cpu
"""
import os
import sys
import json
import time
import io
import argparse

import torch
import numpy as np
from thop import profile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from models import build_model, count_parameters


def measure_latency(model, device, input_size=(1, 3, 224, 224), warmup=10, runs=50):
    """测单张推理延迟（毫秒）。"""
    x = torch.randn(*input_size).to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / runs
    return dt * 1000


def measure_throughput(model, device, batch=32, runs=10):
    """测批量吞吐（张/秒）。"""
    x = torch.randn(batch, 3, 224, 224).to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return (batch * runs) / dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--cpu-too", action="store_true", help="同时测 CPU 延迟（较慢）")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"测试设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    results = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "torch": torch.__version__,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {},
    }

    for name in config.BACKBONES:
        print(f"\n--- {name} ---")
        model = build_model(name, num_classes=len(config.get_classes()), pretrained=False).to(device)
        total_p, _ = count_parameters(model)

        # FLOPs
        try:
            dummy = torch.randn(1, 3, 224, 224).to(device)
            flops, _params = profile(model, inputs=(dummy,), verbose=False)
            flops_g = flops / 1e9
        except Exception as e:
            flops_g = None
            print("  FLOPs 计算失败:", e)

        ckpt_path = os.path.join(config.CKPT_DIR, f"{name}_best.pth")
        size_mb = os.path.getsize(ckpt_path) / 1024**2 if os.path.exists(ckpt_path) else None

        lat = measure_latency(model, device)
        thr = measure_throughput(model, device)
        print(f"  参数量 {total_p:.2f}M | FLOPs {flops_g:.2f}G | 单张 {lat:.1f}ms | 吞吐 {thr:.1f} 张/s")

        entry = {
            "params_m": round(total_p, 2),
            "flops_g": round(flops_g, 2) if flops_g else None,
            "model_size_mb": round(size_mb, 1) if size_mb else None,
            "latency_ms": round(lat, 2),
            "throughput_img_s": round(thr, 1),
        }

        if args.cpu_too and device.type == "cuda":
            cpu = torch.device("cpu")
            m2 = model.to(cpu)
            entry["cpu_latency_ms"] = round(measure_latency(m2, cpu, runs=10), 2)
            print(f"  CPU 单张 {entry['cpu_latency_ms']:.1f}ms")
            model.to(device)

        results["models"][name] = entry

    out = os.path.join(config.METRIC_DIR, "benchmark.json")
    with io.open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
