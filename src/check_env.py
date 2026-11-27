# -*- coding: utf-8 -*-
"""
check_env.py — 环境自检

部署到新机器后先跑这个，确认依赖与权重是否就位。
"""
import os
import sys
import platform

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def main():
    print("=" * 55)
    print("  植物叶片病害智能诊断系统 —— 环境自检")
    print("=" * 55)

    print(f"操作系统 : {platform.system()} {platform.release()}")
    print(f"Python   : {sys.version.split()[0]}")

    ok = True

    try:
        import torch
        print(f"PyTorch  : {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA     : 可用")
            print(f"GPU      : {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"显存     : {props.total_memory/1024**3:.1f} GB")
            print("推理模式 : GPU 加速")
        else:
            print("CUDA     : 不可用，将使用 CPU 推理（较慢）")
            print("推理模式 : CPU")
    except Exception as e:
        print("PyTorch  : 未安装 ——", e)
        ok = False

    for pkg in ["torchvision", "gradio", "cv2", "sklearn", "matplotlib", "thop"]:
        try:
            __import__(pkg)
            print(f"{pkg:<15s}: OK")
        except Exception:
            print(f"{pkg:<15s}: 缺失")
            ok = False

    print("-" * 55)
    print("模型权重：")
    for m in config.BACKBONES:
        p = os.path.join(config.CKPT_DIR, f"{m}_best.pth")
        if os.path.exists(p):
            print(f"  {m:<22s} OK  ({os.path.getsize(p)/1024**2:.1f} MB)")
        else:
            print(f"  {m:<22s} 缺失")
            ok = False

    print("-" * 55)
    print("数据清单：", "OK" if os.path.exists(config.SPLITS_JSON) else "缺失")

    print("=" * 55)
    print("自检结果:", "通过，可以启动" if ok else "存在问题，请检查上面的缺失项")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
