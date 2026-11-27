# -*- coding: utf-8 -*-
"""
utils.py — 通用工具：随机种子、日志、计时
"""
import os
import time
import random

import numpy as np
import torch


def set_seed(seed: int):
    """固定所有随机源，保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class Logger:
    """同时写控制台与文件。"""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


class Timer:
    """计时上下文管理器。"""

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.t0

    def __str__(self):
        return f"{self.elapsed:.1f}s"
