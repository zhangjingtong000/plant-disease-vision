# -*- coding: utf-8 -*-
"""域可分性量化：用线性分类器判断两个域在特征空间能否区分。"""
import sys
import os
import json
import io

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from models import build_model
from cross_domain_protocol import CLS_MAP, find_root

TF = transforms.Compose([
    transforms.Resize(int(config.IMG_SIZE * 1.14)),
    transforms.CenterCrop(config.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(config.MEAN, config.STD),
])


def _open(p):
    if os.name == "nt" and not p.startswith("\\\\?\\"):
        p = "\\\\?\\" + os.path.abspath(p)
    return Image.open(p).convert("RGB")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(os.path.join(config.CKPT_DIR, "resnet50_best.pth"),
                  map_location=device, weights_only=False)
model = build_model(ckpt["model_name"],
                    len(config.load_splits()["classes"]),
                    pretrained=False)
model.load_state_dict(ckpt["state_dict"])
model.to(device).eval()

store = {}


def hook(m, i, o):
    store["f"] = o.detach().flatten(1)


h = model.avgpool.register_forward_hook(hook)


@torch.no_grad()
def feats(paths):
    out = []
    for i in range(0, len(paths), 32):
        batch = paths[i:i + 32]
        xs = torch.stack([TF(_open(p)) for p in batch]).to(device)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            model(xs)
        out.append(store["f"].cpu().numpy())
    return np.concatenate(out)


# 域 A：测试集抽样
sp = config.load_splits()
rng = np.random.RandomState(0)
a_recs = rng.choice(sp["test"], 400, replace=False)
a_paths = [os.path.join(config.DATA_DIR, r["path"]) for r in a_recs]

# 域 B
root = find_root()
b_paths = []
for d in sorted(os.listdir(root)):
    sd = os.path.join(root, d)
    if not os.path.isdir(sd):
        continue
    for f in sorted(os.listdir(sd)):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            b_paths.append(os.path.join(sd, f))
b_paths = b_paths[:400]

print(f"域 A {len(a_paths)} 张 | 域 B {len(b_paths)} 张")
Fa = feats(a_paths)
Fb = feats(b_paths)
h.remove()

X = np.concatenate([Fa, Fb])
y = np.array([0] * len(Fa) + [1] * len(Fb))
X = (X - X.mean(0)) / (X.std(0) + 1e-6)

clf = LogisticRegression(max_iter=2000, C=1.0)
scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
acc = float(scores.mean())

res = {
    "n_A": len(Fa), "n_B": len(Fb),
    "domain_classifier_acc": acc,
    "cv_scores": [float(s) for s in scores],
    "interpretation": (
        "两域高度可分" if acc > 0.95 else
        "两域明显可分" if acc > 0.85 else
        "两域部分可分" if acc > 0.70 else
        "两域高度重叠"
    ),
}
print(json.dumps(res, ensure_ascii=False, indent=1))
with io.open(os.path.join(config.METRIC_DIR, "domain_classifier.json"), "w",
             encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print("已保存 metrics/domain_classifier.json")
