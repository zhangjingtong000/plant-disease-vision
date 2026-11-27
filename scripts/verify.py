import json, io, os, random
from PIL import Image
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with io.open(os.path.join(BASE, "data", "splits.json"), encoding="utf-8") as f:
    rec = json.load(f)
rng = random.Random(0)
bad = []
for split in ("train", "val", "test"):
    sample = rng.sample(rec[split], 30)
    for r in sample:
        p = os.path.join(BASE, "data", r["path"])
        try:
            im = Image.open(p); im.verify()
        except Exception as e:
            bad.append((p, str(e)))
    print(split, "抽查 30 张，异常", len([b for b in bad]))
print("总异常:", len(bad))
# 尺寸抽查
p = os.path.join(BASE, "data", rec["train"][0]["path"])
print("示例:", p, Image.open(p).size, Image.open(p).mode)
print("类别数:", len(rec["classes"]), " 作物数:", len(rec["crops"]))
