import json, io, os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with io.open(os.path.join(BASE, "data", "splits.json"), encoding="utf-8") as f:
    rec = json.load(f)

keep = set()
for split in ("train", "val", "test"):
    for r in rec[split]:
        keep.add(os.path.normpath(os.path.join(BASE, "data", r["path"])))

root = os.path.join(BASE, "data", "images")
removed_files = 0
removed_dirs = 0
for dirpath, dirnames, filenames in os.walk(root, topdown=False):
    for fn in filenames:
        fp = os.path.normpath(os.path.join(dirpath, fn))
        if fp not in keep:
            os.remove(fp); removed_files += 1
    if not os.listdir(dirpath):
        os.rmdir(dirpath); removed_dirs += 1

print("删除多余文件:", removed_files, " 删除空目录:", removed_dirs)

files = []
for dirpath, _, filenames in os.walk(root):
    for fn in filenames:
        files.append(os.path.join(dirpath, fn))
print("清理后图片总数:", len(files))
print("类别目录数:", len([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]))
