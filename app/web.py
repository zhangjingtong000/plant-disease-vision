# -*- coding: utf-8 -*-
"""
web.py — 植物叶片病害智能诊断系统

视觉规范来源：GitHub `longbridge/developers` 的 DESIGN.md（见 docs/DESIGN-longbridge.md）
  - 主强调色 Teal #00ADA2，主按钮 Black pill 999px
  - 卡片 White + 1px #EAEBEC 边框 + 16px 圆角 + 默认无阴影
  - 间距基准 4px，卡片内边距 28px/24px，容器 1200px
  - 正文 #3C3C43 16px，标题 #0A0E19 600
"""
import os
import sys
import csv
import json
import io
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import config
from models import build_model
from gradcam import GradCAM, leaf_mask, locate_lesion, overlay_cam_on_image, occlusion_test
import knowledge

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 启动前置检查：缺数据给明确提示并退出；缺权重只告警（不伪造权重）
# 注意：放在函数里而不是模块顶层——这样 import web 只做依赖检查，不会在
# pytest 收集阶段被 SystemExit 带崩。真正的启动入口（app/web_agent.py）负责调用。
import preflight  # noqa: E402  （src/ 已在 sys.path 上）


def preflight_or_exit(exit_code: int = 2) -> bool:
    """缺数据时打印提示并以 exit_code 退出；缺权重只告警。返回是否通过。"""
    blocking, msgs = preflight.check(
        config.DATA_DIR, config.CKPT_DIR, config.BACKBONES)
    for msg in msgs:
        print(msg, file=sys.stderr, flush=True)
    if blocking:
        raise SystemExit(exit_code)
    return True


try:
    CLASSES = config.get_classes()
except Exception:  # 缺 data/splits.json 时不阻塞 import，界面启动前会再检查
    CLASSES = []
HISTORY_CSV = os.path.join(config.OUTPUT_DIR, "history.csv")
HISTORY_IMG_DIR = os.path.join(config.OUTPUT_DIR, "history_img")

# 示例图随仓库放在 assets/samples/；老版本放在 data/examples/（该目录从未入库），
# 两个位置都找，找不到就走「没有示例图」而不是抛 FileNotFoundError。
EXAMPLES_DIRS = (
    os.path.join(config.BASE_DIR, "assets", "samples"),
    os.path.join(config.DATA_DIR, "examples"),
)
EXAMPLES = {
    "番茄晚疫病": "tomato_late_blight.jpg",
    "苹果黑星病": "apple_scab.jpg",
    "玉米锈病": "corn_rust.jpg",
    "土豆健康": "potato_healthy.jpg",
}

# =====================================================================
#  模型与推理
# =====================================================================
_MODEL_CACHE = {}


def get_model(name: str):
    if name in _MODEL_CACHE:
        return _MODEL_CACHE[name]
    ckpt_path = os.path.join(config.CKPT_DIR, f"{name}_best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"未找到模型权重：{ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    try:
        model = build_model(ckpt["model_name"], num_classes=len(CLASSES), pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
    except Exception as e:
        # 权重与当前类别数不匹配（例如旧版 38 类模型）时，降级到主模型
        fallback = config.BACKBONES[0]
        if name != fallback:
            return get_model(fallback)
        raise RuntimeError(
            f"模型权重与当前 {len(CLASSES)} 类不匹配，请重新训练：{name}（{str(e)[:80]}）"
        )
    model.to(DEVICE).eval()
    _MODEL_CACHE[name] = (model, ckpt["model_name"])
    return _MODEL_CACHE[name]


def preprocess(pil_img: Image.Image) -> torch.Tensor:
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize(int(config.IMG_SIZE * 1.14)),
        transforms.CenterCrop(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD),
    ])
    return tf(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)


@torch.no_grad()
def infer(pil_img, model_name="resnet50"):
    model, _ = get_model(model_name)
    x = preprocess(pil_img)
    with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
        logits = model(x)
    probs = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
    order = probs.argsort()[::-1][:5]
    top5 = [(CLASSES[i], float(probs[i])) for i in order]
    return top5, probs, CLASSES[int(probs.argmax())]


def cn_name(cls: str):
    info = knowledge.get_info(cls)
    return f"{info['crop_cn']} - {info['disease_cn']}"


def render_status():
    dev = "GPU" if DEVICE.type == "cuda" else "CPU"
    try:
        get_model("resnet50")
        ready = "ResNet50"
    except Exception:
        ready = "未加载"
    return (f'<div class="pb-nav"><span><i></i>{dev}</span>'
            f'<span>38 类</span><span>{ready}</span></div>')


def load_example(key):
    """按 key 取示例图；assets/samples 与 data/examples 都找不到时返回 None。"""
    name = EXAMPLES.get(key)
    if not name:
        return None
    for d in EXAMPLES_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return Image.open(p).convert("RGB")
    return None


def render_result_html(pred, top5, conf, info, low, model_name, elapsed):
    sev = info["severity"]
    tag_cls = "pb-status-tag"
    if sev in ("高", "极高"):
        tag_cls = "pb-status-tag danger"
    if low:
        tag_cls = "pb-status-tag warn"
    tag = "可信判定" if not low else "建议人工复核"
    return f'''
<div>
  <div class="pb-result-crop">{info['crop_cn']}</div>
  <div class="pb-result-name">{info['disease_cn']}</div>
  <div class="pb-score">
    <span class="pb-score-num">{conf*100:.1f}</span>
    <span class="pb-score-unit">% 置信度</span>
  </div>
  <div class="pb-progress"><i style="width:{conf*100:.1f}%"></i></div>
  <div class="pb-facts">
    <div><div class="pb-fact-label">病原类型</div>
      <div class="pb-fact-value">{info['pathogen']}</div></div>
    <div><div class="pb-fact-label">危害程度</div>
      <div class="pb-fact-value">{sev}</div></div>
    <div><div class="pb-fact-label">推理耗时</div>
      <div class="pb-fact-value">{elapsed*1000:.0f} ms</div></div>
  </div>
  <div class="{tag_cls}">{tag} · {model_name}</div>
</div>'''


def render_top5_html(top5):
    rows = []
    for i, (c, p) in enumerate(top5):
        rows.append(
            f'<div class="pb-rank-row">'
            f'<span class="pb-rank-name">{cn_name(c)}</span>'
            f'<span class="pb-rank-track"><i style="width:{max(p*100, .8):.1f}%"></i></span>'
            f'<span class="pb-rank-pct">{p*100:.2f}%</span></div>')
    return '<div class="pb-rank-title">Top-5 候选</div>' + "".join(rows)


def _save_history_image(pil_img=None, src_path=None):
    """把这次识别用的图片存一份到 history_img/，返回相对路径。

    历史记录要能回看当时那张图，所以不能只存临时文件路径。
    """
    try:
        os.makedirs(HISTORY_IMG_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        name = f"{stamp}.jpg"
        dst = os.path.join(HISTORY_IMG_DIR, name)
        if pil_img is not None:
            pil_img.convert("RGB").save(dst, quality=88)
        elif src_path and os.path.exists(str(src_path)):
            from PIL import Image as _Img
            _Img.open(src_path).convert("RGB").save(dst, quality=88)
        else:
            return ""
        return os.path.relpath(dst, config.OUTPUT_DIR).replace("\\", "/")
    except Exception:
        return ""


def _log_history(model_name, pred, conf, pil_img=None, src_path=None):
    try:
        img_rel = _save_history_image(pil_img=pil_img, src_path=src_path)
        new = not os.path.exists(HISTORY_CSV)
        with io.open(HISTORY_CSV, "a", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["时间", "模型", "预测类别", "中文名", "置信度", "图片"])
            w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), model_name,
                        pred, cn_name(pred), f"{conf:.4f}", img_rel])
    except Exception:
        pass


def render_detail_md(info):
    """只渲染症状与防治，避免和结果卡、Top-5 重复。"""
    lines = [f"**症状**　{info['symptoms']}", "", "**防治建议**"]
    for i, c in enumerate(info["control"], 1):
        lines.append(f"{i}. {c}")
    return "\n".join(lines)
def is_leaf_like(pil_img):
    """判断输入是否像一张叶片照片，避免对任意图片强行给出病害结论。

    主要判据是「绿色区域的连通性」：真实叶片的绿色像素连成一片，
    而随机噪声、文字截图、纯色块的绿色像素是散点。
    返回 (bool, dict)。
    """
    import cv2
    rgb = np.array(pil_img.convert("RGB").resize((224, 224)))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0].astype(np.int32), hsv[..., 1], hsv[..., 2]

    green = ((h >= 30) & (h <= 95) & (s >= 35) & (v >= 25)).astype(np.uint8)
    green_ratio = float(green.mean())

    # 病叶往往发黄、褐变，光看绿色会漏判。
    # 把黄绿到褐黄的色相一并算作"植物色"，但要求有一定饱和度，避免误收木色背景。
    plantish = (((h >= 15) & (h <= 100)) & (s >= 45) & (v >= 30)).astype(np.uint8)
    plant_ratio = float(plantish.mean())

    n, labels, stats, _ = cv2.connectedComponentsWithStats(green, 8)
    if n > 1:
        largest = int(stats[1:, cv2.CC_STAT_AREA].max())
        conn = float(largest / max(int(green.sum()), 1))
    else:
        conn = 0.0

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    ent = float(-(p * np.log2(p)).sum())

    bg = (s < 32) & (v > 195)
    fg = float(1.0 - bg.mean())

    # 原判据只看绿色比例，病叶（黄化/褐变）会被误判成"不是叶片"。
    # 改为：绿色达到 5%，或者植物色（含黄褐）达到 25%，其余条件不变。
    color_ok = (green_ratio >= 0.05) or (plant_ratio >= 0.25)
    ok = color_ok and (conn >= 0.30) and (ent > 2.5) and (fg >= 0.05)
    return ok, {"green_ratio": green_ratio, "plant_ratio": plant_ratio,
                "conn": conn, "entropy": ent, "fg_ratio": fg}


def diagnose_view(pil_img, model_name, conf_thresh):
    if pil_img is None:
        return EMPTY_HTML, "", ""
    t0 = time.time()
    top5, probs, pred = infer(pil_img, model_name)
    elapsed = time.time() - t0
    conf = top5[0][1]
    low = conf < conf_thresh
    info = knowledge.get_info(pred)
    _log_history(model_name, pred, conf, pil_img=pil_img)
    return (render_result_html(pred, top5, conf, info, low, model_name, elapsed),
            render_detail_md(info),
            render_top5_html(top5))


def locate_raw(pil_img, model_name, thresh=0.6):
    """病斑定位的原始版本：返回热力图、外框图与统计数据，供 PDF 使用。"""
    import cv2
    model, mname = get_model(model_name)
    x = preprocess(pil_img)
    gc = GradCAM(model, mname)
    try:
        cam, cls_idx = gc(x)
    finally:
        gc.close()

    rgb = np.array(pil_img.convert("RGB").resize((config.IMG_SIZE, config.IMG_SIZE)))
    mask = leaf_mask(rgb)
    overlay = overlay_cam_on_image(rgb, cam, mask)
    lesion = locate_lesion(cam, mask, thresh=thresh)

    vis = rgb.copy()
    if lesion["bbox"] is not None:
        x0, y0, w, h = lesion["bbox"]
        cv2.rectangle(vis, (x0, y0), (x0 + w, y0 + h), (0, 173, 162), 3)
        vis[lesion["lesion_mask"]] = (vis[lesion["lesion_mask"]] * 0.45 +
                                      np.array([0, 173, 162]) * 0.55).astype(np.uint8)
    ratio = float(lesion["area_ratio"]) * 100
    if ratio == 0:
        level = "未见明显病斑"
    elif ratio < 10:
        level = "轻微"
    elif ratio < 25:
        level = "中等"
    else:
        level = "严重"
    return {
        "overlay": overlay, "vis": vis, "ratio": ratio, "level": level,
        "has_bbox": lesion["bbox"] is not None,
        "pred_cls": CLASSES[cls_idx],
    }


def locate_view(pil_img, model_name, thresh):
    if pil_img is None:
        return None, None, "", gr.update(visible=False)
    model, mname = get_model(model_name)
    x = preprocess(pil_img)
    gc = GradCAM(model, mname)
    try:
        cam, cls_idx = gc(x)
    finally:
        gc.close()

    rgb = np.array(pil_img.convert("RGB").resize((config.IMG_SIZE, config.IMG_SIZE)))
    mask = leaf_mask(rgb)
    overlay = overlay_cam_on_image(rgb, cam, mask)

    lesion = locate_lesion(cam, mask, thresh=thresh)
    vis = rgb.copy()
    if lesion["bbox"] is not None:
        import cv2
        x0, y0, w, h = lesion["bbox"]
        cv2.rectangle(vis, (x0, y0), (x0 + w, y0 + h), (0, 173, 162), 3)
        vis[lesion["lesion_mask"]] = (vis[lesion["lesion_mask"]] * 0.45 +
                                      np.array([0, 173, 162]) * 0.55).astype(np.uint8)

    info = knowledge.get_info(CLASSES[cls_idx])
    ratio = lesion["area_ratio"] * 100
    if ratio == 0:
        text = f"预测 {cn_name(CLASSES[cls_idx])}，未检测到明显病斑区域。"
    else:
        level = "轻微" if ratio < 10 else ("中等" if ratio < 25 else "严重")
        text = (f"预测 {cn_name(CLASSES[cls_idx])}　·　病斑占叶面 {ratio:.1f}%　·　"
                f"危害 {level}　·　病原 {info['pathogen']}")
    return overlay, vis, f'<div class="pb-hint">{text}</div>', gr.update(visible=True)


def demo_all():
    """一键演示只跑诊断，病斑定位留给用户按需触发，避免等待过久。"""
    img = load_example("番茄晚疫病")
    if img is None:
        return None, EMPTY_HTML, "", "", gr.update(visible=False)
    html, detail, top5 = diagnose_view(img, config.BACKBONES[0], 0.6)
    return img, html, detail, top5, gr.update(visible=False)


def batch_diagnose(files, model_name, conf_thresh):
    if not files:
        return [], None
    rows = []
    for f in files:
        try:
            p = f.name if hasattr(f, "name") else f
            pil = Image.open(p).convert("RGB")
            # 与单张诊断一致：先判断是否像叶片
            leaf_ok, _ = is_leaf_like(pil)
            if not leaf_ok:
                rows.append([os.path.basename(p), "未识别到叶片", "-", "无法识别"])
                continue
            top5, probs, pred = infer(pil, model_name)
            conf = top5[0][1]
            if conf < conf_thresh:
                flag = "待复核"
            else:
                flag = "可信"
            rows.append([os.path.basename(p), cn_name(pred), f"{conf*100:.2f}%", flag])
            # G1：批量结果同样写入识别历史
            try:
                _log_history(model_name, pred, conf, src_path=p)
            except Exception:
                pass
        except Exception as e:
            rows.append([os.path.basename(str(f)), f"处理失败: {e}", "-", "失败"])

    out_csv = os.path.join(config.OUTPUT_DIR, f"batch_{datetime.now():%Y%m%d_%H%M%S}.csv")
    with io.open(out_csv, "w", encoding="utf-8-sig", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["文件名", "诊断结果", "置信度", "状态"])
        w.writerows(rows)
    return rows, out_csv


def get_class_images(class_name, n=3, source="lab"):
    """取某类别的示例图片路径（来自项目自带数据集）。"""
    out = []
    if source == "lab":
        d = os.path.join(config.IMAGES_DIR, class_name)
        if os.path.isdir(d):
            files = sorted(f for f in os.listdir(d)
                           if f.lower().endswith((".jpg", ".jpeg", ".png")))
            step = max(1, len(files) // max(n, 1))
            for i in range(0, len(files), step):
                out.append(os.path.join(d, files[i]))
                if len(out) >= n:
                    break
    else:
        try:
            import cross_domain_protocol as cdp
            rev = {v: k for k, v in cdp.CLS_MAP.items()}
            dirname = rev.get(class_name)
            if dirname:
                root = cdp.find_root()
                if root:
                    sd = os.path.join(root, dirname)
                    if os.path.isdir(sd):
                        files = sorted(f for f in os.listdir(sd)
                                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
                        out = [os.path.join(sd, f) for f in files[:n]]
        except Exception:
            pass
    return out


def kb_lookup(cls_name):
    if not cls_name:
        return "请选择类别。"
    return knowledge.format_report(cls_name, [])


def eval_gallery(model_name):
    mapping = [
        (f"02_训练曲线_{model_name}.png", "训练曲线"),
        (f"03_混淆矩阵_{model_name}.png", "混淆矩阵"),
        (f"03_混淆矩阵_番茄_{model_name}.png", "番茄子矩阵"),
        (f"05_各类F1_{model_name}.png", "各类别 F1"),
        (f"06_ROC_病害vs健康_{model_name}.png", "病害/健康 ROC"),
        (f"07_退化鲁棒性_{model_name}.png", "退化鲁棒性"),
        (f"10_遮挡实验_{model_name}.png", "遮挡实验"),
    ]
    imgs = []
    for fname, label in mapping:
        p = os.path.join(config.FIG_DIR, fname)
        if os.path.exists(p):
            imgs.append((p, label))
    return imgs


def eval_metrics_text(model_name):
    p = os.path.join(config.METRIC_DIR, f"eval_{model_name}.json")
    if not os.path.exists(p):
        return "暂无该模型的评估数据。"
    with io.open(p, encoding="utf-8") as f:
        m = json.load(f)
    return (f"Accuracy **{m['accuracy']*100:.2f}%**　·　"
            f"macro-F1 **{m['macro_f1']*100:.2f}%**　·　"
            f"Top-3 **{m['top3_accuracy']*100:.2f}%**　·　"
            f"AUC **{m['healthy_vs_disease_auc']:.4f}**　·　"
            f"测试样本 {m['total_test']}")


def compare_table():
    """三模型对比：精度 + 参数量 + FLOPs + 延迟（性能来自 benchmark.json）。"""
    bench = {}
    bp = os.path.join(config.METRIC_DIR, "benchmark.json")
    if os.path.exists(bp):
        try:
            with io.open(bp, encoding="utf-8") as f:
                _d = json.load(f)
            bench = _d.get("models", _d) if isinstance(_d, dict) else {}
        except Exception:
            bench = {}
    rows = []
    for name in config.BACKBONES:
        p = os.path.join(config.METRIC_DIR, f"eval_{name}.json")
        t = os.path.join(config.METRIC_DIR, f"train_{name}.json")
        if not os.path.exists(p):
            continue
        with io.open(p, encoding="utf-8") as f:
            m = json.load(f)
        params = "-"
        if os.path.exists(t):
            with io.open(t, encoding="utf-8") as f:
                params = f"{json.load(f)['params_m']:.2f}M"
        b = bench.get(name, {}) if isinstance(bench, dict) else {}
        flops = f"{b['flops_g']:.2f}G" if b.get("flops_g") else "-"
        lat = f"{b['latency_ms']:.1f}ms" if b.get("latency_ms") else "-"
        rows.append([name, f"{m['accuracy']*100:.2f}%", f"{m['macro_f1']*100:.2f}%",
                     params, flops, lat, f"{m['total_test']}"])
    return rows


REVIEW_CSV = os.path.join(config.OUTPUT_DIR, "reviews.csv")


def record_review(cls, action, note=""):
    """记录人工复核结论（确认 / 更正 / 驳回）。"""
    new = not os.path.exists(REVIEW_CSV)
    try:
        with io.open(REVIEW_CSV, "a", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["时间", "predict", "复核结论", "备注"])
            w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        cls, action, note])
        return True
    except Exception:
        return False


def load_history(limit=200):
    """读取识别历史，返回含表头的行列表。"""
    if not os.path.exists(HISTORY_CSV):
        return []
    with io.open(HISTORY_CSV, encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    head = list(rows[0])
    body = [list(r) for r in rows[1:]]
    # 旧记录只有 5 列，补齐「图片」列为空，避免前端越界
    if "图片" not in head:
        head.append("图片")
    width = len(head)
    for r in body:
        while len(r) < width:
            r.append("")
    if limit:
        body = body[-limit:]
    return [head] + body


def filter_history(crop=None, model=None, conf_min=None):
    """按作物、模型、最低置信度筛选历史。"""
    rows = load_history(limit=0)
    if len(rows) <= 1:
        return rows
    head, body = rows[0], rows[1:]
    out = []
    for r in body:
        if len(r) < 5:
            continue
        name = r[3]
        r_crop = name.split(" - ")[0] if " - " in name else name
        if crop and crop != "全部" and r_crop != crop:
            continue
        if model and model != "全部" and r[1] != model:
            continue
        if conf_min is not None:
            try:
                if float(r[4]) < conf_min:
                    continue
            except Exception:
                pass
        out.append(r)
    return [head] + out


def delete_history(indices, crop=None, model=None):
    """按表格显示序号删除（表格倒序，#1 为最新），并遵循当前筛选。

    返回 (删除条数, 剩余条数)。
    """
    if not os.path.exists(HISTORY_CSV):
        return 0, 0
    with io.open(HISTORY_CSV, encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if len(rows) <= 1:
        return 0, 0
    head, body = rows[0], rows[1:]

    # 复现界面上的筛选与倒序，保证显示的 # 与删除的 # 一致
    shown = []
    for r in body:
        if len(r) < 5:
            continue
        r_crop = r[3].split(" - ")[0] if " - " in r[3] else r[3]
        if crop and crop != "全部" and r_crop != crop:
            continue
        if model and model != "全部" and r[1] != model:
            continue
        shown.append(r)
    shown = list(reversed(shown))

    idxs = []
    for part in str(indices or "").replace("，", ",").split(","):
        part = part.strip()
        if part.isdigit():
            idxs.append(int(part))

    del_ids = {id(shown[n - 1]) for n in idxs if 1 <= n <= len(shown)}
    keep = [r for r in body if id(r) not in del_ids]
    removed = len(body) - len(keep)

    with io.open(HISTORY_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(keep)
    return removed, len(keep)


def clear_history(backup=True):
    """清空识别历史；默认先把现有内容备份为 history.csv.bak。

    返回 (是否成功, 备份路径或空字符串)。
    """
    if not os.path.exists(HISTORY_CSV):
        return True, ""
    bak = HISTORY_CSV + ".bak"
    if backup:
        try:
            import shutil
            shutil.copy2(HISTORY_CSV, bak)
        except Exception:
            bak = ""
    try:
        with io.open(HISTORY_CSV, encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
        head = rows[0] if rows else ["时间", "模型", "预测类别", "中文名", "置信度"]
        with io.open(HISTORY_CSV, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(head)
        return True, bak
    except Exception:
        return False, ""


def restore_history():
    """从 .bak 恢复历史。返回恢复的条数（不含表头）。"""
    bak = HISTORY_CSV + ".bak"
    if not os.path.exists(bak):
        return 0
    import shutil
    shutil.copy2(bak, HISTORY_CSV)
    with io.open(HISTORY_CSV, encoding="utf-8-sig") as f:
        return max(0, len(list(csv.reader(f))) - 1)


def history_chart():
    rows = load_history()
    if len(rows) <= 1:
        return None, None, "暂无历史记录。"
    data = rows[1:]

    import collections
    crop_cnt = collections.Counter()
    confs = []
    for r in data:
        if len(r) < 5:
            continue
        name = r[3]
        crop = name.split(" - ")[0] if " - " in name else name
        crop_cnt[crop] += 1
        try:
            confs.append(float(r[4]))
        except Exception:
            pass

    stamp = datetime.now().strftime("%H%M%S")
    p1 = os.path.join(config.OUTPUT_DIR, f"_chart_crop_{stamp}.png")
    p2 = os.path.join(config.OUTPUT_DIR, f"_chart_conf_{stamp}.png")

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.pie(list(crop_cnt.values()), labels=list(crop_cnt.keys()), autopct="%1.0f%%",
           startangle=90, textprops={"fontsize": 9})
    ax.set_title("作物分布", fontsize=11)
    plt.tight_layout(); plt.savefig(p1, dpi=130, bbox_inches="tight"); plt.close()

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(confs, bins=10, range=(0, 1), color="#00ADA2", edgecolor="white")
    ax.set_xlabel("置信度"); ax.set_ylabel("次数")
    ax.set_title(f"置信度分布（{len(confs)} 次诊断）", fontsize=11)
    plt.tight_layout(); plt.savefig(p2, dpi=130, bbox_inches="tight"); plt.close()

    return p1, p2, f"共 {len(data)} 条记录。"


def make_pdf(pil_img, model_name, conf_thresh=0.0, meta=None):
    """生成 PDF 诊断报告。

    conf_thresh > 0 时，若置信度低于阈值则不生成报告（与页面拒识保持一致）。
    """
    if pil_img is None:
        return None, "请先上传图片。"

    # H1: 与页面使用同一个叶片检查
    leaf_ok, _ = is_leaf_like(pil_img)
    if not leaf_ok:
        return None, "未识别到叶片，未生成报告。"

    t0 = time.time()
    top5, probs, pred = infer(pil_img, model_name)
    elapsed = time.time() - t0
    conf = top5[0][1]

    # H1: 置信度不足时拒绝导出，避免"页面说无法确定、PDF 给结论"
    if conf_thresh and conf < conf_thresh:
        return None, f"置信度 {conf*100:.1f}% 低于阈值 {conf_thresh*100:.0f}%，未生成报告。"

    info = knowledge.get_info(pred)

    out_path = os.path.join(
        config.OUTPUT_DIR, f"诊断报告_{datetime.now():%Y%m%d_%H%M%S}.pdf")
    lesion = None
    try:
        lesion = locate_raw(pil_img, model_name, thresh=0.5)
    except Exception:
        lesion = None

    try:
        from pdf_report import build_report
        task_no = datetime.now().strftime("%m%d-%H%M")
        build_report(pil_img, info, top5, model_name, out_path, cn_name,
                     elapsed=elapsed, task_no=task_no, lesion=lesion, meta=meta)
    except Exception as e:
        return None, f"生成失败：{e}"
    return out_path, "报告已生成"


# =====================================================================
#  界面
# =====================================================================
# UI 层已迁移至 app/web_agent.py
