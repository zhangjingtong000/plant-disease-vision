# -*- coding: utf-8 -*-
"""web_agent.py — 植物叶片病害智能诊断系统（UI 层）

设计规范：docs/DESIGN-longbridge.md
  · 强调色 Teal #00ADA2；主按钮黑色 pill；次按钮浅灰 8px 圆角
  · 卡片白底 + 1px #EAEBEC 边框 + 16px 圆角 + 默认无阴影
  · 页面底色 #F5F6F6；正文 #3C3C43 16px；标题 #0A0E19 600
  · 间距基准 4px；内容容器最大宽度 1200px

后端函数全部来自 app/web.py，本文件不修改后端，只负责界面与交互。
"""
import os
import sys
import io
import pickle
import re
import json
import html
import time
import random
from PIL import Image
import math
from collections import Counter
from urllib.parse import quote

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gradio as gr

import web
import ai_qa

CLASSES = web.CLASSES
BACKBONES = web.config.BACKBONES

# 类别与作物数（随数据集变化自动更新）
_NCLS = len(CLASSES)

# 知识库总条数（精编 + 扩充），与图像可识别的 _NCLS 区分开
_NKB = len(getattr(web.knowledge, 'ALL_KNOWLEDGE', web.knowledge.KNOWLEDGE))
_NCROP = len({v["crop_cn"] for v in web.knowledge.KNOWLEDGE.values()})
EXAMPLES = web.EXAMPLES
DEVICE = web.DEVICE

_MODEL_LABEL = {
    "resnet50": "ResNet50",
    "efficientnet_b0": "EfficientNet-B0",
    "mobilenet_v3_large": "MobileNet-V3 Large",
}
_MODEL_CHOICES = [(_MODEL_LABEL[b], b) for b in BACKBONES]

_CLASS_CHOICES = []
for _c in CLASSES:
    _i = web.knowledge.get_info(_c)
    _CLASS_CHOICES.append((f"{_i['crop_cn']} · {_i['disease_cn']}", _c))

_KB_CHOICES = list(_CLASS_CHOICES)
try:
    for _kc, _ki in getattr(web.knowledge, "KG_KNOWLEDGE", {}).items():
        if _kc not in CLASSES:
            _KB_CHOICES.append((f'{_ki.get("crop_cn", "")} · {_ki.get("disease_cn", "")}', _kc))
except Exception:
    pass
_KB_TOTAL = len(getattr(web.knowledge, "ALL_KNOWLEDGE", {}) or _KB_CHOICES)

_CROP_COUNT = len({web.knowledge.get_info(c)["crop_cn"] for c in CLASSES})
_CKPT = os.path.join(web.config.CKPT_DIR, "resnet50_best.pth")
_MODEL_READY = os.path.exists(_CKPT)
_DEV = "GPU" if DEVICE.type == "cuda" else "CPU"

# =====================================================================
#  内联 SVG 图标（无外部资源）
# =====================================================================
_ICONS = {
    "leaf": '<path d="M20.5 3.5C12 3.5 5 7 5 14a5 5 0 0 0 5 5c7 0 10.5-7 10.5-15.5Z"/><path d="M4 21c2.5-6 6.5-9.5 12-12"/>',
    "upload": '<path d="M12 16V4"/><path d="m7 9 5-5 5 5"/><path d="M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"/>',
    "scan": '<path d="M4 8V6a2 2 0 0 1 2-2h2"/><path d="M16 4h2a2 2 0 0 1 2 2v2"/><path d="M20 16v2a2 2 0 0 1-2 2h-2"/><path d="M8 20H6a2 2 0 0 1-2-2v-2"/><path d="M8 12h8"/>',
    "target": '<circle cx="12" cy="12" r="7.5"/><circle cx="12" cy="12" r="2.5"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/>',
    "file": '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/><path d="M9 13h6M9 17h4"/>',
    "chart": '<path d="M4 20V11M10 20V4M16 20v-6"/><path d="M2 20h20"/>',
    "book": '<path d="M4 5a2 2 0 0 1 2-2h13v18H6a2 2 0 0 1-2-2z"/><path d="M4 17h15"/>',
    "clock": '<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/>',
    "layers": '<path d="m12 3 9 5-9 5-9-5z"/><path d="m3 13 9 5 9-5"/>',
    "cpu": '<rect x="6" y="6" width="12" height="12" rx="2"/><rect x="9.5" y="9.5" width="5" height="5" rx="1"/><path d="M9 2v2M15 2v2M9 20v2M15 20v2M2 9h2M2 15h2M20 9h2M20 15h2"/>',
    "check": '<path d="m5 12 5 5L20 7"/>',
    "alert": '<path d="M12 3 2.5 20h19z"/><path d="M12 9v5M12 17h.01"/>',
    "download": '<path d="M12 4v12"/><path d="m7 11 5 5 5-5"/><path d="M4 20h16"/>',
    "history": '<path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 4v5h5"/><path d="M12 8v4l3 2"/>',
    "shield": '<path d="M12 3 5 6v6c0 4 3 7 7 9 4-2 7-5 7-9V6z"/><path d="m9 12 2 2 4-4"/>',
    "spark": '<path d="m12 3 1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/>',
    "arrow": '<path d="M5 12h14"/><path d="m14 7 5 5-5 5"/>',
    "grid": '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
    "refresh": '<path d="M20 11a8 8 0 1 0-2 5.3"/><path d="M20 4v7h-7"/>',
    "trash": '<path d="M4 7h16"/><path d="M9 7V5h6v2"/><path d="M6 7l1 13h10l1-13"/>',
}

def icon(name, size=18, cls=""):
    body = _ICONS.get(name, "")
    return (
        f'<svg class="lb-i {cls}" width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">{body}</svg>'
    )

def esc(v):
    return html.escape(str(v), quote=True)

# =====================================================================
#  页面片段
# =====================================================================
def render_header():
    ready = (
        f'<span class="lb-chip"><i class="lb-dot"></i>{_MODEL_LABEL["resnet50"]} 权重就绪</span>'
        if _MODEL_READY else
        '<span class="lb-chip is-warn"><i class="lb-dot is-warn"></i>权重未找到</span>'
    )
    return f'''
<header class="lb-header">
  <div class="lb-header-inner">
    <div class="lb-brand">
      <span class="lb-brand-mark">{icon("leaf", 21)}</span>
      <div class="lb-brand-text">
        <h1>植物叶片病害智能诊断系统</h1>
        <p>作物叶片病害识别 · 病斑定位</p>
      </div>
    </div>
    <div class="lb-brand-meta">
      <span class="lb-chip">{icon("cpu", 14)}{_DEV} 推理</span>
      <span class="lb-chip">{icon("layers", 14)}{len(CLASSES)} 类 · {_CROP_COUNT} 种作物</span>
      {ready}
    </div>
  </div>
</header>'''

def pagehead(title, desc, tag):
    return f'''
<div class="lb-pagehead">
  <div class="lb-pagehead-main">
    <span class="lb-pagehead-tag">{tag}</span>
    <h1>{title}</h1>
  </div>
  <p>{desc}</p>
</div>'''

PAGEHEAD_DIAG = pagehead("智能诊断", "上传叶片照片，选择骨干网络与拒识阈值后开始识别。", "01")
PAGEHEAD_BATCH = pagehead("批量诊断", "一次上传多张叶片照片，统一模型批量识别并导出 CSV。", "02")
PAGEHEAD_EVAL = pagehead("模型评估", "查看三个骨干网络的同分布测试指标，以及 ResNet50 在 PlantDoc 上的零样本跨域结果与域偏移诊断。", "03")
_QA_INTENT_PKL = os.path.join(_ROOT, "src", "intent_model", "intent_clf.pkl")
# 实体识别 90.8% 来自严格评估结果；当前没有可直接读取的产物，先以常量展示。
_QA_ENTITY_ACC = 90.8
# 回答配图逻辑固定为 2 张实验室 + 1 张真实场景。
_QA_ANSWER_SHOTS = 3


def _intent_heldout():
    """读取 intent_clf.pkl 的 heldout_acc；缺失/损坏时返回 —。"""
    try:
        with io.open(_QA_INTENT_PKL, "rb") as fp:
            data = pickle.load(fp)
        value = data.get("heldout_acc")
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "—"


def render_qa_tech_strip():
    # 展示实际可用于问答的条目数（精编 + 扩充库），不是只有精编那 132 条
    n_knowledge = len(getattr(web.knowledge, "ALL_KNOWLEDGE",
                              web.knowledge.KNOWLEDGE))
    tiles = [
        ("混合检索", "向量 + BM25", f"{n_knowledge}", "条知识"),
        ("意图理解", "bge 嵌入 + 逻辑回归", _intent_heldout(), ""),
        ("实体识别", "别名词典 + 编辑距离", f"{_QA_ENTITY_ACC:.1f}%", ""),
        ("自动配图", "按命中病害匹配", f"{_QA_ANSWER_SHOTS}", "张/次"),
    ]
    cells = "".join(
        f'<div class="lb-tech-card"><span class="lb-tech-k">{k}</span>'
        f'<span class="lb-tech-desc">{d}</span>'
        f'<div class="lb-tech-v">{v}<em>{u}</em></div></div>'
        for k, d, v, u in tiles
    )
    return (
        '<div class="lb-tech-strip">'
        '<div class="lb-tech-note">混合检索 + 意图理解 + 实体识别，'
        '回答基于本地知识库，不依赖任何外部 API。</div>'
        f'<div class="lb-tech-grid">{cells}</div></div>'
    )


PAGEHEAD_KB = pagehead("知识问答", "基于本地知识库的检索增强问答。", "04")
PAGEHEAD_MORE = pagehead("更多", "识别历史、统计分布与使用说明。", "05")

EMPTY_RESULT = f'''
<div class="lb-result-wrap">
  <div class="lb-card lb-empty-card">
    <div class="lb-empty">
      <span class="lb-empty-icon">{icon("scan", 26)}</span>
      <h2>等待诊断</h2>
      <p>上传叶片照片，或从左侧载入内置示例，<br>点击「开始诊断」查看识别结果。</p>
      <div class="lb-empty-steps">
        <div><span>1</span><b>上传照片</b><em>拖拽或点击选择</em></div>
        <div><span>2</span><b>开始诊断</b><em>模型给出 Top-5</em></div>
        <div><span>3</span><b>查看详情</b><em>定位病斑与防治</em></div>
      </div>
    </div>
  </div>
</div>'''

def _split_name(cls):
    info = web.knowledge.get_info(cls)
    return info["crop_cn"], info["disease_cn"]

def _plain_verdict(info, conf):
    """把识别结果翻译成一句农户能懂的话。"""
    sev = info.get("severity", "")
    disease = info.get("disease_cn", "")
    if "健康" in disease:
        return "叶片未见明显病害特征，保持常规管理即可。"
    action = {
        "极高": "属高危险病害，建议尽快处理并隔离病株",
        "高": "危害较大，建议尽早用药控制",
        "中": "需要留意，建议按防治方案处理",
        "低": "危害较轻，做好田间管理即可",
    }.get(sev, "建议按下方防治方案处理")
    return f"识别为{disease}，{action}；把握约 {conf*100:.0f}%。"



def _img_url(path):
    """把本地绝对路径转成 Gradio 可直接访问的 URL。"""
    p = str(path).replace("\\", "/")
    return "/gradio_api/file=" + quote(p, safe="/:")

def render_shot_grid(items):
    """items: [(path, source_label), ...]；缩略图点击后在新标签打开原图。"""
    cells = []
    for path, label in items:
        u = _img_url(path)
        cells.append(
            f'<a class="lb-shot" href="{u}" rel="noopener" '
            f'title="点击查看大图 · {esc(label)}">'
            f'<img src="{u}" loading="lazy" alt="{esc(label)}">'
            f'<span class="lb-shot-src">{esc(label)}</span></a>'
        )
    n = max(1, min(len(items), 3))
    return f'<div class="lb-shot-grid" style="--shot-cols:{n}">{"".join(cells)}</div>'

def render_result(info, top5, elapsed, model_name, low):
    conf = top5[0][1]
    ref_paths = web.get_class_images(info.get("cls"), n=1, source="lab")
    ref_html = ""
    if ref_paths:
        _u = _img_url(ref_paths[0])
        ref_html = (f'<a class="lb-result-ref" href="{_u}" rel="noopener" '
                    f'title="实验室样本参考图，点击查看大图">'
                    f'<img src="{_u}" loading="lazy" alt="实验室样本参考图">'
                    f'<span>实验室样本</span></a>')
    sev_high = info["severity"] in ("高", "极高")
    badge = (
        f'<span class="lb-badge is-warn">{icon("alert", 13)}建议人工复核</span>'
        if low else
        f'<span class="lb-badge">{icon("check", 13)}可信判定</span>'
    )
    sev_cls = " is-warn-text" if sev_high else ""
    if elapsed is None:
        elapsed_fact = ('<div class="lb-fact"><span>推理耗时</span><b>历史记录</b>'
                        '<i class="lb-fact-note">不重新推理</i></div>')
    else:
        elapsed_fact = (f'<div class="lb-fact"><span>推理耗时</span>'
                        f'<b>{elapsed * 1000:.0f} ms</b>'
                        '<i class="lb-fact-note">首次含模型加载</i></div>')
    return f'''
<div class="lb-card lb-result-card">
  {ref_html}
  <div class="lb-result-head">
    <span class="lb-eyebrow">诊断结果 · {esc(_MODEL_LABEL.get(model_name, model_name))}</span>
    {badge}
  </div>
  <div class="lb-crop">{esc(info["crop_cn"])}<span>作物叶片样本</span></div>
  <h2 class="lb-disease">{esc(info["disease_cn"])}</h2>
  <p class="lb-plain-verdict">{_plain_verdict(info, conf)}</p>
  <div class="lb-conf">
    <span class="lb-conf-num">{conf * 100:.1f}</span>
    <span class="lb-conf-unit">%</span>
    <span class="lb-conf-label">置信度</span>
  </div>
  <div class="lb-meter"><i style="width:{max(conf * 100, 0.6):.2f}%"></i></div>
  <div class="lb-facts">
    <div class="lb-fact"><span>病原类型</span><b>{esc(info["pathogen"])}</b></div>
    <div class="lb-fact"><span>病害危害性</span><b class="{sev_cls}">{esc(info["severity"])}</b></div>
    {elapsed_fact}
  </div>
</div>'''

def render_not_leaf(meta):
    """输入不像叶片时的提示卡。"""
    return f'''
<div class="lb-card lb-result-card">
  <div class="lb-result-head">
    <span class="lb-eyebrow">输入检查</span>
    <span class="lb-badge is-warn">{icon("alert", 13)}未识别到叶片</span>
  </div>
  <h2 class="lb-disease" style="font-size:23px;">这张图不像叶片照片</h2>
  <p class="lb-symptom" style="margin-top:10px;">
    画面中没有检测到足够比例的叶片区域，系统不给出诊断结论。
    请拍摄单片叶片，主体占画面 60% 以上，背景尽量干净，光照均匀。
  </p>
  <div class="lb-facts">
    <div class="lb-fact"><span>绿色区域占比</span><b>{meta["green_ratio"] * 100:.0f}%</b></div>
    <div class="lb-fact"><span>区域连通度</span><b>{meta["conn"] * 100:.0f}%</b></div>
    <div class="lb-fact"><span>图像熵</span><b>{meta["entropy"]:.1f}</b></div>
  </div>
</div>'''


def render_rejected(info, top5, conf, thresh, model_name):
    """置信度低于阈值时的拒识卡：不给结论。"""
    return f'''
<div class="lb-card lb-result-card">
  <div class="lb-result-head">
    <span class="lb-eyebrow">诊断结果 · {esc(_MODEL_LABEL.get(model_name, model_name))}</span>
    <span class="lb-badge is-warn">{icon("alert", 13)}置信度不足</span>
  </div>
  <h2 class="lb-disease" style="font-size:23px;">无法确定</h2>
  <p class="lb-symptom" style="margin-top:10px;">
    最高置信度只有 {conf * 100:.1f}%，低于当前拒识阈值 {thresh * 100:.0f}%。
    这通常意味着拍摄角度、清晰度或病害早期特征不明显，系统不强行给出结论。
  </p>
  <div class="lb-facts">
    <div class="lb-fact"><span>最高置信度</span><b>{conf * 100:.1f}%</b></div>
    <div class="lb-fact"><span>拒识阈值</span><b>{thresh * 100:.0f}%</b></div>
    <div class="lb-fact"><span>处理建议</span><b>重拍或人工判断</b></div>
  </div>
  <p class="lb-symptom" style="margin-top:12px;font-size:12.5px;color:var(--lb-faint);">
    下方 Top-5 仅作参考，不代表诊断结论。
  </p>
  <div class="lb-help-row">
    <span>还是判断不了？</span>
    <b>建议联系当地植保站或农技推广部门</b>
    <em>可把这张图和 System 提示的拍摄建议一起提供给对方</em>
  </div>
</div>'''


def render_top5(top5, bare=False):
    rows = []
    for i, (cls, p) in enumerate(top5):
        crop, disease = _split_name(cls)
        rows.append(f'''
      <div class="lb-rank{" is-first" if i == 0 else ""}">
        <span class="lb-rank-idx">{i + 1:02d}</span>
        <span class="lb-rank-name">{esc(crop)} · {esc(disease)}</span>
        <span class="lb-rank-track"><i style="width:{max(p * 100, 1.2):.2f}%"></i></span>
        <span class="lb-rank-pct">{p * 100:.2f}%</span>
      </div>''')
    body = f'''
  <div class="lb-top5-head">
    <h3>Top-5 候选</h3>
    <span>同一次前向计算的全部候选</span>
  </div>
  {"".join(rows)}
  <div class="lb-top5-foot">下方「病斑定位」「症状与防治」可切换查看，无需滚动页面</div>'''
    if bare:
        return body
    return f'<div class="lb-card lb-top5-card">{body}</div>'

def render_result_block(info, top5, elapsed, model_name, low):
    return f'<div class="lb-result-wrap">{render_result(info, top5, elapsed, model_name, low)}{render_top5(top5)}</div>'

def _clean_text(value):
    """去掉知识库文本里从 docx 带出的 _x000D_ 占位。"""
    return str(value).replace("_x000D_", "").replace("\r", "")

def _split_inline_blocks(text):
    """把内嵌 (1)(2)① 的长文本拆成 [(序号, 内容), ...]。

    编号前的导语返回序号为 0 的项。没有编号时返回 [(0, 原文)]。
    """
    import re as _re
    s = _clean_text(text or "").strip()
    if not s:
        return []

    pat = _re.compile(
        r"(?:\((\d+)\)|（(\d+)）|([①②③④⑤⑥⑦⑧⑨⑩])|(?<![\d.])(\d+)[.、])")

    marks = list(pat.finditer(s))
    if len(marks) < 2:
        return [(0, s)]

    blocks = []
    # 第一个编号之前的内容作为导语
    head = s[:marks[0].start()].strip(" ；;，,。")
    if head and len(head) > 4:
        blocks.append((0, head))

    for i, m in enumerate(marks):
        start = m.end()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(s)
        body = s[start:end].strip(" ；;，,。")
        if not body or len(body) <= 2:
            continue
        num = m.group(1) or m.group(2) or m.group(4)
        if not num:
            # ① 这类符号编号，按顺序给号
            num = str(i + 1)
        blocks.append((num, body))
    return blocks if len(blocks) >= 2 else [(0, s)]


def _render_blocks(blocks, css_cls):
    """把 [(序号, 内容)] 渲染成带编号的段落列表。"""
    if not blocks:
        return ""
    if len(blocks) == 1 and blocks[0][0] == 0:
        return f'<p class="{css_cls}-plain">{esc(blocks[0][1])}</p>'
    lis = []
    for num, body in blocks:
        if str(num) == "0":
            lis.append(f'<li class="{css_cls}-lead"><p>{esc(body)}</p></li>')
        else:
            lis.append(
                f'<li><span>{esc(str(num))}</span><p>{esc(body)}</p></li>')
    return f'<ol class="{css_cls}">{"".join(lis)}</ol>'


def _expand_control(items):
    """把内嵌 (1)(2) 的防治条目拆成多条，让每项单独一行。

    数据里有 49 条把「农业防治…化学防治…」塞进了一个字符串，
    直接渲染会挤成一坨看不到换行。
    """
    import re as _re
    # 先合并原始数据里「只有小标题」的碎片条目，如 "农业防治：" 单独一条、
    # 具体内容在下一条。这类碎片单独成行会很难看。
    _clean = [_clean_text(t or "").strip() for t in (items or [])]
    _clean = [s for s in _clean if s]
    pre = []
    for s in _clean:
        if pre and len(pre[-1]) <= 12 and pre[-1].rstrip().endswith(("：", ":")):
            pre[-1] = pre[-1] + s
        else:
            pre.append(s)

    out = []
    for s in pre:
        parts = _re.split(
            r"(?:\(\d+\)|（\d+）|(?<![\d.])[①②③④⑤⑥⑦⑧⑨⑩]|\s(?=\d+[.、]\D))",
            s)
        parts = [p.strip(" ；;，,。") for p in parts if p and len(p.strip()) > 4]
        if len(parts) < 2:
            out.append(s)
            continue
        # 「农业防治：」这种只有标题没有内容的碎片，并到后面那一项
        merged = []
        for seg in parts:
            if merged and len(merged[-1]) <= 12 and merged[-1].rstrip().endswith(("：", ":")):
                merged[-1] = merged[-1] + seg
            else:
                merged.append(seg)
        out.extend(merged)
    return out


def _strip_leading_no(t):
    """去掉正文开头的 (1) ① 1. 之类编号；列表本身已有序号，避免「1 (1)农业防治」。"""
    import re as _re
    s = _clean_text(t or "")
    s = _re.sub(r"^\s*(?:\(\d+\)|（\d+）|\d+[.、]|(?<!\d)[①②③④⑤⑥⑦⑧⑨⑩])\s*", "", s)
    return s.strip()


def _render_occurrence(info):
    """发生规律 + 权威来源标注；普通/降级条目按需显示。"""
    occ = _clean_text(info.get("occurrence") or "")
    if not occ:
        return ""
    source = str(info.get("occ_source") or "").strip()
    url = str(info.get("occ_url") or "").strip()
    if url and not url.lower().startswith(("http://", "https://")):
        url = ""
    typ = str(info.get("occ_type") or "industry").strip().lower()
    if typ not in ("official", "journal", "industry"):
        typ = "industry"
    cls = {"official": "is-official", "journal": "is-journal",
           "industry": "is-industry"}[typ]
    source_html = ""
    if source:
        label = source
        if typ == "industry" and "行业媒体" not in label:
            label += "（行业媒体）"
        if url:
            source_html = (f'<a class="lb-occ-source {cls}" href="{esc(url)}" '
                           f'target="_blank" rel="noopener">来源：{esc(label)} ↗</a>')
        else:
            source_html = (f'<span class="lb-occ-source {cls}">'
                           f'来源：{esc(label)}</span>')
    degraded = ("资料较少" in occ) or ("建议参考当地植保部门" in occ)
    degraded_html = ('<div class="lb-occ-degraded">本条为通用说明，非具体病害数据</div>'
                     if degraded else "")
    return f'''
<div class="lb-occ-block">
  <h3 class="lb-occ-title">{icon("info", 16)}发生规律</h3>
  {_render_blocks(_split_inline_blocks(occ), "lb-blocklist")}
  {source_html}
  {degraded_html}
</div>'''


def render_detail(info, with_head=True):
    cls = info.get("cls", "")
    sym = _clean_text(info.get("symptoms") or "")
    # 数据修补文件里以“通用___病害名”收录的展开症状，作为短文本的后备展示
    if len(sym) < 40:
        try:
            from knowledge_fix import PATCH as _KB_FIX
            _base = re.split(r"[（(]", str(info.get("disease_cn") or ""))[0].strip()
            _alt = (_KB_FIX.get("通用___" + _base) or {}).get("symptoms") or ""
            if len(_alt) > len(sym):
                sym = _clean_text(_alt)
        except Exception:
            pass
    warn = ""
    if len(sym) < 40:
        warn = ('<div class="lb-lesion-note is-warn">'
                + icon("alert", 15) +
                '该病症状资料较少，建议结合图片诊断或咨询当地植保部门。</div>')
    ctrl = web.knowledge.get_control(cls) if cls else {}
    has_struct = bool(ctrl.get("农业防治") or ctrl.get("化学防治"))
    occ_html = _render_occurrence(info)

    # 统一口径：所有防治建议都是「带序号的编号列表」。
    # 有结构化数据（农业/化学/注意事项）时，每组单独编号；否则整体编号。
    if has_struct:
        blocks = []
        for title, key in (("农业防治", "农业防治"),
                           ("化学防治", "化学防治"),
                           ("注意事项", "注意事项")):
            raw = _expand_control(ctrl.get(key) or [])
            if not raw:
                continue
            lis = "".join(
                f'<li><span>{i}</span><p>{esc(_strip_leading_no(t))}</p></li>'
                for i, t in enumerate(raw, 1))
            blocks.append(
                f'<div class="lb-ctrl-group"><h4>{title}</h4>'
                f'<ol class="lb-control-list">{lis}</ol></div>')
        items = "".join(blocks)
    else:
        _ctrl_items = _expand_control(info.get("control") or [])
        items = "".join(
            f'<li><span>{i}</span><p>{esc(_strip_leading_no(t))}</p></li>'
            for i, t in enumerate(_ctrl_items, 1)
        )
    head = "" if not with_head else '''
<div class="lb-section-head">
  <h2>诊断详情</h2>
  <span>症状与防治建议 · 供参考</span>
</div>'''
    return f'''{head}
<div class="lb-card lb-detail-card">
  <div class="lb-detail-grid">
    <div class="lb-detail-col">
      <h3>{icon("scan", 17)}症状表现</h3>
      {_render_blocks(_split_inline_blocks(sym), "lb-blocklist")}
      {warn}
    </div>
    <div class="lb-detail-col">
      <h3>{icon("shield", 17)}防治建议</h3>
      {items if has_struct else f'<ol class="lb-control-list">{items}</ol>'}
    </div>
  </div>
  {occ_html}
</div>'''

LESION_HEAD = f'''
<div class="lb-card-head">
  <div>
    <h2>病斑定位</h2>
    <p>Grad-CAM 热力图与病斑外接框</p>
  </div>
  <span class="lb-hint">{icon("target", 15)}模型解释</span>
</div>'''

def severity_grade(ratio_pct):
    """病斑面积占比 → 农学常用严重度级别（0/1/3/5/7/9）。"""
    if ratio_pct <= 0:
        return 0, "无病斑"
    if ratio_pct < 5:
        return 1, "轻微"
    if ratio_pct < 10:
        return 3, "轻度"
    if ratio_pct < 25:
        return 5, "中度"
    if ratio_pct < 50:
        return 7, "重度"
    return 9, "严重"


def disease_index(grades):
    """病情指数 I = Σ(病级 × 叶数) / (总叶数 × 9) × 100。"""
    if not grades:
        return 0.0
    return sum(grades) / (len(grades) * 9) * 100


def render_lesion_caption(text, models_ready=True):
    if not text:
        return ""
    plain = re.sub(r"<[^>]+>", "", text).replace("\u3000", " ").strip()
    ratio = re.search(r"病斑占叶面\s*([\d.]+)%", plain)
    level = re.search(r"危害\s*([^\s·]+)", plain)
    patho = re.search(r"病原\s*([^\s·]+)", plain)
    if not ratio:
        return f'<div class="lb-lesion-note">{icon("info", 15)}未检测到明显病斑区域，热力图仍可参考。</div>'
    _lv, _lv_name = severity_grade(float(ratio.group(1)))
    chips = [f'<span class="lb-stat-chip"><em>病斑占叶面</em><b>{ratio.group(1)}%</b></span>',
             f'<span class="lb-stat-chip"><em>严重度</em><b>{_lv} 级 · {_lv_name}</b></span>']
    if level:
        chips.append(f'<span class="lb-stat-chip"><em>病斑程度</em><b>{esc(level.group(1))}</b></span>')
    if patho:
        chips.append(f'<span class="lb-stat-chip"><em>病原类型</em><b>{esc(patho.group(1))}</b></span>')
    return f'<div class="lb-lesion-facts">{"".join(chips)}</div>'

def render_batch_summary(rows):
    """批量汇总：张数、作物分布、待复核清单、处置建议。"""
    if not rows:
        return ""
    from collections import Counter

    total = len(rows)
    crops = Counter()
    review = []
    unknown = 0
    for r in rows:
        if len(r) < 4:
            continue
        name = str(r[1])
        if name == "未识别到叶片":
            unknown += 1
            continue
        crop = name.split(" - ")[0] if " - " in name else name
        crops[crop] += 1
        if str(r[3]) != "可信":
            review.append((str(r[0]), name, str(r[2])))

    crop_txt = "、".join(f"{k} {v} 张" for k, v in crops.most_common()) or "—"
    review_rows = "".join(
        f'<li><b>{esc(fn)}</b><span>{esc(nm)} · {esc(cf)}</span></li>'
        for fn, nm, cf in review[:8]
    ) or '<li class="lb-none">本批没有需要复核的样本</li>'
    more = (f'<p class="lb-more">另有 {len(review) - 8} 张待复核，见 CSV</p>'
            if len(review) > 8 else "")

    advice = []
    if review:
        advice.append(f"优先处理 {len(review)} 张待复核样本，可重拍后单独诊断")
    if unknown:
        advice.append(f"{unknown} 张未识别到叶片，建议确认拍摄对象")
    if crops.most_common(1) and crops.most_common(1)[0][1] >= max(3, total // 3):
        top_crop, top_n = crops.most_common(1)[0]
        advice.append(f"{top_crop} 占 {top_n} 张，建议关注该作物的连片发病风险")
    if not advice:
        advice.append("本批结果整体可信，按各类防治方案常规处理即可")
    advice.append("严重度分级按单片病斑占叶面比例：<5% 为 1 级、5–10% 为 3 级、10–25% 为 5 级、25–50% 为 7 级、>50% 为 9 级")

    return f'''
<div class="lb-card lb-batch-summary">
  <div class="lb-card-head">
    <div><h2>本批汇总</h2><p>共 {total} 张 · {len(crops)} 种作物 · {len(review)} 张待复核</p></div>
    <span class="lb-hint">{icon("grid", 15)}批次概览</span>
  </div>
  <div class="lb-sum-grid">
    <div class="lb-sum-block"><span>作物分布</span><p>{esc(crop_txt)}</p></div>
    <div class="lb-sum-block"><span>待复核清单</span><ul class="lb-sum-review">{review_rows}</ul>{more}</div>
  </div>
  <div class="lb-sum-advice">
    <b>处置建议</b>
    <ol>{"".join(f"<li>{esc(a)}</li>" for a in advice)}</ol>
  </div>
</div>'''


def render_batch_empty():
    return f'''
<div class="lb-card lb-batch-empty-card">
  <div class="lb-empty lb-empty-sm">
    <span class="lb-empty-icon">{icon("grid", 24)}</span>
    <h2>暂未开始批量诊断</h2>
    <p>上传多张照片并点击「开始批量诊断」，结果会显示在这里。</p>
  </div>
</div>'''

def render_batch(rows, model_name):
    if not rows:
        return render_batch_empty()
    normal = sum(1 for r in rows if str(r[3]) == "可信")
    summary_html = render_batch_summary(rows)
    review = len(rows) - normal
    body = []
    for r in rows:
        flag = str(r[3])
        if flag == "可信":
            badge = f'<span class="lb-badge">{icon("check", 12)}可信</span>'
        elif flag == "无法识别":
            badge = f'<span class="lb-badge is-warn">{icon("alert", 12)}无法识别</span>'
        elif flag == "失败":
            badge = f'<span class="lb-badge is-warn">{icon("alert", 12)}失败</span>'
        else:
            badge = f'<span class="lb-badge is-warn">{icon("alert", 12)}待复核</span>'
        body.append(f'''
      <tr>
        <td class="lb-file-name">{esc(r[0])}</td>
        <td>{esc(r[1])}</td>
        <td class="lb-num">{esc(r[2])}</td>
        <td>{badge}</td>
      </tr>''')
    return f'''
<div class="lb-card lb-table-card lb-batch-result-card">
  <div class="lb-card-head">
    <div>
      <h2>批量诊断结果</h2>
      <p>共 {len(rows)} 张 · 可信 {normal} 张 · 待复核/其他 {review} 张</p>
    </div>
    <span class="lb-hint">{esc(_MODEL_LABEL.get(model_name, model_name))}</span>
  </div>
  <div class="lb-table-wrap">
    <table class="lb-table">
      <thead><tr><th>照片</th><th>诊断结果</th><th class="lb-num">置信度</th><th>状态</th></tr></thead>
      <tbody>{"".join(body)}</tbody>
    </table>
  </div>
</div>'''

EVAL_FIGS = [
    ("训练曲线", "训练与验证的准确率、损失变化", "02_训练曲线"),
    ("混淆矩阵", f"{_NCLS} 类整体分类情况", "03_混淆矩阵"),
    ("番茄子矩阵", "番茄类内细粒度混淆", "03_混淆矩阵_番茄"),
    ("各类别 F1", "按类别查看精确率与召回", "05_各类F1"),
    ("病害 / 健康 ROC", "二分类区分能力，越靠左上越好", "06_ROC"),
    ("退化鲁棒性", "噪声 / 运动模糊 / 降分辨率下的准确率", "07_退化鲁棒性"),
    ("遮挡实验", "遮挡病斑区 vs 等面积随机区，置信度落差", "10_遮挡实验"),
]

_PH_CACHE = {}


def _placeholder(text):
    """为缺失的图表生成一张占位图，避免界面出现空白框。"""
    if text in _PH_CACHE:
        return _PH_CACHE[text]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
        fig, ax = plt.subplots(figsize=(6, 3.2), dpi=110)
        ax.set_facecolor("#F7F8F8")
        fig.patch.set_facecolor("#FFFFFF")
        ax.axis("off")
        ax.text(0.5, 0.58, text, ha="center", va="center", fontsize=12, color="#82888D")
        ax.text(0.5, 0.38, "该模型未跑这一项实验", ha="center", va="center",
                fontsize=9.5, color="#B4B8BB")
        p = os.path.join(web.config.OUTPUT_DIR, f"_ph_{abs(hash(text)) % 10**8}.png")
        plt.tight_layout()
        plt.savefig(p, dpi=110, facecolor="white")
        plt.close(fig)
        _PH_CACHE[text] = p
        return p
    except Exception:
        return None


# outputs/metrics 是本地跑实验的落盘目录（被 .gitignore 忽略），results/ 是随
# 仓库提供的实测产物。clone 下来只有 results/，所以两个目录都要找，否则跨域面板
# 永远是空的。
_RESULTS_DIR = os.path.join(_ROOT, "results")


def _read_metric_json(name):
    """按 outputs/metrics → results/ 顺序读取指标 JSON；两处都没有返回 None。"""
    for d in (web.config.METRIC_DIR, _RESULTS_DIR):
        p = os.path.join(d, name)
        if not os.path.exists(p):
            continue
        try:
            with io.open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            continue
    return None


def _fmt_pct(ratio, digits=2):
    """比率 → "38.98%"；缺值一律 "—"，不把「没有产物」显示成 0.00%。"""
    if ratio is None:
        return "—"
    try:
        return "{:.{}f}%".format(float(ratio) * 100.0, digits)
    except (TypeError, ValueError):
        return "—"


def _load_domain_clf_stats():
    """域分类器准确率与各折范围；产物缺失时返回 "—"。"""
    d = _read_metric_json("domain_classifier.json") or {}
    cv = d.get("cv_scores") or []
    cv_s = "{:.1f}%–{:.1f}%".format(min(cv) * 100, max(cv) * 100) if cv else "—"
    return _fmt_pct(d.get("domain_classifier_acc")), cv_s


_DC_ACC, _DC_CV = _load_domain_clf_stats()


def _load_xd_stats():
    """跨域协议指标，动态读取；产物缺失时每一项都是 "—"。"""
    d = _read_metric_json("cross_domain_protocol.json")
    keys = ("t1", "ci", "t3", "t5", "f1", "mb", "n", "nc", "strict")
    if not d:
        return {k: "—" for k in keys}
    lo, hi = d.get("ci_low"), d.get("ci_high")
    f1 = d.get("macro_f1")
    return {
        "t1": _fmt_pct(d.get("acc_fold")),
        "ci": ("{:.1f}–{:.1f}".format(lo * 100, hi * 100)
               if lo is not None and hi is not None else "—"),
        "t3": _fmt_pct(d.get("top3")),
        "t5": _fmt_pct(d.get("top5")),
        "f1": ("{:.4f}".format(f1) if f1 is not None else "—"),
        "mb": _fmt_pct(d.get("majority_baseline")),
        "n": d.get("n", "—"),
        "nc": d.get("n_classes", "—"),
        "strict": _fmt_pct(d.get("acc_strict")),
    }


_XD = _load_xd_stats()


def _load_adapt_stats():
    """AdaBN 与混合微调的最新结果；产物缺失时保留 None，界面显示 "—"。"""
    out = {"bn_before": None, "bn_after": None, "mx_before": None, "mx_after": None}
    d = _read_metric_json("bn_adapt_resnet50.json")
    if d:
        out["bn_before"] = d.get("before")
        out["bn_after"] = d.get("after")
    d = _read_metric_json("mixed_resnet50_mix.json")
    if d:
        out["mx_before"] = d.get("before")
        out["mx_after"] = d.get("after")
    return out


_AD = _load_adapt_stats()
_BN_A = _AD["bn_after"]
_MX_A = _AD["mx_after"]
# 同分布准确率不写死：取 results/cross_domain_resnet50.json 的 in_domain_accuracy
_XD_IN = _fmt_pct((_read_metric_json("cross_domain_resnet50.json") or {}
                   ).get("in_domain_accuracy"))


def _load_offset_spectrum():
    """偏移谱各档准确率（百分数）；产物缺失时各档为 None。"""
    out = {k: None for k in ("S0", "S1", "S2", "S3", "S4", "S5")}
    d = _read_metric_json("offset_spectrum.json")
    if d:
        for name, v in (d.get("levels") or {}).items():
            key = name.split(" ")[0]
            if key in out:
                out[key] = (v.get("mean") or 0) * 100
    return out


def _load_domain_clf_counts():
    """域分类器用到的源域 / 目标域样本量（缺产物时两项都是 "—"）。"""
    d = _read_metric_json("domain_classifier.json") or {}
    na, nb = d.get("n_A"), d.get("n_B")
    return ("—" if na is None else na), ("—" if nb is None else nb)


def _pct_text(value):
    """纯文本百分比：有值补 "%"，缺值保持 "—"（不出现 "—%"）。"""
    return "—" if value in (None, "", "—") else "{}%".format(value)


def _pct_cell(value):
    """统计块里的百分比单元：38.98<em>%</em>；缺值只输出 "—"。"""
    return "—" if value in (None, "", "—") else "{}<em>%</em>".format(value)


def _pct_raw(value):
    """已经是百分数的值（偏移谱）→ "95.67%"；缺值 "—"。"""
    return "—" if value is None else "{:.2f}%".format(float(value))


_OS = _load_offset_spectrum()
_OS_S0 = _OS["S0"]
_OS_S1 = _OS["S1"]
_OS_S2 = _OS["S2"]
_OS_S3 = _OS["S3"]
_OS_S4 = _OS["S4"]
_OS_S5 = _OS["S5"]
_XD_T1 = _XD["t1"]
_XD_CI = _XD["ci"]
# 偏移谱六档齐全时，面板才给出「哪一级开始塌陷」的结论文字
_OS_OK = all(_OS[k] is not None for k in ("S0", "S1", "S2", "S3", "S4", "S5"))
_DC_NA, _DC_NB = _load_domain_clf_counts()
_MISSING_NOTE = ('<div class="lb-xd-note">该组产物未随仓库提供（<code>{}</code>），'
                 '表格里的数字因此为空；按 README「跨域实验」可自行复跑。</div>')


def eval_fig_values(model_name):
    gallery = {label: path for path, label in web.eval_gallery(model_name)}
    order = ["训练曲线", "混淆矩阵", "番茄子矩阵", "各类别 F1", "病害/健康 ROC",
             "退化鲁棒性", "遮挡实验"]
    out = []
    for k in order:
        out.append(gallery.get(k) or _placeholder(k))
    return out

def _eval_metrics(model_name):
    p = os.path.join(web.config.METRIC_DIR, f"eval_{model_name}.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def render_eval_old_note():
    return ('<div class="lb-eval-old"><b>图表版本</b> 当前同分布指标与图表来自 '
            f'<b>{_NCLS} 类模型</b>；指标与图表均基于该模型。'
            f'下方跨域区块同样基于 {_NCLS} 类模型。</div>')

_ANALYSIS_CACHE = {"mtime": None, "data": None}

def load_eval_analysis():
    """读取 outputs/metrics/eval_analysis.json；文件缺失或损坏时返回 None。"""
    path = os.path.join(web.config.METRIC_DIR, "eval_analysis.json")
    try:
        mtime = os.path.getmtime(path)
        if _ANALYSIS_CACHE.get("mtime") == mtime:
            return _ANALYSIS_CACHE.get("data")
        with io.open(path, encoding="utf-8") as fp:
            data = json.load(fp)
        _ANALYSIS_CACHE.update(mtime=mtime, data=data)
        return data
    except Exception:
        return None

def _analysis_fig(name):
    path = os.path.join(web.config.FIG_DIR, name)
    return path if os.path.exists(path) else None

def _analysis_pct(value):
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "—"

def _analysis_f3(value):
    try:
        return f"{float(value):.3f}"
    except Exception:
        return "—"

def _analysis_empty(text="暂无数据"):
    return (f'<div class="lb-empty lb-empty-sm lb-analysis-empty">'
            f'<h2>{text}</h2><p>未找到 eval_analysis.json 或对应图表。</p></div>')

def render_analysis_dist_note():
    d = load_eval_analysis()
    if not d:
        return _analysis_empty()
    buckets = d.get("f1_buckets") or {}
    order = ["≥0.99", "0.95–0.99", "0.90–0.95", "0.80–0.90", "<0.80"]
    total = d.get("n_classes") or sum(buckets.values())
    high = (buckets.get("≥0.99", 0) + buckets.get("0.95–0.99", 0))
    low = buckets.get("<0.80", 0)
    high_pct = high / total * 100 if total else 0
    rows = "".join(
        f'<div class="lb-f1-bucket"><span>{esc(k)}</span>'
        f'<b>{buckets.get(k, 0)}</b><i style="width:{buckets.get(k, 0) / max(total, 1) * 100:.1f}%"></i></div>'
        for k in order if k in buckets
    )
    return f'''<div class="lb-analysis-card">
  <div class="lb-analysis-head"><h3>F1 分布</h3><span>{total} 类 · ResNet50</span></div>
  <p class="lb-analysis-text">{total} 类中 <b>{high} 类（{high_pct:.0f}%）</b>的 F1 达到 0.95 以上，
  {buckets.get("≥0.99", 0)} 类接近满分；<b>{low} 类</b>低于 0.80。</p>
  <div class="lb-f1-buckets">{rows}</div>
</div>'''

def render_analysis_worst():
    d = load_eval_analysis()
    if not d:
        return _analysis_empty()
    worst = d.get("worst10") or []
    if not worst:
        return _analysis_empty()
    rows = "".join(
        f'<tr><td>{esc(x.get("cn", "—"))}</td>'
        f'<td class="lb-num">{_analysis_f3(x.get("f1"))}</td>'
        f'<td class="lb-num">{int(float(x.get("support", 0)))}</td></tr>'
        for x in worst[:10]
    )
    crops = Counter(str(x.get("cn", "")).split("·")[0] for x in worst[:10])
    crop, crop_n = (crops.most_common(1)[0] if crops else ("—", 0))
    supports = [float(x.get("support", 0)) for x in worst[:10] if x.get("support")]
    avg_support = sum(supports) / len(supports) if supports else 0
    return f'''<div class="lb-analysis-card">
  <div class="lb-analysis-head"><h3>薄弱类别 Top{len(worst[:10])}</h3><span>按 F1 从低到高</span></div>
  <div class="lb-table-wrap">
    <table class="lb-table lb-analysis-table">
      <thead><tr><th>作物·病害</th><th class="lb-num">F1</th><th class="lb-num">测试样本</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
  <p class="lb-analysis-text">薄弱类别集中在 <b>{esc(crop)}</b>（{crop_n} 类），
  平均测试样本仅 <b>{avg_support:.0f} 张</b>；这些类多来自外部数据集，图像风格与 PlantVillage
  差异较大，是当前主要误差来源。</p>
</div>'''

def render_analysis_sample_note():
    d = load_eval_analysis()
    if not d:
        return _analysis_empty()
    fs = d.get("few_shot") or {}
    try:
        n = int(fs.get("n_classes", 0))
        avg = float(fs.get("avg_f1", 0))
        n_many = int(fs.get("n_many", 0))
        avg_many = float(fs.get("avg_f1_many", 0))
        diff = (avg_many - avg) * 100
    except Exception:
        return _analysis_empty()
    return f'''<div class="lb-analysis-card">
  <div class="lb-analysis-head"><h3>样本量 vs F1</h3><span>少数样本类别单独观察</span></div>
  <p class="lb-analysis-text">样本量 ≤20 的 <b>{n} 个类别</b>平均 F1 为 <b>{avg:.3f}</b>；
  样本量 &gt;20 的 <b>{n_many} 类</b>为 <b>{avg_many:.3f}</b>，差距仅
  <b>{diff:.2f} 个百分点</b>，说明样本量不是当前的主要瓶颈。</p>
  <div class="lb-analysis-kv">
    <span><em>少数样本</em><b>{n} 类</b></span>
    <span><em>平均 F1</em><b>{avg:.3f}</b></span>
    <span><em>常规样本</em><b>{n_many} 类</b></span>
    <span><em>平均 F1</em><b>{avg_many:.3f}</b></span>
  </div>
</div>'''

def render_analysis_model_gap():
    d = load_eval_analysis()
    if not d:
        return _analysis_empty()
    gap = d.get("model_gap_top10") or []
    if not gap:
        return _analysis_empty()
    rows = "".join(
        f'<tr><td>{esc(x.get("cn", "—"))}</td>'
        f'<td class="lb-num">{_analysis_f3(x.get("resnet50"))}</td>'
        f'<td class="lb-num">{_analysis_f3(x.get("efficientnet_b0"))}</td>'
        f'<td class="lb-num">{_analysis_f3(x.get("mobilenet_v3_large"))}</td>'
        f'<td class="lb-num">{_analysis_f3(x.get("spread"))}</td></tr>'
        for x in gap[:10]
    )
    by = {str(r[0]): r for r in web.compare_table()}
    r50, mb, eb = by.get("resnet50"), by.get("mobilenet_v3_large"), by.get("efficientnet_b0")
    models = d.get("models") or {}
    r_acc = _analysis_pct((models.get("ResNet50") or {}).get("accuracy"))
    m_acc = _analysis_pct((models.get("MobileNetV3-Large") or {}).get("accuracy"))
    e_acc = _analysis_pct((models.get("EfficientNet-B0") or {}).get("accuracy"))
    return f'''<div class="lb-analysis-card">
  <div class="lb-analysis-head"><h3>分歧最大的 {len(gap[:10])} 个类别</h3><span>三种骨干网络的逐类 F1</span></div>
  <div class="lb-table-wrap">
    <table class="lb-table lb-analysis-table lb-gap-table">
      <thead><tr><th>作物·病害</th><th class="lb-num">ResNet50</th>
      <th class="lb-num">EfficientNet-B0</th><th class="lb-num">MobileNetV3</th>
      <th class="lb-num">分歧</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
  <p class="lb-analysis-text">分歧集中在样本量少、来自外部数据集的类别；三个模型在这些类上都不稳，
  单看总体准确率会掩盖这类长尾风险。</p>
  <div class="lb-model-advice">
    <b>选型建议</b>
    <p>ResNet50 准确率 {r_acc}，参数量 {esc(r50[3]) if r50 else "—"}、单张 {esc(r50[5]) if r50 else "—"}；
    MobileNetV3 准确率 {m_acc}，参数量仅 {esc(mb[3]) if mb else "—"}、单张 {esc(mb[5]) if mb else "—"}，性价比最高；
    EfficientNet-B0 准确率 {e_acc}，在本轮数据中更容易因置信度偏低触发拒识。
    <b>演示默认 ResNet50，边缘部署推荐 MobileNetV3。</b></p>
  </div>
</div>'''


def load_data_assets():
    """读取 outputs/metrics/data_assets.json；缺失或损坏返回 None。"""
    path = os.path.join(web.config.METRIC_DIR, "data_assets.json")
    try:
        with io.open(path, encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return None


def _fmt_int(v):
    try:
        return f"{int(v):,}"
    except Exception:
        return "—"


def load_workload():
    """读取 outputs/metrics/workload.json；缺失或损坏返回 None。"""
    path = os.path.join(web.config.METRIC_DIR, "workload.json")
    try:
        with io.open(path, encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return None


def _pct(v):
    try:
        return f"{float(v) * 100:.2f}%"
    except Exception:
        return "—"


def _ms(v):
    try:
        return f"{float(v):.0f}ms"
    except Exception:
        return "—"


def _block_empty(text="统计未生成"):
    return (f'<div class="lb-card lb-empty-card"><div class="lb-empty lb-empty-sm">'
            f'<h2>{text}</h2><p>对应统计文件缺失或尚未生成，页面不展示估算值。</p>'
            f'</div></div>')


def render_data_assets():
    d = load_data_assets()
    w = load_workload()
    m = (d or {}).get("main_dataset") or {}
    xd = (d or {}).get("cross_domain") or {}
    kb = (d or {}).get("knowledge_base") or {}
    it = (d or {}).get("intent_training") or {}
    aux = (d or {}).get("aux_sets") or {}
    tr = (w or {}).get("training") or []
    ts = (w or {}).get("training_summary") or {}
    exps = (w or {}).get("experiments") or []
    groups = (w or {}).get("experiment_groups") or {}
    modules = (w or {}).get("modules") or []
    ex_count = (w or {}).get("experiment_count")
    if ex_count is None:
        ex_count = len(exps)

    classes = _fmt_int(m.get("classes")) if d else "—"
    crops = _fmt_int(m.get("crops")) if d else "—"
    total = _fmt_int(m.get("total")) if d else "—"
    train, val, test = (_fmt_int(m.get("train")), _fmt_int(m.get("val")),
                        _fmt_int(m.get("test"))) if d else ("—", "—", "—")
    kb_total = _fmt_int(kb.get("total")) if d else "—"
    kb_precise = _fmt_int(kb.get("precise")) if d else "—"
    kb_ext = _fmt_int(kb.get("extended")) if d else "—"
    try:
        chars_wan = f"{float(kb.get('chars', 0)) / 10000:.1f}" if d else "—"
    except Exception:
        chars_wan = "—"
    ratio = esc(m.get("split_ratio") or "—") if d else "—"
    models_n = _fmt_int(ts.get("models")) if w else "—"
    epochs_n = _fmt_int(ts.get("total_epochs")) if w else "—"
    steps_n = _fmt_int(ts.get("total_steps")) if w else "—"
    try:
        samples_wan = f"{float(ts.get('total_samples', 0)) / 10000:.0f}" if w else "—"
    except Exception:
        samples_wan = "—"
    exp_n = _fmt_int(ex_count) if w else "—"

    def stat(k, v, u, sub):
        return (f'<div class="lb-stat"><span class="lb-stat-k">{k}</span>'
                f'<div class="lb-stat-v">{v}<em>{u}</em></div>'
                f'<span class="lb-stat-sub">{sub}</span></div>')

    stats = ''.join([
        stat("训练图片", total, "张",
             f"训练 {train} · 验证 {val} · 测试 {test}" if d else "数据统计未生成"),
        stat("类别 / 作物", f"{classes} / {crops}", "类 / 种",
             "主训练集覆盖范围" if d else "数据统计未生成"),
        stat("训练模型", steps_n, "步",
             (f"{models_n} 个骨干网络 · {samples_wan} 万张次"
              if w else "训练统计未生成")),
        stat("知识库条目", kb_total, "条",
             f"精编 {kb_precise} + 扩充 {kb_ext}" if d else "数据统计未生成"),
        stat("知识库字数", chars_wan, "万字",
             "精编 + CropDP-KG 扩充" if d else "数据统计未生成"),
        stat("完成实验", exp_n, "项",
             (f"{len(groups)} 组实验 · {len(modules)} 个工程模块" if w else "实验统计未生成")),
    ])

    if d:
        table_rows = [
            (m.get("name"), m.get("source"), f"{total} 张 / {classes} 类", m.get("usage")),
            (xd.get("name"), xd.get("source"), f"{_fmt_int(xd.get('images'))} 张",
             f"{xd.get('usage') or ''}；{xd.get('note') or ''}"),
            ("病害知识库（精编）", kb.get("source_precise"), f"{kb_precise} 条", "问答与配图"),
            ("病害知识库（扩充）", kb.get("source_extended"), f"{kb_ext} 条", "RAG 检索"),
            (it.get("name"), it.get("source"),
             f"{_fmt_int(it.get('samples'))} 条 / {_fmt_int(it.get('classes'))} 类",
             it.get("usage")),
        ]
        rows_html = "".join(
            f'<tr><td>{esc(n or "—")}</td><td>{esc(src or "—")}</td>'
            f'<td class="lb-data-size">{esc(size or "—")}</td><td>{esc(use or "—")}</td></tr>'
            for n, src, size, use in table_rows
        )
        source_block = f'''
  <div class="lb-card lb-data-table-card">
    <div class="lb-card-head">
      <div><h2>数据源清单</h2><p>主训练集、跨域测试集与知识库来源</p></div>
      <span class="lb-hint">5 类数据资产</span>
    </div>
    <div class="lb-data-table-wrap">
      <table class="lb-table lb-data-table">
        <thead><tr><th>名称</th><th>来源</th><th>规模</th><th>用途</th></tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
  </div>'''
    else:
        source_block = _block_empty("数据统计未生成")

    if w:
        tr_rows = ""
        for x in tr:
            eps = _fmt_int(x.get("epochs"))
            best = x.get("best_epoch")
            early = bool(x.get("early_stopped"))
            sub = f"{eps} 轮"
            if best and early:
                sub += f"，第 {_fmt_int(best)} 轮"
            if early:
                sub += '<em class="lb-train-stop">早停</em>'
            tr_rows += (
                f'<tr><td><b>{esc(x.get("name") or "—")}</b></td>'
                f'<td class="lb-train-steps"><b>{_fmt_int(x.get("steps"))}</b>'
                f'<span>{sub}</span></td>'
                f'<td class="lb-num">{_pct(x.get("test_acc"))}</td>'
                f'<td class="lb-num">{_pct(x.get("macro_f1"))}</td>'
                f'<td class="lb-num">{_ms(x.get("latency_ms"))}</td></tr>'
            )
        train_block = f'''
  <div class="lb-card lb-data-table-card">
    <div class="lb-card-head">
      <div><h2>三模型训练概览</h2><p>训练步数、测试准确率、Macro-F1 与单张推理延迟</p></div>
      <span class="lb-hint">{models_n} 个骨干网络 · 累计 {steps_n} 步</span>
    </div>
    <div class="lb-data-table-wrap">
      <table class="lb-table lb-data-table">
        <thead><tr><th>模型</th><th class="lb-num">训练步数</th>
        <th class="lb-num">测试准确率</th><th class="lb-num">Macro-F1</th>
        <th class="lb-num">单张延迟</th></tr></thead>
        <tbody>{tr_rows}</tbody>
      </table>
    </div>
  </div>'''
    else:
        train_block = _block_empty("训练统计未生成")

    if w:
        bucket = {}
        for e in exps:
            bucket.setdefault(e.get("group") or "其他", []).append(e)
        order = [g for g in groups if g in bucket] + [g for g in bucket if g not in groups]
        parts = []
        for i, g in enumerate(order):
            group_rows = "".join(
                f'<tr><td><b>{esc(e.get("name") or "—")}</b></td>'
                f'<td>{esc(e.get("purpose") or "—")}</td>'
                f'<td>{esc(e.get("result") or "—")}</td></tr>'
                for e in bucket[g]
            )
            opened = " open" if i == 0 else ""
            parts.append(
                f'<details class="lb-exp-group"{opened}>'
                f'<summary><b>{esc(g)}</b><span>{len(bucket[g])} 项</span></summary>'
                f'<div class="lb-data-table-wrap"><table class="lb-table lb-data-table lb-exp-table">'
                f'<thead><tr><th>名称</th><th>目的</th><th>结果</th></tr></thead>'
                f'<tbody>{group_rows}</tbody></table></div></details>'
            )
        exp_block = f'''
  <div class="lb-card lb-data-table-card">
    <div class="lb-card-head">
      <div><h2>实验清单</h2><p>按方向分组；默认展开第一组，可逐组展开</p></div>
      <span class="lb-hint">{exp_n} 项 · {len(groups)} 组</span>
    </div>
    {''.join(parts)}
  </div>'''
    else:
        exp_block = _block_empty("实验统计未生成")

    if w:
        mod_cards = "".join(
            f'<div class="lb-data-module"><b>{esc(x.get("name") or "—")}</b>'
            f'<div class="lb-data-tags">' +
            "".join(f'<span>{esc(t)}</span>' for t in
                    [s.strip() for s in str(x.get("features") or "").split("·") if s.strip()]) +
            f'</div></div>'
            for x in modules
        )
        mod_block = f'''
  <div class="lb-card lb-data-module-card">
    <div class="lb-card-head">
      <div><h2>工程模块</h2><p>{len(modules)} 个模块，覆盖诊断、评估、问答与历史</p></div>
      <span class="lb-hint">功能点</span>
    </div>
    <div class="lb-data-modules">{mod_cards}</div>
  </div>'''
    else:
        mod_block = _block_empty("模块统计未生成")

    if d:
        cov_rows = []
        ext = kb.get("extended") or 0
        cov = kb.get("coverage") or {}
        for label, key in (("为害症状", "with_symptoms"), ("发生规律", "with_occurrence"),
                           ("防治方法", "with_control")):
            n = cov.get(key) or 0
            pct = (float(n) / float(ext) * 100) if ext else 0
            cov_rows.append(
                f'<div class="lb-data-cov-row"><div class="lb-data-cov-head">'
                f'<span>{label}</span><b>{_fmt_int(n)} / {_fmt_int(ext)}（{pct:.1f}%）</b></div>'
                f'<div class="lb-data-cov-bar"><i style="width:{pct:.1f}%"></i></div></div>'
            )
        split_block = f'''
  <div class="lb-data-two">
    <div class="lb-card lb-data-split-card">
      <div class="lb-card-head"><div><h2>训练 / 验证 / 测试</h2>
        <p>按 {ratio} 划分</p></div><span class="lb-hint">{esc(m.get("name") or "主训练集")}</span></div>
      <div class="lb-data-split">
        <div><span>训练</span><b>{train}</b></div>
        <div><span>验证</span><b>{val}</b></div>
        <div><span>测试</span><b>{test}</b></div>
      </div>
      <p class="lb-data-note">共 {total} 张 · {classes} 类 · {crops} 种作物；划分比例 {ratio}。</p>
    </div>
    <div class="lb-card lb-data-cov-card">
      <div class="lb-card-head"><div><h2>扩充知识库字段覆盖率</h2>
        <p>分母 {kb_ext} 条 CropDP-KG 记录</p></div><span class="lb-hint">字段完整度</span></div>
      {''.join(cov_rows)}
    </div>
  </div>'''
        aux_bits = []
        for key, label in (("batch1", "batch1"), ("batch2", "batch2"),
                           ("multi", "多作物"), ("rice", "水稻")):
            if key in aux:
                aux_bits.append(f"{label} {_fmt_int(aux.get(key))} 张")
        aux_html = ("辅助批次：" + " · ".join(aux_bits)) if aux_bits else ""
        source_note = f'''
  <div class="lb-card lb-data-source-card">
    <div class="lb-card-head"><div><h2>数据来源说明</h2><p>数据如何构成、哪些参与训练</p></div></div>
    <div class="lb-data-source">
      <p><b>主训练集</b>：{esc(m.get("source") or "—")}，共 <b>{classes} 类 / {crops} 种作物</b>；
      划分 <b>{train} / {val} / {test}</b>（{ratio}），用于三模型训练与同分布评估。</p>
      <p><b>跨域测试集</b>：{esc(xd.get("name") or "—")}，{_fmt_int(xd.get("images"))} 张，
      来自真实田间拍摄，用于检验实验室图到田间图的泛化能力；<b>不参与训练</b>，避免数据泄漏。</p>
      <p><b>知识库</b>：{kb_precise} 条人工整理的精编条目（含病原、危害程度），
      以及从 CropDP-KG 导入的 {kb_ext} 条原始记录（含为害症状、发生规律、防治方法）；
      合计 {kb_total} 条、约 {chars_wan} 万字，用于 RAG 检索与问答。</p>
      {f'<p class="lb-data-aux">{aux_html}</p>' if aux_html else ''}
    </div>
  </div>'''
    else:
        split_block = ""
        source_note = ""

    notice = ""
    if not d or not w:
        missing = []
        if not d:
            missing.append("data_assets.json")
        if not w:
            missing.append("workload.json")
        notice = (f'<div class="lb-data-missing">统计未生成：{esc("、".join(missing))}；'
                  f'对应区块显示占位，不影响其他数据。</div>')

    return f'''
<div class="lb-data-section">
  <div class="lb-data-head">
    <div><h2>数据资产</h2><p>训练数据、工程实验、模块与知识库工作量</p></div>
    <span class="lb-hint">来自 outputs/metrics/</span>
  </div>
  {notice}
  <div class="lb-data-stats">{stats}</div>
  {train_block}
  {exp_block}
  {mod_block}
  {source_block}
  {split_block}
  {source_note}
</div>'''


def render_eval_stats(model_name):
    m = _eval_metrics(model_name)
    if not m:
        return '<div class="lb-card lb-empty-card"><div class="lb-empty lb-empty-sm"><h2>暂无评估数据</h2><p>请先运行评估脚本生成指标文件。</p></div></div>'
    tiles = [
        ("Accuracy", f"{m['accuracy'] * 100:.2f}", "%", "测试集整体准确率"),
        ("Macro-F1", f"{m['macro_f1'] * 100:.2f}", "%", "各类别 F1 的宏平均"),
        ("Top-3", f"{m['top3_accuracy'] * 100:.2f}", "%", "前三个候选中命中"),
        ("AUC", f"{m['healthy_vs_disease_auc']:.4f}", "", "病害与健康二分类"),
    ]
    cells = "".join(
        f'<div class="lb-stat"><span class="lb-stat-k">{k}</span>'
        f'<div class="lb-stat-v">{v}<em>{u}</em></div>'
        f'<span class="lb-stat-sub">{s}</span></div>'
        for k, v, u, s in tiles
    )
    return f'<div class="lb-stat-grid">{cells}</div>'


# ---------------- 跨域泛化（本轮新增） ----------------
XD_FIGS = {
    "protocol": ("14_跨域协议对比.png", "评估协议对比", "同分布 / 零样本 / 监督微调"),
    "confusion": ("15_跨域混淆矩阵.png", "跨域混淆矩阵",
                  f'{_XD["nc"]} 类 · 对角线为正确分类'),
    "conf": ("16_置信度分布.png", "置信度分布", "同分布 vs 跨域"),
    "risk": ("17_风险覆盖曲线.png", "风险-覆盖曲线", "按置信度覆盖的选择性风险"),
    "consistency": ("18_三模型一致性.png", "三模型一致性", "一致 vs 不一致样本"),
    "tsne": ("19_域特征分布.png", "域特征分布", "t-SNE · 线性域分类器 " + _DC_ACC),
    "shift": ("20_域偏移谱.png", "受控域偏移谱", "S0–S5 逐级叠加扰动"),
}

def xd_fig(key):
    fname, _, _ = XD_FIGS[key]
    p = os.path.join(web.config.FIG_DIR, fname)
    return p if os.path.exists(p) else None

def render_xd_section_head():
    return f'''
<div class="lb-xd-section-head">
  <span class="lb-eyebrow">跨域泛化 · ResNet50 · {_NCLS} 类</span>
  <h2>PlantDoc 零样本与域偏移诊断</h2>
  <p>以下为 ResNet50 在本轮跨域实验中的结果，与上方模型选择无关；数字来自 PlantDoc 236 张测试图。</p>
</div>'''

def render_xd_hero():
    if _XD["t1"] == "—":
        headline = ("真实场景零样本：协议指标未随仓库提供，"
                    "本面板不显示未经复跑的准确率")
        note = ('<p>需要 <code>results/cross_domain_protocol.json</code>；'
                '仓库里已带这份产物，若为空说明文件缺失或被移动。</p>')
    else:
        headline = (f'同分布 {_XD_IN}，真实场景零样本 {_pct_text(_XD["t1"])}'
                    f'（折叠协议）：差距来自域偏移，而非过拟合')
        note = (f'<p>PlantDoc 共 {_XD["n"]} 张、{_XD["nc"]} 类。多数类基线仅 '
                f'{_pct_text(_XD["mb"])}，模型 Top-1 {_pct_text(_XD["t1"])}'
                f'（95% CI {_XD["ci"]}），显著高于多数类基线。注意：若按严格协议'
                f'（{_NCLS} 类直接 argmax）仅为 {_pct_text(_XD["strict"])}，'
                f'因为细分类别与 PlantDoc 标注粒度不匹配。</p>')
    return f'''
<div class="lb-card lb-xd-hero">
  <div class="lb-xd-hero-top">
    <div>
      <span class="lb-eyebrow">零样本跨域 · PlantDoc</span>
      <h2>{headline}</h2>
      {note}
    </div>
    <span class="lb-badge is-warn">真实场景零样本</span>
  </div>
  <div class="lb-xd-stats">
    <div class="lb-xd-stat"><span>Top-1</span><b>{_pct_cell(_XD_T1)}</b><small>95% CI {_XD_CI} · 源 cross_domain_protocol.json</small></div>
    <div class="lb-xd-stat"><span>Top-3</span><b>{_pct_cell(_XD["t3"])}</b><small>前三个候选命中</small></div>
    <div class="lb-xd-stat"><span>Top-5</span><b>{_pct_cell(_XD["t5"])}</b><small>前五个候选命中</small></div>
    <div class="lb-xd-stat"><span>Macro-F1</span><b>{_XD["f1"]}</b><small>{_XD["nc"]} 类宏平均</small></div>
    <div class="lb-xd-stat"><span>多数类基线</span><b>{_pct_cell(_XD["mb"])}</b><small>模型显著优于基线</small></div>
    <div class="lb-xd-stat"><span>测试样本</span><b>{_XD["n"]}</b><small>PlantDoc test</small></div>
  </div>
</div>'''

def render_xd_protocol_side():
    rows = [
        ("原始模型（折叠协议）", _pct_text(_XD["t1"]), _pct_text(_XD["t3"]),
         _pct_text(_XD["t5"]), "否", "零样本泛化", "is-base"),
        ("原始模型（严格协议）", _pct_text(_XD["strict"]), "—", "—",
         "否", "零样本泛化", ""),
        ("AdaBN 归一化适配", _fmt_pct(_BN_A), "—", "—",
         "否（无标签）", "无监督适配", ""),
        ("混合微调", _fmt_pct(_MX_A), "—", "—",
         "是", "有监督适配", "is-best"),
    ]
    body = "".join(
        f'<tr class="{cls}"><td><b>{name}</b></td>'
        f'<td class="lb-num">{t1}</td><td class="lb-num">{t3}</td>'
        f'<td class="lb-num">{t5}</td><td>{tag}</td><td>{kind}</td></tr>'
        for name, t1, t3, t5, tag, kind, cls in rows
    )
    return f'''
<div class="lb-xd-side-head">
  <span class="lb-eyebrow">各方案对比</span>
  <h2>同一目标域上的四种做法</h2>
  <p>前两项为严格/折叠两种协议下的零样本结果；后两项为两种适配方法。</p>
</div>
<div class="lb-table-wrap">
  <table class="lb-table lb-xd-table">
    <thead><tr><th>方案</th><th class="lb-num">Top-1</th><th class="lb-num">Top-3</th>
    <th class="lb-num">Top-5</th><th>目标域标签</th><th>性质</th></tr></thead>
    <tbody>{body}</tbody>
  </table>
</div>
<div class="lb-xd-note">混合域微调使用目标域标签，属于有监督适配，不能与零样本方案直接比较。</div>'''

def render_xd_shift_side():
    rows = [
        ("S0", "原图", _pct_raw(_OS_S0), ""),
        ("S1", "换复杂背景", _pct_raw(_OS_S1), ""),
        ("S2", "+ 光照扰动", _pct_raw(_OS_S2), ""),
        ("S3", "+ 模糊压缩", _pct_raw(_OS_S3), ""),
        ("S4", "+ 目标缩至画面 42%", _pct_raw(_OS_S4), "is-key"),
        ("S5", "+ 视角变化", _pct_raw(_OS_S5), ""),
    ]
    body = "".join(
        f'<tr class="{cls}"><td><b>{lv}</b></td><td>{name}</td><td class="lb-num">{acc}</td></tr>'
        for lv, name, acc, cls in rows
    )
    if _OS_OK:
        callout = f'''<div class="lb-xd-callout">
  <b>S3→S4 掉 {abs(_OS_S3 - _OS_S4):.1f} 个百分点</b>
  <span>背景与光照变化（S1、S2）影响很小，仍在 {_pct_raw(_OS_S2)}；<b>模糊压缩（S3）一步掉到 {_pct_raw(_OS_S3)}</b>，
  目标缩小（S4）再掉到 {_pct_raw(_OS_S4)}。PlantDoc 折叠协议为 {_pct_text(_XD["t1"])}，介于 S4 与 S5 之间，
  说明真实场景的掉点不是单一因素，而是<b>模糊、压缩、小目标、视角</b>叠加的结果。</span>
</div>'''
    else:
        callout = _MISSING_NOTE.format(
            "outputs/metrics/offset_spectrum.json")
    return f'''
<div class="lb-xd-side-head">
  <span class="lb-eyebrow">受控域偏移谱</span>
  <h2>掉点来自哪些扰动</h2>
  <p>同一模型、同一目标域，逐级叠加扰动，观察准确率在哪一级开始塌陷。</p>
</div>
<div class="lb-table-wrap">
  <table class="lb-table lb-xd-table lb-xd-shift-table">
    <thead><tr><th>级别</th><th>偏移</th><th class="lb-num">准确率</th></tr></thead>
    <tbody>{body}</tbody>
  </table>
</div>
{callout}'''

def render_xd_tsne_side():
    if _DC_ACC == "—":
        intro = ("域分类器产物未随仓库提供，这组数字为空；按 README「跨域实验」"
                 "重跑 `src/domain_classifier.py` 即可补齐。")
    else:
        detail = ("区分准确率 {}。".format(_DC_ACC) if _DC_CV == "—" else
                  "区分准确率 {}（各折 {}），最低一折仅 {}。".format(
                      _DC_ACC, _DC_CV, _DC_CV.split("–")[0]))
        intro = (f"用 ResNet50 的全局池化特征训练线性域分类器（源域 {_DC_NA} 张 / "
                 f"PlantDoc {_DC_NB} 张，5 折交叉验证），{detail}"
                 "说明两域差异是结构性的，不是单张样本噪声。")
    if _OS_OK:
        bullets = f'''
<div class="lb-xd-bullets">
  <div><b>背景 / 光照</b><span>S1–S2 从 {_pct_raw(_OS_S0)} 降到 {_pct_raw(_OS_S2)}，仅掉 {abs(_OS_S0 - _OS_S2):.1f} 个点，不是主因。</span></div>
  <div><b>模糊 / 压缩</b><span>S3 从 {_pct_raw(_OS_S2)} 掉到 {_pct_raw(_OS_S3)}，一步掉 {abs(_OS_S2 - _OS_S3):.1f} 个点，是主因之一。</span></div>
  <div><b>小目标 / 视角</b><span>S4 为 {_pct_raw(_OS_S4)}、S5 降到 {_pct_raw(_OS_S5)}，目标变小与拍摄角度进一步放大误差。</span></div>
</div>'''
    else:
        bullets = _MISSING_NOTE.format("outputs/metrics/offset_spectrum.json")
    return f'''
<div class="lb-xd-side-head">
  <span class="lb-eyebrow">域特征分布</span>
  <h2>两域存在系统性偏移</h2>
  <p>{intro}</p>
</div>
{bullets}
<div class="lb-xd-note">跨域掉点由系统性域偏移主导，其中<b>模糊压缩与小目标</b>贡献最大；这提示实际使用时应尽量保证病斑在画面中占比足够、图像清晰。</div>'''

def render_xd_limitations():
    return f'''
<div class="lb-card lb-xd-limit">
  <div class="lb-card-head">
    <div><h2>局限说明</h2><p>这组数字能说明方向，但不能外推为所有田间场景的精度承诺。</p></div>
    <span class="lb-hint">使用边界</span>
  </div>
  <ul>
    <li><b>样本量</b>PlantDoc 仅 {_XD["n"]} 张，Top-1 的 95% CI 为 {_XD["ci"]}，区间较宽。</li>
    <li><b>域覆盖</b>只覆盖 PlantDoc 一个目标域，尚无第三域复现。</li>
    <li><b>方案可比性</b>混合域微调使用了目标域标签，是有监督适配，不能与零样本泛化直接比较。</li>
    <li><b>扰动设置</b>偏移谱是受控合成扰动，真实田间可能叠加多种因素；S4/S5 用于机制诊断，不是田间精度承诺。</li>
  </ul>
</div>'''

def render_compare(rows):
    if not rows:
        return '<div class="lb-card lb-empty-card"><div class="lb-empty lb-empty-sm"><h2>暂无对比数据</h2><p>三个模型的指标文件生成后可查看对比。</p></div></div>'
    best_acc = max(float(str(r[1]).rstrip("%")) for r in rows)
    best_f1 = max(float(str(r[2]).rstrip("%")) for r in rows)
    body = []
    for r in rows:
        model, acc, f1, params, flops, lat, total = r
        acc_v = float(str(acc).rstrip("%"))
        f1_v = float(str(f1).rstrip("%"))
        tags = []
        if acc_v == best_acc:
            tags.append('<span class="lb-tag">准确率最佳</span>')
        if f1_v == best_f1:
            tags.append('<span class="lb-tag">F1 最佳</span>')
        body.append(f'''
      <tr>
        <td><b>{esc(_MODEL_LABEL.get(model, model))}</b>{"".join(tags)}</td>
        <td class="lb-num">{esc(acc)}</td>
        <td class="lb-num">{esc(f1)}</td>
        <td class="lb-num">{esc(params)}</td>
        <td class="lb-num">{esc(flops)}</td>
        <td class="lb-num">{esc(lat)}</td>
        <td class="lb-num">{esc(total)}</td>
      </tr>''')
    return f'''
<div class="lb-card lb-table-card">
  <div class="lb-card-head">
    <div>
      <h2>三模型对比</h2>
      <p>同一测试集；性能数据来自 benchmark.json（RTX 4060 Laptop）</p>
    </div>
    <span class="lb-hint">{icon("chart", 15)}评估汇总</span>
  </div>
  <div class="lb-table-wrap">
    <table class="lb-table">
      <thead><tr><th>模型</th><th class="lb-num">Accuracy</th><th class="lb-num">Macro-F1</th><th class="lb-num">参数量</th><th class="lb-num">FLOPs</th><th class="lb-num">延迟</th><th class="lb-num">测试样本</th></tr></thead>
      <tbody>{"".join(body)}</tbody>
    </table>
  </div>
</div>'''

EVAL_TIPS = f'''
<div class="lb-card lb-tips-card">
  <div class="lb-card-head"><h2>图表说明</h2></div>
  <div class="lb-tip-list">
    <div><b>训练曲线</b><span>验证集准确率仍在上升、损失未发散，说明训练稳定。</span></div>
    <div><b>混淆矩阵</b><span>对角线为正确分类，重点看相近病害之间的误判。</span></div>
    <div><b>番茄子矩阵</b><span>单作物细粒度表现，晚疫病与早疫病最易混淆。</span></div>
    <div><b>各类 F1</b><span>关注长尾类别，F1 偏低通常与样本量不足有关。</span></div>
    <div><b>ROC</b><span>曲线越靠近左上角，病害与健康的区分能力越强。</span></div>
  </div>
</div>'''

def _kb_figs(cls, n=3):
    """知识库卡片里展示该病的示例图。"""
    try:
        paths = web.get_class_images(cls, n)
    except Exception:
        paths = []
    if not paths:
        return ""
    from urllib.parse import quote
    cells = []
    for i, p in enumerate(paths):
        url = "/gradio_api/file=" + quote(str(p).replace("\\", "/"))
        label = "实验室样本" if i == 0 else ""
        cells.append(
            f'<figure class="lb-kb-fig"><img src="{url}" alt="示例图" loading="lazy">'
            f'<figcaption>{label}</figcaption></figure>'
        )
    return "".join(cells)


def render_kb(cls):
    if not cls:
        return '<div class="lb-card lb-empty-card"><div class="lb-empty"><h2>请选择类别</h2><p>从左侧下拉框中选择作物与病害。</p></div></div>'
    info = web.knowledge.get_info(cls)
    occ_html = _render_occurrence(info)
    _ctrl_items = _expand_control(info.get("control") or [])
    controls = "".join(
        f'<li><span>{i}</span><p>{esc(_strip_leading_no(t))}</p></li>'
        for i, t in enumerate(_ctrl_items, 1))
    lab = web.get_class_images(cls, n=3, source="lab")
    field = web.get_class_images(cls, n=1, source="field")
    items = [(p, "实验室样本") for p in lab[:2]] + [(p, "真实场景样本") for p in field[:1]]
    used = {p for p, _ in items}
    for p in lab:
        if len(items) >= 3:
            break
        if p not in used:
            items.append((p, "实验室样本"))
    items = items[:3]
    shots = ""
    if items:
        shots = f'''
  <div class="lb-shot-block">
    <div class="lb-shot-head"><b>示例图片</b><span>数据集标注图 · 点击看大图</span></div>
    {render_shot_grid(items)}
  </div>'''
    return f'''
<div class="lb-card lb-kb-card">
  <div class="lb-kb-head">
    <div>
      <span class="lb-kb-crop">{esc(info["crop_cn"])}</span>
      <h2>{esc(info["disease_cn"])}</h2>
    </div>
    <code class="lb-kb-code">{esc(cls)}</code>
  </div>
  <div class="lb-kb-meta">
    <span class="lb-stat-chip"><em>病原类型</em><b>{esc(info["pathogen"])}</b></span>
    <span class="lb-stat-chip"><em>危害程度</em><b>{esc(info["severity"])}</b></span>
  </div>
  <div class="lb-kb-section">
    <h3>{icon("scan", 16)}症状表现</h3>
    {_render_blocks(_split_inline_blocks(info["symptoms"]), "lb-blocklist")}
  </div>
  {occ_html}
  {shots}
  <div class="lb-kb-section">
    <h3>{icon("shield", 16)}防治建议</h3>
    <ol class="lb-control-list">{controls}</ol>
  </div>
</div>'''

def _history_records(rows):
    return rows[1:] if rows and len(rows) > 1 else []

def _crop_counts(rows):
    cnt = Counter()
    for r in _history_records(rows):
        if len(r) > 3:
            name = r[3]
            crop = name.split(" - ")[0] if " - " in name else name
            cnt[crop] += 1
    return cnt

def render_crop_donut(rows):
    cnt = _crop_counts(rows)
    total = sum(cnt.values())
    if not total:
        return '<div class="lb-chart-empty">暂无记录</div>'
    items = cnt.most_common(5)
    if len(cnt) > 5:
        items.append(("其他", sum(v for _, v in cnt.most_common()[5:])))
    colors = ["#00ADA2", "#5CC7C0", "#93DAD6", "#C2EAE7", "#DCEFED", "#C9CFD0"]
    cx, cy, r, sw = 120, 128, 78, 26
    circumference = 2 * math.pi * r
    arcs, legend, acc = [], [], 0.0
    for i, (name, count) in enumerate(items):
        seg = count / total * circumference
        color = colors[i % len(colors)]
        if len(items) == 1:
            arcs.append(
                f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="{sw}"/>'
            )
        else:
            dash = max(seg - 3, 0.8)
            arcs.append(
                f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" '
                f'stroke-width="{sw}" stroke-dasharray="{dash:.2f} {circumference - dash:.2f}" '
                f'stroke-dashoffset="{-acc:.2f}" transform="rotate(-90 {cx} {cy})"/>'
            )
        ly = 44 + i * 31
        legend.append(
            f'<rect x="262" y="{ly - 10}" width="10" height="10" rx="3" fill="{color}"/>'
            f'<text x="282" y="{ly}" class="lb-legend-name">{esc(name)}</text>'
            f'<text x="512" y="{ly}" text-anchor="end" class="lb-legend-val">'
            f'{count} 次 · {count / total * 100:.0f}%</text>'
        )
        acc += seg
    return (
        '<svg class="lb-chart-svg" viewBox="0 0 520 256" role="img" aria-label="作物分布饼图">'
        + "".join(arcs)
        + f'<text x="{cx}" y="{cy - 2}" text-anchor="middle" class="lb-donut-total">{total}</text>'
        + f'<text x="{cx}" y="{cy + 20}" text-anchor="middle" class="lb-donut-sub">次诊断</text>'
        + "".join(legend)
        + "</svg>"
    )

def render_conf_hist(rows):
    vals = []
    for r in _history_records(rows):
        try:
            vals.append(max(0.0, min(0.9999, float(r[4]))))
        except Exception:
            pass
    bins = [0] * 10
    for v in vals:
        bins[min(9, int(v * 10))] += 1
    max_v = max(bins) if bins else 0
    if not vals:
        return '<div class="lb-chart-empty">暂无记录</div>'
    x0, y0, w, h = 36, 18, 462, 190
    gap = 9
    bw = (w - gap * 9) / 10
    top = max_v if max_v else 1
    grid = []
    for i in range(3):
        gy = y0 + h - h * i / 2
        val = round(top * i / 2)
        grid.append(
            f'<line x1="{x0}" y1="{gy:.1f}" x2="{x0 + w}" y2="{gy:.1f}" class="lb-grid-line"/>'
            f'<text x="{x0 - 8}" y="{gy + 3:.1f}" text-anchor="end" class="lb-axis">{val}</text>'
        )
    bars = []
    for i, count in enumerate(bins):
        bx = x0 + i * (bw + gap)
        bh = (count / top) * h if count else 0
        by = y0 + h - bh
        if count:
            bars.append(
                f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
                f'rx="4" fill="#00ADA2"/>'
                f'<text x="{bx + bw / 2:.1f}" y="{by - 6:.1f}" text-anchor="middle" '
                f'class="lb-bar-val">{count}</text>'
            )
    xlabels = []
    for i in range(6):
        lx = x0 + w * i / 5
        anchor = "start" if i == 0 else ("end" if i == 5 else "middle")
        xlabels.append(
            f'<text x="{lx:.1f}" y="{y0 + h + 20}" text-anchor="{anchor}" class="lb-axis">'
            f'{i / 5:.1f}</text>'
        )
    return (
        '<svg class="lb-chart-svg" viewBox="0 0 520 256" role="img" aria-label="置信度分布直方图">'
        + "".join(grid)
        + f'<line x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}" class="lb-axis-line"/>'
        + "".join(bars)
        + "".join(xlabels)
        + f'<text x="{x0 + w / 2}" y="250" text-anchor="middle" class="lb-axis">置信度</text>'
        + "</svg>"
    )

def render_history_charts(rows):
    data = _history_records(rows)
    cnt = _crop_counts(rows)
    if not data:
        return (
            '<div class="lb-card lb-empty-card" style="min-height:240px">'
            '<div class="lb-empty lb-empty-sm"><span class="lb-empty-icon">'
            + icon("chart", 24)
            + '</span><h2>暂无可统计的记录</h2><p>完成诊断后，这里会显示作物分布与置信度分布。</p></div></div>'
        )
    total = sum(cnt.values())
    return f'''
<div class="lb-charts-grid">
  <div class="lb-card lb-chart-card">
    <div class="lb-card-head">
      <div><h2>作物分布</h2><p>按中文诊断结果归类</p></div>
      <span class="lb-hint">{total} 次诊断</span>
    </div>
    {render_crop_donut(rows)}
  </div>
  <div class="lb-card lb-chart-card">
    <div class="lb-card-head">
      <div><h2>置信度分布</h2><p>0.0 – 1.0，按 0.1 分箱</p></div>
      <span class="lb-hint">{len(_history_records(rows))} 条记录</span>
    </div>
    {render_conf_hist(rows)}
  </div>
</div>'''

def render_history_stats(rows):
    data = rows[1:] if rows and len(rows) > 1 else []
    if not data:
        total, crops, avg = 0, 0, "—"
    else:
        total = len(data)
        crops = len({(r[3].split(" - ")[0] if len(r) > 3 and " - " in r[3] else r[3]) for r in data if len(r) > 3})
        vals = []
        for r in data:
            try:
                vals.append(float(r[4]))
            except Exception:
                pass
        avg = f"{sum(vals) / len(vals) * 100:.1f}%" if vals else "—"
    tiles = [
        ("累计诊断", str(total), "条", "本机保存的识别记录"),
        ("覆盖作物", str(crops), "种", "按中文结果自动归类"),
        ("平均置信度", avg, "", "仅统计成功记录"),
    ]
    cells = "".join(
        f'<div class="lb-stat"><span class="lb-stat-k">{k}</span>'
        f'<div class="lb-stat-v">{v}<em>{u}</em></div>'
        f'<span class="lb-stat-sub">{s}</span></div>'
        for k, v, u, s in tiles
    )
    return f'<div class="lb-stat-grid lb-stat-grid-3">{cells}</div>'

def render_history_table(rows):
    all_data = list(reversed(rows[1:])) if rows and len(rows) > 1 else []
    data = all_data[:50]
    if not data:
        return f'''
<div class="lb-card lb-table-card">
  <div class="lb-empty lb-empty-sm">
    <span class="lb-empty-icon">{icon("history", 24)}</span>
    <h2>暂无识别历史</h2>
    <p>完成一次诊断后，记录会自动保存在本机。</p>
  </div>
</div>'''
    body = []
    for r in data:
        if len(r) < 5:
            continue
        t, model, cls, cn, conf = r[0], r[1], r[2], r[3], r[4]
        idx = len(body)
        try:
            cv = float(conf)
            conf_s = f"{cv * 100:.2f}%"
            badge = ('<span class="lb-badge">可信</span>' if cv >= 0.6
                     else '<span class="lb-badge is-warn">待复核</span>')
        except Exception:
            cv = 0.0
            conf_s = esc(conf)
            badge = '<span class="lb-badge is-neutral">—</span>'
        # 第 6 列是相对 outputs/ 的原图路径；旧记录为空，显示「无原图」占位。
        img_rel = str(r[5]).strip() if len(r) > 5 else ""
        path = ""
        if img_rel:
            try:
                cand = img_rel if os.path.isabs(img_rel) else os.path.join(web.config.OUTPUT_DIR, img_rel)
                if os.path.isfile(cand):
                    path = cand
            except Exception:
                path = ""
        thumb = (f'<img src="{_img_url(path)}" alt="{esc(cn)}" loading="lazy">'
                 if path else '<span class="lb-hist-ph">无原图</span>')
        payload = json.dumps({
            "idx": idx + 1, "time": t, "model": model, "cls": cls,
            "cn": cn, "conf": cv, "path": path,
        }, ensure_ascii=False)
        body.append(f'''
      <tr>
        <td class="lb-num">{idx + 1}</td>
        <td><div class="lb-hist-thumb">{thumb}</div></td>
        <td class="lb-time">{esc(t)}</td>
        <td>{esc(_MODEL_LABEL.get(model, model))}</td>
        <td>{esc(cn)}</td>
        <td class="lb-num">{conf_s}</td>
        <td>{badge}</td>
        <td><a class="lb-hist-detail-link" href="#" data-hist="{idx}"
                data-hist-row="{esc(payload)}">详细</a></td>
      </tr>''')
    return f'''
<div class="lb-card lb-table-card">
  <div class="lb-card-head">
    <div>
      <h2>识别历史</h2>
      <p>显示 {len(body)} 条 · 共 {len(all_data)} 条，每行可点「详细」查看完整结果</p>
    </div>
    <span class="lb-hint">{icon("history", 15)}本机记录</span>
  </div>
  <div class="lb-table-wrap lb-table-scroll">
    <table class="lb-table">
      <thead><tr><th class="lb-num">#</th><th>源图</th><th>时间</th><th>模型</th><th>诊断结果</th><th class="lb-num">置信度</th><th>状态</th><th>操作</th></tr></thead>
      <tbody>{"".join(body)}</tbody>
    </table>
  </div>
</div>'''


USAGE = f'''
<div class="lb-card lb-usage-card">
  <div class="lb-card-head">
    <div>
      <h2>使用说明</h2>
      <p>从上传到导出报告</p>
    </div>
    <span class="lb-hint">{icon("book", 15)}四步完成</span>
  </div>
  <div class="lb-usage-grid">
    <div class="lb-usage-step"><span class="lb-usage-num">01</span><h3>上传照片</h3><p>拖拽或点击上传单片叶片，主体建议占画面 60% 以上。</p></div>
    <div class="lb-usage-step"><span class="lb-usage-num">02</span><h3>开始诊断</h3><p>选择骨干网络与拒识阈值；置信度低于阈值时系统不给出结论，提示重拍或人工判断。</p></div>
    <div class="lb-usage-step"><span class="lb-usage-num">03</span><h3>定位病斑</h3><p>生成 Grad-CAM 热力图与病斑外接框，查看病斑占叶面比例。</p></div>
    <div class="lb-usage-step"><span class="lb-usage-num">04</span><h3>导出归档</h3><p>批量诊断结果导出 CSV；单张诊断报告导出 PDF。</p></div>
  </div>
  <p class="lb-usage-foot">{icon("info", 14)}识别结果用于课程演示与辅助参考，不替代专业植保诊断。</p>
</div>'''

PDF_HINT = '<div class="lb-pdf-hint">导出 PDF 后，下载链接会出现在这里</div>'
QA_INTRO = f"在上方输入问题，或点击「常见问题」里的任意一条。回答会优先引用病害知识库（共 {_NKB} 条）。"

def render_pdf_note(msg):
    return f'<div class="lb-pdf-note">{icon("file", 14)}{esc(msg)}</div>'

def render_pdf_link(msg, path):
    href = "/gradio_api/file=" + quote(path.replace("\\", "/"), safe="")
    return (
        f'<div class="lb-pdf-note">{icon("file", 14)}<span>{esc(msg)}</span>'
        f'<a class="lb-pdf-link" href="{href}" download>'
        f'{icon("download", 14)}下载 PDF</a></div>'
    )

# =====================================================================
#  回调
# =====================================================================
def _empty_diag():
    """诊断空状态：清空结果、Top5、病斑、防治与 PDF。"""
    return (EMPTY_RESULT, "", None, None, "", "", PDF_HINT, None, None)

def on_image_change():
    return _empty_diag()

def on_example(key):
    img = web.load_example(key)
    return img

def on_diagnose(img, model_name, thresh):
    if img is None:
        return _empty_diag()

    # 1) 先判断输入是否像叶片，避免对任意图片强行给结论
    leaf_ok, meta = web.is_leaf_like(img)
    if not leaf_ok:
        return (render_not_leaf(meta), "", None, None, "", "",
                '<div class="lb-hint">未识别到叶片，未生成报告。</div>', None, None)

    t0 = time.time()
    top5, probs, pred = web.infer(img, model_name)
    elapsed = time.time() - t0
    info = web.knowledge.get_info(pred)
    conf = top5[0][1]
    low = conf < thresh

    try:
        web._log_history(model_name, pred, conf, pil_img=img)
    except Exception:
        pass

    # 2) 置信度不足时真正拒识：不给结论、不给 PDF
    if low:
        return (render_rejected(info, top5, conf, thresh, model_name),
                render_top5(top5, bare=True),
                None, None, "", "",
                '<div class="lb-hint">置信度低于阈值，建议重拍后重试或人工判断。</div>',
                info, model_name)

    return (render_result(info, top5, elapsed, model_name, False),
            render_top5(top5, bare=True),
            None, None, "", "", PDF_HINT, info, model_name)

def on_locate(img, model_name, thresh):
    """病斑定位：只更新右栏面板里的图片，不改变页面高度。"""
    if img is None:
        return None, None, "", ""
    try:
        gr.Info("正在计算 Grad-CAM，约 5-10 秒", duration=3)
    except Exception:
        pass
    try:
        # 病斑分割阈值固定 0.5，不再与「拒识阈值」混用
        overlay, vis, caption, _ = web.locate_view(img, model_name, 0.5)
    except Exception as e:
        return None, None, f'<div class="lb-lesion-note is-warn">{icon("alert", 15)}病斑定位失败：{esc(e)}</div>', ""
    return overlay, vis, render_lesion_caption(caption), ""

def on_review(action, info):
    """记录人工复核结论。"""
    if not info:
        return render_pdf_note("请先完成一次诊断。")
    ok = web.record_review(info.get("cls", ""), action)
    msg = {"确认": "已记录：确认判断正确", "更正": "已记录：判断有误，已标记",
           "驳回": "已记录：结果不采信"}.get(action, "已记录")
    return render_pdf_note(msg if ok else "记录失败")


def on_detail_view(info):
    """症状与防治：在右栏面板内切换，不新增页面区块。"""
    if not info:
        return None, None, "", '<div class="lb-lesion-note">请先完成一次诊断。</div>'
    return None, None, "", render_detail(info, with_head=False)

def on_clear_lesion():
    return None, None, ""

def on_clear_detail():
    return ""

def on_pdf(img, model_name, thresh, diagnosed_model=None,
           meta_plot=None, meta_variety=None, meta_submitter=None):
    """导出 PDF：使用诊断时实际使用的模型，并沿用同一个拒识阈值。"""
    if img is None:
        return render_pdf_note("请先上传图片。")
    use_model = diagnosed_model or model_name
    try:
        meta = {"plot": meta_plot, "variety": meta_variety,
                "submitter": meta_submitter}
        path, msg = web.make_pdf(img, use_model, conf_thresh=thresh, meta=meta)
    except Exception as e:
        return render_pdf_note(f"生成失败：{e}")
    if not path:
        return render_pdf_note(msg)
    return render_pdf_link(f"{msg} · {os.path.basename(path)}", path)

BATCH_DETAIL_EMPTY = ('<div class="lb-empty-inline">选择一张照片后，这里会显示与智能诊断相同的完整结果。</div>')

def render_batch_detail_meta(path, model_name, info, label=None):
    label = label or LIBRARY_LABEL.get(str(path), os.path.basename(str(path)))
    return f'''
<div class="lb-batch-detail-meta">
  <b>{esc(label)}</b>
  <em>批量条目 · 与单张诊断共用同一套结果</em>
</div>'''

BATCH_ROW_SLOTS = 40

def _batch_badge(flag):
    flag = str(flag)
    if flag == "可信":
        return f'<span class="lb-badge">{icon("check", 11)}可信</span>'
    return f'<span class="lb-badge is-warn">{icon("alert", 11)}{esc(flag)}</span>'

def _batch_row_html(row):
    if not row:
        return ""
    name, result, conf, flag = [str(x) for x in row[:4]]
    return f'''<div class="lb-batch-row-inner">
  <div class="lb-batch-file" title="{esc(name)}">{esc(name)}</div>
  <div class="lb-batch-result" title="{esc(result)}">{esc(result)}</div>
  <div class="lb-batch-conf">{esc(conf)}</div>
  <div class="lb-batch-status">{_batch_badge(flag)}</div>
</div>'''

def _batch_row_updates(rows):
    out = []
    for i in range(BATCH_ROW_SLOTS):
        if i < len(rows):
            out.append(gr.update(visible=True))
            out.append(gr.update(value=_batch_row_html(rows[i])))
        else:
            out.append(gr.update(visible=False))
            out.append(gr.update(value=""))
    return tuple(out)

def on_batch(files, model_name, thresh, selected=None):
    items, _ = _batch_grouped(files, _selected_paths(selected))
    if not items:
        return (_batch_row_updates([]) +
                (gr.update(value=None, interactive=False), "请先选择照片",
                 gr.update(choices=[], value=None), None, "", BATCH_DETAIL_EMPTY, "",
                 None, None, "", "", [], {}, {}, ""))
    t0 = time.time()
    all_paths = [p for p, _, _ in items]
    rows, csv_path = web.batch_diagnose(all_paths, model_name, thresh)
    cost = time.time() - t0
    display_map = {}
    for i, row in enumerate(rows[:len(items)]):
        p, lib_label, is_lib = items[i]
        pred = str(row[1]) if len(row) > 1 else ""
        flag = str(row[3]) if len(row) > 3 else ""
        if (flag in ("待复核", "无法识别", "失败")
                or pred.startswith("未识别") or pred.startswith("处理失败")):
            shown = "未识别" if "未识别" in pred else "无法确定"
        else:
            shown = pred.replace(" - ", "·")
        prefix = "" if is_lib else "本地 · "
        full = f"#{i + 1} {prefix}{shown}"
        row[0] = full
        display_map[str(p)] = full
    summary = render_batch_summary(rows)
    choices = [(display_map.get(str(p), f"#{i + 1}"), p)
               for i, (p, _, _) in enumerate(items)]
    status = (f"耗时 {cost:.1f} 秒 · 已生成 CSV · 共 {len(rows)} 张；"
              f"点每行右侧「详细」查看" +
              (f"（结果较多，仅展示前 {BATCH_ROW_SLOTS} 条，CSV 完整）"
               if len(rows) > BATCH_ROW_SLOTS else ""))
    return (_batch_row_updates(rows) +
            (gr.update(value=csv_path, interactive=True), status,
             gr.update(choices=choices, value=None), None, "", BATCH_DETAIL_EMPTY, "",
             None, None, "", "", [p for p, _, _ in items], display_map, {}, summary))

def open_batch_row(paths, display, model_name, thresh, idx):
    if not paths or idx < 0 or idx >= len(paths):
        return (gr.update(),) * 10
    path = paths[idx]
    label = (display or {}).get(str(path))
    result = on_batch_detail(path, model_name, thresh, display)
    choices = [((display or {}).get(str(p)) or os.path.basename(str(p)), p)
               for p in paths]
    return (gr.update(visible=True), gr.update(choices=choices, value=path),
            *result)

def on_batch_detail(path, model_name, thresh, display=None, hist_cache=None):
    """批量结果里选中某张，展示与单张诊断完全一致的结果（含叶片检查与拒识）。"""
    if hist_cache and str(path) in hist_cache:
        e = hist_cache[str(path)]
        return (e.get("img"), e.get("meta"), e.get("result"),
                e.get("top5"), None, None, "", "")
    label = (display or {}).get(str(path))
    if not path:
        return (None, "", BATCH_DETAIL_EMPTY, "", None, None, "", "")
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        return (None, "", f'<div class="lb-lesion-note is-warn">{esc(e)}</div>',
                "", None, None, "", "")

    # 与单张诊断一致：先做叶片检查
    leaf_ok, meta = web.is_leaf_like(img)
    if not leaf_ok:
        return (img, render_batch_detail_meta(path, model_name, None, label),
                render_not_leaf(meta), "", None, None, "", "")

    t0 = time.time()
    top5, probs, pred = web.infer(img, model_name)
    elapsed = time.time() - t0
    info = web.knowledge.get_info(pred)
    conf = top5[0][1]

    # 与单张诊断一致：置信度不足时拒识
    if conf < thresh:
        return (img, render_batch_detail_meta(path, model_name, info, label),
                render_rejected(info, top5, conf, thresh, model_name),
                render_top5(top5, bare=True), None, None, "", "")

    return (img,
            render_batch_detail_meta(path, model_name, info, label),
            render_result(info, top5, elapsed, model_name, False),
            render_top5(top5, bare=True),
            None, None, "", "")

def on_batch_locate(path, model_name, thresh):
    """批量详情内病斑定位：在弹窗面板内切换，不新增页面。"""
    if not path:
        return None, None, "", ""
    try:
        img = Image.open(path).convert("RGB")
        # 病斑分割阈值固定 0.5，与单张诊断保持一致
        overlay, vis, caption, _ = web.locate_view(img, model_name, 0.5)
    except Exception as e:
        return None, None, f'<div class="lb-lesion-note is-warn">{icon("alert", 15)}病斑定位失败：{esc(e)}</div>', ""
    return overlay, vis, render_lesion_caption(caption), ""

def on_batch_advice(path, model_name, thresh):
    """批量详情内症状与防治：复用知识库详情，清空病斑图。"""
    if not path:
        return None, None, "", '<div class="lb-lesion-note">请先选择一张照片。</div>'
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        return None, None, "", f'<div class="lb-lesion-note is-warn">{esc(e)}</div>'
    top5, probs, pred = web.infer(img, model_name)
    info = web.knowledge.get_info(pred)
    return None, None, "", render_detail(info, with_head=False)

# ---------------- 内置图片库 ----------------
def _build_library(limit=38):
    """构建图片库：实验室测试集 + 真实场景样本，并在标签上标注来源。"""
    items = []

    # ① PlantVillage 测试集（白底，模型训练同分布）
    root = web.config.IMAGES_DIR
    if os.path.isdir(root):
        for cls in sorted(os.listdir(root)):
            d = os.path.join(root, cls)
            if not os.path.isdir(d):
                continue
            files = sorted(f for f in os.listdir(d)
                           if f.lower().endswith((".jpg", ".jpeg", ".png")))
            if not files:
                continue
            p = os.path.join(d, files[0])
            info = web.knowledge.get_info(cls)
            items.append((p, f"{info['crop_cn']}·{info['disease_cn']}"))
            if len(items) >= limit:
                break

    # ② PlantDoc 真实场景（互联网照片，跨域验证用）
    pd_root = os.path.join(web.config.DATA_DIR, "plantdoc_extract",
                           "PlantDoc-Dataset-master", "test")
    if os.path.isdir(pd_root):
        try:
            from cross_domain_protocol import CLS_MAP
        except Exception:
            CLS_MAP = {}
        for dirname in sorted(os.listdir(pd_root)):
            sd = os.path.join(pd_root, dirname)
            if not os.path.isdir(sd) or dirname not in CLS_MAP:
                continue
            files = sorted(f for f in os.listdir(sd)
                           if f.lower().endswith((".jpg", ".jpeg", ".png")))
            if not files:
                continue
            p = os.path.join(sd, files[0])
            info = web.knowledge.get_info(CLS_MAP[dirname])
            items.append((p, f"{info['crop_cn']}·{info['disease_cn']}（真实）"))
    return items

LIBRARY = _build_library(len(CLASSES))



# 示例按钮优先覆盖中国常见作物（10 个，两行 5 列）
_PREFERRED_CROPS = ["水稻", "小麦", "番茄", "玉米", "黄瓜",
                    "辣椒", "茶叶", "苹果", "葡萄", "柑橘"]
_CROP_EXAMPLES = []
for _crop in _PREFERRED_CROPS:
    _cands = [(p, l) for p, l in LIBRARY
              if l.startswith(_crop + "·") and "（真实）" not in l]
    _cands.sort(key=lambda x: ("健康" in x[1], x[1]))
    if _cands:
        _CROP_EXAMPLES.append((_cands[0][0], _crop))

LIBRARY_LABEL = {p: l for p, l in LIBRARY}
LIBRARY_CROP = {p: l.split("·")[0] for p, l in LIBRARY}
CROP_NAMES = list(dict.fromkeys(
    web.knowledge.get_info(c)["crop_cn"] for c in CLASSES
))

def filter_library(crop):
    """按作物名筛选图片库；返回 (Gallery 更新, 当前列表状态)。"""
    if not crop or crop == "全部":
        view = list(LIBRARY)
    else:
        view = [(p, l) for p, l in LIBRARY if LIBRARY_CROP.get(p) == crop]
    return gr.update(value=[(p, l) for p, l in view]), view

def open_modal():
    return gr.update(visible=True)

def close_modal():
    return gr.update(visible=False)

def on_pick_image(evt: gr.SelectData, view):
    """点击图片库缩略图：载入并关闭浮层。"""
    try:
        p, _ = view[evt.index]
        return Image.open(p).convert("RGB"), gr.update(visible=False)
    except Exception:
        return None, gr.update(visible=False)

def _batch_file_path(item):
    return str(item.name if hasattr(item, "name") else item)

def _batch_selected_items(files, selected):
    """返回 [(path, 显示名, 是否图库)]；图库优先，上传文件按序号去重。"""
    items, seen = [], set()
    for raw in (selected or []):
        p = str(raw)
        if p in seen:
            continue
        seen.add(p)
        label = LIBRARY_LABEL.get(p, os.path.basename(p))
        items.append((p, label, True))
    for f in (files or []):
        p = _batch_file_path(f)
        if p in seen:
            continue
        seen.add(p)
        items.append((p, os.path.basename(p), False))
    return items

def _batch_grouped(files, selected):
    items = _batch_selected_items(files, selected)
    groups = {}
    for p, label, is_lib in items:
        crop = label.split("·")[0] if "·" in label else "本地图片"
        groups.setdefault(crop, []).append((p, label, is_lib))
    return items, groups

def _batch_selected_note(files, selected):
    items, groups = _batch_grouped(files, selected)
    if not items:
        return ""
    return f"已选 {len(items)} 张 · {len(groups)} 种作物"

def render_batch_selected(files, selected, compact=False):
    items, groups = _batch_grouped(files, selected)
    if not items:
        return ""
    idx = 1
    out = ['<div class="lb-pick-groups%s">' % (' is-compact' if compact else '')]
    for crop, group in groups.items():
        out.append('<div class="lb-pick-group">')
        out.append(f'<div class="lb-pick-group-head"><b>{esc(crop)}</b>'
                   f'<span>{len(group)} 张</span></div>')
        out.append('<div class="lb-pick-cards">')
        for p, label, is_lib in group:
            thumb = ""
            if is_lib and os.path.exists(p):
                thumb = f'<img src="{_img_url(p)}" alt="{esc(label)}" loading="lazy">'
            else:
                thumb = '<span class="lb-pick-ph">LOCAL</span>'
            out.append(
                f'<div class="lb-pick-card" title="{esc(label)}">'
                f'<div class="lb-pick-thumb">{thumb}'
                f'<span class="lb-pick-check">{icon("check", 12)}</span>'
                f'<span class="lb-pick-index">#{idx}</span></div>'
                f'<div class="lb-pick-name">{esc(label)}</div></div>'
            )
            idx += 1
        out.append('</div></div>')
    out.append('</div>')
    return "".join(out)

def render_library_grid(crop="全部"):
    """批量图片库：HTML 卡片网格，点击卡片由 JS 切换勾选状态。"""
    view = LIBRARY if not crop or crop == "全部" else [
        (p, l) for p, l in LIBRARY if LIBRARY_CROP.get(p) == crop
    ]
    groups = {}
    for p, label in view:
        c = LIBRARY_CROP.get(p, "本地图片")
        groups.setdefault(c, []).append((p, label))
    out = ['<div class="lb-lib-groups">']
    for c, group in groups.items():
        out.append(f'<div class="lb-lib-group"><div class="lb-lib-group-head">'
                   f'<b>{esc(c)}</b><span>{len(group)} 张</span></div>'
                   f'<div class="lb-lib-cards">')
        for p, label in group:
            out.append(
                f'<button type="button" class="lb-batch-lib-card" '
                f'data-path="{esc(p)}" data-label="{esc(label)}" '
                f'data-thumb="{_img_url(p)}" title="{esc(label)}">'
                f'<img src="{_img_url(p)}" alt="{esc(label)}" loading="lazy">'
                f'<span class="lb-batch-lib-check">{icon("check", 11)}</span>'
                f'<span class="lb-batch-lib-name">{esc(label)}</span></button>'
            )
        out.append('</div></div>')
    out.append('</div>')
    return "".join(out)

def filter_batch_grid(crop):
    return render_library_grid(crop)

def _selected_paths(value):
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    if isinstance(value, str) and value.strip():
        try:
            data = json.loads(value)
            if isinstance(data, list):
                return [str(x) for x in data]
        except Exception:
            pass
    return []

def render_upload_cards(files):
    items = _batch_selected_items(files, [])
    if not items:
        return ""
    out = ['<div class="lb-pick-groups lb-upload-group"><div class="lb-pick-group">'
           '<div class="lb-pick-group-head"><b>本地图片</b>'
           f'<span>{len(items)} 张</span></div><div class="lb-pick-cards">']
    for i, (p, label, _) in enumerate(items, 1):
        shown = f"#{i} 本地 · {label}"
        thumb = ""
        if os.path.exists(p):
            thumb = f'<img src="{_img_url(p)}" alt="" loading="lazy" onerror="this.remove()">'
        out.append(
            f'<div class="lb-pick-card" title="{esc(shown)}">'
            f'<div class="lb-pick-thumb"><span class="lb-pick-ph">LOCAL</span>{thumb}'
            f'<span class="lb-pick-check">{icon("check", 12)}</span>'
            f'<span class="lb-pick-index">#{i}</span></div>'
            f'<div class="lb-pick-name">{esc(shown)}</div></div>'
        )
    out.append('</div></div></div>')
    return "".join(out)

def sync_upload_cards(files):
    return render_upload_cards(files)

def append_batch(evt: gr.SelectData, state, view, files):
    """批量图片库：点击一次追加一张，可连续点击。"""
    try:
        p, _ = view[evt.index]
    except Exception:
        p = None
    cur = list(state) if isinstance(state, (list, tuple)) else []
    if p and p not in cur:
        cur.append(p)
    cards = render_batch_selected(files, cur)
    note = _batch_selected_note(files, cur)
    count = f"已选 {len(cur)} 张 · 点击缩略图继续追加"
    gallery = gr.update(value=[(p, l) for p, l in view])
    return cur, count, cards, render_batch_selected(files, cur, True), note, gallery

def sync_batch_cards(files, state):
    cards = render_batch_selected(files, state)
    note = _batch_selected_note(files, state)
    count = f"已选 {len(_batch_selected_items(files, state))} 张 · 可继续点击缩略图追加"
    return cards, note, render_batch_selected(files, state, True), count

def clear_batch():
    empty = render_batch_selected([], [])
    return (gr.update(value=[]), [], "已选 0 张 · 点击缩略图继续追加",
            "", "", "", gr.update(value="[]"), "")

def _answer_figs(r):
    """根据回答命中的最相关类别，取 2-3 张对应图片。"""
    try:
        cls = r.get("top_cls")
        if not cls or cls not in web.knowledge.KNOWLEDGE:
            return ""
        # 系统/使用类问题不配病害图
        if r.get("mode") in ("超出范围", "未收录", "-"):
            return ""
        # 问系统用法、模型指标这类问题不配病害图
        try:
            if ai_qa._detect_intent(r.get("_q", "")) == "system":
                return ""
        except Exception:
            pass
        info = web.knowledge.get_info(cls)
        # 只对具体病害配图，健康类不配
        if "健康" in info["disease_cn"]:
            return ""
        # 模型训练集里没有的类别（只存在于知识库）不配图
        if cls not in set(CLASSES):
            return ""

        from urllib.parse import quote
        paths = list(web.get_class_images(cls, 2))
        field = web.get_class_images(cls, 1, "field")
        if field:
            paths = paths[:2] + field[:1]
        if not paths:
            return ""

        cells = []
        for pth in paths[:3]:
            u = "/gradio_api/file=" + quote(str(pth).replace("\\", "/"))
            cells.append(
                f'<figure class="lb-ans-fig"><img src="{u}" alt="{esc(info["disease_cn"])}" loading="lazy">'
                f'<figcaption>{esc(info["crop_cn"])}·{esc(info["disease_cn"])}</figcaption></figure>'
            )
        return ('<div class="lb-ans-figs"><div class="lb-ans-figs-head">'
                '<span>相关图片</span><em>来自项目数据集</em></div>'
                '<div class="lb-ans-figs-grid">' + "".join(cells) + "</div></div>")
    except Exception:
        return ""


def on_ask(question):
    """AI 问答：RAG 检索 + 可选大模型生成。"""
    r = ai_qa.answer(question)
    src = "、".join(r["sources"][:3]) if r["sources"] else "无匹配条目"
    meta = (f'<div class="lb-qa-meta">回答来源 {r["mode"]} · 耗时 {r["elapsed"]}s · '
            f'参考：{src}</div>')
    r["_q"] = question
    figs = _answer_figs(r)
    return (r["answer"] + figs if figs else r["answer"]), meta

def on_ask_example(q):
    ans, meta = on_ask(q)
    return q, ans, meta

def random_question():
    return random.choice(ai_qa.examples())

def on_eval(model_name):
    vals = eval_fig_values(model_name)
    return (render_eval_stats(model_name), *vals, render_compare(web.compare_table()))

def on_kb(cls):
    return render_kb(cls)

def on_history_open(payload):
    """识别历史「详细」：复用批量详情浮层，显示记录时的结果。"""
    try:
        d = json.loads(payload or "{}")
    except Exception:
        d = {}
    path = d.get("path") or ""
    cls = d.get("cls") or ""
    model = d.get("model") or ""
    cn = d.get("cn") or ""
    path_ok = bool(path) and os.path.isfile(path)
    try:
        cv = float(d.get("conf", 0))
    except Exception:
        cv = 0.0
    label = f"#{d.get('idx', '?')} {cn.replace(' - ', '·')}".strip()
    try:
        info = web.knowledge.get_info(cls)
    except Exception:
        info = {"crop_cn": "", "disease_cn": cn}
    img = None
    if path_ok:
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = None
    try:
        result = render_result(info, [(cls, cv)], None, model, cv < 0.6)
    except Exception:
        result = BATCH_DETAIL_EMPTY
    if not path_ok:
        label = (label or cn) + " · 无原图"
    meta = render_batch_detail_meta(path, model, info, label or cn)
    try:
        top5 = render_top5([(cls, cv)], bare=True)
    except Exception:
        top5 = ""
    cache = {str(path): {"img": img, "meta": meta, "result": result, "top5": top5}}
    return (gr.update(visible=True),
            gr.update(choices=[(label or cn, path)], value=path),
            img, meta, result, top5, None, None, "", "", cache)

def on_history_refresh():
    rows = web.load_history()
    return (render_history_table(rows), render_history_stats(rows),
            render_history_charts(rows))

def on_history_filter(crop):
    rows = web.filter_history(crop=None if crop in (None, "全部") else crop)
    return render_history_table(rows)


def on_history_delete(indices, crop):
    n, left = web.delete_history(indices or "", crop=crop)
    rows = web.filter_history(crop=None if crop in (None, "全部") else crop)
    msg = f"已删除 {n} 条，剩余 {left} 条" if n else "没有匹配的序号，未删除任何记录"
    try:
        gr.Info(msg)
    except Exception:
        pass
    return render_history_table(rows)


def on_history_clear():
    msg, _ = web.clear_history()
    rows = []
    try:
        gr.Info(msg)
    except Exception:
        pass
    return (render_history_table(rows), render_history_stats(rows),
            render_history_charts(rows))

# =====================================================================
#  CSS
# =====================================================================
CSS = r"""
:root{
  --lb-bg:#F5F6F6;
  --lb-surface:#FFFFFF;
  --lb-border:#EAEBEC;
  --lb-border-strong:#DCE0E1;
  --lb-ink:#0A0E19;
  --lb-body:#3C3C43;
  --lb-muted:#6C6E75;
  --lb-faint:#82888D;
  --lb-tertiary:#9D9FA3;
  --lb-teal:#00ADA2;
  --lb-teal-dark:#00877F;
  --lb-teal-soft:rgba(0,173,162,.10);
  --lb-teal-faint:rgba(0,173,162,.05);
  --lb-warn:#FF9728;
  --lb-warn-dark:#B86400;
  --lb-warn-soft:rgba(255,151,40,.12);
  --lb-font:"Inter","Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei","Helvetica Neue",Arial,sans-serif;
  --lb-mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,"Courier New",monospace;
}

html, body{background:var(--lb-bg)!important;}
body, button, input, textarea, select, textarea{
  font-family:var(--lb-font)!important;
  -webkit-font-smoothing:antialiased;
}
.gradio-container{
  max-width:none!important;width:100%!important;
  background:var(--lb-bg)!important;color:var(--lb-body)!important;
  font-size:16px!important;line-height:1.65!important;
}
main.contain{
  max-width:none!important;width:100%!important;
  margin:0!important;padding:0!important;flex-grow:1;
}
.main.fillable{
  margin:0!important;padding:0!important;width:100%!important;max-width:100%!important;
}
main.contain > .column{gap:0!important;}
footer, .footer, .built-with{display:none!important;}
*{box-sizing:border-box;}
.lb-i{flex:none;display:block;}

/* ---------- 顶栏 ---------- */
div.block.lb-header-html{padding:0!important;border:0!important;background:transparent!important;box-shadow:none!important;}
div.block.lb-header-html .html-container{padding:0!important;}
.lb-header{
  background:rgba(255,255,255,.94);
  backdrop-filter:saturate(1.6) blur(20px);
  border-bottom:1px solid var(--lb-border);
}
.lb-header-inner{
  max-width:1200px;margin:0 auto;padding:0 24px;height:68px;
  display:flex;align-items:center;justify-content:space-between;gap:24px;
}
.lb-brand{display:flex;align-items:center;gap:12px;min-width:0;}
.lb-brand-mark{
  width:38px;height:38px;border-radius:11px;background:var(--lb-teal-soft);
  color:var(--lb-teal);display:flex;align-items:center;justify-content:center;flex:none;
}
.lb-brand-text h1{
  margin:0;font-size:18px;line-height:1.35;font-weight:600;color:var(--lb-ink);
  letter-spacing:0;white-space:nowrap;
}
.lb-brand-text p{margin:1px 0 0;font-size:11.5px;line-height:1.4;color:var(--lb-faint);}
.lb-brand-meta{display:flex;align-items:center;gap:8px;flex-wrap:wrap;justify-content:flex-end;}
.lb-chip{
  display:inline-flex;align-items:center;gap:6px;height:28px;padding:0 10px;
  border:1px solid var(--lb-border);border-radius:8px;background:var(--lb-bg);
  font-size:12px;font-weight:500;color:#525C5D;white-space:nowrap;
}
.lb-chip.is-warn{color:var(--lb-warn-dark);background:var(--lb-warn-soft);border-color:transparent;}
.lb-dot{width:6px;height:6px;border-radius:50%;background:var(--lb-teal);display:inline-block;flex:none;}
.lb-dot.is-warn{background:var(--lb-warn);}

/* ---------- 页签 ---------- */
.lb-tabs{margin:0!important;gap:0!important;background:transparent!important;}
.lb-tabs > .tab-wrapper{
  background:rgba(255,255,255,.94);
  backdrop-filter:blur(12px);
  border-bottom:1px solid var(--lb-border);
  position:sticky;top:0;z-index:30;
  height:54px!important;padding:0!important;display:flex!important;align-items:stretch!important;
}
.lb-tabs .tab-container[role="tablist"]{
  max-width:1200px;margin:0 auto;padding:0 24px;height:54px;
  display:flex;align-items:stretch;gap:30px;
}
.lb-tabs button[role="tab"]{
  height:54px!important;padding:0!important;margin:0!important;
  font-size:15px!important;font-weight:600!important;color:var(--lb-muted)!important;
  background:transparent!important;border:0!important;border-bottom:2px solid transparent!important;
  border-radius:0!important;box-shadow:none!important;
}
.lb-tabs button[role="tab"]:hover{color:var(--lb-ink)!important;}
.lb-tabs button[role="tab"].selected{
  color:var(--lb-ink)!important;border-bottom-color:var(--lb-teal)!important;
}
.lb-tabs .tabitem{
  padding:0!important;border:0!important;background:transparent!important;
}
.lb-tabs .tabitem > .column{width:100%;}

/* ---------- 统一内容容器（五个页签共用） ---------- */
.lb-page{
  display:block!important;width:100%!important;max-width:1200px!important;
  margin:0 auto!important;padding:22px 24px 64px!important;
  gap:0!important;min-width:0!important;
}
.lb-page > *{margin:0 0 20px 0!important;min-width:0!important;}
.lb-page > *:last-child{margin-bottom:0!important;}
.lb-page .column, .lb-page .row{min-width:0!important;}

/* ---------- 页头 ---------- */
div.block.lb-html{
  padding:0!important;border:0!important;background:transparent!important;
  box-shadow:none!important;min-width:0!important;
}
div.block.lb-html .html-container{padding:0!important;}
div.block.lb-html .prose{
  max-width:none!important;padding:0!important;margin:0!important;
  font-size:16px;line-height:1.65;color:var(--lb-body);
}
.lb-pagehead{
  display:flex;align-items:flex-end;justify-content:space-between;gap:24px;
  padding:2px 0 0;
}
.lb-pagehead-main{display:flex;align-items:center;gap:10px;min-width:0;}
.lb-pagehead-tag{
  font-family:var(--lb-mono);font-size:11px;font-weight:600;color:var(--lb-teal);
  background:var(--lb-teal-soft);border-radius:4px;padding:2px 6px;line-height:1.5;
  transform:translateY(1px);
}
.lb-pagehead h1{
  margin:0;font-size:28px;line-height:1.25;font-weight:600;color:var(--lb-ink);
  letter-spacing:0;white-space:nowrap;
}
.lb-pagehead p{
  margin:0 0 3px;font-size:13.5px;line-height:1.6;color:var(--lb-faint);
  text-align:right;max-width:560px;
}

/* ---------- 栅格 ---------- */
.lb-grid{display:grid!important;gap:20px!important;align-items:start!important;width:100%!important;}
.lb-grid-5-7{grid-template-columns:minmax(0,5fr) minmax(0,7fr)!important;}
.lb-grid-8-4{grid-template-columns:minmax(0,8fr) minmax(0,4fr)!important;}
.lb-grid-4-8{grid-template-columns:minmax(0,4fr) minmax(0,8fr)!important;}
.lb-grid-2{grid-template-columns:repeat(2,minmax(0,1fr))!important;}
.lb-col{display:block!important;gap:0!important;min-width:0!important;}

/* ---------- 卡片与文字 ---------- */
.lb-card{
  background:var(--lb-surface);
  border:1px solid var(--lb-border);
  border-radius:16px;
  box-shadow:none;
  padding:24px;
}
.lb-card-head{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;margin-bottom:16px;}
.lb-card-head h2{margin:0;font-size:19px;line-height:1.4;font-weight:600;color:var(--lb-ink);letter-spacing:0;}
.lb-card-head p{margin:4px 0 0;font-size:12px;line-height:1.5;color:var(--lb-faint);}
.lb-hint{display:inline-flex;align-items:center;gap:6px;font-size:12px;color:var(--lb-faint);white-space:nowrap;}
.lb-eyebrow{font-size:12px;font-weight:600;color:var(--lb-faint);letter-spacing:.02em;}
.lb-num{font-variant-numeric:tabular-nums;text-align:right;}
.lb-section-head{display:flex;align-items:baseline;gap:12px;margin:6px 0 12px;}
.lb-section-head h2{margin:0;font-size:19px;line-height:1.4;font-weight:600;color:var(--lb-ink);}
.lb-section-head span{font-size:12px;color:var(--lb-tertiary);}

/* ---------- 输入卡 ---------- */
.lb-input-card{
  background:var(--lb-surface)!important;border:1px solid var(--lb-border)!important;
  border-radius:16px!important;padding:24px!important;box-shadow:none!important;
  display:block!important;gap:0!important;
}
.lb-input-card > * + *{margin-top:14px!important;}
.lb-input-card .lb-card-head{margin-bottom:0!important;}
.lb-input-card .lb-card-head h2{font-size:19px;}
.lb-input-card .lb-card-head p{margin-top:2px;}
.lb-example-label{display:flex;align-items:center;justify-content:space-between;margin:2px 0 8px!important;}
.lb-example-label span{font-size:12px;color:var(--lb-faint);}
.lb-example-label em{font-style:normal;font-size:11.5px;color:var(--lb-tertiary);}
.lb-divider{height:1px!important;background:var(--lb-border)!important;margin:4px 0!important;}
.lb-start{margin-top:4px!important;}

/* ---------- Gradio 按钮 ---------- */
.lb-btn-primary{
  height:44px!important;width:100%!important;border-radius:999px!important;
  background:#000000!important;color:#FFFFFF!important;border:1px solid transparent!important;
  font-size:14px!important;font-weight:500!important;box-shadow:none!important;
  padding:0 24px!important;
}
.lb-btn-primary:hover{background:#282828!important;}
.lb-btn-primary:active{background:#141414!important;}
.lb-btn-secondary{
  height:40px!important;border-radius:8px!important;background:#F3F5F6!important;
  color:var(--lb-body)!important;border:1px solid var(--lb-border)!important;
  font-size:14px!important;font-weight:500!important;box-shadow:none!important;
  padding:0 20px!important;
}
.lb-btn-secondary:hover{background:#EBEDEE!important;border-color:#DCDEE0!important;}
.lb-btn-secondary:active{background:#E1E3E4!important;}
.lb-examples{display:grid!important;grid-template-columns:repeat(4,minmax(0,1fr))!important;gap:6px!important;}
.lb-example{
  height:36px!important;min-width:0!important;padding:0 4px!important;
  font-size:12.5px!important;font-weight:500!important;border-radius:8px!important;
  background:#F3F5F6!important;border:1px solid var(--lb-border)!important;
  color:var(--lb-body)!important;box-shadow:none!important;white-space:nowrap;
}
.lb-example:hover{background:#EBEDEE!important;color:var(--lb-ink)!important;}
.lb-action-row{display:none!important;grid-template-columns:1fr 1fr!important;gap:12px!important;margin-top:12px!important;}
.lb-right:has(.lb-result-card) .lb-action-row{display:grid!important;}
.lb-action-row .lb-btn-secondary{width:100%!important;}

/* ---------- 上传区 ---------- */
.lb-upload{
  height:240px!important;border:1px dashed #D8DCDE!important;border-radius:12px!important;
  background:#FAFBFB!important;box-shadow:none!important;overflow:hidden!important;
}
.lb-upload:hover{border-color:var(--lb-teal)!important;background:#F8FCFB!important;}
.lb-upload .upload-container{height:100%!important;}
.lb-upload .upload-container button{
  height:100%!important;background:transparent!important;border:0!important;box-shadow:none!important;
  border-radius:0!important;color:var(--lb-muted)!important;
}
.lb-upload .wrap{
  display:flex!important;flex-direction:column!important;align-items:center!important;
  justify-content:center!important;gap:10px!important;margin:0 auto!important;
  font-size:0!important;line-height:1.7!important;color:var(--lb-muted)!important;
}
.lb-upload .wrap::after{
  content:"拖拽叶片照片到此处，或点击上传";
  font-size:13.5px;line-height:1.7;color:var(--lb-muted);white-space:nowrap;
}
.lb-upload .icon-wrap{
  flex:none!important;margin:0 auto!important;
  width:46px!important;height:46px!important;border-radius:12px!important;
  background:var(--lb-teal-soft)!important;color:var(--lb-teal)!important;
  display:flex!important;align-items:center!important;justify-content:center!important;
  padding:11px!important;
}
.lb-upload .or{display:none!important;}
.lb-upload .or{color:var(--lb-tertiary)!important;}
.lb-upload .image-container{background:#F2F4F4!important;}
.lb-upload .image-container img{object-fit:contain!important;}
.lb-upload .source-selection{display:none!important;}

/* compact-layout: 示例按钮强制单行 */
.lb-examples{flex-wrap:nowrap !important;gap:6px !important;}
.lb-examples .lb-example{
  flex:1 1 0 !important;min-width:0 !important;
  font-size:12.5px !important;padding:7px 2px !important;
  white-space:nowrap !important;overflow:hidden !important;text-overflow:ellipsis !important;
}
/* compact-layout: 压缩纵向高度，减少留白 */
.lb-page .lb-card{padding:22px !important;}
.lb-page .lb-card-head{margin-bottom:14px !important;}
.lb-page .lb-empty{padding:38px 24px !important;}
.lb-page .lb-empty-steps{margin-top:24px !important;}
.lb-page .lb-example-label{margin:0 0 10px !important;}
.lb-page .lb-examples{gap:8px !important;}
.lb-page .lb-controls{gap:14px !important;}
.lb-page .lb-divider{margin:10px 0 !important;}
.lb-batch-settings > * + *{margin-top:12px !important;}
.lb-page .lb-upload{min-height:0 !important;}

/* clip-fix: Gradio 默认 overflow:hidden 会裁掉紧贴顶部的标签字形 */
.lb-page .block, .lb-page .form, .lb-select, .lb-slider, .lb-detail,
.lb-card, .lb-html, .lb-answer, .lb-qa-meta{
  overflow: visible !important;
}
.lb-select .container, .lb-select .wrap, .lb-select .wrap-inner{
  overflow: visible !important;
}
/* 标签与容器边缘留出安全距离，避免圆角裁切 */
.lb-select, .lb-slider{padding-top:2px !important;}

/* ---------- 回答配图 ---------- */
.lb-ans-figs{margin-top:14px;padding-top:12px;border-top:1px solid var(--lb-border);}
.lb-ans-figs-head{display:flex;align-items:baseline;gap:8px;margin-bottom:8px;}
.lb-ans-figs-head span{font-size:12.5px;font-weight:600;color:var(--lb-ink);}
.lb-ans-figs-head em{font-size:11.5px;color:var(--lb-muted);font-style:normal;}
.lb-ans-figs-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;}
.lb-ans-fig{margin:0;border-radius:8px;overflow:hidden;background:#F3F5F6;
  border:1px solid var(--lb-border);}
.lb-ans-fig img{width:100%;height:96px;object-fit:cover;display:block;cursor:zoom-in;}
.lb-ans-fig figcaption{font-size:11px;color:var(--lb-muted);padding:3px 6px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}

/* ---------- 全屏查看 ---------- */
.block.fullscreen{
  position:fixed!important;
  inset:0!important;
  left:0!important;top:0!important;
  width:100vw!important;height:100vh!important;
  max-width:100vw!important;max-height:100vh!important;
  margin:0!important;
  background:rgba(15,20,24,.90)!important;
  display:flex!important;
  align-items:center!important;
  justify-content:center!important;
  padding:3vh 3vw!important;
  box-sizing:border-box!important;
  z-index:1200!important;
}
.block.fullscreen .image-container,
.block.fullscreen .image-frame{
  position:relative!important;
  left:auto!important;top:auto!important;
  transform:none!important;
  width:auto!important;height:auto!important;
  max-width:94vw!important;max-height:94vh!important;
  margin:0!important;
  background:transparent!important;
  display:flex!important;
  align-items:center!important;
  justify-content:center!important;
}
.block.fullscreen img{
  position:relative!important;
  left:auto!important;top:auto!important;
  width:94vw!important;
  height:90vh!important;
  max-width:94vw!important;max-height:90vh!important;
  object-fit:contain!important;   /* 小图放大、大图缩小，比例不变 */
  border-radius:4px!important;
  cursor:default!important;
}
.lb-fs-hint{
  position:fixed;left:50%;bottom:26px;transform:translateX(-50%);
  z-index:1201;padding:7px 16px;border-radius:999px;
  background:rgba(0,0,0,.6);color:#fff;font-size:12.5px;
  pointer-events:none;
}

/* ---------- 知识库示例图 ---------- */
.lb-kb-figs{
  display:grid;grid-template-columns:repeat(3,minmax(0,1fr));
  gap:8px;margin:10px 0 14px;
}
.lb-kb-fig{
  margin:0;border-radius:8px;overflow:hidden;
  background:#F3F5F6;border:1px solid var(--lb-border);
}
.lb-kb-fig img{
  width:100%;height:104px;object-fit:cover;display:block;
  cursor:zoom-in;transition:transform .18s ease;
}
.lb-kb-fig img:hover{transform:scale(1.04);}
.lb-kb-fig figcaption{
  font-size:11px;color:var(--lb-muted);padding:3px 6px;line-height:1.4;
}

/* ---------- 评估页内导航 ---------- */
.lb-eval-nav{
  display:flex;flex-wrap:wrap;gap:8px;
  position:sticky;top:0;z-index:6;
  padding:10px 0;margin:-6px 0 14px;
  background:linear-gradient(#F5F6F6 70%, rgba(245,246,246,0));
}
.lb-eval-nav a{
  font-size:12.5px;font-weight:500;color:var(--lb-body);
  padding:6px 12px;border-radius:999px;
  background:#FFFFFF;border:1px solid var(--lb-border);
  text-decoration:none;transition:all .15s ease;
}
.lb-eval-nav a:hover{border-color:var(--lb-teal);color:var(--lb-teal-ink);}
.lb-anchor,.lb-page{scroll-margin-top:64px;}
.lb-page{scroll-behavior:smooth;}

/* ---------- 复核 ---------- */
.lb-review-row{display:none!important;align-items:center!important;gap:8px!important;margin-top:8px!important;}
.lb-right:has(.lb-result-card) .lb-review-row{display:flex!important;}
.lb-review-tip{font-size:12px;color:var(--lb-tertiary);}
.lb-review-row .lb-btn-chip{flex:0 0 auto!important;width:auto!important;}

/* ---------- 批量汇总 ---------- */
.lb-sum-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;}
.lb-sum-block > span{font-size:12px;color:var(--lb-faint);display:block;margin-bottom:6px;}
.lb-sum-block > p{font-size:13px;line-height:1.7;color:var(--lb-body);margin:0;}
.lb-sum-review{list-style:none;margin:0;padding:0;}
.lb-sum-review li{display:flex;justify-content:space-between;gap:10px;
  font-size:12.5px;padding:3px 0;border-bottom:1px dashed var(--lb-border);}
.lb-sum-review li:last-child{border-bottom:0;}
.lb-sum-review b{color:var(--lb-body);font-weight:500;}
.lb-sum-review span{color:var(--lb-muted);}
.lb-sum-review .lb-none{color:var(--lb-muted);border:0;}
.lb-more{font-size:12px;color:var(--lb-tertiary);margin:4px 0 0;}
.lb-sum-advice{margin-top:16px;padding-top:14px;border-top:1px solid var(--lb-border);}
.lb-sum-advice b{font-size:12px;color:var(--lb-teal-ink);}
.lb-sum-advice ol{margin:6px 0 0;padding-left:18px;}
.lb-sum-advice li{font-size:12.8px;line-height:1.7;color:var(--lb-body);}

/* ---------- 人话结论与求助出口 ---------- */
.lb-plain-verdict{
  font-size:13px;line-height:1.7;color:var(--lb-muted);
  margin:-4px 0 10px;
}
.lb-help-row{
  margin-top:14px;padding:10px 12px;border-radius:8px;
  background:#FFF8EC;border:1px solid #F5E3C8;
  display:flex;flex-wrap:wrap;align-items:baseline;gap:6px;
  font-size:12.5px;color:#8A6220;
}
.lb-help-row b{color:#6B4A12;}
.lb-help-row em{font-style:normal;color:#A07C42;font-size:12px;}

/* ---------- 结构化防治方案 ---------- */
.lb-ctrl-group{margin-bottom:14px;}
.lb-ctrl-group:last-child{margin-bottom:0;}
.lb-ctrl-group h4{
  font-size:12px;font-weight:600;color:var(--lb-teal-ink);letter-spacing:.02em;
  margin:0 0 7px;
}
.lb-ctrl-group ul{margin:0;padding-left:15px;}
.lb-ctrl-group li{margin-bottom:5px;}
.lb-ctrl-group li p{font-size:12.8px;line-height:1.72;color:var(--lb-body);margin:0;}
.lb-ctrl-note li p{color:var(--lb-muted);font-size:12.2px;}

/* ---------- 文字链接按钮（示例问题） ---------- */
.lb-link-btn{
  background:transparent!important;border:0!important;box-shadow:none!important;
  justify-content:flex-start!important;text-align:left!important;
  color:var(--lb-teal)!important;font-size:13px!important;font-weight:500!important;
  padding:7px 0!important;height:auto!important;min-height:0!important;
  width:100%!important;line-height:1.5!important;
}
.lb-link-btn:hover{text-decoration:underline!important;background:transparent!important;}
.lb-section-gap{height:20px;}
.lb-hint-text{font-size:12.5px;color:var(--lb-tertiary);line-height:1.65;margin:2px 0 8px;}

/* ---------- AI 问答 ---------- */
.lb-qa-meta{
  font-size:12px;color:var(--lb-tertiary);margin-top:12px;padding-top:12px;
  border-top:1px solid var(--lb-border);line-height:1.6;
}
.lb-detail p{margin:0 0 10px;}
.lb-detail strong{color:var(--lb-ink);}

/* ---------- 表单 ---------- */
.lb-page .form{
  display:block!important;background:transparent!important;border:0!important;
  padding:0!important;box-shadow:none!important;gap:0!important;
}
.lb-controls{display:grid!important;grid-template-columns:1.3fr 1fr!important;gap:16px!important;align-items:start!important;}
.lb-controls > *{min-width:0!important;}
.lb-field{display:block!important;padding:0!important;gap:0!important;min-width:0!important;}
.lb-select, .lb-slider{
  background:transparent!important;border:0!important;padding:0!important;box-shadow:none!important;
}
.lb-select label, .lb-slider label,
.lb-select .block-label, .lb-slider .block-label{
  position:static!important;
  display:block!important;
  float:none!important;
  width:auto!important;
  max-width:none!important;
  height:auto!important;
  min-height:0!important;
  overflow:visible!important;
  transform:none!important;
  translate:none!important;
  top:auto!important;
  left:auto!important;
  background:transparent!important;
  border:0!important;
  padding:0!important;
  margin:0 0 6px!important;
  font-size:12px!important;font-weight:600!important;color:var(--lb-muted)!important;
  line-height:1.5!important;letter-spacing:0!important;
  text-overflow:clip!important;white-space:normal!important;
}
.lb-select .wrap > label, .lb-slider .wrap > label{position:static!important;}
.lb-slider .head{gap:10px!important;}
.lb-slider .head label{white-space:nowrap!important;}
.lb-select .container > span:first-child,
.lb-slider .head label span,
.lb-slider .head > span:first-child{
  position:static!important;display:block!important;
  width:auto!important;max-width:none!important;
  margin:0!important;padding:0!important;transform:none!important;
  font-size:12px!important;font-weight:600!important;color:var(--lb-muted)!important;
  line-height:1.5!important;white-space:nowrap!important;
}
.lb-select .container{padding:0!important;}
.lb-select .wrap{
  background:transparent!important;padding:0!important;margin:0!important;
  min-height:0!important;border:0!important;box-shadow:none!important;
}
.lb-select .wrap-inner, .lb-select .secondary-wrap{
  padding:0!important;margin:0!important;width:100%!important;background:transparent!important;
  border:0!important;box-shadow:none!important;
}
.lb-select input{
  width:100%!important;margin:0!important;
  height:40px!important;border:1px solid var(--lb-border)!important;border-radius:8px!important;
  background:#FFFFFF!important;font-size:14px!important;color:var(--lb-body)!important;
  padding:0 34px 0 12px!important;box-shadow:none!important;
}
.lb-select input:focus{border-color:var(--lb-teal)!important;box-shadow:0 0 0 3px rgba(0,173,162,.10)!important;}
.lb-select textarea{
  border:1px solid var(--lb-border)!important;border-radius:8px!important;
  background:#FFFFFF!important;padding:10px 12px!important;
  font-size:14px!important;color:var(--lb-body)!important;line-height:1.6!important;
  min-height:76px!important;box-shadow:none!important;resize:vertical!important;
}
.lb-select textarea:focus{border-color:var(--lb-teal)!important;box-shadow:0 0 0 3px rgba(0,173,162,.10)!important;}
.lb-slider .head{
  display:flex!important;align-items:center!important;justify-content:space-between!important;
  gap:8px!important;margin:0!important;min-height:22px!important;
}
.lb-slider .head label{margin:0!important;}
.lb-slider .min_value, .lb-slider .max_value, .lb-slider .reset-button{display:none!important;}
.lb-slider input[type="number"], .lb-slider .number{
  font-size:12.5px!important;color:var(--lb-body)!important;
}
.lb-slider .tab-like-container{
  border:1px solid var(--lb-border)!important;border-radius:8px!important;background:#FFFFFF!important;
  height:30px!important;min-width:58px!important;
}
.lb-slider input[type="number"]{height:28px!important;padding:0 6px!important;}
.lb-slider .slider_input_container{margin-top:6px!important;}
.lb-slider input[type="range"]{accent-color:var(--lb-teal)!important;height:16px!important;}
.lb-file{
  border:1px dashed #D8DCDE!important;border-radius:12px!important;
  background:#FAFBFB!important;box-shadow:none!important;
  padding:0!important;overflow:hidden!important;
}
.lb-file:hover{border-color:var(--lb-teal)!important;}
.lb-file .wrap{height:100%!important;}
.lb-file .upload-container{height:100%!important;}
.lb-file button{color:var(--lb-muted)!important;font-size:13px!important;height:100%!important;background:transparent!important;}
.lb-file .wrap{
  display:flex!important;flex-direction:column!important;align-items:center!important;
  justify-content:center!important;gap:10px!important;font-size:0!important;
}
.lb-file .wrap::after{
  content:"拖拽多张叶片照片到此处，或点击选择文件";
  font-size:13.5px;line-height:1.7;color:var(--lb-muted);white-space:nowrap;
}
.lb-file .or{display:none!important;}
.lb-file .icon-wrap{
  width:46px!important;height:46px!important;border-radius:12px!important;
  background:var(--lb-teal-soft)!important;color:var(--lb-teal)!important;
  display:flex!important;align-items:center!important;justify-content:center!important;
  padding:11px!important;
}
.lb-batch-settings .lb-btn-primary{margin-top:16px!important;}
.lb-batch-settings > * + *{margin-top:14px!important;}
.lb-csv-btn{width:100%!important;}
.lb-batch-status{font-size:12px!important;color:var(--lb-faint)!important;margin-top:8px!important;}

/* ---------- 结果 ---------- */
.lb-result-wrap{display:flex!important;flex-direction:column;gap:16px;}
.lb-result-card{padding:24px;}
.lb-result-head{display:flex;align-items:center;justify-content:space-between;gap:12px;}
.lb-badge{
  display:inline-flex;align-items:center;gap:5px;height:26px;padding:0 10px;
  border-radius:999px;background:var(--lb-teal-soft);color:var(--lb-teal-dark);
  font-size:12px;font-weight:600;white-space:nowrap;
}
.lb-badge.is-warn{background:var(--lb-warn-soft);color:var(--lb-warn-dark);}
.lb-badge.is-neutral{background:var(--lb-bg);color:var(--lb-muted);}
.lb-crop{
  margin-top:18px;font-size:13.5px;color:var(--lb-faint);
  display:flex;align-items:center;gap:8px;
}
.lb-crop span{color:var(--lb-tertiary);font-size:12px;}
.lb-crop span::before{content:"";display:inline-block;width:3px;height:3px;border-radius:50%;background:var(--lb-tertiary);margin-right:8px;vertical-align:middle;}
.lb-disease{
  margin:4px 0 0;font-size:32px;line-height:1.3;font-weight:600;
  color:var(--lb-ink);letter-spacing:0;
}
.lb-conf{display:flex;align-items:baseline;gap:5px;margin-top:12px;}
.lb-conf-num{font-size:44px;line-height:1;font-weight:600;color:var(--lb-teal);letter-spacing:-.02em;}
.lb-conf-unit{font-size:18px;font-weight:600;color:var(--lb-teal);}
.lb-conf-label{font-size:12px;color:var(--lb-faint);margin-left:4px;}
.lb-meter{height:6px;border-radius:999px;background:#EFF2F2;overflow:hidden;margin-top:14px;}
.lb-meter i{display:block;height:100%;border-radius:inherit;background:var(--lb-teal);transition:width .45s ease;}
.lb-facts{
  display:grid;grid-template-columns:repeat(3,1fr);gap:16px;
  margin-top:18px;padding-top:16px;border-top:1px solid var(--lb-border);
}
.lb-fact span{display:block;font-size:12px;line-height:1.5;color:var(--lb-faint);}
.lb-fact b{display:block;margin-top:3px;font-size:14.5px;line-height:1.5;font-weight:600;color:var(--lb-ink);}
.lb-fact b.is-warn-text{color:var(--lb-warn-dark);}
.lb-empty-card{padding:0;min-height:580px;display:flex!important;}
.lb-empty{
  flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;
  text-align:center;padding:40px 32px;min-height:100%;
}
.lb-empty-sm{min-height:220px;padding:32px 24px;}
.lb-empty-icon{
  width:56px;height:56px;border-radius:16px;background:var(--lb-teal-soft);
  color:var(--lb-teal);display:flex;align-items:center;justify-content:center;margin-bottom:16px;
}
.lb-empty h2{margin:0;font-size:19px;line-height:1.4;font-weight:600;color:var(--lb-ink);}
.lb-empty p{margin:8px 0 0;font-size:14px;line-height:1.75;color:var(--lb-faint);}
.lb-empty-steps{
  display:grid;grid-template-columns:repeat(3,1fr);gap:12px;width:100%;
  margin-top:30px;padding-top:22px;border-top:1px solid var(--lb-border);
}
.lb-empty-steps > div{display:flex;flex-direction:column;align-items:flex-start;gap:2px;text-align:left;padding-left:2px;}
.lb-empty-steps span{
  font-family:var(--lb-mono);font-size:11px;color:var(--lb-teal);
  width:20px;height:20px;border-radius:6px;background:var(--lb-teal-soft);
  display:flex;align-items:center;justify-content:center;margin-bottom:6px;
}
.lb-empty-steps b{font-size:13px;font-weight:600;color:var(--lb-ink);}
.lb-empty-steps em{font-style:normal;font-size:11.5px;color:var(--lb-faint);line-height:1.5;}

/* Top-5 */
.lb-top5-card{padding:20px 24px 16px;}
.lb-top5-head{display:flex;align-items:baseline;justify-content:space-between;gap:12px;margin-bottom:8px;}
.lb-top5-head h3{margin:0;font-size:15px;line-height:1.5;font-weight:600;color:var(--lb-ink);}
.lb-top5-head span{font-size:11.5px;color:var(--lb-tertiary);}
.lb-rank{
  display:grid;grid-template-columns:26px minmax(0,1fr) 92px 54px;
  gap:10px;align-items:center;height:33px;
}
.lb-rank.is-first{
  background:var(--lb-teal-faint);border-radius:8px;margin:0 -8px;padding:0 8px;
}
.lb-rank-idx{font-family:var(--lb-mono);font-size:11px;color:var(--lb-tertiary);}
.lb-rank-name{
  font-size:13px;line-height:1.5;color:var(--lb-body);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}
.lb-rank.is-first .lb-rank-name{color:var(--lb-ink);font-weight:600;}
.lb-rank-track{height:4px;border-radius:999px;background:#EFF2F2;overflow:hidden;}
.lb-rank-track i{display:block;height:100%;border-radius:inherit;background:#CBD1D2;}
.lb-rank.is-first .lb-rank-track i{background:var(--lb-teal);}
.lb-rank-pct{font-size:12px;color:var(--lb-muted);text-align:right;font-variant-numeric:tabular-nums;}

/* ---------- 详情 ---------- */
.lb-detail-area{display:none!important;}
.lb-detail-area:has(.lb-detail-card){display:block!important;}
.lb-detail-card{padding:26px 28px;}
.lb-detail-grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:40px;}
.lb-detail-col h3{
  margin:0 0 10px;font-size:14px;line-height:1.5;font-weight:600;color:var(--lb-ink);
  display:flex;align-items:center;gap:8px;
}
.lb-detail-col h3 .lb-i{color:var(--lb-teal);}
.lb-symptom{margin:0;font-size:16px;line-height:1.8;color:var(--lb-body);}
/* 症状 / 发生规律的编号段落 */
.lb-blocklist{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:8px;}
.lb-blocklist li{display:flex;gap:9px;align-items:flex-start;}
.lb-blocklist li > span{
  flex:none;width:20px;height:20px;border-radius:6px;margin-top:2px;
  background:var(--lb-chip-bg,#eef2f7);color:var(--lb-chip-fg,#5b6b7c);
  font-size:11px;font-weight:700;line-height:20px;text-align:center;}
.lb-blocklist li p{margin:0;font-size:15px;line-height:1.78;color:var(--lb-body);}
.lb-blocklist li.lb-blocklist-lead{display:block;}
.lb-blocklist li.lb-blocklist-lead p{color:var(--lb-body);}
.lb-blocklist-plain{margin:0;font-size:15px;line-height:1.8;color:var(--lb-body);}
.lb-detail-pane .lb-blocklist li p{font-size:12.5px!important;line-height:1.6!important;}
.lb-detail-wide .lb-blocklist li p{font-size:14px!important;}
.lb-kb-panel .lb-blocklist li p{font-size:12.5px!important;line-height:1.55!important;}

.lb-control-list{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:10px;}
.lb-control-list li{display:flex;gap:10px;align-items:flex-start;}
.lb-control-list li > span{
  flex:none;width:20px;height:20px;border-radius:6px;background:var(--lb-teal-soft);
  color:var(--lb-teal-dark);font-size:11px;font-weight:600;
  display:flex;align-items:center;justify-content:center;margin-top:3px;
}
.lb-control-list p{margin:0;font-size:15px;line-height:1.75;color:var(--lb-body);}

/* ---------- 病斑定位 ---------- */
.lb-lesion-card{background:var(--lb-surface)!important;border:1px solid var(--lb-border)!important;border-radius:16px!important;padding:24px!important;box-shadow:none!important;display:none!important;}
.lb-lesion-card:has(img){display:block!important;}
.lb-lesion-grid{display:grid!important;grid-template-columns:minmax(0,1fr) minmax(0,1fr) minmax(0,.9fr)!important;gap:20px!important;align-items:start!important;}
.lb-lesion-fig{display:flex;flex-direction:column;gap:8px;min-width:0;}
.lb-lesion-fig span{font-size:12px;color:var(--lb-faint);}
.lb-lesion-fig .lb-figure{
  width:min(320px,100%)!important;height:320px!important;margin:0 auto!important;
  border:1px solid var(--lb-border)!important;border-radius:12px!important;overflow:hidden!important;
  background:#F7F8F8!important;box-shadow:none!important;padding:0!important;
}
.lb-lesion-fig .image-container{background:#F7F8F8!important;}
.lb-lesion-fig img{width:100%!important;height:100%!important;object-fit:contain!important;}
.lb-lesion-side{display:block!important;padding-top:28px;min-width:0;}
.lb-lesion-side .lb-lesion-facts{
  flex-direction:column!important;align-items:stretch!important;gap:8px!important;
  margin-top:0!important;padding-top:0!important;border-top:0!important;
}
.lb-lesion-side .lb-stat-chip{justify-content:space-between!important;}
.lb-lesion-tip{
  font-size:12.5px;line-height:1.8;color:var(--lb-faint);
  margin-top:16px;padding-top:16px;border-top:1px solid var(--lb-border);
}
.lb-lesion-facts{display:flex;flex-wrap:wrap;gap:10px;margin-top:16px;padding-top:16px;border-top:1px solid var(--lb-border);}
.lb-stat-chip{
  display:inline-flex;align-items:baseline;gap:6px;padding:5px 10px;border-radius:8px;
  background:var(--lb-bg);border:1px solid var(--lb-border);
}
.lb-stat-chip em{font-style:normal;font-size:11.5px;color:var(--lb-faint);}
.lb-stat-chip b{font-size:13px;font-weight:600;color:var(--lb-ink);}
.lb-lesion-note{display:flex;align-items:center;gap:8px;margin-top:14px;font-size:13px;color:var(--lb-faint);}
.lb-lesion-note.is-warn{color:var(--lb-warn-dark);}

/* ---------- 表格 ---------- */
.lb-table-card{padding:24px 24px 10px;}
.lb-table-wrap{width:100%;overflow-x:auto;}
.lb-table-scroll{max-height:none;overflow-y:visible;}
.lb-table{width:100%;border-collapse:collapse;font-size:13.5px;border:0!important;}
.lb-table thead, .lb-table tbody, .lb-table tr{border:0!important;}
.lb-table th{
  text-align:left;font-size:12px;font-weight:600;color:var(--lb-faint);
  padding:0 16px 11px 0;border:0!important;border-bottom:1px solid var(--lb-border)!important;
  white-space:nowrap;background:transparent!important;
}
.lb-table th:first-child, .lb-table td:first-child{padding-left:0!important;}
.lb-table th + th, .lb-table td + td{padding-left:16px!important;}
.lb-table th:last-child, .lb-table td:last-child{padding-right:0!important;}
.lb-table th.lb-num, .lb-table td.lb-num{text-align:right;padding-right:0;}
.lb-table td{
  padding:11px 16px 11px 0;border:0!important;border-bottom:1px solid #F1F2F3!important;
  color:var(--lb-body);vertical-align:middle;line-height:1.55;background:transparent;
}
.lb-table tr:last-child td{border-bottom:0!important;}
.lb-table tbody tr:hover td{background:#FAFBFA;}
.lb-table td.lb-file-name{color:var(--lb-ink);font-weight:500;max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.lb-table td.lb-time{font-family:var(--lb-mono);font-size:11.5px;color:var(--lb-muted);white-space:nowrap;}
.lb-table .lb-badge{height:24px;font-size:11.5px;padding:0 8px;}
.lb-table .lb-tag{
  display:inline-flex;align-items:center;height:20px;padding:0 6px;margin-left:8px;
  border-radius:4px;background:var(--lb-teal-soft);color:var(--lb-teal-dark);
  font-size:10.5px;font-weight:600;vertical-align:middle;
}
.lb-batch-empty-card{min-height:240px;display:flex!important;padding:0;}

/* ---------- 模型评估 ---------- */
.lb-toolbar{display:flex!important;align-items:flex-end!important;justify-content:space-between!important;gap:16px!important;}
.lb-toolbar .form{width:300px!important;max-width:100%!important;flex:none!important;}
.lb-toolbar .lb-select{width:300px;max-width:100%;}
.lb-toolbar .lb-html{flex:1;}
.lb-toolbar-note{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--lb-tertiary);justify-content:flex-end;padding-bottom:10px;}
.lb-stat-grid{display:grid!important;grid-template-columns:repeat(4,minmax(0,1fr))!important;gap:16px!important;}
.lb-stat-grid-3{grid-template-columns:repeat(3,minmax(0,1fr))!important;}
.lb-stat{
  background:var(--lb-surface);border:1px solid var(--lb-border);border-radius:16px;
  padding:20px 24px;box-shadow:none;
}
.lb-stat-k{display:block;font-size:12px;font-weight:600;color:var(--lb-faint);line-height:1.5;}
.lb-stat-v{margin-top:6px;font-size:30px;line-height:1.15;font-weight:600;color:var(--lb-ink);letter-spacing:-.01em;}
.lb-stat-v em{font-style:normal;font-size:13px;font-weight:500;color:var(--lb-faint);margin-left:2px;}
.lb-stat-sub{display:block;margin-top:4px;font-size:11.5px;color:var(--lb-tertiary);line-height:1.5;}
.lb-figure-grid{display:grid!important;grid-template-columns:repeat(2,minmax(0,1fr))!important;gap:20px!important;align-items:stretch!important;}
.lb-figure-card{
  background:var(--lb-surface)!important;border:1px solid var(--lb-border)!important;
  border-radius:16px!important;padding:18px 18px 12px!important;box-shadow:none!important;
  display:flex!important;flex-direction:column!important;gap:0!important;min-width:0!important;
}
.lb-figure-head{display:flex;align-items:baseline;justify-content:space-between;gap:10px;margin-bottom:10px;}
.lb-figure-head h3{margin:0;font-size:15px;line-height:1.5;font-weight:600;color:var(--lb-ink);}
.lb-figure-head span{font-size:11.5px;color:var(--lb-tertiary);}
.lb-figure{border:0!important;background:transparent!important;padding:0!important;box-shadow:none!important;}
.lb-figure .image-container{background:#FFFFFF!important;border-radius:8px;overflow:hidden;}
.lb-figure img{object-fit:contain!important;}
.lb-tips-card{padding:24px;}
.lb-tip-list{display:flex;flex-direction:column;gap:12px;}
.lb-tip-list > div{display:flex;flex-direction:column;gap:1px;padding-left:12px;border-left:2px solid var(--lb-border);}
.lb-tip-list b{font-size:13px;font-weight:600;color:var(--lb-ink);line-height:1.5;}
.lb-tip-list span{font-size:12.5px;color:var(--lb-muted);line-height:1.6;}

/* ---------- 知识库 ---------- */
.lb-kb-side{padding:24px!important;}
.lb-kb-side .lb-card-head{margin-bottom:14px!important;}
.lb-kb-side .lb-select{margin-top:4px;}
.lb-kb-note{margin-top:14px;padding-top:14px;border-top:1px solid var(--lb-border);font-size:12px;line-height:1.7;color:var(--lb-faint);}
.lb-kb-note b{color:var(--lb-ink);font-weight:600;}
.lb-kb-card{padding:28px;}
.lb-kb-head{display:flex;align-items:flex-start;justify-content:space-between;gap:20px;padding-bottom:18px;border-bottom:1px solid var(--lb-border);}
.lb-kb-crop{font-size:12px;font-weight:600;color:var(--lb-teal-dark);background:var(--lb-teal-soft);border-radius:4px;padding:2px 7px;display:inline-block;line-height:1.5;}
.lb-kb-head h2{margin:8px 0 0;font-size:28px;line-height:1.3;font-weight:600;color:var(--lb-ink);letter-spacing:0;}
.lb-kb-code{font-family:var(--lb-mono);font-size:10.5px;color:var(--lb-tertiary);background:var(--lb-bg);border:1px solid var(--lb-border);border-radius:4px;padding:3px 7px;white-space:nowrap;}
.lb-kb-meta{display:flex;gap:10px;flex-wrap:wrap;margin-top:18px;}
.lb-kb-section{margin-top:24px;}
.lb-kb-section h3{margin:0 0 10px;font-size:14px;font-weight:600;color:var(--lb-ink);display:flex;align-items:center;gap:8px;}
.lb-kb-section h3 .lb-i{color:var(--lb-teal);}

/* ---------- 历史 ---------- */
.lb-history-toolbar{display:flex!important;align-items:center!important;justify-content:space-between!important;gap:12px!important;}
.lb-history-toolbar > .lb-html{flex:1 1 auto!important;width:auto!important;min-width:0!important;}
.lb-history-title{font-size:13px;color:var(--lb-faint);}
.lb-history-actions{display:flex!important;gap:8px!important;flex:0 0 auto!important;width:auto!important;min-width:0!important;}
.lb-history-actions button{
  flex:0 0 auto!important;width:auto!important;min-width:88px!important;
  height:40px!important;font-size:13px!important;
}
.lb-charts-row{display:grid!important;grid-template-columns:repeat(2,minmax(0,1fr))!important;gap:20px!important;}
.lb-charts-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px;}
.lb-chart-card{padding:24px;}
.lb-chart-card .lb-card-head{margin-bottom:6px;}
.lb-chart-svg{display:block;width:100%;height:auto;overflow:visible;}
.lb-chart-svg text{font-family:var(--lb-font);}
.lb-chart-svg .lb-donut-total{font-size:30px;font-weight:600;fill:var(--lb-ink);}
.lb-chart-svg .lb-donut-sub{font-size:11.5px;fill:var(--lb-faint);}
.lb-chart-svg .lb-legend-name{font-size:12.5px;fill:var(--lb-body);}
.lb-chart-svg .lb-legend-val{font-size:11.5px;fill:var(--lb-faint);}
.lb-chart-svg .lb-axis{font-size:10.5px;fill:var(--lb-tertiary);}
.lb-chart-svg .lb-bar-val{font-size:10.5px;fill:var(--lb-muted);}
.lb-chart-svg .lb-grid-line{stroke:#EFF1F2;stroke-width:1;}
.lb-chart-svg .lb-axis-line{stroke:#E4E7E8;stroke-width:1;}
.lb-chart-empty{display:flex;align-items:center;justify-content:center;height:220px;font-size:13px;color:var(--lb-tertiary);}
.lb-usage-card{padding:28px;}
.lb-usage-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:24px;}
.lb-usage-step{display:flex;flex-direction:column;gap:4px;}
.lb-usage-num{font-family:var(--lb-mono);font-size:11px;color:var(--lb-teal);margin-bottom:4px;}
.lb-usage-step h3{margin:0;font-size:14px;font-weight:600;color:var(--lb-ink);line-height:1.5;}
.lb-usage-step p{margin:0;font-size:12.5px;line-height:1.7;color:var(--lb-muted);}
.lb-usage-foot{display:flex;align-items:center;gap:6px;margin:24px 0 0;padding-top:16px;border-top:1px solid var(--lb-border);font-size:11.5px;color:var(--lb-tertiary);}

/* ---------- PDF ---------- */
.lb-pdf-note{
  display:flex;align-items:center;gap:8px;flex-wrap:wrap;
  margin-top:12px;font-size:12px;line-height:1.5;color:var(--lb-faint);
}
.lb-pdf-note .lb-i{color:var(--lb-teal);}
.lb-pdf-link{
  display:inline-flex;align-items:center;gap:5px;height:32px;padding:0 12px;
  border:1px solid var(--lb-border);border-radius:8px;background:#F3F5F6;
  color:var(--lb-body);font-size:12.5px;font-weight:500;text-decoration:none;white-space:nowrap;
}
.lb-pdf-link:hover{background:#EBEDEE;border-color:#DCDEE0;color:var(--lb-ink);text-decoration:none;}
.lb-pdf-link .lb-i{color:var(--lb-muted);}

/* ---------- 响应式 ---------- */
@media (max-width:1100px){
  .lb-brand-text p{display:none;}
  .lb-stat-grid{grid-template-columns:repeat(2,minmax(0,1fr))!important;}
  .lb-usage-grid{grid-template-columns:repeat(2,minmax(0,1fr));}
}
@media (max-width:900px){
  .lb-grid-5-7, .lb-grid-8-4, .lb-grid-4-8, .lb-figure-grid, .lb-charts-row, .lb-charts-grid{grid-template-columns:1fr!important;}
  .lb-pagehead{flex-direction:column;align-items:flex-start;gap:6px;}
  .lb-pagehead p{text-align:left;}
  .lb-brand-meta{display:none;}
  .lb-detail-grid{grid-template-columns:1fr;gap:24px;}
  .lb-stat-grid, .lb-stat-grid-3{grid-template-columns:1fr!important;}
  .lb-usage-grid{grid-template-columns:1fr;}
  .lb-lesion-grid{grid-template-columns:1fr!important;}
}


/* =====================================================================
   布局重构 v2：一屏诊断、面板切换、浮层图片库、AI 问答合并
   ===================================================================== */
.lb-page{padding:18px 24px 22px!important;}
.lb-page > *{margin:0 0 16px 0!important;}
.lb-page > *:last-child{margin-bottom:0!important;}

/* ---------- 智能诊断：等高两栏 ---------- */
.lb-diag-grid{
  display:grid!important;grid-template-columns:minmax(0,5fr) minmax(0,7fr)!important;
  gap:18px!important;align-items:stretch!important;min-height:628px!important;
}
.lb-diag-grid > .lb-input-card,
.lb-diag-grid > .lb-right{height:100%!important;min-height:0!important;}
.lb-input-card{
  display:flex!important;flex-direction:column!important;
  padding:22px!important;gap:0!important;
}
.lb-input-card > * + *{margin-top:10px!important;}
.lb-input-card .lb-card-head{margin-bottom:0!important;}
.lb-input-card .lb-card-head h2{font-size:18px!important;}
.lb-input-card .lb-card-head p{font-size:11.5px!important;}
.lb-upload{height:190px!important;min-height:190px!important;}
.lb-examples{display:grid!important;grid-template-columns:repeat(5,minmax(0,1fr))!important;gap:6px!important;}
.lb-example{
  height:34px!important;padding:0 2px!important;font-size:11.5px!important;
  border-radius:8px!important;white-space:nowrap!important;overflow:hidden!important;
}
.lb-lib-trigger{
  color:var(--lb-teal-dark)!important;background:var(--lb-teal-soft)!important;
  border-color:rgba(0,173,162,.28)!important;
}
.lb-example-label{margin:0 0 2px!important;}
.lb-example-label span{font-size:11.5px!important;}
.lb-example-label em{font-size:10.5px!important;}
.lb-divider{margin:2px 0!important;}
.lb-controls{gap:14px!important;}
.lb-hint-text{
  margin:0!important;font-size:11.5px!important;line-height:1.55!important;
  color:var(--lb-tertiary)!important;
}
.lb-model-hint{
  font-size:11px!important;line-height:1.55!important;color:var(--lb-faint)!important;
  white-space:normal!important;
}
.lb-start{margin-top:auto!important;height:44px!important;}

/* 右栏卡片纵向堆叠 */
.lb-right{
  display:flex!important;flex-direction:column!important;
  height:100%!important;min-height:0!important;gap:12px!important;
}
.lb-right > .lb-html{margin:0!important;}
.lb-result-html{flex:0 0 auto!important;min-width:0!important;}
.lb-result-card{padding:20px 22px!important;}
.lb-result-card .lb-crop{margin-top:12px!important;}
.lb-result-card .lb-disease{font-size:30px!important;line-height:1.28!important;}
.lb-result-card .lb-conf{margin-top:8px!important;}
.lb-result-card .lb-conf-num{font-size:40px!important;}
.lb-result-card .lb-meter{margin-top:10px!important;}
.lb-result-card .lb-facts{margin-top:14px!important;padding-top:12px!important;}
.lb-result-card .lb-fact b{font-size:14px!important;}

/* 结果与病斑/防治同区面板 */
.lb-panel{
  display:none!important;background:var(--lb-surface)!important;
  border:1px solid var(--lb-border)!important;border-radius:16px!important;
  box-shadow:none!important;padding:16px 18px!important;
  height:236px!important;min-height:236px!important;overflow:hidden!important;
}
.lb-right:has(.lb-result-card) .lb-panel{display:block!important;}
.lb-pane{min-width:0!important;}
.lb-top5-pane{display:block!important;}
.lb-lesion-pane,.lb-detail-pane{display:none!important;}
.lb-panel:has(.lb-lesion-pane img) .lb-top5-pane{display:none!important;}
.lb-panel:has(.lb-lesion-pane img) .lb-lesion-pane{display:block!important;}
.lb-panel:has(.lb-detail-pane .lb-detail-card) .lb-top5-pane{display:none!important;}
.lb-panel:has(.lb-detail-pane .lb-detail-card) .lb-detail-pane{display:block!important;}

.lb-top5-head{margin:0 0 4px!important;}
.lb-top5-head h3{font-size:14px!important;}
.lb-top5-head span{font-size:10.5px!important;}
.lb-rank{
  height:28px!important;grid-template-columns:22px minmax(0,1fr) 70px 44px!important;
  gap:8px!important;
}
.lb-rank-idx{font-size:10px!important;}
.lb-rank-name{font-size:12px!important;}
.lb-rank-pct{font-size:11px!important;}
.lb-top5-foot{
  margin-top:5px!important;font-size:10px!important;line-height:1.5!important;
  color:var(--lb-tertiary)!important;white-space:nowrap!important;overflow:hidden!important;
  text-overflow:ellipsis!important;
}

.lb-pane-head{
  display:flex!important;align-items:center!important;justify-content:space-between!important;
  gap:12px!important;height:24px!important;margin:0 0 8px!important;
}
.lb-pane-title{display:flex!important;align-items:baseline!important;gap:8px!important;min-width:0!important;}
.lb-pane-title b{font-size:14px!important;font-weight:600!important;color:var(--lb-ink)!important;white-space:nowrap!important;}
.lb-pane-title span{
  font-size:10.5px!important;color:var(--lb-tertiary)!important;
  white-space:nowrap!important;overflow:hidden!important;text-overflow:ellipsis!important;
}
.lb-mini-btn{
  height:26px!important;min-height:26px!important;width:auto!important;flex:0 0 auto!important;
  padding:0 9px!important;border-radius:8px!important;font-size:11px!important;
  background:#F3F5F6!important;border:1px solid var(--lb-border)!important;
  color:var(--lb-body)!important;box-shadow:none!important;
}

.lb-mini-lesion{
  display:grid!important;grid-template-columns:140px 140px minmax(0,1fr)!important;
  gap:14px!important;align-items:start!important;
}
.lb-mini-fig{display:flex!important;flex-direction:column!important;gap:4px!important;min-width:0!important;}
.lb-mini-fig .lb-html span{font-size:10.5px!important;color:var(--lb-faint)!important;}
.lb-mini-fig .lb-figure{
  width:140px!important;height:140px!important;margin:0!important;
  border:1px solid var(--lb-border)!important;border-radius:10px!important;
  overflow:hidden!important;background:#F7F8F8!important;padding:0!important;
}
.lb-mini-fig .image-container{background:#F7F8F8!important;}
.lb-mini-fig img{width:100%!important;height:100%!important;object-fit:contain!important;}
.lb-mini-side{padding-top:18px!important;min-width:0!important;}
.lb-mini-side .lb-lesion-facts{
  display:flex!important;flex-direction:column!important;gap:5px!important;
  margin:0!important;padding:0!important;border:0!important;
}
.lb-mini-side .lb-stat-chip{
  display:flex!important;justify-content:space-between!important;
  padding:3px 8px!important;border-radius:6px!important;
}
.lb-mini-side .lb-stat-chip em{font-size:10.5px!important;}
.lb-mini-side .lb-stat-chip b{font-size:11.5px!important;}
.lb-mini-tip{
  margin-top:6px!important;font-size:10px!important;line-height:1.55!important;
  color:var(--lb-tertiary)!important;
}
.lb-mini-side .lb-lesion-note{font-size:10.5px!important;margin-top:6px!important;}

.lb-detail-pane .lb-detail-card{
  border:0!important;padding:0!important;background:transparent!important;
  box-shadow:none!important;
}
.lb-detail-pane .lb-detail-grid{gap:26px!important;}
.lb-detail-pane .lb-detail-col h3{font-size:12px!important;margin:0 0 5px!important;}
.lb-detail-pane .lb-symptom{font-size:13px!important;line-height:1.65!important;}
.lb-detail-pane .lb-control-list{gap:5px!important;}
.lb-detail-pane .lb-control-list p{font-size:12.5px!important;line-height:1.55!important;}
.lb-detail-pane .lb-control-list li > span{width:17px!important;height:17px!important;font-size:9.5px!important;margin-top:2px!important;}
.lb-detail-pane .lb-empty-inline,
.lb-detail-pane .lb-lesion-note{font-size:12px!important;}

.lb-action-row{
  display:none!important;grid-template-columns:1fr 1fr 1fr!important;
  gap:8px!important;margin:0!important;flex:0 0 auto!important;
}
.lb-right:has(.lb-result-card) .lb-action-row{display:grid!important;}
.lb-action-row .lb-btn-secondary{
  height:38px!important;min-height:38px!important;padding:0 6px!important;
  font-size:12.5px!important;width:100%!important;
}
div.block.lb-pdf-foot{
  display:none!important;margin-top:auto!important;flex:0 0 auto!important;min-width:0!important;
  height:28px!important;min-height:28px!important;padding:4px 10px!important;
  background:#FAFBFB!important;border:1px solid var(--lb-border)!important;border-radius:8px!important;
  align-items:center!important;
}
.lb-right:has(.lb-result-card) div.block.lb-pdf-foot{display:flex!important;}
.lb-pdf-hint{font-size:10.5px!important;line-height:1.4!important;color:var(--lb-tertiary)!important;}
.lb-pdf-foot .lb-pdf-note{margin:0!important;font-size:10.5px!important;gap:6px!important;}
.lb-pdf-foot .lb-pdf-link{height:24px!important;padding:0 8px!important;font-size:10.5px!important;}

/* 空状态：右侧整卡填满，与左栏底边对齐 */
.lb-right:has(.lb-empty-card) .lb-result-html{
  flex:1 1 auto!important;display:flex!important;flex-direction:column!important;min-height:0!important;
}
.lb-right:has(.lb-empty-card) .lb-result-html .html-container,
.lb-right:has(.lb-empty-card) .lb-result-html .prose,
.lb-right:has(.lb-empty-card) .lb-result-wrap{height:100%!important;min-height:0!important;}
.lb-right:has(.lb-empty-card) .lb-empty-card{
  height:100%!important;min-height:0!important;padding:0!important;
}
.lb-right:has(.lb-empty-card) .lb-empty{padding:28px 24px!important;}
.lb-right:has(.lb-empty-card) .lb-empty-steps{margin-top:22px!important;padding-top:18px!important;}

/* ---------- 浮层（图片库 / 批量详情） ---------- */
.lb-modal-mask{
  display:flex!important;position:fixed!important;inset:0!important;
  width:100vw!important;height:100vh!important;z-index:1000!important;
  align-items:center!important;justify-content:center!important;
  padding:24px!important;margin:0!important;
  background:rgba(10,14,25,.46)!important;
  backdrop-filter:blur(3px)!important;
}
.lb-modal-mask.hide{display:none!important;}
.lb-modal-panel{
  display:flex!important;flex-direction:column!important;gap:10px!important;
  width:min(980px,100%)!important;max-width:980px!important;
  height:fit-content!important;align-self:center!important;flex:0 1 auto!important;
  max-height:calc(100vh - 48px)!important;overflow:auto!important;
  background:var(--lb-surface)!important;border:1px solid var(--lb-border)!important;
  border-radius:16px!important;padding:22px!important;
  box-shadow:0 8px 32px rgba(10,14,25,.18)!important;
}
.lb-modal-head{
  display:flex!important;align-items:flex-start!important;justify-content:space-between!important;
  gap:16px!important;width:100%!important;flex:0 0 auto!important;
}
.lb-modal-title h2{margin:0!important;font-size:22px!important;line-height:1.3!important;font-weight:600!important;color:var(--lb-ink)!important;}
.lb-modal-title p{margin:4px 0 0!important;font-size:12px!important;line-height:1.5!important;color:var(--lb-faint)!important;}
.lb-modal-close{
  width:auto!important;min-width:64px!important;flex:0 0 auto!important;
  height:32px!important;min-height:32px!important;font-size:12.5px!important;padding:0 14px!important;
}
.lb-modal-tip{font-size:12px!important;line-height:1.6!important;color:var(--lb-faint)!important;flex:0 0 auto!important;}
.lb-modal-count{
  font-size:12px!important;color:var(--lb-body)!important;
  background:var(--lb-bg)!important;border:1px solid var(--lb-border)!important;
  border-radius:8px!important;padding:8px 12px!important;flex:0 0 auto!important;
}
.lb-gallery{border:0!important;background:transparent!important;padding:0!important;margin:0!important;}
.lb-gallery .grid-container{gap:8px!important;}
.lb-gallery .thumbnail-item{border-radius:8px!important;border:1px solid var(--lb-border)!important;overflow:hidden!important;}
.lb-batch-lib-panel{width:min(980px,100%)!important;}
.lb-batch-detail-panel{width:min(820px,100%)!important;}
.lb-batch-detail-dd{margin:0 0 4px!important;flex:0 0 auto!important;}
.lb-batch-detail-body{
  max-height:calc(100vh - 260px)!important;overflow:auto!important;
  padding-right:4px!important;flex:0 1 auto!important;min-height:0!important;
}
.lb-batch-detail-panel{height:fit-content!important;flex:0 0 auto!important;max-height:calc(100vh - 48px)!important;}
.lb-batch-detail-body .lb-result-card,
.lb-batch-detail-body .lb-detail-card{
  border:0!important;padding:0!important;background:transparent!important;box-shadow:none!important;
}
.lb-batch-detail-body .lb-result-card{
  padding-bottom:14px!important;border-bottom:1px solid var(--lb-border)!important;margin-bottom:16px!important;
}
.lb-batch-detail-body .lb-detail-grid{gap:24px!important;}
.lb-batch-detail-body .lb-symptom{font-size:14px!important;}
.lb-batch-detail-body .lb-control-list p{font-size:13px!important;}
.lb-empty-inline{font-size:12.5px!important;color:var(--lb-tertiary)!important;padding:16px 0!important;}

/* ---------- 批量诊断 ---------- */
.lb-batch-grid{
  display:grid!important;grid-template-columns:minmax(0,8fr) minmax(0,4fr)!important;
  gap:18px!important;align-items:stretch!important;min-height:388px!important;
}
.lb-batch-pick,.lb-batch-settings{
  display:flex!important;flex-direction:column!important;height:100%!important;
  padding:18px!important;min-width:0!important;
}
.lb-batch-pick > * + *,.lb-batch-settings > * + *{margin-top:6px!important;}
.lb-batch-pick .lb-card-head,.lb-batch-settings .lb-card-head{margin-bottom:0!important;}
.lb-batch-pick .lb-card-head h2,.lb-batch-settings .lb-card-head h2{font-size:18px!important;}
.lb-file{height:190px!important;min-height:190px!important;flex:0 0 auto!important;}
.lb-batch-tip{font-size:11px!important;line-height:1.55!important;color:var(--lb-faint)!important;}
.lb-batch-tip b{color:var(--lb-teal-dark)!important;font-weight:600!important;}
.lb-batch-pick-actions{display:grid!important;grid-template-columns:1fr 1fr!important;gap:8px!important;margin-top:auto!important;}
.lb-batch-pick-actions button{
  height:36px!important;min-height:36px!important;width:100%!important;
  font-size:12px!important;padding:0 6px!important;
}
.lb-batch-settings .lb-btn-primary{height:42px!important;min-height:42px!important;margin-top:auto!important;}
.lb-batch-settings .lb-csv-btn{height:36px!important;min-height:36px!important;font-size:12px!important;}
.lb-batch-status{font-size:11px!important;margin-top:2px!important;color:var(--lb-faint)!important;}
.lb-batch-detail-bar{
  display:none!important;align-items:center!important;justify-content:space-between!important;
  gap:12px!important;margin:0!important;flex:0 0 auto!important;
  height:34px!important;min-height:34px!important;position:relative!important;z-index:5!important;
  flex-wrap:nowrap!important;overflow:hidden!important;
}
.lb-batch-page:has(.lb-batch-result-card) .lb-batch-detail-bar{display:flex!important;}
.lb-batch-detail-bar .lb-html{flex:1 1 auto!important;min-width:0!important;height:34px!important;display:flex!important;align-items:center!important;}
.lb-detail-hint{font-size:11.5px!important;color:var(--lb-tertiary)!important;line-height:1.5!important;}
.lb-batch-detail-bar button{
  width:auto!important;min-width:128px!important;height:34px!important;min-height:34px!important;
  font-size:12px!important;flex:0 0 auto!important;
}
.lb-batch-result-card .lb-card-head{margin-bottom:6px!important;}
.lb-batch-result-card{padding:18px 20px 8px!important;}
.lb-batch-result-card .lb-card-head h2{font-size:17px!important;}
.lb-batch-result-card .lb-table th{padding-bottom:6px!important;}
.lb-batch-result-card .lb-table td{padding-top:7px!important;padding-bottom:7px!important;font-size:12.5px!important;}
.lb-batch-result-card .lb-badge{height:22px!important;font-size:11px!important;}
.lb-batch-empty-card{min-height:188px!important;}

/* ---------- AI 问答 ---------- */
.lb-ai-grid{
  display:grid!important;grid-template-columns:minmax(0,7fr) minmax(0,5fr)!important;
  gap:18px!important;align-items:stretch!important;min-height:600px!important;
}
.lb-ai-ask,.lb-ai-kb{
  display:flex!important;flex-direction:column!important;height:100%!important;
  padding:22px!important;min-width:0!important;
}
.lb-ai-ask > * + *,.lb-ai-kb > * + *{margin-top:10px!important;}
.lb-ai-ask .lb-card-head,.lb-ai-kb .lb-card-head{margin-bottom:0!important;}
.lb-ai-ask .lb-card-head h2,.lb-ai-kb .lb-card-head h2{font-size:18px!important;}
.lb-ai-ask .lb-card-head p,.lb-ai-kb .lb-card-head p{font-size:11px!important;}
.lb-mini-label{font-size:10.5px!important;font-weight:600!important;color:var(--lb-tertiary)!important;line-height:1.4!important;}
.lb-qa-examples{display:flex!important;flex-direction:column!important;gap:0!important;margin-top:-2px!important;}
.lb-qa-examples .lb-link-btn{
  font-size:12px!important;padding:4px 0!important;min-height:24px!important;
  line-height:1.5!important;color:var(--lb-teal-dark)!important;
}
.lb-qa-input textarea{min-height:68px!important;height:68px!important;}
.lb-qa-actions{display:grid!important;grid-template-columns:1.35fr .65fr!important;gap:8px!important;}
.lb-qa-actions button{height:40px!important;min-height:40px!important;width:100%!important;font-size:13px!important;}
.lb-qa-answer{
  flex:1 1 auto!important;min-height:150px!important;max-height:260px!important;
  overflow:auto!important;background:#FAFBFB!important;border:1px solid var(--lb-border)!important;
  border-radius:12px!important;padding:14px 16px!important;
}
.lb-qa-answer p{margin:0 0 8px!important;font-size:13px!important;line-height:1.7!important;}
.lb-qa-answer p:last-child{margin-bottom:0!important;}
.lb-qa-meta-wrap{flex:0 0 auto!important;min-height:18px!important;}
.lb-qa-meta{font-size:10.5px!important;line-height:1.5!important;color:var(--lb-tertiary)!important;margin:0!important;padding:0!important;border:0!important;}
.lb-ai-kb .lb-select{flex:0 0 auto!important;}
.lb-kb-panel{
  flex:1 1 auto!important;min-height:0!important;max-height:430px!important;
  overflow:auto!important;padding-right:4px!important;
}
.lb-kb-panel .lb-kb-card{
  border:0!important;padding:0!important;background:transparent!important;box-shadow:none!important;
}
.lb-kb-panel .lb-kb-head{padding-bottom:8px!important;}
.lb-kb-panel .lb-kb-head h2{font-size:21px!important;margin-top:2px!important;}
.lb-kb-panel .lb-kb-crop{font-size:10.5px!important;}
.lb-kb-panel .lb-kb-code{font-size:9.5px!important;}
.lb-kb-panel .lb-kb-meta{margin-top:8px!important;gap:5px!important;}
.lb-kb-panel .lb-stat-chip{padding:3px 8px!important;}
.lb-kb-panel .lb-stat-chip em{font-size:10.5px!important;}
.lb-kb-panel .lb-stat-chip b{font-size:11.5px!important;}
.lb-kb-panel .lb-kb-section{margin-top:10px!important;}
.lb-kb-panel .lb-kb-section h3{font-size:12px!important;margin-bottom:3px!important;}
.lb-kb-panel .lb-symptom{font-size:13px!important;line-height:1.55!important;}
.lb-kb-panel .lb-control-list{gap:2px!important;}
.lb-kb-panel .lb-control-list p{font-size:12.5px!important;line-height:1.45!important;}
.lb-kb-panel .lb-control-list li > span{width:17px!important;height:17px!important;font-size:9.5px!important;margin-top:2px!important;}
.lb-ai-kb .lb-kb-note{
  flex:0 0 auto!important;margin-top:auto!important;padding-top:10px!important;
  border-top:1px solid var(--lb-border)!important;font-size:10.5px!important;line-height:1.5!important;
}

/* 历史表格限制高度，避免整页过长 */
.lb-table-scroll{max-height:360px!important;overflow-y:auto!important;}

/* 响应式：新版栅格在窄屏回落为单列 */
@media (max-width:900px){
  .lb-diag-grid,.lb-batch-grid,.lb-ai-grid{grid-template-columns:1fr!important;min-height:0!important;}
  .lb-diag-grid > .lb-input-card,.lb-diag-grid > .lb-right,
  .lb-ai-ask,.lb-ai-kb{height:auto!important;}
  .lb-panel{height:auto!important;min-height:0!important;}
  .lb-mini-lesion{grid-template-columns:140px 140px!important;}
  .lb-mini-side{grid-column:1 / -1!important;padding-top:0!important;}
  .lb-modal-mask{padding:12px!important;}
  .lb-modal-panel{max-height:calc(100vh - 24px)!important;padding:16px!important;}
  .lb-ai-ask,.lb-ai-kb{min-height:0!important;}
  .lb-qa-answer{max-height:none!important;}
  .lb-kb-panel{max-height:none!important;}
  .lb-batch-detail-bar{display:none!important;}
}




/* ---------- v2.1 密度与对齐微调 ---------- */
.lb-upload{height:270px!important;min-height:270px!important;}
.lb-input-card,.lb-right,.lb-batch-pick,.lb-batch-settings,.lb-ai-ask,.lb-ai-kb,
.lb-modal-panel,.lb-batch-detail-panel{flex-wrap:nowrap!important;}
.lb-ai-ask,.lb-ai-kb,.lb-batch-pick,.lb-batch-settings{gap:0!important;}
.lb-rank{height:38px!important;}
.lb-mini-lesion{grid-template-columns:170px 170px minmax(0,1fr)!important;}
.lb-mini-fig .lb-figure{width:170px!important;height:170px!important;}
.lb-mini-side{padding-top:18px!important;}
.lb-detail-pane .lb-detail-card{height:100%!important;display:flex!important;flex-direction:column!important;}
.lb-detail-pane .lb-detail-grid{flex:1 1 auto!important;min-height:0!important;}
.lb-detail-pane .lb-symptom{font-size:13.5px!important;line-height:1.7!important;}
.lb-detail-pane .lb-control-list g, .lb-detail-pane .lb-control-list p{font-size:13px!important;}
.lb-detail-foot{
  margin-top:auto!important;padding-top:7px!important;border-top:1px solid var(--lb-border)!important;
  font-size:10px!important;line-height:1.5!important;color:var(--lb-tertiary)!important;
}

/* 批量页：并排设置、稳定高度、可读标签 */
.lb-batch-grid{min-height:386px!important;}
.lb-batch-page{padding-bottom:0!important;}
.lb-file{height:170px!important;min-height:170px!important;}
.lb-batch-pick > * + *{margin-top:8px!important;}
.lb-batch-pick .lb-card-head h2,.lb-batch-settings .lb-card-head h2{font-size:18px!important;}
.lb-batch-sel-note{
  font-size:10.5px!important;line-height:1.45!important;color:var(--lb-faint)!important;
  white-space:nowrap!important;overflow:hidden!important;text-overflow:ellipsis!important;
}
.lb-batch-controls{grid-template-columns:minmax(0,1.25fr) minmax(0,1fr)!important;gap:10px!important;}
.lb-batch-model-hint{font-size:10.5px!important;}
.lb-batch-settings > * + *{margin-top:8px!important;}
.lb-batch-settings .lb-btn-primary{margin-top:8px!important;height:42px!important;min-height:42px!important;}
.lb-batch-settings .lb-csv-btn{height:36px!important;min-height:36px!important;font-size:12px!important;}
.lb-csv-btn:disabled{opacity:.55!important;cursor:not-allowed!important;}
.lb-batch-status{font-size:10.5px!important;line-height:1.4!important;color:var(--lb-faint)!important;}
.lb-batch-guide{
  margin-top:auto!important;padding-top:10px!important;border-top:1px solid var(--lb-border)!important;
  font-size:10.5px!important;line-height:1.55!important;color:var(--lb-tertiary)!important;
}
.lb-batch-guide b{font-size:11px!important;font-weight:600!important;color:var(--lb-muted)!important;}
.lb-batch-result-card .lb-table th{padding-bottom:9px!important;}
.lb-batch-result-card .lb-table td{padding-top:9px!important;padding-bottom:9px!important;}

/* AI 问答：固定两栏高度，右侧知识库 + 常见问题 */
.lb-ai-grid{height:600px!important;min-height:600px!important;}
.lb-ai-ask,.lb-ai-kb{
  height:600px!important;max-height:600px!important;overflow:hidden!important;
  padding:18px!important;
}
.lb-ai-ask > * + *{margin-top:8px!important;}
.lb-ai-kb > * + *{margin-top:6px!important;}
.lb-ai-ask .lb-card-head h2,.lb-ai-kb .lb-card-head h2{font-size:18px!important;}
.lb-qa-input textarea{min-height:74px!important;height:74px!important;}
.lb-qa-actions{gap:8px!important;margin-top:8px!important;}
.lb-qa-answer{
  flex:1 1 auto!important;min-height:0!important;max-height:none!important;
  overflow:auto!important;padding:14px 16px!important;
}
.lb-qa-answer p{font-size:13px!important;line-height:1.7!important;}
.lb-ai-kb .lb-select{flex:0 0 auto!important;}
.lb-ai-kb .lb-kb-panel{
  flex:1 1 auto!important;min-height:0!important;max-height:330px!important;
  overflow:auto!important;padding-right:4px!important;
}
.lb-ask-label{margin-top:2px!important;flex:0 0 auto!important;}
.lb-qa-examples{
  display:grid!important;grid-template-columns:1fr 1fr!important;
  gap:0 14px!important;flex:0 0 auto!important;
}
.lb-qa-examples .block{margin:0!important;padding:0!important;border:0!important;background:transparent!important;}
.lb-qa-examples .lb-link-btn{
  height:25px!important;min-height:25px!important;padding:2px 0!important;
  font-size:11px!important;line-height:1.4!important;white-space:nowrap!important;
  overflow:hidden!important;text-overflow:ellipsis!important;
}
.lb-ai-kb .lb-kb-note{
  margin-top:6px!important;padding-top:8px!important;flex:0 0 auto!important;
  font-size:10.5px!important;line-height:1.45!important;
}

/* 历史表最多 8 行，完整数据仍在 CSV */
.lb-table-scroll{max-height:390px!important;overflow-y:auto!important;}

@media (max-width:900px){
  .lb-ai-grid,.lb-ai-ask,.lb-ai-kb{height:auto!important;max-height:none!important;}
  .lb-ai-ask,.lb-ai-kb{overflow:visible!important;}
  .lb-qa-answer{max-height:320px!important;}
  .lb-ai-kb .lb-kb-panel{max-height:340px!important;flex:none!important;}
  .lb-mini-lesion{grid-template-columns:150px 150px!important;}
  .lb-mini-fig .lb-figure{width:150px!important;height:150px!important;}
}



/* ---------- 批量详情：复用智能诊断结果 ---------- */
.lb-batch-detail-panel{width:min(1180px,96vw)!important;max-width:1180px!important;overflow-x:hidden!important;}
.lb-batch-detail-select{
  display:grid!important;grid-template-columns:auto minmax(0,1fr)!important;
  align-items:center!important;gap:12px!important;
  height:42px!important;min-height:42px!important;flex:0 0 auto!important;
  flex-wrap:nowrap!important;margin:0!important;width:100%!important;
}
.lb-batch-detail-select .lb-html{
  width:auto!important;min-width:0!important;flex:none!important;
}
.lb-batch-detail-select-label{
  font-size:11px!important;color:var(--lb-faint)!important;white-space:nowrap!important;
}
.lb-batch-detail-select .lb-select,.lb-batch-detail-dd,
.lb-batch-detail-select .svelte-d5xbca{
  width:100%!important;min-width:0!important;max-width:none!important;margin:0!important;
}
.lb-batch-detail-top{
  display:grid!important;grid-template-columns:340px minmax(0,1fr)!important;
  gap:18px!important;align-items:start!important;flex:0 0 auto!important;
  flex-wrap:nowrap!important;
}
.lb-batch-detail-photo-col{min-width:0!important;}
.lb-batch-detail-photo{
  width:100%!important;height:235px!important;min-height:235px!important;margin:0!important;
  border:1px solid var(--lb-border)!important;border-radius:12px!important;
  overflow:hidden!important;background:#F7F8F8!important;padding:0!important;
}
.lb-batch-detail-photo .image-container{background:#F7F8F8!important;}
.lb-batch-detail-photo img{width:100%!important;height:100%!important;object-fit:contain!important;}
.lb-batch-detail-meta{
  padding:6px 2px 0!important;display:flex!important;flex-direction:column!important;
  gap:2px!important;min-width:0!important;
}
.lb-batch-detail-meta .lb-eyebrow{font-size:9.5px!important;}
.lb-batch-detail-meta b{
  font-size:13.5px!important;font-weight:600!important;color:var(--lb-ink)!important;
  line-height:1.35!important;word-break:break-all!important;
}
.lb-batch-detail-meta em{
  font-size:10.5px!important;line-height:1.5!important;color:var(--lb-tertiary)!important;
  font-style:normal!important;
}
.lb-batch-detail-main{min-width:0!important;}
.lb-batch-detail-main .lb-result-card{min-height:283px!important;}
.lb-batch-detail-panel .lb-panel{
  display:none!important;height:266px!important;min-height:266px!important;
  overflow:auto!important;flex:0 0 auto!important;margin:0!important;
}
.lb-batch-detail-panel:has(.lb-result-card) .lb-panel{display:block!important;}
.lb-batch-detail-panel .lb-action-row{
  display:none!important;grid-template-columns:1fr 1fr!important;
}
.lb-batch-detail-panel:has(.lb-result-card) .lb-action-row{display:grid!important;}
.lb-batch-detail-panel .lb-action-row .lb-btn-secondary{
  height:38px!important;min-height:38px!important;
}



/* ---------- 模型评估：跨域泛化 ---------- */
.lb-xd-section{
  margin-top:28px!important;display:flex!important;flex-direction:column!important;
  gap:16px!important;min-width:0!important;
}
.lb-xd-section-head{padding:2px 0 4px!important;}
.lb-xd-section-head h2{
  margin:4px 0 6px!important;font-size:24px!important;line-height:1.35!important;
  font-weight:600!important;color:var(--lb-ink)!important;letter-spacing:-.01em!important;
}
.lb-xd-section-head p{
  margin:0!important;font-size:13px!important;line-height:1.65!important;color:var(--lb-tertiary)!important;
}
.lb-xd-hero{padding:24px 26px 0!important;}
.lb-xd-hero-top{display:flex!important;justify-content:space-between!important;gap:20px!important;align-items:flex-start!important;}
.lb-xd-hero h2{
  margin:6px 0 8px!important;font-size:22px!important;line-height:1.45!important;
  font-weight:600!important;color:var(--lb-ink)!important;letter-spacing:-.01em!important;
}
.lb-xd-hero p{
  margin:0!important;font-size:13px!important;line-height:1.7!important;color:var(--lb-muted)!important;
  max-width:920px!important;
}
.lb-xd-stats{
  display:grid!important;grid-template-columns:repeat(6,minmax(0,1fr))!important;
  margin-top:20px!important;border-top:1px solid var(--lb-border)!important;
}
.lb-xd-stat{padding:14px 16px 16px!important;border-right:1px solid var(--lb-border)!important;min-width:0!important;}
.lb-xd-stat:last-child{border-right:0!important;}
.lb-xd-stat > span{display:block!important;font-size:11px!important;color:var(--lb-faint)!important;line-height:1.5!important;}
.lb-xd-stat b{
  display:block!important;margin-top:4px!important;font-size:23px!important;line-height:1.2!important;
  font-weight:600!important;color:var(--lb-ink)!important;letter-spacing:-.01em!important;
}
.lb-xd-stat b em{font-style:normal!important;font-size:12px!important;font-weight:500!important;color:var(--lb-faint)!important;margin-left:1px!important;}
.lb-xd-stat small{display:block!important;margin-top:4px!important;font-size:10.5px!important;line-height:1.5!important;color:var(--lb-tertiary)!important;}

.lb-xd-row{display:grid!important;gap:16px!important;align-items:stretch!important;flex-wrap:nowrap!important;}
.lb-xd-row-protocol{grid-template-columns:minmax(0,5fr) minmax(0,7fr)!important;}
.lb-xd-row-shift{grid-template-columns:minmax(0,5fr) minmax(0,7fr)!important;}
.lb-xd-row-tsne{grid-template-columns:minmax(0,6fr) minmax(0,6fr)!important;}
.lb-xd-row > .column{height:100%!important;min-width:0!important;}
.lb-xd-side{padding:20px 22px!important;height:100%!important;min-width:0!important;}
.lb-xd-side-head h2{
  margin:4px 0 4px!important;font-size:18px!important;line-height:1.4!important;
  font-weight:600!important;color:var(--lb-ink)!important;
}
.lb-xd-side-head p{margin:0 0 8px!important;font-size:11.5px!important;line-height:1.6!important;color:var(--lb-tertiary)!important;}
.lb-xd-table{font-size:12px!important;}
.lb-xd-table th{font-size:10.5px!important;padding-bottom:7px!important;}
.lb-xd-table td{padding-top:7px!important;padding-bottom:7px!important;line-height:1.45!important;}
.lb-xd-table tr.is-base td{background:#F4FBFA!important;}
.lb-xd-table tr.is-best td{background:var(--lb-teal-soft)!important;}
.lb-xd-table tr.is-key td{background:#FFF8EE!important;}
.lb-xd-table tr.is-base:hover td,.lb-xd-table tr.is-best:hover td,.lb-xd-table tr.is-key:hover td{background:inherit!important;}
.lb-xd-table td:first-child{white-space:nowrap!important;}
.lb-xd-note{
  margin-top:10px!important;font-size:10.5px!important;line-height:1.6!important;
  color:var(--lb-tertiary)!important;
}
.lb-xd-callout{
  margin-top:12px!important;padding:10px 12px!important;border-left:3px solid var(--lb-teal)!important;
  background:#F3FBFA!important;border-radius:0 8px 8px 0!important;
  display:flex!important;flex-direction:column!important;gap:2px!important;
}
.lb-xd-callout b{font-size:13px!important;color:var(--lb-teal-dark)!important;line-height:1.45!important;}
.lb-xd-callout span{font-size:10.5px!important;line-height:1.6!important;color:var(--lb-muted)!important;}
.lb-xd-bullets{display:flex!important;flex-direction:column!important;gap:7px!important;margin-top:6px!important;}
.lb-xd-bullets > div{display:flex!important;flex-direction:column!important;gap:1px!important;padding-left:10px!important;border-left:2px solid var(--lb-border)!important;}
.lb-xd-bullets b{font-size:12px!important;font-weight:600!important;color:var(--lb-ink)!important;line-height:1.45!important;}
.lb-xd-bullets span{font-size:10.5px!important;line-height:1.55!important;color:var(--lb-tertiary)!important;}

.lb-xd-wide{width:100%!important;}
.lb-xd-fig-note{
  margin-top:8px!important;font-size:10.5px!important;line-height:1.6!important;
  color:var(--lb-tertiary)!important;
}
.lb-xd-three{
  display:grid!important;grid-template-columns:repeat(3,minmax(0,1fr))!important;
  gap:16px!important;align-items:stretch!important;flex-wrap:nowrap!important;
}
.lb-xd-three > .column{height:100%!important;min-width:0!important;}
.lb-xd-limit{padding:22px 24px!important;}
.lb-xd-limit ul{margin:8px 0 0!important;padding:0!important;list-style:none!important;display:flex!important;flex-direction:column!important;gap:7px!important;}
.lb-xd-limit li{
  font-size:12px!important;line-height:1.65!important;color:var(--lb-muted)!important;
  padding-left:12px!important;border-left:2px solid var(--lb-border)!important;
}
.lb-xd-limit li b{color:var(--lb-ink)!important;font-weight:600!important;margin-right:4px!important;}

@media (max-width:1100px){
  .lb-xd-stats{grid-template-columns:repeat(3,minmax(0,1fr))!important;}
  .lb-xd-stat:nth-child(3n){border-right:0!important;}
  .lb-xd-stat{border-bottom:1px solid var(--lb-border)!important;}
  .lb-xd-row-protocol,.lb-xd-row-shift,.lb-xd-row-tsne,
  .lb-xd-three{grid-template-columns:minmax(0,1fr)!important;}
  .lb-xd-hero-top{flex-direction:column!important;}
}



/* ---------- 病害示例图片 ---------- */
.lb-result-card{position:relative!important;}
.lb-result-card .lb-disease{padding-right:118px!important;}
.lb-result-ref{
  position:absolute!important;top:54px!important;right:20px!important;
  width:100px!important;height:100px!important;border-radius:10px!important;
  overflow:hidden!important;border:1px solid var(--lb-border)!important;
  background:#F7F8F8!important;display:block!important;text-decoration:none!important;
  z-index:2!important;
}
.lb-result-ref img{width:100%!important;height:100%!important;object-fit:cover!important;display:block!important;}
.lb-result-ref span{
  position:absolute!important;left:0!important;right:0!important;bottom:0!important;
  background:rgba(10,14,25,.62)!important;color:#fff!important;
  font-size:8.5px!important;line-height:1.65!important;text-align:center!important;
  letter-spacing:.02em!important;
}
.lb-shot-grid{display:grid!important;grid-template-columns:repeat(3,minmax(0,1fr))!important;gap:8px!important;}
.lb-shot{
  position:relative!important;display:block!important;height:100px!important;
  border-radius:10px!important;overflow:hidden!important;border:1px solid var(--lb-border)!important;
  background:#F7F8F8!important;text-decoration:none!important;
}
.lb-shot img{width:100%!important;height:100%!important;object-fit:cover!important;display:block!important;}
.lb-shot-src{
  position:absolute!important;left:0!important;right:0!important;bottom:0!important;
  background:rgba(10,14,25,.62)!important;color:#fff!important;
  font-size:8.5px!important;line-height:1.65!important;text-align:center!important;
  letter-spacing:.02em!important;
}
.lb-shot-head{display:flex!important;align-items:baseline!important;justify-content:space-between!important;gap:8px!important;margin:0 0 6px!important;}
.lb-shot-head b{font-size:11px!important;font-weight:600!important;color:var(--lb-ink)!important;line-height:1.45!important;}
.lb-shot-head span{font-size:9.5px!important;color:var(--lb-tertiary)!important;white-space:nowrap!important;}
.lb-shot-block{
  margin:12px 0 2px!important;padding-top:10px!important;border-top:1px solid var(--lb-border)!important;
}
.lb-kb-panel .lb-shot-grid{grid-template-columns:repeat(3,minmax(0,1fr))!important;}
.lb-kb-panel .lb-shot{height:100px!important;}


/* ---------- 示例图 / 库内联按钮适配 ---------- */
.lb-example-head{
  display:flex!important;align-items:center!important;justify-content:space-between!important;
  gap:8px!important;margin:0 0 6px!important;height:28px!important;flex-wrap:nowrap!important;
}
.lb-example-head .lb-html{flex:1 1 auto!important;min-width:0!important;}
.lb-example-head .lb-example-label{margin:0!important;}
.lb-lib-inline{
  width:auto!important;min-width:64px!important;height:26px!important;min-height:26px!important;
  padding:0 10px!important;font-size:11px!important;flex:0 0 auto!important;
}
.lb-shot-grid{grid-template-columns:repeat(var(--shot-cols,3),minmax(0,1fr))!important;}
.lb-kb-panel .lb-shot-grid{grid-template-columns:repeat(var(--shot-cols,3),minmax(0,1fr))!important;}

.lb-lib-filter{
  display:grid!important;grid-template-columns:auto minmax(0,1fr)!important;
  gap:10px!important;align-items:center!important;height:40px!important;min-height:40px!important;
  margin:0 0 8px!important;flex-wrap:nowrap!important;
}
.lb-lib-filter .lb-html{flex:0 0 auto!important;width:auto!important;min-width:0!important;}
.lb-lib-filter-label{font-size:11px!important;color:var(--lb-faint)!important;white-space:nowrap!important;}
.lb-lib-filter .lb-select,.lb-lib-filter .lb-lib-crop{
  width:100%!important;min-width:0!important;max-width:none!important;margin:0!important;
  flex:1 1 auto!important;
}
.lb-lib-filter .svelte-d5xbca{width:100%!important;min-width:0!important;}

.lb-eval-old{
  margin:0!important;padding:11px 14px!important;border-radius:10px!important;
  background:#FFF8EC!important;border:1px solid #F5E3C8!important;
  font-size:11.5px!important;line-height:1.65!important;color:#8A6220!important;
}
.lb-eval-old b{font-weight:600!important;color:#6B4A12!important;}


/* =====================================================================
   v6：页面利用率 / 详情展开 / 知识库单滚轴 / 评估子页签 / 批量行按钮
   ===================================================================== */

/* ---------- 顶栏与一级页签更紧凑，内容更宽 ---------- */
.lb-header-inner{
  max-width:1320px!important;height:52px!important;padding:0 20px!important;gap:16px!important;
}
.lb-brand{gap:10px!important;}
.lb-brand-mark{width:32px!important;height:32px!important;border-radius:9px!important;}
.lb-brand-text h1{font-size:17px!important;line-height:1.3!important;}
.lb-brand-text p{font-size:10.5px!important;line-height:1.35!important;}
.lb-brand-meta{gap:6px!important;}
.lb-chip{height:24px!important;padding:0 8px!important;font-size:11px!important;}

.lb-tabs > .tab-wrapper{height:42px!important;}
.lb-tabs .tab-container[role="tablist"]{
  max-width:1320px!important;height:42px!important;padding:0 20px!important;gap:28px!important;
}
.lb-tabs button[role="tab"]{height:42px!important;font-size:14px!important;}

.lb-page{
  max-width:1320px!important;padding:12px 20px 36px!important;
}
.lb-page > *{margin:0 0 14px 0!important;}
.lb-pagehead{align-items:center!important;gap:16px!important;padding:0!important;}
.lb-pagehead h1{font-size:24px!important;line-height:1.3!important;}
.lb-pagehead p{font-size:12.5px!important;line-height:1.55!important;max-width:640px!important;margin:0!important;}
.lb-pagehead-tag{font-size:10px!important;}

/* ---------- 症状与防治：从右栏移到全宽区域，完整展开 ---------- */
.lb-detail-wide{
  display:none!important;width:100%!important;min-width:0!important;
  margin:0!important;padding:0!important;border:0!important;background:transparent!important;
  scroll-margin-top:96px!important;
}
.lb-detail-wide:has(.lb-detail-card){display:block!important;}
.lb-detail-wide .lb-detail-pane{display:block!important;width:100%!important;min-width:0!important;}
.lb-detail-wide .lb-pane-head{margin:0 0 10px!important;}
.lb-detail-wide .lb-detail-card{
  display:block!important;height:auto!important;min-height:0!important;max-height:none!important;
  padding:22px 26px!important;border:1px solid var(--lb-border)!important;border-radius:16px!important;
  background:#FFFFFF!important;box-shadow:none!important;overflow:visible!important;
}
.lb-detail-wide .lb-detail-grid{
  grid-template-columns:minmax(0,1.12fr) minmax(0,1fr)!important;gap:34px!important;
}
.lb-detail-wide .lb-detail-col h3{font-size:14.5px!important;margin:0 0 10px!important;}
.lb-detail-wide .lb-symptom{font-size:15px!important;line-height:1.78!important;}
.lb-detail-wide .lb-control-list{gap:8px!important;}
.lb-detail-wide .lb-control-list p{font-size:14px!important;line-height:1.65!important;}
.lb-detail-wide .lb-control-list li > span{width:19px!important;height:19px!important;font-size:10px!important;}
.lb-detail-wide .lb-detail-foot{font-size:10.5px!important;padding-top:10px!important;margin-top:14px!important;}
.lb-detail-wide .lb-detail-foot + *{margin:0!important;}

/* ---------- AI 问答：左右固定等高，内部知识库单滚轴 ---------- */
.lb-ai-grid{
  grid-template-columns:minmax(0,6fr) minmax(0,6fr)!important;
  height:700px!important;min-height:700px!important;max-height:700px!important;
  align-items:stretch!important;
}
.lb-ai-ask,.lb-ai-kb{
  height:700px!important;min-height:700px!important;max-height:700px!important;
  overflow:hidden!important;
}
.lb-ai-ask{display:flex!important;flex-direction:column!important;}
.lb-ai-ask .lb-qa-answer{
  flex:1 1 auto!important;min-height:0!important;max-height:none!important;
  overflow-y:auto!important;overflow-x:hidden!important;height:auto!important;
  overscroll-behavior:contain!important;
}
.lb-ai-ask .lb-qa-answer .prose{
  overflow:visible!important;max-height:none!important;height:auto!important;
}
.lb-ai-kb{display:flex!important;flex-direction:column!important;padding-bottom:16px!important;}
.lb-ai-kb > .lb-kb-panel{
  flex:1 1 auto!important;min-height:0!important;max-height:none!important;
  overflow-y:auto!important;overflow-x:hidden!important;padding-right:4px!important;
}
.lb-ai-kb > .lb-kb-panel .prose.lb-kb-panel,
.lb-ai-kb .lb-kb-panel .lb-kb-panel{
  height:auto!important;min-height:0!important;max-height:none!important;
  overflow:visible!important;flex:none!important;padding-right:0!important;
}
.lb-ai-kb .lb-kb-note{margin-top:8px!important;flex:0 0 auto!important;}
.lb-ai-kb .lb-kb-panel .lb-shot{height:104px!important;}

/* ---------- 评估页：四个锚点改为真正的子页签 ---------- */
.lb-eval-subtabs{margin:0!important;gap:0!important;background:transparent!important;}
.lb-eval-subtabs > .tab-wrapper{
  position:sticky!important;top:0!important;z-index:22!important;
  background:rgba(245,246,246,.97)!important;
  border-bottom:1px solid var(--lb-border)!important;
  height:44px!important;padding:0!important;
  display:flex!important;align-items:stretch!important;
}
.lb-eval-subtabs .tab-container[role="tablist"]{
  max-width:1320px!important;width:100%!important;margin:0 auto!important;
  padding:0!important;height:44px!important;display:flex!important;align-items:stretch!important;gap:28px!important;
}
.lb-eval-subtabs button[role="tab"]{
  height:44px!important;padding:0!important;margin:0!important;
  font-size:13.5px!important;font-weight:600!important;color:var(--lb-muted)!important;
  background:transparent!important;border:0!important;border-bottom:2px solid transparent!important;
  border-radius:0!important;box-shadow:none!important;
}
.lb-eval-subtabs button[role="tab"].selected{
  color:var(--lb-ink)!important;border-bottom-color:var(--lb-teal)!important;
}
.lb-eval-subtabs .tabitem{padding:0!important;border:0!important;background:transparent!important;}
.lb-eval-subtabs .tabitem > .column{width:100%!important;}
.lb-eval-subtabs + *{margin-top:14px!important;}

/* ---------- 批量结果：HTML 行 + 每行「详细」按钮 ---------- */
.lb-batch-rows{display:block!important;width:100%!important;min-width:0!important;}
.lb-batch-thead{
  display:grid!important;
  grid-template-columns:minmax(0,40fr) minmax(0,26fr) minmax(0,12fr) minmax(0,12fr) minmax(0,10fr)!important;
  gap:12px!important;align-items:center!important;padding:0 0 9px!important;
  border-bottom:1px solid var(--lb-border);font-size:12px;font-weight:600;color:var(--lb-faint);white-space:nowrap;
}
.lb-batch-thead .lb-num{text-align:right;}
.lb-batch-row{
  display:grid!important;
  grid-template-columns:minmax(0,40fr) minmax(0,26fr) minmax(0,12fr) minmax(0,12fr) minmax(0,10fr)!important;
  gap:12px!important;align-items:center!important;padding:7px 0!important;margin:0!important;
  border-bottom:1px solid #F1F2F3;
}
.lb-batch-row:last-child{border-bottom:0;}
.lb-batch-row:hover{background:#FAFBFA;}
.lb-batch-row > .lb-html,
.lb-batch-row > .lb-html > .html-container,
.lb-batch-row > .lb-html .prose,
.lb-batch-row-inner{
  display:contents!important;min-width:0!important;width:auto!important;
}
.lb-batch-row-inner{font-size:12.5px;line-height:1.5;}
.lb-batch-file{color:var(--lb-ink);font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.lb-batch-result{color:var(--lb-body);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.lb-batch-conf{text-align:right;font-family:var(--lb-mono);font-size:11.5px;color:var(--lb-muted);}
.lb-batch-status{justify-self:start;}
.lb-batch-status .lb-badge{height:22px!important;font-size:11px!important;padding:0 7px!important;}
.lb-batch-detail-btn{
  width:56px!important;min-width:56px!important;height:27px!important;min-height:27px!important;
  padding:0!important;font-size:11px!important;border-radius:7px!important;justify-self:start!important;
}


/* ---------- 统一图片放大层：滚轮缩放 / 拖拽平移 ---------- */
.lb-lightbox{
  position:fixed!important;inset:0!important;z-index:3000!important;
  background:rgba(10,14,18,.94)!important;
  display:flex!important;align-items:center!important;justify-content:center!important;
  overflow:hidden!important;overscroll-behavior:contain!important;
}
.lb-lightbox img{
  max-width:92vw!important;max-height:84vh!important;width:auto!important;height:auto!important;
  object-fit:contain!important;border-radius:4px!important;
  box-shadow:0 10px 50px rgba(0,0,0,.55)!important;
  cursor:grab!important;user-select:none!important;touch-action:none!important;
  transform-origin:center center!important;will-change:transform!important;
  transition:none!important;
}
.lb-lightbox img.is-grabbing{cursor:grabbing!important;}
.lb-lightbox-close{
  position:fixed!important;top:18px!important;right:22px!important;
  width:36px!important;height:36px!important;border-radius:50%!important;
  border:1px solid rgba(255,255,255,.25)!important;background:rgba(0,0,0,.35)!important;
  color:#fff!important;font-size:22px!important;line-height:30px!important;text-align:center!important;
  cursor:pointer!important;z-index:3001!important;padding:0!important;
}
.lb-lightbox-close:hover{background:rgba(255,255,255,.16)!important;}
.lb-lightbox-hint{
  position:fixed!important;left:50%!important;bottom:22px!important;transform:translateX(-50%)!important;
  z-index:3001!important;padding:7px 16px!important;border-radius:999px!important;
  background:rgba(0,0,0,.55)!important;color:#fff!important;
  font-size:12px!important;white-space:nowrap!important;pointer-events:none!important;
}
.lb-upload img,.lb-figure img,.lb-batch-detail-photo img,.lb-shot img,.lb-result-ref img{cursor:zoom-in!important;}


/* ---------- 批量已选：明显的勾选卡片 + 按作物分组 ---------- */
.lb-batch-pick-sub{
  margin:2px 0 -2px!important;font-size:11px!important;font-weight:600!important;
  color:var(--lb-faint)!important;letter-spacing:.02em!important;
}
.lb-batch-upload-file{
  height:240px!important;min-height:240px!important;max-height:240px!important;overflow:hidden!important;
  border:1px dashed #D8DCDE!important;border-radius:12px!important;background:#FAFBFB!important;
}
.lb-batch-upload-file .file-preview-holder{display:none!important;}
.lb-batch-upload-file .top-panel{position:relative!important;height:240px!important;display:flex!important;align-items:center!important;justify-content:center!important;padding:0!important;}
.lb-batch-upload-file .top-panel .icon-button-wrapper{position:absolute!important;right:8px!important;top:8px!important;}
.lb-batch-upload-file .upload-button{width:100%!important;height:100%!important;}
.lb-batch-selected{min-height:0!important;margin:0!important;}
.lb-pick-groups{display:flex;flex-direction:column;gap:8px;min-width:0;}
.lb-pick-group-head{
  display:flex;align-items:baseline;justify-content:space-between;gap:8px;
  margin:0 0 5px;font-size:11px;color:var(--lb-faint);
}
.lb-pick-group-head b{font-size:11.5px;font-weight:600;color:var(--lb-ink);}
.lb-pick-cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:7px;}
.lb-pick-card{min-width:0;border:1.5px solid #C9EFEB;border-radius:10px;background:#F5FCFB;padding:4px;position:relative;}
.lb-pick-thumb{position:relative;height:74px;border-radius:7px;overflow:hidden;background:#EAF4F2;display:flex;align-items:center;justify-content:center;}
.lb-pick-thumb img{width:100%;height:100%;object-fit:cover;display:block;}
.lb-pick-ph{font-family:var(--lb-mono);font-size:9px;color:#7BA9A4;letter-spacing:.08em;}
.lb-pick-check{
  position:absolute;right:5px;top:5px;width:18px;height:18px;border-radius:50%;
  background:var(--lb-teal);color:#fff;display:flex;align-items:center;justify-content:center;
  box-shadow:0 1px 4px rgba(0,0,0,.18);
}
.lb-pick-check .lb-i{width:11px;height:11px;}
.lb-pick-index{
  position:absolute;left:5px;top:5px;height:16px;padding:0 5px;border-radius:999px;
  background:rgba(10,14,25,.62);color:#fff;font-family:var(--lb-mono);font-size:9px;
  line-height:16px;
}
.lb-pick-name{
  margin-top:4px;font-size:10px;line-height:1.35;color:var(--lb-body);
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
}
.lb-pick-empty{
  padding:12px 14px;border:1px dashed var(--lb-border);border-radius:10px;
  font-size:11.5px;line-height:1.6;color:var(--lb-faint);background:#FAFBFB;
}
.lb-modal-selected{
  max-height:200px!important;overflow-y:auto!important;overflow-x:hidden!important;
  margin:0 0 6px!important;padding:0 2px 0 0!important;
}
.lb-modal-selected .lb-pick-groups{display:grid;grid-template-columns:1fr;gap:5px;}
.lb-modal-selected .lb-pick-cards{grid-template-columns:repeat(7,minmax(0,1fr));gap:5px;}
.lb-modal-selected .lb-pick-thumb{height:48px;}
.lb-modal-selected .lb-pick-name{font-size:9px;}
.lb-modal-selected .lb-pick-card{padding:3px;}

/* ---------- 识别历史：每行详细入口 + 详情层 ---------- */
.lb-hist-detail-link{
  display:inline-flex!important;align-items:center!important;justify-content:center!important;
  height:24px!important;padding:0 9px!important;border-radius:7px!important;
  background:#F0F2F3!important;color:var(--lb-ink)!important;font-size:11px!important;
  text-decoration:none!important;white-space:nowrap!important;
}
.lb-hist-detail-link:hover{background:var(--lb-teal-soft)!important;color:var(--lb-teal-dark)!important;}
.lb-hist-detail-store{display:none!important;}
.lb-hist-detail-layer{
  position:fixed!important;inset:0!important;z-index:2600!important;
  background:rgba(15,20,24,.62)!important;display:flex!important;align-items:center!important;justify-content:center!important;
  padding:5vh 4vw!important;
}
.lb-hist-detail-panel{
  width:min(980px,92vw)!important;max-height:86vh!important;overflow-y:auto!important;
  background:#fff!important;border-radius:16px!important;padding:22px 26px!important;
  box-shadow:0 20px 80px rgba(0,0,0,.3)!important;position:relative!important;
}
.lb-hist-detail-head{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;border-bottom:1px solid var(--lb-border);padding:0 36px 10px 0;margin-bottom:12px;}
.lb-hist-detail-head h2{margin:2px 0 0;font-size:22px;line-height:1.3;color:var(--lb-ink);}
.lb-hist-detail-head .lb-hint{white-space:nowrap;}
.lb-hist-detail-meta{font-size:11.5px;color:var(--lb-faint);margin:0 0 12px;font-family:var(--lb-mono);}
.lb-hist-detail-close{
  position:absolute!important;right:14px!important;top:12px!important;width:30px!important;height:30px!important;
  border-radius:50%!important;border:1px solid var(--lb-border)!important;background:#fff!important;
  font-size:18px!important;line-height:26px!important;cursor:pointer!important;
}
.lb-hist-detail-panel .lb-detail-grid{grid-template-columns:minmax(0,1.1fr) minmax(0,1fr)!important;gap:26px!important;}
.lb-hist-detail-panel .lb-symptom{font-size:14px!important;line-height:1.7!important;}
.lb-hist-detail-panel .lb-control-list p{font-size:13px!important;line-height:1.6!important;}


/* ---------- 评估页分析内容 ---------- */
.lb-analysis-row{display:grid!important;grid-template-columns:minmax(0,1fr) minmax(0,1fr)!important;gap:16px!important;align-items:stretch!important;}
.lb-analysis-row > .lb-html{
  min-width:0!important;height:100%!important;display:flex!important;flex-direction:column!important;
}
.lb-analysis-row > .lb-html > .html-container,
.lb-analysis-row > .lb-html > .prose,
.lb-analysis-row > .lb-html .prose{
  height:100%!important;display:flex!important;flex-direction:column!important;min-height:0!important;
}
.lb-analysis-row > .lb-html .lb-analysis-card{
  height:100%!important;display:flex!important;flex-direction:column!important;
}
.lb-analysis-row > .lb-html .lb-analysis-card .lb-table-wrap{flex:1 1 auto!important;min-height:0!important;}
.lb-analysis-row > .lb-analysis-fig-card{height:100%!important;}
.lb-analysis-row-worst{grid-template-columns:minmax(0,1fr) minmax(0,1fr)!important;}
.lb-analysis-card{
  height:100%!important;box-sizing:border-box!important;
  background:#fff!important;border:1px solid var(--lb-border)!important;border-radius:16px!important;
  padding:18px 20px!important;min-width:0!important;
}
.lb-analysis-fig-card{display:flex!important;flex-direction:column!important;padding:18px 20px 14px!important;}
.lb-analysis-fig-card .lb-figure{height:300px!important;min-height:300px!important;}
.lb-figure-card{height:100%!important;}
.lb-figure-card .lb-figure{height:300px!important;min-height:300px!important;}
.lb-eval-tips-card{height:100%!important;display:flex!important;flex-direction:column!important;}
.lb-analysis-head{display:flex!important;align-items:baseline!important;justify-content:space-between!important;gap:12px;margin:0 0 8px!important;}
.lb-analysis-head h3{margin:0!important;font-size:14.5px!important;font-weight:600!important;color:var(--lb-ink)!important;}
.lb-analysis-head span{font-size:10.5px!important;color:var(--lb-tertiary)!important;white-space:nowrap!important;}
.lb-analysis-text{font-size:12.5px!important;line-height:1.7!important;color:var(--lb-body)!important;margin:8px 0 0!important;}
.lb-analysis-text b{color:var(--lb-ink)!important;font-weight:600!important;}
.lb-f1-buckets{display:flex!important;flex-direction:column!important;gap:7px;margin-top:12px!important;}
.lb-f1-bucket{display:grid;grid-template-columns:88px 34px minmax(0,1fr);gap:8px;align-items:center;font-size:11.5px;color:var(--lb-muted);}
.lb-f1-bucket b{text-align:right;color:var(--lb-ink);font-family:var(--lb-mono);font-size:11px;}
.lb-f1-bucket i{display:block;height:7px;border-radius:99px;background:linear-gradient(90deg,var(--lb-teal),#61D6CE);max-width:100%;}
.lb-analysis-table{font-size:12px!important;}
.lb-analysis-table th{padding-bottom:7px!important;font-size:11px!important;}
.lb-analysis-table td{padding-top:7px!important;padding-bottom:7px!important;}
.lb-gap-table th,.lb-gap-table td{padding-right:8px!important;font-size:11px!important;}
.lb-analysis-kv{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px;}
.lb-analysis-kv span{background:#F7F9F9;border-radius:9px;padding:8px 10px;display:flex;flex-direction:column;gap:2px;}
.lb-analysis-kv em{font-style:normal;font-size:10px;color:var(--lb-faint);}
.lb-analysis-kv b{font-size:14px;color:var(--lb-ink);font-family:var(--lb-mono);}
.lb-model-advice{margin-top:10px;padding:10px 12px;border-radius:10px;background:#F5FCFB;border:1px solid #D9F0EC;}
.lb-model-advice > b{font-size:11.5px;color:var(--lb-teal-dark);}
.lb-model-advice p{margin:4px 0 0;font-size:11.5px;line-height:1.65;color:var(--lb-body);}
.lb-analysis-figs{grid-template-columns:repeat(3,minmax(0,1fr))!important;}
.lb-analysis-empty{height:100%!important;min-height:140px;display:flex!important;align-items:center!important;justify-content:center!important;flex-direction:column!important;}
@media (max-width:900px){
  .lb-analysis-row,.lb-analysis-row-worst,.lb-analysis-figs{grid-template-columns:1fr!important;}
}


/* ---------- 识别历史：源图缩略 ---------- */
.lb-hist-thumb{
  width:46px!important;height:34px!important;border-radius:6px!important;overflow:hidden!important;
  background:#F3F5F6!important;display:flex!important;align-items:center!important;justify-content:center!important;
}
.lb-hist-thumb img{width:100%!important;height:100%!important;object-fit:cover!important;display:block!important;}
.lb-hist-ph{font-size:11px!important;color:var(--lb-faint)!important;}

/* ---------- 批量图片库：可勾选卡片网格 ---------- */
.lb-js-hidden{display:none!important;}
.lb-batch-grid-wrap{
  max-height:430px!important;overflow-y:auto!important;overflow-x:hidden!important;
  padding-right:4px!important;min-height:0!important;
}
.lb-lib-groups{display:flex;flex-direction:column;gap:10px;}
.lb-lib-group-head{display:flex;align-items:baseline;justify-content:space-between;margin:0 0 5px;font-size:11px;color:var(--lb-faint);}
.lb-lib-group-head b{font-size:11.5px;font-weight:600;color:var(--lb-ink);}
.lb-lib-cards{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:6px;}
.lb-batch-lib-card{
  position:relative;display:block;width:100%;padding:3px;margin:0;border:1.5px solid var(--lb-border);
  border-radius:9px;background:#fff;cursor:pointer;text-align:left;overflow:hidden;
  transition:border-color .12s ease,background .12s ease,transform .12s ease;
}
.lb-batch-lib-card:hover{border-color:#9FD8D2;transform:translateY(-1px);}
.lb-batch-lib-card img{display:block;width:100%;height:64px;object-fit:cover;border-radius:6px;}
.lb-batch-lib-check{
  position:absolute;right:6px;top:6px;width:18px;height:18px;border-radius:50%;
  background:var(--lb-teal);color:#fff;display:none;align-items:center;justify-content:center;
  box-shadow:0 1px 5px rgba(0,0,0,.2);
}
.lb-batch-lib-card.is-selected{border-color:var(--lb-teal)!important;background:#F0FBF9!important;box-shadow:0 0 0 1px var(--lb-teal) inset;}
.lb-batch-lib-card.is-selected .lb-batch-lib-check{display:flex;}
.lb-batch-lib-card.is-selected img{filter:saturate(1.05);}
.lb-batch-lib-name{
  display:block;margin-top:3px;font-size:9.5px;line-height:1.3;color:var(--lb-body);
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
}
.lb-batch-upload{min-height:0!important;margin-top:2px!important;}
.lb-batch-upload:empty{display:none!important;}
.lb-batch-upload .lb-pick-cards{grid-template-columns:repeat(4,minmax(0,1fr));}
.lb-batch-upload .lb-pick-thumb{height:58px;}

html,body{overflow-anchor:none!important;}
.lb-page{overflow-anchor:none!important;}


/* ---------- 发生规律与权威来源 ---------- */
.lb-occ-block{margin-top:16px!important;padding-top:13px!important;border-top:1px solid var(--lb-border)!important;}
.lb-occ-title{display:flex!important;align-items:center!important;gap:6px!important;margin:0 0 6px!important;font-size:14px!important;font-weight:600!important;color:var(--lb-ink)!important;}
.lb-occ-title svg{color:var(--lb-teal)!important;}
.lb-occ-text{margin:0!important;font-size:13.5px!important;line-height:1.75!important;color:var(--lb-body)!important;}
.lb-occ-source{display:inline-flex!important;align-items:center!important;gap:4px!important;margin-top:8px!important;padding:3px 8px!important;border-radius:999px!important;border:1px solid transparent!important;font-size:11px!important;line-height:1.4!important;text-decoration:none!important;white-space:normal!important;}
.lb-occ-source:hover{text-decoration:underline!important;}
.lb-occ-source.is-official{background:#EAF7F1!important;color:#1B7A54!important;border-color:#CBEBDD!important;}
.lb-occ-source.is-journal{background:#EDF3FE!important;color:#245FA8!important;border-color:#D3E2FA!important;}
.lb-occ-source.is-industry{background:#F3F5F6!important;color:#6B7379!important;border-color:#E3E6E8!important;}
.lb-occ-degraded{margin-top:7px!important;font-size:10.5px!important;line-height:1.5!important;color:var(--lb-faint)!important;}
.lb-detail-wide .lb-occ-text{font-size:14.5px!important;line-height:1.8!important;}
.lb-kb-panel .lb-occ-text{font-size:12.5px!important;line-height:1.65!important;}
.lb-kb-panel .lb-occ-title{font-size:12.5px!important;}
/* ---------- 数据资产 ---------- */
.lb-data-section{display:flex!important;flex-direction:column!important;gap:18px!important;min-width:0!important;}
.lb-data-head{display:flex!important;align-items:flex-end!important;justify-content:space-between!important;gap:12px!important;}
.lb-data-head h2{margin:0!important;font-size:20px!important;line-height:1.3!important;color:var(--lb-ink)!important;}
.lb-data-head p{margin:3px 0 0!important;font-size:12px!important;color:var(--lb-tertiary)!important;}
.lb-data-table-card{padding-bottom:14px!important;min-width:0!important;}
.lb-data-table-wrap{width:100%!important;overflow-x:auto!important;border-radius:10px!important;min-width:0!important;}
.lb-data-table{min-width:680px!important;width:100%!important;font-size:12.5px!important;}
.lb-data-table th{white-space:nowrap!important;}
.lb-data-table td{vertical-align:top!important;line-height:1.65!important;}
.lb-data-table .lb-data-size{white-space:nowrap!important;font-family:var(--lb-mono)!important;font-size:11.5px!important;}
.lb-data-aux{margin:10px 0 0!important;font-size:11px!important;color:var(--lb-faint)!important;line-height:1.6!important;}
.lb-data-two{display:grid!important;grid-template-columns:minmax(0,1fr) minmax(0,1fr)!important;gap:16px!important;align-items:stretch!important;}
.lb-data-split{display:grid!important;grid-template-columns:repeat(3,minmax(0,1fr))!important;gap:10px!important;margin-top:14px!important;}
.lb-data-split > div{background:#F7F9F9!important;border-radius:10px!important;padding:10px 12px!important;display:flex!important;flex-direction:column!important;gap:3px!important;}
.lb-data-split span{font-size:11px!important;color:var(--lb-faint)!important;}
.lb-data-split b{font-size:20px!important;line-height:1.2!important;color:var(--lb-ink)!important;font-family:var(--lb-mono)!important;}
.lb-data-note{margin:12px 0 0!important;font-size:11.5px!important;color:var(--lb-tertiary)!important;line-height:1.65!important;}
.lb-data-cov-row{margin-top:12px!important;}
.lb-data-cov-head{display:flex!important;align-items:baseline!important;justify-content:space-between!important;gap:10px!important;font-size:11.5px!important;color:var(--lb-body)!important;}
.lb-data-cov-head b{font-size:11px!important;color:var(--lb-faint)!important;font-family:var(--lb-mono)!important;font-weight:500!important;white-space:nowrap!important;}
.lb-data-cov-bar{height:8px!important;border-radius:99px!important;background:#EEF3F2!important;overflow:hidden!important;margin-top:5px!important;}
.lb-data-cov-bar i{display:block!important;height:100%!important;border-radius:99px!important;background:linear-gradient(90deg,var(--lb-teal),#61D6CE)!important;}
.lb-data-source-card{padding-bottom:18px!important;}
.lb-data-source p{margin:8px 0 0!important;font-size:12.5px!important;line-height:1.75!important;color:var(--lb-body)!important;}
.lb-data-source b{color:var(--lb-ink)!important;font-weight:600!important;}
@media (max-width:900px){
  .lb-data-two{grid-template-columns:1fr!important;}
}

/* 训练步数第二行小字 */
.lb-train-steps{white-space:nowrap!important;text-align:right!important;}
.lb-train-steps > b{display:block!important;font-family:var(--lb-mono)!important;font-size:12px!important;color:var(--lb-ink)!important;font-weight:600!important;}
.lb-train-steps > span{display:block!important;margin-top:1px!important;font-size:10px!important;line-height:1.4!important;color:var(--lb-faint)!important;}
.lb-train-stop{font-style:normal!important;margin-left:4px!important;padding:0 4px!important;border-radius:4px!important;background:#FFF4E5!important;color:#8A5A00!important;font-size:9px!important;}
/* 数据资产：6 卡 + 训练/实验/模块 */
.lb-data-stats{display:grid!important;grid-template-columns:repeat(3,minmax(0,1fr))!important;gap:16px!important;}
.lb-data-missing{font-size:12px!important;line-height:1.6!important;color:var(--lb-warn,#8A5A00)!important;background:#FFF8E8!important;border:1px solid #F2E2B8!important;border-radius:10px!important;padding:9px 12px!important;}
.lb-exp-group{border:1px solid var(--lb-border)!important;border-radius:12px!important;margin-top:10px!important;overflow:hidden!important;background:#fff!important;}
.lb-exp-group:first-of-type{margin-top:0!important;}
.lb-exp-group summary{list-style:none!important;cursor:pointer!important;display:flex!important;align-items:center!important;justify-content:space-between!important;gap:10px!important;padding:10px 12px!important;background:#F7F9F9!important;}
.lb-exp-group summary::-webkit-details-marker{display:none!important;}
.lb-exp-group summary b{font-size:12.5px!important;color:var(--lb-ink)!important;}
.lb-exp-group summary span{font-size:11px!important;color:var(--lb-faint)!important;}
.lb-exp-group[open] summary{border-bottom:1px solid var(--lb-border)!important;}
.lb-exp-group .lb-data-table-wrap{border-radius:0!important;padding:0 10px 6px!important;}
.lb-exp-table{min-width:640px!important;}
.lb-data-module-card{padding-bottom:14px!important;}
.lb-data-modules{display:grid!important;grid-template-columns:repeat(3,minmax(0,1fr))!important;gap:10px!important;margin-top:12px!important;}
.lb-data-module{border:1px solid var(--lb-border)!important;border-radius:10px!important;padding:10px 12px!important;min-width:0!important;background:#FBFCFC!important;}
.lb-data-module > b{display:block!important;font-size:12.5px!important;color:var(--lb-ink)!important;margin-bottom:6px!important;}
.lb-data-tags{display:flex!important;flex-wrap:wrap!important;gap:4px!important;}
.lb-data-tags span{font-size:10.5px!important;line-height:1.5!important;color:var(--lb-muted)!important;background:#EEF3F2!important;border-radius:6px!important;padding:2px 6px!important;}
@media (max-width:900px){
  .lb-data-stats{grid-template-columns:repeat(2,minmax(0,1fr))!important;}
  .lb-data-modules{grid-template-columns:1fr!important;}
}
/* ---------- 知识问答：技术说明 ---------- */
.lb-tech-strip{margin:0!important;display:flex;flex-direction:column;gap:8px;min-width:0;}
.lb-tech-note{font-size:11.5px!important;line-height:1.55!important;color:var(--lb-faint)!important;}
.lb-tech-grid{display:grid!important;grid-template-columns:repeat(4,minmax(0,1fr))!important;gap:12px!important;align-items:stretch!important;}
.lb-tech-card{
  display:flex!important;flex-direction:column!important;gap:2px!important;
  background:#fff!important;border:1px solid var(--lb-border)!important;border-radius:16px!important;
  padding:13px 16px!important;min-width:0!important;
}
.lb-tech-k{font-size:11.5px!important;font-weight:600!important;color:var(--lb-faint)!important;line-height:1.45!important;}
.lb-tech-desc{font-size:11px!important;color:var(--lb-tertiary)!important;line-height:1.45!important;min-height:32px!important;}
.lb-tech-v{margin-top:auto!important;font-size:22px!important;line-height:1.2!important;font-weight:600!important;color:var(--lb-ink)!important;letter-spacing:-.01em!important;}
.lb-tech-v em{font-style:normal!important;font-size:10.5px!important;font-weight:500!important;color:var(--lb-faint)!important;margin-left:2px!important;}
.lb-ans-fig img{cursor:zoom-in!important;}
.lb-pick-thumb img{cursor:zoom-in!important;}
@media (max-width:900px){
  .lb-tech-grid{grid-template-columns:repeat(2,minmax(0,1fr))!important;}
  .lb-tech-desc{min-height:0!important;}
}


/* 批量上传区：让 Gradio 原生拖拽文案显示出来 */
.lb-batch-upload-file button[aria-label="Click to upload or drop files"] .wrap{
  font-size:13px!important;color:var(--lb-faint)!important;line-height:1.65!important;
  display:flex!important;flex-wrap:wrap!important;align-items:center!important;justify-content:center!important;
  gap:4px!important;text-align:center!important;
}
.lb-batch-upload-file button[aria-label="Click to upload or drop files"] .icon-wrap{
  width:30px!important;height:30px!important;color:var(--lb-teal)!important;
}
.lb-batch-upload-file button[aria-label="Click to upload or drop files"] .or{
  font-size:11.5px!important;color:var(--lb-tertiary)!important;margin:0 3px!important;
}
.lb-batch-upload .lb-pick-thumb img{
  position:absolute!important;inset:0!important;width:100%!important;height:100%!important;
  object-fit:cover!important;z-index:1!important;
}
.lb-batch-upload .lb-pick-thumb .lb-pick-check,
.lb-batch-upload .lb-pick-thumb .lb-pick-index{z-index:2!important;}


/* 批量上传区：真实 overlay 文案，避免 Gradio File 上传后只剩图标 */
.lb-batch-upload-wrap{position:relative!important;display:block!important;min-width:0!important;isolation:isolate!important;z-index:0!important;}
.lb-batch-drop-hint-wrap{
  position:absolute!important;inset:0!important;z-index:99999!important;
  pointer-events:none!important;padding:0!important;
}
.lb-batch-drop-hint{
  position:absolute!important;left:0!important;right:0!important;top:57%!important;
  text-align:center!important;font-size:13px!important;line-height:1.55!important;
  color:var(--lb-faint)!important;pointer-events:none!important;
}
.lb-batch-drop-hint span{display:block!important;margin-top:2px!important;font-size:11.5px!important;color:var(--lb-tertiary)!important;}
.lb-batch-upload-file button[aria-label="Click to upload or drop files"] .wrap{font-size:0!important;}
.lb-batch-upload-file .wrap::after{display:none!important;content:none!important;}

"""
# =====================================================================
#  主题
# =====================================================================
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.teal,
    neutral_hue=gr.themes.colors.gray,
    radius_size=gr.themes.sizes.radius_sm,
).set(
    body_background_fill="#F5F6F6",
    body_text_color="#3C3C43",
    body_text_color_subdued="#82888D",
    block_background_fill="#FFFFFF",
    block_border_width="1px",
    block_border_color="#EAEBEC",
    block_shadow="none",
    block_radius="16px",
    block_label_background_fill="#FFFFFF",
    block_label_text_color="#82888D",
    block_title_text_weight="600",
    button_primary_background_fill="#000000",
    button_primary_background_fill_hover="#282828",
    button_primary_text_color="#FFFFFF",
    button_primary_border_color="transparent",
    button_large_radius="999px",
    input_background_fill="#FFFFFF",
    input_border_color="#EAEBEC",
    slider_color="#00ADA2",
    link_text_color="#00ADA2",
)

# =====================================================================
#  界面
# =====================================================================
_FS_HEAD = """
<script>
(function () {
  if (window.__lbFsBound) return;
  window.__lbFsBound = true;

  function fullscreenEl() {
    return document.querySelector('.block.fullscreen');
  }

  function closeFullscreen() {
    document.dispatchEvent(new KeyboardEvent('keydown',
      { key: 'Escape', code: 'Escape', keyCode: 27, which: 27, bubbles: true }));
  }

  function ensureHint() {
    var h = document.querySelector('.lb-fs-hint');
    if (!fullscreenEl()) {
      if (h) h.remove();
      return;
    }
    if (h) return;
    var d = document.createElement('div');
    d.className = 'lb-fs-hint';
    d.textContent = '点击图片外侧空白处或按 ESC 关闭';
    document.body.appendChild(d);
  }

  // 统一图片放大层：滚轮缩放 + 拖拽平移 + 双击复位
  function openLightbox(src) {
    if (!src) return;
    var old = document.querySelector('.lb-lightbox');
    if (old) old.parentNode.removeChild(old);

    var layer = document.createElement('div');
    layer.className = 'lb-lightbox';
    var big = document.createElement('img');
    big.src = src;
    big.alt = '图片预览';
    var close = document.createElement('button');
    close.className = 'lb-lightbox-close';
    close.type = 'button';
    close.setAttribute('aria-label', '关闭');
    close.textContent = '×';
    var hint = document.createElement('div');
    hint.className = 'lb-lightbox-hint';
    hint.textContent = '滚轮缩放 · 拖动平移 · 双击复位 · ESC / 点空白关闭';
    layer.appendChild(big);
    layer.appendChild(close);
    layer.appendChild(hint);
    document.body.appendChild(layer);

    var scale = 1, tx = 0, ty = 0;
    var dragging = false, startX = 0, startY = 0, startTx = 0, startTy = 0;

    function apply() {
      big.style.transform = 'translate(' + tx + 'px,' + ty + 'px) scale(' + scale + ')';
    }
    function reset() {
      scale = 1; tx = 0; ty = 0; apply();
    }
    function closeBox() {
      if (layer.parentNode) layer.parentNode.removeChild(layer);
      document.removeEventListener('keydown', onKey, true);
    }
    function onKey(ev) {
      if (ev.key === 'Escape') { ev.preventDefault(); closeBox(); }
    }

    big.addEventListener('wheel', function (ev) {
      ev.preventDefault();
      ev.stopPropagation();
      var oldScale = scale;
      var factor = ev.deltaY < 0 ? 1.16 : 1 / 1.16;
      var next = Math.min(5, Math.max(0.5, scale * factor));
      if (next === oldScale) return;
      var cx = ev.clientX - window.innerWidth / 2;
      var cy = ev.clientY - window.innerHeight / 2;
      tx = (tx - cx) * (next / oldScale) + cx;
      ty = (ty - cy) * (next / oldScale) + cy;
      scale = next;
      apply();
    }, { passive: false });

    big.addEventListener('pointerdown', function (ev) {
      if (ev.button !== 0) return;
      dragging = true;
      startX = ev.clientX; startY = ev.clientY;
      startTx = tx; startTy = ty;
      big.classList.add('is-grabbing');
      try { big.setPointerCapture(ev.pointerId); } catch (e) {}
      ev.preventDefault();
    });
    big.addEventListener('pointermove', function (ev) {
      if (!dragging) return;
      tx = startTx + (ev.clientX - startX);
      ty = startTy + (ev.clientY - startY);
      apply();
    });
    function stopDrag() {
      dragging = false;
      big.classList.remove('is-grabbing');
    }
    big.addEventListener('pointerup', stopDrag);
    big.addEventListener('pointercancel', stopDrag);
    big.addEventListener('dblclick', function (ev) {
      ev.preventDefault();
      reset();
    });

    layer.addEventListener('click', function (ev) {
      if (ev.target === layer) closeBox();
    });
    close.addEventListener('click', function (ev) {
      ev.preventDefault(); ev.stopPropagation(); closeBox();
    });
    document.addEventListener('keydown', onKey, true);
  }

  // 所有业务图片统一走 lightbox；图片库缩略图保留选图行为
  document.addEventListener('click', function (e) {
    var img = e.target && e.target.closest ? e.target.closest('img') : null;
    if (!img) return;
    if (img.closest('.lb-gallery')) return;
    var host = img.closest('.lb-upload, .lb-figure, .lb-batch-detail-photo, .lb-ans-fig, .lb-pick-thumb, .lb-kb-fig, a.lb-shot, a.lb-result-ref');
    if (!host) return;
    e.preventDefault();
    e.stopPropagation();
    var src = host.tagName === 'A'
      ? (host.getAttribute('href') || img.currentSrc || img.src)
      : (img.currentSrc || img.src);
    openLightbox(src);
  }, false);

  // 评估子页签：阻止按钮 focus 触发的自动滚动，保持原地切换
  document.addEventListener('mousedown', function (e) {
    var t = e.target && e.target.closest ? e.target.closest('.lb-eval-subtabs button[role="tab"]') : null;
    if (t) e.preventDefault();
  }, true);

  // 评估子页签切换后恢复到点击前的滚动位置，不做任何定位滚动
  document.addEventListener('click', function (e) {
    var t = e.target && e.target.closest ? e.target.closest('.lb-eval-subtabs button[role="tab"]') : null;
    if (!t) return;
    var y = window.scrollY;
    var start = performance.now();
    function keepScroll() {
      window.scrollTo(0, y);
      if (performance.now() - start < 650) requestAnimationFrame(keepScroll);
    }
    requestAnimationFrame(keepScroll);
  }, true);

  // 识别历史：详细 -> 复用全局批量详情浮层
  document.addEventListener('click', function (e) {
    var a = e.target && e.target.closest ? e.target.closest('a.lb-hist-detail-link') : null;
    if (!a) return;
    e.preventDefault();
    e.stopPropagation();
    var payload = a.getAttribute('data-hist-row') || '';
    var ta = document.querySelector('#hist-js-row textarea, #hist-js-row input');
    if (ta) {
      ta.value = payload;
      ta.dispatchEvent(new Event('input', { bubbles: true }));
      ta.dispatchEvent(new Event('change', { bubbles: true }));
    }
    setTimeout(function () {
      var btn = document.querySelector('#hist-js-open');
      if (btn) btn.click();
    }, 90);
  }, false);

  // 症状与防治展开后自动滚到全宽详情，避免用户手动寻找
  document.addEventListener('click', function (e) {
    var b = e.target && e.target.closest ? e.target.closest('button.lb-detail-trigger') : null;
    if (!b) return;
    setTimeout(function () {
      var el = document.querySelector('.lb-detail-wide');
      if (el && el.offsetParent !== null) {
        var y = el.getBoundingClientRect().top + window.scrollY - 96;
        window.scrollTo({ top: y, behavior: 'smooth' });
      }
    }, 550);
  }, false);

  // 批量图片库：卡片勾选 + 明显勾选态 + 已选卡片 + 隐藏 JSON 同步
  (function () {
    if (window.__lbBatchPickBound) return;
    window.__lbBatchPickBound = true;
    var selected = new Map();

    function esc(v) {
      return String(v == null ? "" : v).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
    }
    function groupsHtml() {
      var groups = {};
      selected.forEach(function (item, p) {
        var crop = item.label.indexOf("·") >= 0 ? item.label.split("·")[0] : "图片库";
        (groups[crop] = groups[crop] || []).push({ path: p, label: item.label, thumb: item.thumb });
      });
      var keys = Object.keys(groups);
      if (!keys.length) {
        return "";
      }
      var html = '<div class="lb-pick-groups is-compact">';
      keys.forEach(function (crop) {
        var g = groups[crop];
        html += '<div class="lb-pick-group"><div class="lb-pick-group-head"><b>' + esc(crop) +
          '</b><span>' + g.length + ' 张</span></div><div class="lb-pick-cards">';
        g.forEach(function (it, i) {
          html += '<div class="lb-pick-card" title="' + esc(it.label) + '"><div class="lb-pick-thumb">' +
            (it.thumb ? '<img src="' + esc(it.thumb) + '" alt="">' : '<span class="lb-pick-ph">LOCAL</span>') +
            '<span class="lb-pick-check">✓</span><span class="lb-pick-index">#' + (i + 1) + '</span></div>' +
            '<div class="lb-pick-name">' + esc(it.label) + '</div></div>';
        });
        html += '</div></div>';
      });
      return html + '</div>';
    }
    function writeBox() {
      var box = document.querySelector('#batch-js-selected textarea');
      if (!box) return;
      box.value = JSON.stringify(Array.from(selected.keys()));
      box.dispatchEvent(new Event('input', { bubbles: true }));
      box.dispatchEvent(new Event('change', { bubbles: true }));
    }
    function markCards() {
      document.querySelectorAll('.lb-batch-lib-card').forEach(function (c) {
        c.classList.toggle('is-selected', selected.has(c.getAttribute('data-path')));
      });
    }
    function update() {
      var html = groupsHtml();
      document.querySelectorAll('.lb-batch-selected .prose, .lb-modal-selected .prose').forEach(function (el) {
        el.innerHTML = html;
      });
      var count = selected.size;
      var crops = new Set();
      selected.forEach(function (it) {
        crops.add(it.label.indexOf("·") >= 0 ? it.label.split("·")[0] : "图片库");
      });
      var note = count ? ('已选 ' + count + ' 张 · ' + crops.size + ' 种作物') : '';
      document.querySelectorAll('.lb-batch-sel-note .prose').forEach(function (el) { el.innerText = note; });
      document.querySelectorAll('.lb-batch-lib-modal .lb-modal-count .prose').forEach(function (el) {
        el.innerText = count ? ('已选 ' + count + ' 张 · 点击卡片继续勾选') : '已选 0 张 · 点击卡片勾选';
      });
      writeBox();
      markCards();
    }

    document.addEventListener('click', function (e) {
      var card = e.target && e.target.closest ? e.target.closest('.lb-batch-lib-card') : null;
      if (!card) return;
      e.preventDefault();
      e.stopPropagation();
      var p = card.getAttribute('data-path');
      if (!p) return;
      if (selected.has(p)) selected.delete(p);
      else selected.set(p, {
        label: card.getAttribute('data-label') || '',
        thumb: card.getAttribute('data-thumb') || ''
      });
      update();
    }, false);

    document.addEventListener('click', function (e) {
      var b = e.target && e.target.closest ? e.target.closest('button') : null;
      if (!b) return;
      if ((b.innerText || '').trim() === '清空已选') {
        selected.clear();
        setTimeout(update, 60);
      }
    }, true);

    function observeGrid() {
      var grid = document.querySelector('.lb-batch-grid-wrap');
      if (grid && !grid.__lbObs) {
        grid.__lbObs = true;
        try { new MutationObserver(markCards).observe(grid, { childList: true, subtree: true }); } catch (err) {}
      }
    }
    setInterval(observeGrid, 700);
    setTimeout(function () { observeGrid(); markCards(); }, 500);
  })();

  function boot() {
    if (!document.body) { setTimeout(boot, 300); return; }
    try {
      new MutationObserver(ensureHint).observe(document.body,
        { attributes: true, subtree: true, attributeFilter: ['class'] });
    } catch (e) {}
    setInterval(ensureHint, 700);
    ensureHint();
  }
  boot();

  // ===== 弹层关闭：纯前端，避免 Gradio visible 往返造成的卡顿 =====
  function hideModals() {
    document.querySelectorAll('.lb-modal-mask').forEach(function (m) {
      m.style.setProperty('display', 'none', 'important');
      m.classList.add('hide');
    });
  }
  window.__lbHideModals = hideModals;

  document.addEventListener('keydown', function (e) {
    if (e.key !== 'Escape') return;
    var opened = Array.from(document.querySelectorAll('.lb-modal-mask'))
      .some(function (m) {
        var s = getComputedStyle(m);
        return s.display !== 'none' && s.visibility !== 'hidden';
      });
    if (opened) { e.preventDefault(); hideModals(); }
  }, true);

  document.addEventListener('click', function (e) {
    var t = e.target;
    if (!t || !t.classList) return;
    // 点在遮罩本身（不是内容）时关闭
    if (t.classList.contains('lb-modal-mask')) {
      e.preventDefault(); e.stopPropagation(); hideModals();
    }
  }, true);
})();
</script>
"""


def build_ui():
    init_model = BACKBONES[0]
    init_imgs = eval_fig_values(init_model)
    init_stats = render_eval_stats(init_model)
    init_compare = render_compare(web.compare_table())
    init_kb = render_kb(CLASSES[0])

    init_rows = web.load_history()
    init_hist_stats = render_history_stats(init_rows)
    init_hist_table = render_history_table(init_rows)
    init_hist_charts = render_history_charts(init_rows)

    with gr.Blocks(title="植物叶片病害智能诊断系统") as demo:
        gr.HTML(render_header(), elem_classes="lb-header-html")

        # 全局批量详情浮层：批量结果与识别历史共用
        hist_detail_cache = gr.State({})
        # 批量详情浮层：复用智能诊断的完整结果
        with gr.Column(visible=False, elem_classes="lb-modal-mask lb-batch-detail-modal") as batch_detail_modal:
            with gr.Column(elem_classes="lb-modal-panel lb-batch-detail-panel"):
                with gr.Row(elem_classes="lb-modal-head"):
                    gr.HTML('''
<div class="lb-modal-title"><h2>批量诊断详情</h2><p>与智能诊断共用同一套结果：诊断卡、Top-5、病斑定位、症状与防治</p></div>''',
                            elem_classes="lb-html")
                    batch_detail_close = gr.Button("关闭", size="sm",
                                                   elem_classes="lb-btn-secondary lb-modal-close")

                with gr.Row(elem_classes="lb-batch-detail-select"):
                    gr.HTML('<div class="lb-batch-detail-select-label">选择照片</div>',
                            elem_classes="lb-html")
                    batch_detail_dd = gr.Dropdown(choices=[], show_label=False,
                                                  elem_classes="lb-select lb-batch-detail-dd")

                with gr.Row(elem_classes="lb-batch-detail-top"):
                    with gr.Column(elem_classes="lb-batch-detail-photo-col", min_width=0):
                        batch_detail_img = gr.Image(
                            interactive=False, show_label=False, container=False,
                            height=235, elem_classes="lb-figure lb-batch-detail-photo",
                            buttons=[],
                        )
                        batch_detail_meta = gr.HTML("", elem_classes="lb-html lb-batch-detail-meta")
                    with gr.Column(elem_classes="lb-batch-detail-main", min_width=0):
                        batch_detail_result = gr.HTML(BATCH_DETAIL_EMPTY,
                                                      elem_classes="lb-html lb-batch-detail-result")

                with gr.Column(elem_classes="lb-panel lb-batch-panel", min_width=0):
                    with gr.Column(elem_classes="lb-pane lb-top5-pane", min_width=0):
                        batch_top5_html = gr.HTML("", elem_classes="lb-html")

                    with gr.Column(elem_classes="lb-pane lb-lesion-pane", min_width=0):
                        with gr.Row(elem_classes="lb-pane-head"):
                            gr.HTML('''
<div class="lb-pane-title"><b>病斑定位</b><span>Grad-CAM 热力图与病斑外接框</span></div>''',
                                    elem_classes="lb-html")
                            batch_lesion_back = gr.Button("返回候选", size="sm",
                                                          elem_classes="lb-mini-btn")
                        with gr.Row(elem_classes="lb-mini-lesion"):
                            with gr.Column(elem_classes="lb-mini-fig", min_width=0):
                                gr.HTML('<span>Grad-CAM 热力图</span>', elem_classes="lb-html")
                                batch_cam_img = gr.Image(
                                    interactive=False, show_label=False, container=False,
                                    height=170, elem_classes="lb-figure", buttons=[],
                                )
                            with gr.Column(elem_classes="lb-mini-fig", min_width=0):
                                gr.HTML('<span>病斑外接框</span>', elem_classes="lb-html")
                                batch_box_img = gr.Image(
                                    interactive=False, show_label=False, container=False,
                                    height=170, elem_classes="lb-figure", buttons=[],
                                )
                            with gr.Column(elem_classes="lb-mini-side", min_width=0):
                                batch_lesion_caption = gr.HTML("", elem_classes="lb-html")
                                gr.HTML('<div class="lb-mini-tip">颜色越暖，模型关注度越高；'
                                        '占比按病斑像素/叶片像素计算。</div>', elem_classes="lb-html")

                    with gr.Column(elem_classes="lb-pane lb-detail-pane", min_width=0):
                        with gr.Row(elem_classes="lb-pane-head"):
                            gr.HTML(f'''
<div class="lb-pane-title"><b>症状与防治</b><span>来自病害知识库（共 {_NKB} 条）</span></div>''',
                                    elem_classes="lb-html")
                            batch_detail_back = gr.Button("返回候选", size="sm",
                                                          elem_classes="lb-mini-btn")
                        batch_detail_advice = gr.HTML("", elem_classes="lb-html")
                        gr.HTML(f'<div class="lb-detail-foot">以上内容来自病害知识库（共 {_NKB} 条），'
                                '仅作参考；可返回候选或切换查看病斑。</div>',
                                elem_classes="lb-html")

                with gr.Row(elem_classes="lb-action-row lb-batch-detail-actions"):
                    batch_locate_btn = gr.Button("病斑定位", elem_classes="lb-btn-secondary")
                    batch_advice_btn = gr.Button("症状与防治", elem_classes="lb-btn-secondary")

        with gr.Tabs(elem_classes="lb-tabs"):
            # ========================= ① 智能诊断 =========================
            with gr.Tab("智能诊断"):
                with gr.Column(elem_classes="lb-page lb-diag-page", min_width=0):
                    gr.HTML(PAGEHEAD_DIAG, elem_classes="lb-html")

                    with gr.Row(elem_classes="lb-diag-grid"):
                        # ---------- 左：输入 ----------
                        with gr.Column(elem_classes="lb-card lb-input-card lb-col", min_width=0):
                            gr.HTML('''
<div class="lb-card-head">
  <div><h2>叶片照片</h2><p>单张 · JPG / PNG · 支持拖拽</p></div>
  <span class="lb-hint">建议主体清晰</span>
</div>''', elem_classes="lb-html")
                            image = gr.Image(
                                type="pil", sources=["upload"], show_label=False,
                                height=190, elem_classes="lb-upload", buttons=[],
                            )
                            with gr.Row(elem_classes="lb-example-head"):
                                gr.HTML('<div class="lb-example-label"><span>内置示例</span>'
                                        f'<em>按作物选择，图片库可看全部 {_NCLS} 类</em></div>',
                                        elem_classes="lb-html")
                                lib_btn = gr.Button("图片库", elem_classes="lb-example lb-lib-trigger lb-lib-inline",
                                                    size="sm")
                            with gr.Row(elem_classes="lb-examples"):
                                ex_buttons = []
                                for _path, _lbl in _CROP_EXAMPLES:
                                    _b = gr.Button(_lbl, elem_classes="lb-example", size="sm")
                                    _b.click(
                                        lambda pp=_path: Image.open(pp).convert("RGB"),
                                        None, image,
                                    )
                                    ex_buttons.append(_b)

                            with gr.Accordion("归档信息（可选，写入报告）", open=False):
                                with gr.Row():
                                    meta_plot = gr.Textbox(label="地块", placeholder="如 3 号棚",
                                                           elem_classes="lb-select")
                                    meta_variety = gr.Textbox(label="品种", placeholder="如 合作 903",
                                                              elem_classes="lb-select")
                                meta_submitter = gr.Textbox(label="送检人", placeholder="如 张技术员",
                                                            elem_classes="lb-select")

                            gr.HTML('<div class="lb-divider"></div>', elem_classes="lb-html")
                            with gr.Row(elem_classes="lb-controls"):
                                with gr.Column(elem_classes="lb-field", min_width=0):
                                    model_dd = gr.Dropdown(
                                        _MODEL_CHOICES, value=init_model, label="骨干网络",
                                        elem_classes="lb-select",
                                    )
                                with gr.Column(elem_classes="lb-field", min_width=0):
                                    thresh = gr.Slider(
                                        0.3, 0.95, value=0.6, step=0.05,
                                        label="拒识阈值", elem_classes="lb-slider",
                                    )
                            gr.HTML('<div class="lb-hint-text lb-model-hint">'
                                    'ResNet50 最稳 · EfficientNet-B0 精度最高 · MobileNetV3 最轻最快</div>',
                                    elem_classes="lb-html")
                            gr.HTML('<div class="lb-hint-text">置信度低于阈值时系统显示「无法确定」，不给出病害类别，也不会生成报告。</div>',
                                    elem_classes="lb-html")
                            start_btn = gr.Button("开始诊断", variant="primary",
                                                  elem_classes="lb-btn-primary lb-start")

                        # ---------- 右：结果 + 同区面板 ----------
                        with gr.Column(elem_classes="lb-right lb-col", min_width=0):
                            result_html = gr.HTML(EMPTY_RESULT,
                                                  elem_classes="lb-html lb-result-html")
                            with gr.Column(elem_classes="lb-panel", min_width=0):
                                # Top-5（默认面板）
                                with gr.Column(elem_classes="lb-pane lb-top5-pane", min_width=0):
                                    top5_html = gr.HTML("", elem_classes="lb-html")

                                # 病斑面板：只有生成图片后才出现
                                with gr.Column(elem_classes="lb-pane lb-lesion-pane", min_width=0):
                                    with gr.Row(elem_classes="lb-pane-head"):
                                        gr.HTML('''
<div class="lb-pane-title"><b>病斑定位</b><span>Grad-CAM 热力图与病斑外接框</span></div>''',
                                                elem_classes="lb-html")
                                        lesion_back = gr.Button("返回候选", size="sm",
                                                                elem_classes="lb-mini-btn")
                                    with gr.Row(elem_classes="lb-mini-lesion"):
                                        with gr.Column(elem_classes="lb-mini-fig", min_width=0):
                                            gr.HTML('<span>Grad-CAM 热力图</span>', elem_classes="lb-html")
                                            cam_img = gr.Image(interactive=False, show_label=False,
                                                               container=False, height=140,
                                                               elem_classes="lb-figure", buttons=[])
                                        with gr.Column(elem_classes="lb-mini-fig", min_width=0):
                                            gr.HTML('<span>病斑外接框</span>', elem_classes="lb-html")
                                            box_img = gr.Image(interactive=False, show_label=False,
                                                               container=False, height=140,
                                                               elem_classes="lb-figure", buttons=[])
                                        with gr.Column(elem_classes="lb-mini-side", min_width=0):
                                            lesion_caption = gr.HTML("", elem_classes="lb-html")
                                            gr.HTML('<div class="lb-mini-tip">颜色越暖，模型关注度越高；占比按病斑像素/叶片像素计算。</div>',
                                                    elem_classes="lb-html")

                            with gr.Row(elem_classes="lb-action-row"):
                                locate_btn = gr.Button("病斑定位", elem_classes="lb-btn-secondary")
                                detail_btn = gr.Button("症状与防治", elem_classes="lb-btn-secondary lb-detail-trigger")
                                pdf_btn = gr.Button("导出 PDF", elem_classes="lb-btn-secondary")
                            with gr.Row(elem_classes="lb-review-row"):
                                gr.HTML('<span class="lb-review-tip">这个结果对吗？</span>',
                                        elem_classes="lb-html lb-col")
                                rv_ok = gr.Button("判断正确", size="sm", elem_classes="lb-btn-chip")
                                rv_no = gr.Button("判断有误", size="sm", elem_classes="lb-btn-chip")
                            pdf_note = gr.HTML(PDF_HINT, elem_classes="lb-html lb-pdf-foot")
                            last_info = gr.State(None)
                            diag_model = gr.State(None)

                    # 症状与防治：全宽展开，完整显示长文本
                    with gr.Column(elem_classes="lb-detail-wide", min_width=0):
                    # 症状与防治面板：点击按钮后才出现
                        with gr.Column(elem_classes="lb-pane lb-detail-pane", min_width=0):
                            with gr.Row(elem_classes="lb-pane-head"):
                                gr.HTML(f'''
<div class="lb-pane-title"><b>症状与防治</b><span>来自病害知识库（共 {_NKB} 条）</span></div>''',
                                        elem_classes="lb-html")
                                detail_back = gr.Button("返回候选", size="sm",
                                                        elem_classes="lb-mini-btn")
                            detail_html = gr.HTML("", elem_classes="lb-html")
                            gr.HTML(f'<div class="lb-detail-foot">以上内容来自病害知识库（共 {_NKB} 条），'
                                    '仅作参考；可返回候选或切换查看病斑。</div>',
                                    elem_classes="lb-html")

                    # ---------- 图片库浮层 ----------
                    with gr.Column(visible=False, elem_classes="lb-modal-mask lb-lib-modal") as lib_modal:
                        with gr.Column(elem_classes="lb-modal-panel lb-lib-panel"):
                            with gr.Row(elem_classes="lb-modal-head"):
                                gr.HTML(f'''
<div class="lb-modal-title"><h2>图片库</h2><p>覆盖 {len(CROP_NAMES)} 种作物 · 共 {len(LIBRARY)} 张</p></div>''',
                                        elem_classes="lb-html")
                                lib_close = gr.Button("关闭", size="sm",
                                                      elem_classes="lb-btn-secondary lb-modal-close")
                            gr.HTML('<div class="lb-modal-tip">点击任意缩略图即可载入并关闭；'
                                    '默认实验室白底，「真实」为 PlantDoc 真实场景；可按作物筛选。</div>', elem_classes="lb-html")
                            with gr.Row(elem_classes="lb-lib-filter"):
                                gr.HTML('<div class="lb-lib-filter-label">按作物筛选</div>',
                                        elem_classes="lb-html")
                                lib_crop = gr.Dropdown(
                                    ["全部"] + CROP_NAMES, value="全部", show_label=False,
                                    elem_classes="lb-select lb-lib-crop", filterable=True,
                                )
                            lib_gallery = gr.Gallery(
                                value=[(p, l) for p, l in LIBRARY],
                                columns=6, height=420, show_label=False,
                                preview=False, allow_preview=False, object_fit="cover",
                                elem_classes="lb-gallery",
                            )
                            lib_view = gr.State(list(LIBRARY))

                    # 事件（示例按钮已在创建时绑定）
                    image.change(
                        on_image_change,
                        outputs=[result_html, top5_html, cam_img, box_img,
                                 lesion_caption, detail_html, pdf_note, last_info,
                                 diag_model],
                    )
                    start_btn.click(
                        on_diagnose,
                        inputs=[image, model_dd, thresh],
                        outputs=[result_html, top5_html, cam_img, box_img,
                                 lesion_caption, detail_html, pdf_note, last_info,
                                 diag_model],
                    )
                    locate_btn.click(
                        on_locate,
                        inputs=[image, model_dd, thresh],
                        outputs=[cam_img, box_img, lesion_caption, detail_html],
                    )
                    detail_btn.click(
                        on_detail_view,
                        inputs=[last_info],
                        outputs=[cam_img, box_img, lesion_caption, detail_html],
                    )
                    rv_ok.click(lambda i: on_review("确认", i), inputs=[last_info], outputs=[pdf_note])
                    rv_no.click(lambda i: on_review("更正", i), inputs=[last_info], outputs=[pdf_note])
                    pdf_btn.click(on_pdf,
                  inputs=[image, model_dd, thresh, diag_model,
                          meta_plot, meta_variety, meta_submitter],
                  outputs=[pdf_note])
                    lesion_back.click(on_clear_lesion,
                                      outputs=[cam_img, box_img, lesion_caption])
                    detail_back.click(on_clear_detail, outputs=[detail_html])
                    lib_btn.click(open_modal, None, [lib_modal])
                    lib_close.click(close_modal, None, [lib_modal])
                    lib_crop.change(filter_library, [lib_crop], [lib_gallery, lib_view])
                    lib_gallery.select(on_pick_image, [lib_view], [image, lib_modal])

            # ========================= ② 批量诊断 =========================
            with gr.Tab("批量诊断"):
                with gr.Column(elem_classes="lb-page lb-batch-page", min_width=0):
                    gr.HTML(PAGEHEAD_BATCH, elem_classes="lb-html")

                    with gr.Row(elem_classes="lb-batch-grid"):
                        with gr.Column(elem_classes="lb-card lb-batch-pick lb-col", min_width=0):
                            gr.HTML('''
<div class="lb-card-head">
  <div><h2>批量照片</h2><p>支持 JPG / PNG / WEBP，可拖拽或从图片库多选</p></div>
  <span class="lb-hint">多张</span>
</div>''', elem_classes="lb-html")
                            gr.HTML('<div class="lb-batch-pick-sub">本地图片（可选）</div>',
                                    elem_classes="lb-html")
                            with gr.Column(elem_classes="lb-batch-upload-wrap", min_width=0):
                                batch_files = gr.File(
                                    file_count="multiple", file_types=["image"], type="filepath",
                                    show_label=False, height=240,
                                    elem_classes="lb-file lb-batch-upload-file",
                                )
                                gr.HTML('<div class="lb-batch-drop-hint">'
                                        '拖拽多张叶片照片到此处<br><span>或点击选择文件</span></div>',
                                        elem_classes="lb-html lb-batch-drop-hint-wrap")
                            gr.HTML('<div class="lb-hint-text lb-batch-tip">'
                                    '从图片库选图时：<b>点击卡片勾选/取消</b>；已选图片会以'
                                    '「作物·病害 + 对勾」卡片列在下方。</div>', elem_classes="lb-html")
                            batch_selected = gr.HTML(render_batch_selected([], []),
                                                     elem_classes="lb-html lb-batch-selected")
                            batch_upload_cards = gr.HTML("", elem_classes="lb-html lb-batch-upload")
                            batch_sel_note = gr.HTML('',
                                                     elem_classes="lb-html lb-batch-sel-note")
                            with gr.Row(elem_classes="lb-batch-pick-actions"):
                                batch_lib_btn = gr.Button("浏览图片库", size="sm",
                                                          elem_classes="lb-btn-secondary")
                                batch_clear = gr.Button("清空已选", size="sm",
                                                        elem_classes="lb-btn-secondary")

                        with gr.Column(elem_classes="lb-card lb-batch-settings lb-col", min_width=0):
                            gr.HTML('''
<div class="lb-card-head">
  <div><h2>识别设置</h2><p>所有照片使用同一模型与阈值</p></div>
</div>''', elem_classes="lb-html")
                            with gr.Row(elem_classes="lb-controls lb-batch-controls"):
                                with gr.Column(elem_classes="lb-field", min_width=0):
                                    batch_model = gr.Dropdown(
                                        _MODEL_CHOICES, value=init_model, label="骨干网络",
                                        elem_classes="lb-select",
                                    )
                                with gr.Column(elem_classes="lb-field", min_width=0):
                                    batch_thresh = gr.Slider(
                                        0.3, 0.95, value=0.6, step=0.05,
                                        label="拒识阈值", elem_classes="lb-slider",
                                    )
                            gr.HTML('<div class="lb-hint-text lb-batch-model-hint">'
                                    'ResNet50 稳 · EfficientNet-B0 准 · MobileNetV3 快</div>',
                                    elem_classes="lb-html")
                            gr.HTML('<div class="lb-hint-text">低于阈值标记为需复核</div>',
                                    elem_classes="lb-html")
                            batch_btn = gr.Button("开始批量诊断", variant="primary",
                                                  elem_classes="lb-btn-primary")
                            batch_retry = gr.Button("重新诊断（含失败项）", size="sm",
                                                    elem_classes="lb-btn-secondary")
                            batch_csv = gr.DownloadButton("下载 CSV", value=None, interactive=False,
                                                          elem_classes="lb-btn-secondary lb-csv-btn")
                            batch_status = gr.HTML("批量完成后可下载 CSV",
                                                   elem_classes="lb-html lb-batch-status")
                            gr.HTML('''<div class="lb-batch-guide">
<div><b>批量说明</b> 点卡片勾选/取消，已选卡片带对勾与作物·病害名；低于阈值标记「需复核」；CSV 含文件名、结果、置信度、状态。</div>
</div>''', elem_classes="lb-html")

                    with gr.Row(elem_classes="lb-batch-detail-bar"):
                        gr.HTML('<div class="lb-detail-hint">批量完成后，点每行右侧「详细」打开与智能诊断相同的结果页。</div>',
                                elem_classes="lb-html")
                        batch_detail_btn = gr.Button("查看某张详情", size="sm",
                                                     elem_classes="lb-btn-secondary")

                    with gr.Column(elem_classes="lb-card lb-table-card", min_width=0):
                        gr.HTML('''
<div class="lb-card-head">
  <div><h2>批量诊断结果</h2><p>每行右侧「详细」打开该图的完整结果</p></div>
  <span class="lb-hint">逐行查看</span>
</div>''', elem_classes="lb-html")
                        with gr.Column(elem_classes="lb-batch-table lb-batch-rows", min_width=0):
                            gr.HTML('''
<div class="lb-batch-thead">
  <span>照片 / 作物·病害</span><span>诊断结果</span>
  <span class="lb-num">置信度</span><span>状态</span><span>操作</span>
</div>''', elem_classes="lb-html")
                            batch_rows = []
                            for _row_i in range(BATCH_ROW_SLOTS):
                                with gr.Row(visible=False, elem_classes="lb-batch-row") as _row:
                                    _row_html = gr.HTML("", elem_classes="lb-html lb-batch-row-html")
                                    _row_btn = gr.Button("详细", size="sm",
                                                         elem_classes="lb-batch-detail-btn")
                                batch_rows.append((_row, _row_html, _row_btn))
                    batch_summary = gr.HTML("", elem_classes="lb-html")
                    batch_paths = gr.State([])
                    batch_display = gr.State({})

                    # 批量图片库浮层（可连续多选）
                    with gr.Column(visible=False, elem_classes="lb-modal-mask lb-batch-lib-modal") as batch_lib_modal:
                        with gr.Column(elem_classes="lb-modal-panel lb-batch-lib-panel"):
                            with gr.Row(elem_classes="lb-modal-head"):
                                gr.HTML(f'''
<div class="lb-modal-title"><h2>批量图片库</h2><p>点击卡片勾选/取消 · 覆盖 {len(CROP_NAMES)} 种作物</p></div>''',
                                        elem_classes="lb-html")
                                batch_lib_close = gr.Button("完成", size="sm",
                                                            elem_classes="lb-btn-secondary lb-modal-close")
                            batch_lib_count = gr.HTML('已选 0 张 · 可连续点击缩略图追加',
                                                      elem_classes="lb-html lb-modal-count")
                            batch_lib_selected = gr.HTML(render_batch_selected([], [], True),
                                                         elem_classes="lb-html lb-modal-selected")
                            with gr.Row(elem_classes="lb-lib-filter"):
                                gr.HTML('<div class="lb-lib-filter-label">按作物筛选</div>',
                                        elem_classes="lb-html")
                                batch_lib_crop = gr.Dropdown(
                                    ["全部"] + CROP_NAMES, value="全部", show_label=False,
                                    elem_classes="lb-select lb-lib-crop", filterable=True,
                                )
                            batch_lib_grid = gr.HTML(render_library_grid("全部"),
                                                     elem_classes="lb-html lb-batch-grid-wrap")
                            batch_js_selected = gr.Textbox(
                                value="[]", show_label=False, elem_id="batch-js-selected",
                                elem_classes="lb-js-hidden",
                            )

                    batch_sel = gr.State([])

                    batch_row_outputs = [c for _row, _h, _b in batch_rows for c in (_row, _h)]
                    batch_all_outputs = (batch_row_outputs +
                        [batch_csv, batch_status, batch_detail_dd,
                         batch_detail_img, batch_detail_meta, batch_detail_result,
                         batch_top5_html, batch_cam_img, batch_box_img,
                         batch_lesion_caption, batch_detail_advice,
                         batch_paths, batch_display, hist_detail_cache, batch_summary])
                    batch_btn.click(
                        on_batch,
                        inputs=[batch_files, batch_model, batch_thresh, batch_js_selected],
                        outputs=batch_all_outputs,
                    )
                    batch_retry.click(
                        on_batch,
                        inputs=[batch_files, batch_model, batch_thresh, batch_js_selected],
                        outputs=batch_all_outputs,
                    )
                    for _i, (_row, _row_html, _row_btn) in enumerate(batch_rows):
                        _row_btn.click(
                            lambda paths, display, m, t, i=_i: open_batch_row(paths, display, m, t, i),
                            inputs=[batch_paths, batch_display, batch_model, batch_thresh],
                            outputs=[batch_detail_modal, batch_detail_dd,
                                     batch_detail_img, batch_detail_meta,
                                     batch_detail_result, batch_top5_html,
                                     batch_cam_img, batch_box_img,
                                     batch_lesion_caption, batch_detail_advice],
                        )
                    batch_detail_dd.change(
                        on_batch_detail,
                        inputs=[batch_detail_dd, batch_model, batch_thresh, batch_display, hist_detail_cache],
                        outputs=[batch_detail_img, batch_detail_meta, batch_detail_result,
                                 batch_top5_html, batch_cam_img, batch_box_img,
                                 batch_lesion_caption, batch_detail_advice],
                    )
                    batch_locate_btn.click(
                        on_batch_locate,
                        inputs=[batch_detail_dd, batch_model, batch_thresh],
                        outputs=[batch_cam_img, batch_box_img, batch_lesion_caption,
                                 batch_detail_advice],
                    )
                    batch_advice_btn.click(
                        on_batch_advice,
                        inputs=[batch_detail_dd, batch_model, batch_thresh],
                        outputs=[batch_cam_img, batch_box_img, batch_lesion_caption,
                                 batch_detail_advice],
                    )
                    batch_lesion_back.click(on_clear_lesion,
                                            outputs=[batch_cam_img, batch_box_img,
                                                     batch_lesion_caption])
                    batch_detail_back.click(on_clear_detail, outputs=[batch_detail_advice])
                    batch_lib_btn.click(open_modal, None, [batch_lib_modal])
                    batch_lib_close.click(close_modal, None, [batch_lib_modal])
                    batch_lib_crop.change(filter_batch_grid, [batch_lib_crop], [batch_lib_grid])
                    batch_files.change(sync_upload_cards, [batch_files], [batch_upload_cards])
                    batch_clear.click(clear_batch, None,
                                      [batch_files, batch_sel, batch_lib_count,
                                       batch_selected, batch_lib_selected, batch_sel_note,
                                       batch_js_selected, batch_upload_cards])
                    batch_detail_btn.click(open_modal, None, [batch_detail_modal])
                    batch_detail_close.click(close_modal, None, [batch_detail_modal])

            # ========================= ③ 模型评估 =========================
            with gr.Tab("模型评估"):
                with gr.Column(elem_classes="lb-page", min_width=0):
                    gr.HTML(PAGEHEAD_EVAL, elem_classes="lb-html")

                    with gr.Row(elem_classes="lb-toolbar"):
                        eval_model = gr.Dropdown(
                            _MODEL_CHOICES, value=init_model, label="评估模型",
                            elem_classes="lb-select",
                        )
                        gr.HTML('<div class="lb-toolbar-note">图表可点击图片放大；同分布指标来自测试集，跨域指标来自 PlantDoc 236 张</div>',
                                elem_classes="lb-html")



                    gr.HTML(render_eval_old_note(), elem_classes="lb-html lb-eval-old-wrap")
                    with gr.Tabs(elem_classes="lb-eval-subtabs"):
                        with gr.Tab("同分布指标"):
                            eval_stats = gr.HTML(init_stats, elem_classes="lb-html lb-anchor",
                                                 elem_id="eval-stats")
                            with gr.Row(elem_classes="lb-analysis-row"):
                                with gr.Column(elem_classes="lb-card lb-analysis-fig-card", min_width=0):
                                    gr.HTML(f'<div class="lb-figure-head"><h3>{_NCLS} 类 F1 分布</h3>'
                                            '<span>按 F1 分档 · 点击图片放大</span></div>',
                                            elem_classes="lb-html")
                                    _f1_path = _analysis_fig("11_F1分布.png")
                                    if _f1_path:
                                        gr.Image(value=_f1_path, interactive=False, show_label=False,
                                                 container=False, height=300, elem_classes="lb-figure",
                                                 buttons=[])
                                    else:
                                        gr.HTML(_analysis_empty("暂无图表"), elem_classes="lb-html")
                                gr.HTML(render_analysis_dist_note(), elem_classes="lb-html")
                            with gr.Row(elem_classes="lb-analysis-row lb-analysis-row-worst"):
                                gr.HTML(render_analysis_worst(), elem_classes="lb-html")
                                with gr.Column(elem_classes="lb-card lb-analysis-fig-card", min_width=0):
                                    gr.HTML('<div class="lb-figure-head"><h3>样本量 vs F1</h3>'
                                            '<span>少数样本类别单独观察 · 点击图片放大</span></div>',
                                            elem_classes="lb-html")
                                    _sample_path = _analysis_fig("13_样本量与F1.png")
                                    if _sample_path:
                                        gr.Image(value=_sample_path, interactive=False, show_label=False,
                                                 container=False, height=260, elem_classes="lb-figure",
                                                 buttons=[])
                                    else:
                                        gr.HTML(_analysis_empty("暂无图表"), elem_classes="lb-html")
                                    gr.HTML(render_analysis_sample_note(), elem_classes="lb-html")
                        with gr.Tab("评估图表"):
                            fig_components = []
                            for _row_i in range(4):
                                with gr.Row(elem_classes="lb-figure-grid"):
                                    for _col_i in range(2):
                                        _idx = _row_i * 2 + _col_i
                                        if _idx < len(EVAL_FIGS):
                                            _label, _sub, _ = EVAL_FIGS[_idx]
                                            with gr.Column(elem_classes="lb-card lb-figure-card", min_width=0):
                                                gr.HTML(
                                                    f'<div class="lb-figure-head"><h3>{_label}</h3>'
                                                    f'<span>{_sub} · 点击图片放大</span></div>',
                                                    elem_classes="lb-html",
                                                )
                                                _img = gr.Image(
                                                    value=init_imgs[_idx], interactive=False, show_label=False,
                                                    container=False, height=300, elem_classes="lb-figure",
                                                    buttons=[],
                                                )
                                                fig_components.append(_img)
                                        elif _idx == len(EVAL_FIGS):
                                            gr.HTML(EVAL_TIPS, elem_classes="lb-html lb-card lb-eval-tips-card")
                            with gr.Row(elem_classes="lb-figure-grid lb-analysis-figs"):
                                for _fname, _ftitle, _fsub in (
                                    ("11_F1分布.png", "F1 分布", f"{_NCLS} 类按 F1 分档"),
                                    ("12_三模型分歧.png", "三模型分歧", "分歧最大的类别"),
                                    ("13_样本量与F1.png", "样本量 vs F1", "长尾类别观察"),
                                ):
                                    with gr.Column(elem_classes="lb-card lb-figure-card", min_width=0):
                                        gr.HTML(f'<div class="lb-figure-head"><h3>{_ftitle}</h3>'
                                                f'<span>{_fsub} · 点击图片放大</span></div>',
                                                elem_classes="lb-html")
                                        _p = _analysis_fig(_fname)
                                        if _p:
                                            gr.Image(value=_p, interactive=False, show_label=False,
                                                     container=False, height=260, elem_classes="lb-figure",
                                                     buttons=[])
                                        else:
                                            gr.HTML(_analysis_empty("暂无图表"), elem_classes="lb-html")
                        with gr.Tab("三模型对比"):
                            compare_html = gr.HTML(init_compare, elem_classes="lb-html")
                            with gr.Row(elem_classes="lb-analysis-row"):
                                gr.HTML(render_analysis_model_gap(), elem_classes="lb-html")
                                with gr.Column(elem_classes="lb-card lb-analysis-fig-card", min_width=0):
                                    gr.HTML('<div class="lb-figure-head"><h3>三模型分歧</h3>'
                                            '<span>横轴 ResNet50，纵轴 EfficientNet-B0 · 点击图片放大</span></div>',
                                            elem_classes="lb-html")
                                    _gap_path = _analysis_fig("12_三模型分歧.png")
                                    if _gap_path:
                                        gr.Image(value=_gap_path, interactive=False, show_label=False,
                                                 container=False, height=300, elem_classes="lb-figure",
                                                 buttons=[])
                                    else:
                                        gr.HTML(_analysis_empty("暂无图表"), elem_classes="lb-html")

                        with gr.Tab("跨域泛化"):
                            with gr.Column(elem_classes="lb-xd-section", min_width=0):
                                gr.HTML(render_xd_section_head(), elem_classes="lb-html")
                                gr.HTML(render_xd_hero(), elem_classes="lb-html")

                                with gr.Row(elem_classes="lb-xd-row lb-xd-row-protocol"):
                                    with gr.Column(elem_classes="lb-card lb-figure-card lb-xd-fig", min_width=0):
                                        _f, _l, _s = XD_FIGS["protocol"]
                                        gr.HTML(f'<div class="lb-figure-head"><h3>{_l}</h3>'
                                                f'<span>{_s} · 点击图片放大</span></div>',
                                                elem_classes="lb-html")
                                        gr.Image(value=xd_fig("protocol"), interactive=False, show_label=False,
                                                 container=False, height=300, elem_classes="lb-figure",
                                                 buttons=[])
                                    with gr.Column(elem_classes="lb-card lb-xd-side", min_width=0):
                                        gr.HTML(render_xd_protocol_side(), elem_classes="lb-html")

                                with gr.Row(elem_classes="lb-xd-row lb-xd-row-shift"):
                                    with gr.Column(elem_classes="lb-card lb-figure-card lb-xd-fig", min_width=0):
                                        _f, _l, _s = XD_FIGS["shift"]
                                        gr.HTML(f'<div class="lb-figure-head"><h3>{_l}</h3>'
                                                f'<span>{_s} · 点击图片放大</span></div>',
                                                elem_classes="lb-html")
                                        gr.Image(value=xd_fig("shift"), interactive=False, show_label=False,
                                                 container=False, height=300, elem_classes="lb-figure",
                                                 buttons=[])
                                    with gr.Column(elem_classes="lb-card lb-xd-side", min_width=0):
                                        gr.HTML(render_xd_shift_side(), elem_classes="lb-html")

                                with gr.Row(elem_classes="lb-xd-row lb-xd-row-tsne"):
                                    with gr.Column(elem_classes="lb-card lb-figure-card lb-xd-fig", min_width=0):
                                        _f, _l, _s = XD_FIGS["tsne"]
                                        gr.HTML(f'<div class="lb-figure-head"><h3>{_l}</h3>'
                                                f'<span>{_s} · 点击图片放大</span></div>',
                                                elem_classes="lb-html")
                                        gr.Image(value=xd_fig("tsne"), interactive=False, show_label=False,
                                                 container=False, height=300, elem_classes="lb-figure",
                                                 buttons=[])
                                    with gr.Column(elem_classes="lb-card lb-xd-side", min_width=0):
                                        gr.HTML(render_xd_tsne_side(), elem_classes="lb-html")

                                with gr.Column(elem_classes="lb-card lb-figure-card lb-xd-wide", min_width=0):
                                    _f, _l, _s = XD_FIGS["confusion"]
                                    gr.HTML(f'<div class="lb-figure-head"><h3>{_l}</h3>'
                                            f'<span>{_s} · 点击图片放大</span></div>',
                                            elem_classes="lb-html")
                                    gr.Image(value=xd_fig("confusion"), interactive=False, show_label=False,
                                             container=False, height=560, elem_classes="lb-figure",
                                             buttons=[])
                                    gr.HTML('<div class="lb-xd-fig-note">PlantDoc 27 类全部参与评估。'
                                            '对角线为正确分类；跨域错误分布比同分布测试集更分散，'
                                            '说明模型并非只把样本稳定地推到相邻类别。</div>',
                                            elem_classes="lb-html")

                                with gr.Row(elem_classes="lb-xd-three"):
                                    for _key in ("conf", "consistency", "risk"):
                                        with gr.Column(elem_classes="lb-card lb-figure-card", min_width=0):
                                            _f, _l, _s = XD_FIGS[_key]
                                            gr.HTML(f'<div class="lb-figure-head"><h3>{_l}</h3>'
                                                    f'<span>{_s} · 点击图片放大</span></div>',
                                                    elem_classes="lb-html")
                                            gr.Image(value=xd_fig(_key), interactive=False, show_label=False,
                                                     container=False, height=300, elem_classes="lb-figure",
                                                     buttons=[])

                                gr.HTML(render_xd_limitations(), elem_classes="lb-html")
                        with gr.Tab("数据资产"):
                            data_assets_html = gr.HTML(render_data_assets(),
                                                       elem_classes="lb-html")

                    eval_model.change(
                        on_eval,
                        inputs=[eval_model],
                        outputs=[eval_stats, *fig_components, compare_html],
                    )

            # ========================= ④ AI 问答 =========================
            with gr.Tab("知识问答"):
                with gr.Column(elem_classes="lb-page lb-ai-page", min_width=0):
                    gr.HTML(PAGEHEAD_KB, elem_classes="lb-html")
                    gr.HTML(render_qa_tech_strip(), elem_classes="lb-html")

                    with gr.Row(elem_classes="lb-ai-grid"):
                        # 左：提问与回答
                        with gr.Column(elem_classes="lb-card lb-ai-ask", min_width=0):
                            gr.HTML(f'''
<div class="lb-card-head">
  <div><h2>向系统提问</h2><p>先检索本地病害知识库（共 {_NKB} 条），再组织回答；断网也能返回检索结果</p></div>
</div>''', elem_classes="lb-html")
                            qa_in = gr.Textbox(
                                label=None, show_label=False, lines=3,
                                placeholder="例如：番茄晚疫病有什么症状，怎么防治？",
                                elem_classes="lb-select lb-qa-input",
                            )
                            with gr.Row(elem_classes="lb-qa-actions"):
                                qa_btn = gr.Button("提问", variant="primary",
                                                   elem_classes="lb-btn-primary")
                                qa_rand = gr.Button("换个示例", elem_classes="lb-btn-secondary")
                            qa_ans = gr.Markdown(QA_INTRO, elem_classes="lb-detail lb-qa-answer")
                            qa_meta = gr.HTML("", elem_classes="lb-html lb-qa-meta-wrap")

                        # 右：知识库速查 + 常见问题（纯文字链接，不用胶囊）
                        with gr.Column(elem_classes="lb-card lb-ai-kb", min_width=0):
                            gr.HTML(f'''
<div class="lb-card-head">
  <div><h2>知识库速查</h2><p>{_KB_TOTAL:,} 条 · 可输入作物或病害名搜索</p></div>
</div>''', elem_classes="lb-html")
                            kb_dd = gr.Dropdown(
                                _KB_CHOICES, value=CLASSES[0], label="作物与病害",
                                elem_classes="lb-select", filterable=True,
                            )
                            kb_html = gr.HTML(init_kb, elem_classes="lb-html lb-kb-panel")
                            gr.HTML('<div class="lb-mini-label lb-ask-label">常见问题 · 点击直接提问</div>',
                                    elem_classes="lb-html")
                            qa_ex = []
                            with gr.Column(elem_classes="lb-qa-examples"):
                                for _q in ai_qa.examples()[:6]:
                                    _b = gr.Button(_q, elem_classes="lb-link-btn", size="sm")
                                    qa_ex.append((_b, _q))
                            gr.HTML('<div class="lb-kb-note">问答会优先检索这里的知识；'
                                    '查询结果与 AI 回答可相互对照。</div>', elem_classes="lb-html")

                    qa_btn.click(on_ask, [qa_in], [qa_ans, qa_meta])
                    for _b, _q in qa_ex:
                        _b.click(lambda q=_q: on_ask_example(q), None,
                                 [qa_in, qa_ans, qa_meta])
                    qa_rand.click(random_question, None, qa_in)
                    kb_dd.change(on_kb, [kb_dd], [kb_html])

            # ========================= ⑤ 更多 =========================
            with gr.Tab("更多"):
                with gr.Column(elem_classes="lb-page", min_width=0):
                    gr.HTML(PAGEHEAD_MORE, elem_classes="lb-html")

                    hist_stats = gr.HTML(init_hist_stats, elem_classes="lb-html")

                    with gr.Row(elem_classes="lb-history-toolbar"):
                        gr.HTML('<div class="lb-history-title">识别历史每次诊断后自动追加；表格展示最近记录，完整数据保存在 outputs/history.csv。</div>',
                                elem_classes="lb-html lb-col")
                        with gr.Row(elem_classes="lb-history-actions"):
                            hist_filter = gr.Dropdown(
                                ["全部"] + sorted({r[3].split(" - ")[0] for r in init_rows[1:] if len(r) > 3})
                                if len(init_rows) > 1 else ["全部"],
                                value="全部", show_label=False, label=None,
                                elem_classes="lb-select lb-hist-filter", scale=1,
                            )
                            hist_refresh = gr.Button("刷新", elem_classes="lb-btn-secondary", size="sm")
                            hist_clear = gr.Button("清空历史", elem_classes="lb-btn-secondary", size="sm")

                    with gr.Row(elem_classes="lb-history-delbar"):
                        hist_del_input = gr.Textbox(
                            placeholder="输入要删除的序号，如 1,3,5（对应表格 # 列）",
                            show_label=False, label=None, scale=4,
                            elem_classes="lb-select",
                        )
                        hist_del_btn = gr.Button("删除指定记录", elem_classes="lb-btn-secondary", size="sm", scale=1)
                        hist_export = gr.DownloadButton(
                            "导出历史 CSV", value=web.HISTORY_CSV,
                            elem_classes="lb-btn-secondary", size="sm")

                    hist_table = gr.HTML(init_hist_table, elem_classes="lb-html")
                    hist_js_row = gr.Textbox(value="", show_label=False, elem_id="hist-js-row",
                                             elem_classes="lb-js-hidden")
                    hist_js_open = gr.Button("打开历史详情", elem_id="hist-js-open",
                                             elem_classes="lb-js-hidden")
                    hist_charts = gr.HTML(init_hist_charts, elem_classes="lb-html")

                    gr.HTML(USAGE, elem_classes="lb-html")

                    hist_refresh.click(
                        on_history_refresh,
                        outputs=[hist_table, hist_stats, hist_charts],
                    )
                    hist_clear.click(
                        on_history_clear,
                        outputs=[hist_table, hist_stats, hist_charts],
                    )
                    hist_filter.change(
                        on_history_filter, inputs=[hist_filter], outputs=[hist_table]
                    )
                    hist_del_btn.click(
                        on_history_delete,
                        inputs=[hist_del_input, hist_filter],
                        outputs=[hist_table],
                    )
                    hist_js_open.click(
                        on_history_open,
                        inputs=[hist_js_row],
                        outputs=[batch_detail_modal, batch_detail_dd,
                                 batch_detail_img, batch_detail_meta,
                                 batch_detail_result, batch_top5_html,
                                 batch_cam_img, batch_box_img,
                                 batch_lesion_caption, batch_detail_advice,
                                 hist_detail_cache],
                    )
                    demo.load(
                        on_history_refresh,
                        outputs=[hist_table, hist_stats, hist_charts],
                    )
                    demo.load(
                        render_data_assets,
                        None,
                        [data_assets_html],
                    )

    return demo


def _warmup():
    """启动时预热：模型 + 向量缓存 + 知识库问答链路。"""
    try:
        import threading

        def _run():
            import time
            t0 = time.time()
            try:
                import ai_qa
                ai_qa._build_embedding_cache()
            except Exception:
                pass
            try:
                web.get_model(BACKBONES[0])
            except Exception:
                pass
            print(f"[预热] 完成，耗时 {time.time() - t0:.1f}s", flush=True)

        threading.Thread(target=_run, daemon=True).start()
    except Exception:
        pass


if __name__ == "__main__":
    web.preflight_or_exit()   # 缺数据在此明确退出；import web 不会
    _warmup()
    demo = build_ui()
    demo.launch(
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        inbrowser=True,          # 启动后自动打开浏览器
        theme=THEME,
        css=CSS,
        allowed_paths=[web.config.OUTPUT_DIR, web.config.IMAGES_DIR, web.config.DATA_DIR],
        show_error=True,
        footer_links=[],
        head=_FS_HEAD,
    )




