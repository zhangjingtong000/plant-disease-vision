# -*- coding: utf-8 -*-
"""
pdf_report.py — 诊断报告 PDF 生成

用 reportlab 的文档流排版，比 matplotlib 画布更接近正式报告：
页眉、标题、结果块、图片并排、信息表、症状、防治建议、页脚。
"""
import os
import re
import io as _io
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, KeepTogether, HRFlowable,
)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

TEAL = colors.HexColor("#00ADA2")
TEAL_LIGHT = colors.HexColor("#E8F6F5")
INK = colors.HexColor("#0A0E19")
BODY = colors.HexColor("#3C3C43")
MUTED = colors.HexColor("#82888D")
LINE = colors.HexColor("#EAEBEC")
BG_SOFT = colors.HexColor("#F5F6F6")

_FONT = "Helvetica"

MODEL_CN = {
    "resnet50": "ResNet50",
    "efficientnet_b0": "EfficientNet-B0",
    "mobilenet_v3_large": "MobileNetV3-Large",
}


def _model_label(name):
    return MODEL_CN.get(name, str(name))


def _read_acc(model_name):
    """从评估指标里读该模型的测试集准确率，读不到就不写。"""
    try:
        import json
        import config
        p = os.path.join(config.METRIC_DIR, f"eval_{model_name}.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return float(json.load(f)["accuracy"])
    except Exception:
        pass
    return None


def _ensure_font():
    """注册中文字体，优先黑体。"""
    global _FONT
    if _FONT != "Helvetica":
        return _FONT
    candidates = [
        (r"C:\Windows\Fonts\simhei.ttf", "SimHei"),
        (r"C:\Windows\Fonts\msyh.ttc", "MicrosoftYaHei"),
        (r"C:\Windows\Fonts\simsun.ttc", "SimSun"),
    ]
    for path, name in candidates:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont(name, path))
                _FONT = name
                return _FONT
            except Exception:
                continue
    return _FONT


def _styles():
    f = _ensure_font()
    return {
        "title": ParagraphStyle("title", fontName=f, fontSize=20, leading=26,
                                textColor=INK, spaceAfter=2),
        "subtitle": ParagraphStyle("subtitle", fontName=f, fontSize=9.5, leading=14,
                                   textColor=MUTED),
        "h2": ParagraphStyle("h2", fontName=f, fontSize=10.5, leading=14,
                             textColor=INK, spaceBefore=6, spaceAfter=4),
        "body": ParagraphStyle("body", fontName=f, fontSize=9.2, leading=14.2,
                               textColor=BODY, alignment=TA_LEFT),
        "small": ParagraphStyle("small", fontName=f, fontSize=8.5, leading=13,
                                textColor=MUTED),
        "disease": ParagraphStyle("disease", fontName=f, fontSize=16, leading=21,
                                  textColor=INK),
        "crop": ParagraphStyle("crop", fontName=f, fontSize=9, leading=13,
                               textColor=TEAL),
        "conf": ParagraphStyle("conf", fontName=f, fontSize=23, leading=27,
                               textColor=TEAL),
        "cell_label": ParagraphStyle("cell_label", fontName=f, fontSize=8,
                                     leading=11, textColor=MUTED),
        "cell_value": ParagraphStyle("cell_value", fontName=f, fontSize=10,
                                     leading=14, textColor=INK),
        "footer": ParagraphStyle("footer", fontName=f, fontSize=7.5, leading=11,
                                 textColor=MUTED, alignment=TA_CENTER),
    }


def _top5_chart(top5, cn_name):
    """生成 Top-5 横向条形图，返回临时 PNG 路径。"""
    fig, ax = plt.subplots(figsize=(4.6, 2.5), dpi=200)
    names = [cn_name(c)[:16] for c, _ in top5][::-1]
    vals = [p * 100 for _, p in top5][::-1]
    colors_ = ["#00ADA2" if i == len(vals) - 1 else "#C9D2D0" for i in range(len(vals))]
    ax.barh(names, vals, color=colors_, height=0.62)
    ax.set_xlim(0, 100)
    ax.set_xlabel("置信度 (%)", fontsize=7.5, color="#3C3C43")
    ax.tick_params(labelsize=7, colors="#3C3C43")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#EAEBEC")
    ax.spines["bottom"].set_color("#EAEBEC")
    ax.grid(axis="x", color="#F0F2F2", lw=0.6)
    ax.set_axisbelow(True)
    for i, v in enumerate(vals):
        ax.text(min(v + 1.5, 97), i, f"{v:.1f}%", va="center", fontsize=6.8, color="#6C6E75")
    plt.tight_layout(pad=0.4)
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "outputs", "_pdf_top5.png")
    plt.savefig(p, dpi=200, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    return p


def build_report(pil_img, info, top5, model_name, out_path, cn_name,
                 elapsed=None, task_no="001", lesion=None, meta=None):
    """生成诊断报告 PDF。"""
    st = _styles()
    f = _ensure_font()
    now = datetime.now()

    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=16 * mm, rightMargin=16 * mm,
        topMargin=13 * mm, bottomMargin=13 * mm,
        title="叶片病害诊断报告", author="植物叶片病害智能诊断系统",
    )
    story = []

    # ---------- 页眉 ----------
    header = Table(
        [[Paragraph("植物叶片病害智能诊断系统", ParagraphStyle(
              "hd", fontName=f, fontSize=10, leading=14, textColor=INK)),
          Paragraph(f"报告编号 {task_no}　·　{now:%Y-%m-%d %H:%M}",
                    ParagraphStyle("hr", fontName=f, fontSize=8.5, leading=14,
                                   textColor=MUTED, alignment=2))]],
        colWidths=[100 * mm, 74 * mm],
    )
    header.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(header)
    story.append(Spacer(1, 3 * mm))
    story.append(HRFlowable(width="100%", thickness=1.6, color=TEAL, spaceAfter=8))

    # ---------- 标题 ----------
    story.append(Paragraph("叶片病害诊断报告", st["title"]))
    story.append(Paragraph(
        "本报告由深度学习模型自动生成，列出识别结果、症状说明与防治建议。", st["subtitle"]))
    story.append(Spacer(1, 2.5 * mm))

    # ---------- 结果块 ----------
    conf = top5[0][1]
    left = [
        Paragraph(info["crop_cn"], st["crop"]),
        Paragraph(f"{info['disease_cn']}", st["disease"]),
        Spacer(1, 2 * mm),
        Paragraph(f"{conf*100:.1f}%  <font size=9 color='#82888D'>置信度</font>", st["conf"]),
    ]
    right_info = [
        [Paragraph("病原类型", st["cell_label"]), Paragraph(info["pathogen"], st["cell_value"])],
        [Paragraph("危害程度", st["cell_label"]), Paragraph(info["severity"], st["cell_value"])],
        [Paragraph("识别模型", st["cell_label"]), Paragraph(_model_label(model_name), st["cell_value"])],
        [Paragraph("推理耗时", st["cell_label"]),
         Paragraph(f"{elapsed*1000:.0f} ms" if elapsed else "—", st["cell_value"])],
    ]
    right = Table(right_info, colWidths=[20 * mm, 32 * mm])
    right.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
    ]))

    result_tbl = Table([[left, right]], colWidths=[110 * mm, 64 * mm])
    result_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FAFBFB")),
        ("BOX", (0, 0), (-1, -1), 0.8, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEBEFORE", (1, 0), (1, -1), 0.8, LINE),
    ]))
    story.append(result_tbl)

    # 归档要素（可选填写）
    if meta:
        bits = []
        for k, label in (("plot", "地块"), ("variety", "品种"), ("submitter", "送检人")):
            v = (meta.get(k) or "").strip()
            if v:
                bits.append(f"{label}：{v}")
        if bits:
            story.append(Spacer(1, 1.5 * mm))
            story.append(Paragraph("　·　".join(bits), st["small"]))

    story.append(Spacer(1, 4 * mm))

    # ---------- 图片 ----------
    tmp_img = os.path.join(os.path.dirname(out_path), "_pdf_input.png")
    pil_img.convert("RGB").save(tmp_img, quality=92)
    chart = _top5_chart(top5, cn_name)

    iw, ih = pil_img.size
    ratio = ih / max(iw, 1)
    img_w = 42 * mm
    img_h = min(img_w * ratio, 30 * mm)

    img_tbl = Table([[
        [Paragraph("送检图像", st["cell_label"]), Spacer(1, 1.5 * mm),
         RLImage(tmp_img, width=img_w, height=img_h)],
        [Paragraph("Top-5 候选", st["cell_label"]), Spacer(1, 1.5 * mm),
         RLImage(chart, width=90 * mm, height=26 * mm)],
    ]], colWidths=[87 * mm, 87 * mm])
    img_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (0, 0), "CENTER"),
    ]))
    story.append(img_tbl)
    story.append(Spacer(1, 2.5 * mm))

    # ---------- 病斑定位 ----------
    if lesion is not None:
        story.append(Paragraph("病斑定位", st["h2"]))
        story.append(HRFlowable(width="100%", thickness=0.6, color=LINE, spaceAfter=4))

        cam_path = os.path.join(os.path.dirname(out_path), "_pdf_cam.png")
        box_path = os.path.join(os.path.dirname(out_path), "_pdf_box.png")
        Image.fromarray(lesion["overlay"]).save(cam_path)
        Image.fromarray(lesion["vis"]).save(box_path)

        loc_tbl = Table([[
            [Paragraph("Grad-CAM 关注区域", st["cell_label"]), Spacer(1, 1.5 * mm),
             RLImage(cam_path, width=34 * mm, height=34 * mm)],
            [Paragraph("病斑外接区域", st["cell_label"]), Spacer(1, 1.5 * mm),
             RLImage(box_path, width=34 * mm, height=34 * mm)],
        ]], colWidths=[87 * mm, 87 * mm])
        loc_tbl.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(loc_tbl)
        story.append(Spacer(1, 2 * mm))

        ratio_txt = f"{lesion['ratio']:.1f}%"
        stat_rows = [
            [Paragraph("病斑占叶面比例", st["cell_label"]),
             Paragraph(ratio_txt, st["cell_value"]),
             Paragraph("危害程度", st["cell_label"]),
             Paragraph(lesion["level"], st["cell_value"])],
        ]
        stat_tbl = Table(stat_rows, colWidths=[36 * mm, 40 * mm, 24 * mm, 50 * mm])
        stat_tbl.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FAFBFB")),
            ("BOX", (0, 0), (-1, -1), 0.8, LINE),
            ("INNERGRID", (0, 0), (-1, -1), 0.4, LINE),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        story.append(stat_tbl)
        story.append(Spacer(1, 1.5 * mm))
        story.append(Paragraph(
            "热力图颜色越深表示模型在该区域提取到的证据越强；外接框标注模型判定的主要病斑范围。"
            "病斑比例由 Heatmap 阈值分割后按叶面面积归一化得到。", st["small"]))
        story.append(Spacer(1, 2 * mm))

    # ---------- 症状与防治（两栏并排，节省纵向空间） ----------
    # 优先用结构化防治方案；读不到时退回旧的三条
    ctrl_struct = {}
    try:
        import knowledge
        ctrl_struct = knowledge.get_control(info.get("cls", ""))
    except Exception:
        ctrl_struct = {}

    entries = []
    if ctrl_struct.get("农业防治") or ctrl_struct.get("化学防治"):
        for grp in ("农业防治", "化学防治", "注意事项"):
            for t in (ctrl_struct.get(grp) or []):
                entries.append((grp, t))
        entries = entries[:4]          # 一页放得下的上限
    else:
        entries = [("", t) for t in info["control"]]

    rows = []
    for i, (grp, t) in enumerate(entries, 1):
        label = f"{grp[:2]} · " if grp else ""
        t = re.sub(r"（来源：[^）]*）|\(来源：[^)]*\)", "", t).strip()
        t = re.sub(r"（依据[^）]*）|\(依据[^)]*\)", "", t).strip()
        rows.append([
            Paragraph(f"<font color='#00ADA2'><b>{i:02d}</b></font>",
                      ParagraphStyle("n", fontName=f, fontSize=8, leading=11.5)),
            Paragraph(label + t, ParagraphStyle("cn", fontName=f, fontSize=7.4,
                                                leading=10.4, textColor=BODY)),
        ])
    ctrl_tbl = Table(rows, colWidths=[6 * mm, 79 * mm])
    ctrl_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
    ]))

    left_col = [
        Paragraph("症状表现", st["h2"]),
        HRFlowable(width="100%", thickness=0.6, color=LINE, spaceAfter=3),
        Paragraph(info["symptoms"], st["body"]),
    ]
    right_col = [
        Paragraph("防治建议", st["h2"]),
        HRFlowable(width="100%", thickness=0.6, color=LINE, spaceAfter=3),
        ctrl_tbl,
    ]
    two_col = Table([[left_col, right_col]], colWidths=[84 * mm, 94 * mm])
    two_col.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (0, 0), 0),
        ("RIGHTPADDING", (0, 0), (0, 0), 7),
        ("LEFTPADDING", (1, 0), (1, 0), 7),
        ("RIGHTPADDING", (1, 0), (1, 0), 0),
        ("LINEBEFORE", (1, 0), (1, 0), 0.6, LINE),
    ]))
    story.append(Spacer(1, 1.5 * mm))
    story.append(two_col)
    story.append(Spacer(1, 4 * mm))

    # ---------- 免责 ----------
    story.append(HRFlowable(width="100%", thickness=0.6, color=LINE, spaceAfter=5))
    story.append(Paragraph(
        "本报告由模型自动生成，仅供辅助参考，不能替代农业植保专业诊断。<br/>"
        f"模型：{_model_label(model_name)}"
        + (f"　·　该模型测试集准确率 {_read_acc(model_name) * 100:.2f}%" if _read_acc(model_name) else "")
        + f"　·　生成时间 {now:%Y-%m-%d %H:%M:%S}",
        st["footer"]))

    doc.build(story)

    cleanup = [tmp_img, chart]
    if lesion is not None:
        cleanup += [os.path.join(os.path.dirname(out_path), "_pdf_cam.png"),
                    os.path.join(os.path.dirname(out_path), "_pdf_box.png")]
    for p in cleanup:
        try:
            os.remove(p)
        except Exception:
            pass
    return out_path
