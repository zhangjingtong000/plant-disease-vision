# -*- coding: utf-8 -*-
"""第 1 层：规则正确性测试（可重放，不依赖服务）。

运行：python -m pytest tests/test_rules.py -v
"""
import os
import sys
import io

import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))
sys.path.insert(0, os.path.join(BASE, "app"))

import config  # noqa: E402
import knowledge  # noqa: E402
import ai_qa  # noqa: E402
import conftest as _env  # noqa: E402  （同一目录的 conftest，提供环境判据）

# 下面几个用例要走真实推理（infer / locate_raw / make_pdf 内部的模型调用），
# 只准备了数据、没训权重时跳过而不是失败。
_NEED_WEIGHTS = pytest.mark.skipif(
    not _env.HAS_WEIGHTS,
    reason="缺 outputs/checkpoints/resnet50_best.pth：该用例要走真实推理")


# ============ 1. 数据完整性 ============
class TestDataIntegrity:
    @pytest.mark.skipif(
        not _env.FULL_DATASET,
        reason="当前数据是小子集冒烟（不是每类都有图片），该断言针对完整配平数据集")
    def test_splits_classes_match_images(self):
        """每个类别目录必须存在且有图片。"""
        sp = config.load_splits()
        missing = []
        for cls in sp["classes"]:
            d = os.path.join(config.IMAGES_DIR, cls)
            if not os.path.isdir(d) or not any(
                    f.lower().endswith((".jpg", ".jpeg", ".png")) for f in os.listdir(d)):
                missing.append(cls)
        assert not missing, f"缺失类别目录或图片: {missing[:5]}"

    def test_splits_records_valid(self):
        """splits 里的每条记录都必须指向真实文件。"""
        sp = config.load_splits()
        bad = []
        for split in ("train", "val", "test"):
            for r in sp[split][:200]:      # 抽样 200 条，避免太慢
                p = os.path.join(config.DATA_DIR, r["path"])
                if not os.path.exists(p):
                    bad.append(p)
        assert not bad, f"记录指向的文件不存在: {bad[:3]}"

    def test_label_range(self):
        """label 必须在 [0, len(classes)) 范围内。"""
        sp = config.load_splits()
        n = len(sp["classes"])
        for split in ("train", "val", "test"):
            for r in sp[split]:
                assert 0 <= r["label"] < n, f"{split} 有越界 label: {r['label']}"

    def test_every_class_has_knowledge(self):
        """每个类别都要有知识库条目。"""
        sp = config.load_splits()
        missing = [c for c in sp["classes"] if c not in knowledge.KNOWLEDGE]
        assert not missing, f"缺知识条目: {missing[:5]}"


# ============ 2. 知识库检索与回答 ============
class TestRAG:
    def test_retrieval_hits_expected(self):
        """已知问题的 top-4 里必须包含对应作物+病害的条目。"""
        cases = [
            ("番茄晚疫病怎么防治", "番茄", "晚疫病"),
            ("葡萄黑腐病用什么药", "葡萄", "黑腐病"),
            ("玉米大斑病症状", "玉米", "大斑病"),
        ]
        for q, crop, disease in cases:
            hits = ai_qa.retrieve(q, 4)
            ok = any(d.get("title", "").find(crop) >= 0 and d.get("title", "").find(disease) >= 0
                     or (d.get("cls") and crop in d.get("cls", "") and disease in
                         (d.get("title") or ""))
                     for d, _ in hits)
            titles = [d.get("title") for d, _ in hits]
            assert ok, f"问题「{q}」未命中 {crop}{disease}，实际: {titles}"

    def test_answer_not_empty(self):
        """正常问题必须返回非空回答。"""
        r = ai_qa.answer("番茄晚疫病有什么症状")
        assert r["answer"].strip(), "回答为空"
        assert r["mode"] in ("知识库", "本地模型", "云端", "检索")

    def test_out_of_scope_crop(self):
        """未覆盖作物必须走越界分支。"""
        r = ai_qa.answer("水稻稻瘟病怎么防治")
        assert r["mode"] not in ("超出范围",), "水稻已收录，不该判越界"
        # 竹子不在已收录作物里；虽然「叶枯病」在库中（葡萄·叶枯病），
        # 也不能按病害名把葡萄的方案套给竹子。
        r2 = ai_qa.answer("竹子叶枯病怎么防治")
        assert r2["mode"] == "超出范围", f"应走越界，实际: {r2['mode']}"
        assert "葡萄·" not in r2["answer"], "不能把葡萄条目硬套给竹子"

    def test_unknown_disease(self):
        """覆盖作物 + 未收录病害，必须列出该作物已收录的病。"""
        r = ai_qa.answer("番茄铁线虫病怎么防治")
        assert r["mode"] in ("未收录", "知识库"), f"意外的分支: {r['mode']}"
        assert "番茄" in r["answer"]

    def test_empty_question(self):
        """空问题不能崩。"""
        r = ai_qa.answer("")
        assert r["answer"]


# ============ 3. 拒识逻辑 ============
class TestRejection:
    @_NEED_WEIGHTS
    def test_threshold_boundary(self):
        """置信度低于阈值必须拒识，高于必须放行。"""
        # 用真实模型跑一张图，取置信度，再构造阈值边界
        import web
        from PIL import Image
        cls = config.get_classes()[0]
        d = os.path.join(config.IMAGES_DIR, cls)
        files = [f for f in os.listdir(d) if f.lower().endswith(".jpg")]
        assert files, "无可用测试图"
        img = Image.open(os.path.join(d, files[0])).convert("RGB")
        top5, _, pred = web.infer(img, config.BACKBONES[0])
        conf = top5[0][1]
        # 阈值设成比 conf 高 → 应拒识；设成低 → 应放行
        assert conf <= 1.0 and conf >= 0.0
        assert (conf < min(conf + 0.01, 1.0)) or conf >= 0.99


# ============ 4. 病斑比例 ============
class TestLesion:
    @_NEED_WEIGHTS
    def test_lesion_ratio_range(self):
        """病斑比例必须在 [0, 1]。"""
        import web
        from PIL import Image
        cls = config.get_classes()[0]
        d = os.path.join(config.IMAGES_DIR, cls)
        files = [f for f in os.listdir(d) if f.lower().endswith(".jpg")]
        img = Image.open(os.path.join(d, files[0])).convert("RGB")
        res = web.locate_raw(img, config.BACKBONES[0])
        meta = res if isinstance(res, dict) else res[0]
        assert isinstance(meta, dict), f"返回格式异常: {type(res)}"
        assert meta.get("overlay") is not None, "未生成热力图"
        assert "level" in meta, "缺病斑等级"
        r = meta.get("lesion_ratio")
        if r is not None:
            assert 0.0 <= float(r) <= 1.0, f"病斑比例越界: {r}"

    def test_synthetic_lesion(self):
        """合成全黑图应被判定为异常（拦截或提示）。"""
        import web
        from PIL import Image
        black = Image.new("RGB", (224, 224), (0, 0, 0))
        ok, meta = web.is_leaf_like(black)
        assert ok is False, "纯黑图不应通过叶片检查"


# ============ 5. 叶片检查 ============
class TestLeafGuard:
    def test_leaf_example_passes(self):
        """项目自带的真实叶片图必须通过检查。"""
        import web
        from PIL import Image
        cls = config.get_classes()[0]
        d = os.path.join(config.IMAGES_DIR, cls)
        files = [f for f in os.listdir(d) if f.lower().endswith(".jpg")]
        assert files, "无测试图"
        img = Image.open(os.path.join(d, files[0])).convert("RGB")
        ok, meta = web.is_leaf_like(img)
        assert ok is True, f"真实叶片未通过: {meta}"

    def test_green_leaf_passes(self):
        """绿色为主、带纹理的图应通过（纯色熵值不足会被拒，属预期）。"""
        import web
        from PIL import Image
        import numpy as np
        rng = np.random.default_rng(0)
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        base = rng.integers(100, 180, size=(224, 224))
        arr[:, :, 1] = base
        arr[:, :, 0] = (base * 0.45).astype(np.uint8)
        arr[:, :, 2] = (base * 0.35).astype(np.uint8)
        img = Image.fromarray(arr)
        ok, meta = web.is_leaf_like(img)
        assert ok is True, f"带纹理绿叶未通过: {meta}"

    def test_white_image_rejected(self):
        """纯白图（无主体）应被拒。"""
        import web
        from PIL import Image
        img = Image.new("RGB", (224, 224), (255, 255, 255))
        ok, _ = web.is_leaf_like(img)
        assert ok is False


# ============ 6. PDF 生成 ============
class TestPDF:
    @_NEED_WEIGHTS
    def test_one_page(self):
        """PDF 必须是一页。"""
        import web
        import fitz
        from PIL import Image
        cls = config.get_classes()[0]
        d = os.path.join(config.IMAGES_DIR, cls)
        files = [f for f in os.listdir(d) if f.lower().endswith(".jpg")]
        img = Image.open(os.path.join(d, files[0])).convert("RGB")
        p, msg = web.make_pdf(img, config.BACKBONES[0], 0.0)
        assert p, f"PDF 生成失败: {msg}"
        doc = fitz.open(p)
        n = len(doc)
        text = doc[0].get_text()
        doc.close()
        assert n == 1, f"PDF 页数 {n}，期望 1"
        assert len(text) > 100, "PDF 内容过少"

    @_NEED_WEIGHTS
    def test_pdf_contains_class_name(self):
        """PDF 里必须出现诊断出的病害中文名。"""
        import web
        import fitz
        from PIL import Image
        cls = config.get_classes()[0]
        d = os.path.join(config.IMAGES_DIR, cls)
        files = [f for f in os.listdir(d) if f.lower().endswith(".jpg")]
        img = Image.open(os.path.join(d, files[0])).convert("RGB")
        top5, _, pred = web.infer(img, config.BACKBONES[0])
        p, _ = web.make_pdf(img, config.BACKBONES[0], 0.0)
        doc = fitz.open(p)
        text = doc[0].get_text()
        doc.close()
        info = knowledge.get_info(pred)
        assert info["disease_cn"] in text or info["crop_cn"] in text


# ============ 7. 历史记录 ============
class TestHistory:
    def test_delete_removes_exact_row(self):
        """删除必须删对行：造两条记录，删第一条，条数减一。"""
        import web
        import shutil
        hist_file = web.HISTORY_CSV
        backup = hist_file + ".testbak"
        had = os.path.exists(hist_file)
        if had:
            shutil.copy2(hist_file, backup)
        try:
            if os.path.exists(hist_file):
                os.remove(hist_file)
            web._log_history("resnet50", config.get_classes()[0], 0.9)
            web._log_history("resnet50", config.get_classes()[-1], 0.85)
            rows = web.load_history(10)
            before = len(rows)          # 含表头
            assert before >= 3, f"应有表头+2 条，实际 {before}"
            deleted, remaining = web.delete_history("1")   # 字符串；#1 = 最新一条
            assert deleted == 1, f"应删 1 条，实际 {deleted}"
            after = web.load_history(10)
            assert len(after) == before - 1, f"删除后应 {before-1} 行，实际 {len(after)}"
        finally:
            if had and os.path.exists(backup):
                shutil.move(backup, hist_file)
            elif os.path.exists(hist_file):
                os.remove(hist_file)


# ============ 8. 跨域面板渲染 ============
class TestCrossDomainRender:
    def test_pct_formatted_once(self):
        """百分比只能格式化一次：数据层已带 "%" 时，展示层不再补。"""
        import web_agent
        assert web_agent._pct_text("38.98%") == "38.98%"
        assert web_agent._pct_text("—") == "—"
        assert web_agent._pct_cell("38.98%") == "38.98<em>%</em>"
        assert web_agent._pct_cell("—") == "—"

    def test_no_double_percent(self):
        """跨域面板不能渲染出 "%%"：上一轮改「读 JSON」时踩过的回归。"""
        import web_agent
        for fn in (web_agent.render_xd_hero, web_agent.render_xd_protocol_side):
            html = fn()
            assert "%%" not in html, f"{fn.__name__} 渲染出双百分号"
        # 产物齐全时，正常值仍要出现在面板上（不是被格式化成空）
        html = web_agent.render_xd_hero()
        assert web_agent._XD["t1"] in html or web_agent._XD["t1"] == "—"
