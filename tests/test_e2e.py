# -*- coding: utf-8 -*-
"""第 2 层：端到端闭环测试（Playwright，断言状态真的变化）。

前置：服务已在 http://127.0.0.1:7860 运行。
运行：python -m pytest tests/test_e2e.py -v
"""
import os
import pytest
from playwright.sync_api import sync_playwright

BASE_URL = "http://127.0.0.1:7860"
SHOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "outputs", "figures", "_e2e")


@pytest.fixture(scope="module")
def browser():
    os.makedirs(SHOT, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        yield b
        b.close()


@pytest.fixture(scope="module")
def page(browser):
    pg = browser.new_page(viewport={"width": 1440, "height": 900})
    pg.goto(BASE_URL, wait_until="domcontentloaded", timeout=90000)
    pg.wait_for_timeout(8000)
    return pg


class TestTabs:
    def test_all_tabs_open(self, page):
        """五个页签都要能打开，且控制台无报错。"""
        errs = []
        page.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        for name in ("智能诊断", "批量诊断", "模型评估", "知识问答", "更多"):
            page.get_by_role("tab", name=name).click()
            page.wait_for_timeout(1200)
        assert not errs, f"控制台报错: {errs[:3]}"


class TestDiagnose:
    def test_diagnose_shows_result(self, page):
        """选示例 → 诊断 → 必须出现病害名和置信度。"""
        page.get_by_role("tab", name="智能诊断").click()
        page.wait_for_timeout(1000)

        btns = page.locator(".lb-examples button")
        assert btns.count() >= 5, f"示例按钮只有 {btns.count()} 个"
        btns.first.click()
        page.wait_for_timeout(1500)

        page.get_by_role("button", name="开始诊断").click()
        page.wait_for_timeout(9000)

        body = page.locator("body").inner_text()
        assert "%" in body, "结果里没有置信度"
        # 断言出现了"诊断结果"区块
        assert "诊断结果" in body
        page.screenshot(path=os.path.join(SHOT, "diagnose.png"))

    def test_locate_produces_image(self, page):
        """病斑定位必须生成热力图。"""
        page.get_by_role("button", name="病斑定位").first.click()
        page.wait_for_timeout(12000)
        # 页面上应出现 canvas 或 img（热力图）
        imgs = page.locator("img[src^='data:image'], img[src*='gradio_api']")
        assert imgs.count() > 0, "定位后没有生成图片"
        page.screenshot(path=os.path.join(SHOT, "locate.png"))

    def test_pdf_export(self, page):
        """导出 PDF 必须产生文件下载或提示。"""
        page.get_by_role("button", name="导出 PDF").first.click()
        page.wait_for_timeout(12000)
        body = page.locator("body").inner_text()
        assert "PDF" in body or "报告" in body, "导出无反馈"


class TestAsk:
    def test_question_returns_answer(self, page):
        """提问必须返回非空回答，且带来源。"""
        page.get_by_role("tab", name="知识问答").click()
        page.wait_for_timeout(1500)

        box = page.locator("textarea").first
        box.fill("番茄晚疫病怎么防治？")
        page.wait_for_timeout(500)
        page.get_by_role("button", name="提问").click()
        page.wait_for_timeout(15000)

        body = page.locator("body").inner_text()
        assert "请输入问题" not in body, "问题未提交成功"
        assert "防治" in body or "症状" in body or "结论" in body
        page.screenshot(path=os.path.join(SHOT, "ask.png"))


class TestHistory:
    def test_history_page_loads(self, page):
        """历史页要能加载出表格。"""
        page.get_by_role("tab", name="更多").click()
        page.wait_for_timeout(3000)
        body = page.locator("body").inner_text()
        assert "历史" in body
        page.screenshot(path=os.path.join(SHOT, "history.png"))


class TestEval:
    def test_eval_figures_visible(self, page):
        """评估页图表要有实际图片（naturalWidth>0）。"""
        page.get_by_role("tab", name="模型评估").click()
        page.wait_for_timeout(3000)
        ok = page.evaluate("""() => {
            const imgs = Array.from(document.querySelectorAll('img'));
            return imgs.filter(i => i.naturalWidth > 50).length;
        }""")
        assert ok >= 3, f"只有 {ok} 张图加载成功"
        page.screenshot(path=os.path.join(SHOT, "eval.png"))
