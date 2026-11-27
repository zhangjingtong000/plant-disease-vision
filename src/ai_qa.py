# -*- coding: utf-8 -*-
"""
ai_qa.py — 植物病害智能问答（RAG + 可选大模型）

结构：
  1. 知识库文档集：38 类病害的症状/防治 + 作物信息 + 系统使用说明
  2. 检索：jieba 分词 + BM25，返回最相关的若干条目
  3. 生成：有 API Key 时调用大模型；无 Key 或断网时，直接返回检索到的知识（降级）
"""
import os
import re
import json
import time
import difflib

import jieba
from rank_bm25 import BM25Okapi

import knowledge

# ---------------- 文档集 ----------------
# 作物中文名补充信息
CROP_PROFILE = {
    "苹果": "苹果叶片病害主要有黑星病、黑腐病和锈病。黑星病在潮湿冷凉条件下高发，锈病需要桧柏等转主寄主。",
    "番茄": "番茄是叶片病害最多的作物之一，常见早疫病、晚疫病、叶霉病、斑枯病、细菌性斑点病、靶斑病，"
            "以及黄化曲叶病毒和花叶病毒两种病毒病。早疫与斑枯症状相似，常被混淆。",
    "土豆": "土豆重点关注晚疫病，它是毁灭性病害，高湿条件下可在一周内毁田；早疫病多从下部老叶开始。",
    "葡萄": "葡萄常见黑腐病、埃斯卡病（黑麻疹）和叶枯病，多雨季节需重点防治。",
    "玉米": "玉米常见灰斑病、普通锈病和大斑病，其中大斑病病斑最长可达十几厘米。",
    "草莓": "草莓叶焦病（叶枯病）会导致叶片焦枯，需控制湿度并及时清理老叶。",
    "桃": "桃树细菌性穿孔病会造成叶片穿孔、果实龟裂。",
    "柑橘": "柑橘黄龙病是毁灭性病害，由木虱传播，目前无有效治疗药剂，只能防虫与拔除病株。",
    "甜椒": "甜椒细菌性斑点病在叶面长时间潮湿时容易流行。",
}

# 系统说明文档
SYSTEM_DOCS = [
    ("系统功能",
     "本系统提供智能诊断、批量诊断、模型评估、病害知识库和 AI 问答五个功能。"
     "智能诊断支持单张上传或载入内置示例，输出 Top-5 候选、置信度、病斑定位和防治建议。"),
    ("模型信息",
     "系统提供三个骨干网络：ResNet50 测试准确率 98.61%，EfficientNet-B0 99.34%，"
     "MobileNetV3-Large 98.74%。ResNet50 参数量 23.59M、单张延迟约 16ms；"
     "MobileNetV3 参数量 4.25M、延迟约 14ms，体积只有 ResNet50 的六分之一。"),
    ("数据集",
     "训练数据来自 PlantVillage 公开数据集，原始 54381 张、38 类，配平后每类约 200 张、共 7552 张，"
     "按 70/10/20 划分为训练 5286、验证 755、测试 1511 张。"),
    ("使用方法",
     "在智能诊断页上传叶片照片或点击示例，选择骨干网络与拒识阈值后点击开始诊断。"
     "置信度低于阈值时系统会提示人工复核。点病斑定位可查看 Grad-CAM 热力图与病斑面积占比，"
     "点导出 PDF 可生成诊断报告。"),
    ("免责声明",
     "诊断结果仅供参考，不能替代农业植保专业判断。"),
]


def _doc_crop(doc):
    """取一条检索结果所属作物，不依赖 cls（扩充库条目没有 cls）。"""
    k = doc.get("kg_key")
    if k:
        info = getattr(knowledge, "KG_KNOWLEDGE", {}).get(k)
        if info:
            return info.get("crop_cn")
    cls = doc.get("cls")
    if cls:
        try:
            return knowledge.get_info(cls).get("crop_cn")
        except Exception:
            pass
    # 兜底：标题形如「作物·病害」
    title = doc.get("title", "")
    if "·" in title:
        return title.split("·")[0]
    return None


def _build_docs():
    """把 38 类病害 + 作物档案 + 系统说明拼成检索文档。"""
    docs = []
    for cls, info in knowledge.KNOWLEDGE.items():
        full = knowledge.get_info(cls)
        parts = [
            f"{info['crop_cn']}{info['disease_cn']}。",
            f"病原类型：{info['pathogen']}。危害程度：{info['severity']}。",
            f"症状：{full.get('symptoms', info['symptoms'])}。",
        ]
        if full.get("occurrence"):
            parts.append(f"发生规律：{full['occurrence']}。")
        if full.get("control_detail"):
            parts.append(f"防治：{full['control_detail']}。")
        else:
            parts.append(f"防治建议：{'；'.join(info['control'])}。")
        text = "".join(parts)
        docs.append({
            "title": f"{info['crop_cn']}·{info['disease_cn']}",
            "text": text,
            "cls": cls,
        })

    # 扩充知识库（cropdpkg 原始数据）：只进问答检索，不参与图像分类
    for key, info in getattr(knowledge, "KG_KNOWLEDGE", {}).items():
        parts = [f"{info['crop_cn']}{info['disease_cn']}。"]
        if info.get("pathogen"):
            parts.append(f"病原：{info['pathogen']}。")
        if info.get("symptoms"):
            parts.append(f"症状：{info['symptoms']}。")
        if info.get("occurrence"):
            parts.append(f"发生规律：{info['occurrence']}。")
        if info.get("control"):
            parts.append(f"防治：{'；'.join(info['control'][:5])}。")
        docs.append({
            "title": f"{info['crop_cn']}·{info['disease_cn']}",
            "text": "".join(parts),
            "cls": None,
            "kg_key": key,
        })

    for crop, text in CROP_PROFILE.items():
        docs.append({"title": f"{crop}病害概览", "text": text, "cls": None})

    for title, text in SYSTEM_DOCS:
        docs.append({"title": title, "text": text, "cls": None})
    return docs


DOCS = _build_docs()
_CORPUS = [list(jieba.cut(d["text"])) for d in DOCS]
_BM25 = BM25Okapi(_CORPUS)

_STOPWORDS = set("的 了 是 在 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 自己 这 那 什么 怎么 如何 吗 呢 啊".split())


EMBED_MODEL = os.environ.get("PLANT_EMBED_MODEL", "bge-m3")
EMBED_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "outputs", "kb_embeddings.json")


def _embed(text, host=None):
    """取一段文本的向量；Ollama 不可用时返回 None。"""
    try:
        import requests
        r = requests.post(
            f"{host or OLLAMA_HOST}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=60,
        )
        if r.status_code != 200:
            return None
        return r.json().get("embedding") or None
    except Exception:
        return None


_LOCAL_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets", "embedding", "bge-small-zh-v1.5")
_local_embedder = None
_local_failed = False


def _get_local_embedder():
    """加载本地 bge-small-zh；不可用返回 None。"""
    global _local_embedder, _local_failed
    if _local_embedder is not None:
        return _local_embedder
    if _local_failed:
        return None
    if not os.path.isdir(_LOCAL_MODEL_DIR):
        _local_failed = True
        return None
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(_LOCAL_MODEL_DIR)
        mdl = AutoModel.from_pretrained(_LOCAL_MODEL_DIR)
        mdl.eval()
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mdl.to(dev)
        _local_embedder = (tok, mdl, dev)
        return _local_embedder
    except Exception:
        _local_failed = True
        return None


def _local_embed(texts):
    """用本地模型编码一批文本，返回归一化向量列表。"""
    emb = _get_local_embedder()
    if not emb:
        return None
    try:
        import torch
        import torch.nn.functional as F
        tok, mdl, dev = emb
        if isinstance(texts, str):
            texts = [texts]

        # 分批编码：一次喂几千条会爆内存，失败后静默回退到别的模型，
        # 导致向量维度不一致（512 vs 1024）。
        BATCH = 32
        out_vecs = []
        with torch.no_grad():
            for i in range(0, len(texts), BATCH):
                chunk = texts[i:i + BATCH]
                inputs = tok(chunk, padding=True, truncation=True,
                             max_length=512, return_tensors="pt").to(dev)
                out = mdl(**inputs)
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                vec = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
                vec = F.normalize(vec, p=2, dim=1)
                out_vecs.append(vec.cpu())
        return torch.cat(out_vecs, 0).numpy().tolist()
    except Exception:
        return None


def _cos(a, b):
    import numpy as np
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def _build_embedding_cache():
    """预计算全部知识条目的向量并落盘；失败返回空。"""
    import json as _json
    if os.path.exists(EMBED_CACHE):
        try:
            with io.open(EMBED_CACHE, encoding="utf-8") as f:
                data = _json.load(f)
            vs = data.get("vectors", [])
            if (len(vs) == len(DOCS)
                    and data.get("model") == "bge-small-zh-local"
                    and vs and len(vs[0]) == 512):
                return vs
        except Exception:
            pass

    texts = [d["title"] + "。" + d["text"][:900] for d in DOCS]

    # ① 优先本地 bge-small-zh（离线、体积 92MB）
    vecs = _local_embed(texts)
    if vecs:
        vectors = vecs
        model_tag = "bge-small-zh-local"
    else:
        # 本地模型不可用时不回退到 bge-m3：两者维度不同（512 vs 1024），
        # 混用会让 _cos 直接崩。没有本地模型就交给 BM25。
        return None

    try:
        os.makedirs(os.path.dirname(EMBED_CACHE), exist_ok=True)
        with io.open(EMBED_CACHE, "w", encoding="utf-8") as f:
            _json.dump({"model": model_tag, "vectors": vectors}, f)
    except Exception:
        pass
    return vectors


_VECTORS = None


def retrieve_vector(query, k=4):
    """向量检索；不可用时返回空表示回退 BM25。"""
    global _VECTORS
    if _VECTORS is None:
        _VECTORS = _build_embedding_cache()
    if not _VECTORS:
        return []

    qv_list = _local_embed([query])
    qv = qv_list[0] if qv_list else _embed(query)
    if not qv:
        return []

    scored = [(_cos(qv, v), i) for i, v in enumerate(_VECTORS)]
    scored.sort(reverse=True)
    out = []
    for score, i in scored[:k]:
        if score < 0.35:          # 相关性阈值，避免塞入无关条目
            continue
        out.append((DOCS[i], float(score)))
    return out
def retrieve(query: str, k: int = 4):
    """混合检索：向量（本地 bge-small）为主，BM25 补充，避免专有名词漏召回。"""
    vec = retrieve_vector(query, k * 2)
    bm = _retrieve_bm25(query, k * 2)

    if not vec and not bm:
        return []

    # 归一化各自分数后加权合并
    def _norm(items):
        if not items:
            return {}
        mx = max(s for _, s in items) or 1.0
        return {d["title"]: s / mx for d, s in items}

    sv, sb = _norm(vec), _norm(bm)
    titles = {d["title"]: d for d, _ in list(vec) + list(bm)}
    merged = []
    for title, doc in titles.items():
        score = 0.7 * sv.get(title, 0.0) + 0.3 * sb.get(title, 0.0)
        merged.append((doc, score))
    merged.sort(key=lambda x: -x[1])

    # 问题里点名了作物时，只保留该作物的条目（消除跨作物混淆）
    want_crop = _extract_crop(query)
    if want_crop:
        keep = [(d, s) for d, s in merged if _doc_crop(d) == want_crop]
        if keep:
            merged = keep
        else:
            # 该作物在候选里一条都没有，退一步：从全量库里捞该作物的条目，
            # 按向量相似度给对方排序。总比答成别的作物强。
            cand = [(i, d) for i, d in enumerate(DOCS)
                    if _doc_crop(d) == want_crop]
            if cand:
                merged = [(d, 0.5) for _, d in cand[:k]]

    # 问题里点名的病害优先（例如"高粱黑穗病"不该答成"甘蔗黑穗病"）

    # 过滤掉相关度太低的
    top = merged[0][1] if merged else 0
    out = [(d, s) for d, s in merged[:k] if s >= max(0.25, top * 0.45)]
    return out


def _retrieve_bm25(query: str, k: int = 4):
    """BM25 检索，返回 [(文档, 分数), ...]。"""
    tokens = [t for t in jieba.cut(query) if t.strip() and t not in _STOPWORDS]
    if not tokens:
        return []
    scores = _BM25.get_scores(tokens)

    # 额外的类别名/作物名精确匹配加成
    for i, d in enumerate(DOCS):
        title = d["title"]
        for tok in tokens:
            if tok in title:
                scores[i] += 3.0

    order = sorted(range(len(DOCS)), key=lambda i: scores[i], reverse=True)
    out = []
    for i in order[:k]:
        if scores[i] <= 0:
            continue
        out.append((DOCS[i], float(scores[i])))
    return out


# ---------------- 大模型调用 ----------------
def _read_env(name):
    v = os.environ.get(name)
    if v:
        return v.strip()
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment") as k:
            return winreg.QueryValueEx(k, name)[0].strip()
    except Exception:
        pass
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            return winreg.QueryValueEx(k, name)[0].strip()
    except Exception:
        return ""


OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("PLANT_OLLAMA_MODEL", "qwen3.5:9b")


def _ollama_tags():
    """列出本机 Ollama 可用模型；不可用时返回空列表。"""
    try:
        import requests
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
        if r.status_code != 200:
            return []
        return [m.get("name", "") for m in r.json().get("models", [])]
    except Exception:
        return []


def _pick_ollama_model():
    """优先配置的模型，其次按质量从高到低挑本机已有的。"""
    tags = _ollama_tags()
    if not tags:
        return None
    if OLLAMA_MODEL in tags:
        return OLLAMA_MODEL
    for cand in ("qwen3.5:9b", "qwen3-vl:4b", "qwen3:4b", "qwen2.5:7b",
                 "qwen2.5:3b", "glm4:9b", "llama3.1:8b"):
        if cand in tags:
            return cand
    return tags[0]


def _call_ollama(messages, timeout=150):
    """通过本机 Ollama 推理；关闭 thinking 以拿到正式回答。"""
    try:
        import requests
        model = _pick_ollama_model()
        if not model:
            return None
        r = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "think": False,
                "options": {"temperature": 0.3, "num_predict": 600},
            },
            timeout=timeout,
        )
        if r.status_code != 200:
            return None
        return (r.json().get("message", {}).get("content") or "").strip() or None
    except Exception:
        return None


def _llm_available():
    return bool(_read_env("DEEPSEEK_API_KEY") or _read_env("OPENAI_API_KEY"))


def _call_llm(messages, timeout=40):
    """优先 DeepSeek，其次 OpenAI。失败返回 None。"""
    try:
        from openai import OpenAI
    except Exception:
        return None

    dk = _read_env("DEEPSEEK_API_KEY")
    ok = _read_env("OPENAI_API_KEY")

    if dk:
        client = OpenAI(api_key=dk, base_url="https://api.deepseek.com", timeout=timeout)
        model = "deepseek-chat"
    elif ok:
        client = OpenAI(api_key=ok, timeout=timeout)
        model = "gpt-4o-mini"
    else:
        return None

    try:
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0.3, max_tokens=800,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None


SYSTEM_PROMPT = (
    "你是植物叶片病害诊断系统的助手。只依据提供的资料回答，不要编造。"
    "如果资料里没有答案，就直接说资料中没有相关内容。"
    "回答用简体中文，先给结论，再给要点，控制在 300 字以内。"
)


# 系统实际覆盖的作物（14 种）
# 支持范围取自已合并的全量知识库（精编 132 条 + 扩充 2536 条）
def _refresh_scope():
    """把已经收录的作物从未覆盖清单里剔除（数据集扩充后自动同步）。"""
    g = globals()
    alt = g.get("CROP_ALTERNATIVE")
    for crop in list(UNSUPPORTED_CROPS):
        if crop in SUPPORTED_CROPS:
            UNSUPPORTED_CROPS.pop(crop, None)
            if alt is not None:
                alt.pop(crop, None)



SUPPORTED_CROPS = {v["crop_cn"] for v in knowledge.ALL_KNOWLEDGE.values()}
SUPPORTED_CROPS.discard("通用")

# 扩充库里有些作物用了别称（粟=谷子），把它们映射到标准名，
# 否则用户问"谷子"会被判成"未收录"。
# 扩充库里这些作物被归到了"通用"（导数据时前缀没拆开），
# 但它们确实是可回答的，所以直接登记为支持范围。
_CROP_SYNONYM = {"粟": "谷子", "糜子": "黍", "马铃薯": "土豆",
                 "甜菜": "甜菜", "谷子": "谷子", "粟": "粟"}
for _std, _alias in list(_CROP_SYNONYM.items()):
    SUPPORTED_CROPS.add(_alias)
    SUPPORTED_CROPS.add(_std)

# 别名补全后再剔除已收录作物，否则"谷子"会被误判为未覆盖

# 常见但系统未收录的作物（用于越界识别）
UNSUPPORTED_CROPS = {
    "水稻": "禾本科",
    "小麦": "禾本科",
    "大麦": "禾本科",
    "高粱": "禾本科",
    "谷子": "禾本科",
    "黄瓜": "葫芦科",
    "西瓜": "葫芦科",
    "甜瓜": "葫芦科",
    "苦瓜": "葫芦科",
    "冬瓜": "葫芦科",
    "茄子": "茄科",
    "辣椒": "茄科",
    "烟草": "茄科",
    "白菜": "十字花科",
    "甘蓝": "十字花科",
    "萝卜": "十字花科",
    "油菜": "十字花科",
    "花生": "豆科",
    "棉花": "锦葵科",
    "茶叶": "山茶科",
    "甘蔗": "禾本科",
    "香蕉": "芭蕉科",
    "芒果": "漆树科",
    "荔枝": "无患子科",
    "猕猴桃": "猕猴桃科",
    "洋葱": "百合科",
    "大蒜": "百合科",
    "韭菜": "百合科",
}

# 别名与未覆盖表都就绪后，剔除已收录作物
_refresh_scope()

# 未收录作物 -> 系统内相近的替代参考
CROP_ALTERNATIVE = {
    "水稻": ["玉米"], "小麦": ["玉米"], "大麦": ["玉米"], "高粱": ["玉米"],
    "谷子": ["玉米"], "甘蔗": ["玉米"],
    "黄瓜": ["南瓜"], "西瓜": ["南瓜"], "甜瓜": ["南瓜"], "苦瓜": ["南瓜"], "冬瓜": ["南瓜"],
    "茄子": ["番茄", "甜椒"], "辣椒": ["甜椒"], "烟草": ["番茄"],
    "白菜": [], "甘蓝": [], "萝卜": [], "油菜": [],
    "花生": ["大豆"], "棉花": [],
    "茶叶": [], "香蕉": [], "芒果": [], "荔枝": [], "猕猴桃": [],
    "洋葱": [], "大蒜": [], "韭菜": [],
}





# 病害领域相关词（用于判断问题是否在系统处理范围内）
_DOMAIN_HINTS = (
    # 作物（从知识库动态补全，见下方函数）
)
_DOMAIN_WORDS = (
    "病", "症", "防治", "用药", "农药", "杀菌", "喷洒", "喷药",
    "叶片", "叶子", "果实", "茎", "根", "花", "苗", "植株",
    "种植", "栽培", "大棚", "田间", "土壤", "施肥", "浇水",
    "病害", "虫害", "感染", "发病", "传染", "枯", "萎", "斑",
    "系统", "模型", "准确率", "数据集", "诊断", "识别", "导出",
    "报告", "历史", "功能", "使用", "上传", "批量", "置信度",
)


# 元问题：用户在问"你是谁/怎么工作/什么原理"
_META_PATTERNS = (
    "你是什么", "你是谁", "你是做什么", "你怎么工作", "你如何工作",
    "你的原理", "什么原理", "怎么实现", "如何实现", "用的什么技术",
    "什么技术", "怎么做的", "如何做到", "你基于", "底层是",
    "你和", "跟chatgpt", "跟gpt", "是什么模型", "什么算法",
    "你能干什么", "你会什么", "你有什么功能", "你能做什么",
)


def _is_meta_question(q: str) -> bool:
    q = (q or "").strip().lower()
    return any(p in q for p in _META_PATTERNS)


def _meta_answer() -> str:
    return (
        "**我是怎么工作的**\n\n"
        "整个问答链路分四步，全部在本地完成，不调用外部 API：\n\n"
        "**① 领域判断**\n"
        "先判断你的问题是否属于植物病害或本系统范畴。\n"
        "无关的问题（比如天气、写代码）会直接说明超出范围，不会硬答。\n\n"
        "**② 实体识别**\n"
        "从问题里抽出作物名和病害名。用别名词典覆盖口语和俗称"
        "（西红柿→番茄、洋芋→土豆），再用编辑距离容错错别字"
        "（晚役病→晚疫病）。\n\n"
        "**③ 意图分类**\n"
        "判断你想问哪方面：怎么防治、有什么症状、发病条件、病害区别，"
        "还是系统用法。用的是 bge 中文嵌入 + 逻辑回归分类器。\n\n"
        "**④ 检索与组织**\n"
        "用向量检索（bge-small-zh，512 维）结合 BM25 关键词检索，"
        "按 0.7:0.3 加权。命中作物后会过滤掉其他作物的条目，"
        "避免「问番茄答土豆」。最后按意图组织成结构化回答，并附上相关图片。\n\n"
        "> 知识库共 132 条，覆盖 28 种作物、127 类病害。"
        "若问题信息不足（比如只问「这叶子咋了」），我会先问你是什么作物，"
        "而不是随便猜一个答案。"
    )


def _is_plant_domain(question: str) -> bool:
    """判断问题是否属于本系统能处理的领域（病害或系统自身）。"""
    q = (question or "").strip()
    if not q:
        return False
    # 提到任一已收录作物 → 相关
    if any(crop in q for crop in SUPPORTED_CROPS):
        return True
    # 提到任一作物别名 → 相关
    if any(alias in q for alias in _CROP_ALIAS):
        return True
    # 含领域关键词 → 相关
    if any(w in q for w in _DOMAIN_WORDS):
        return True
    return False


def _ask_more_message(question: str) -> str:
    """信息不足时，告诉用户还缺什么，而不是随便猜一个病来答。"""
    q = (question or "").strip("？?。 ")
    lines = ["**还需要一点信息才能准确回答**", ""]
    lines.append(f"你问的是「{q}」，但没说清楚是哪种植物的哪种病害。")
    lines.append("")
    lines.append("可以这样补充：")
    lines.append("")
    lines.append("- 加上**作物名**：「**番茄**叶子长斑怎么治」")
    lines.append("- 或直接说**病害名**：「晚疫病怎么防治」")
    lines.append("- 不确定是什么病？用「智能诊断」上传一张叶片照片，我帮你识别")
    lines.append("")
    lines.append("> 我没有随便挑一个病害来答，因为那样很可能答非所问。")
    return "\n".join(lines)


def _off_domain_message() -> str:
    return ("**这个问题超出了本系统的范围**\n\n"
            "我是植物叶片病害诊断助手，可以回答：\n\n"
            "- 某种作物得了什么病、有什么症状\n"
            "- 怎么防治、用什么药、发病条件\n"
            "- 系统的使用方法、模型指标\n\n"
            "可以换个问法，例如「番茄晚疫病怎么防治」"
            "「水稻稻瘟病有什么症状」「三个模型准确率多少」。")


def _detect_out_of_scope(question):
    """判断问题是否超出覆盖范围；返回 (作物, 科属) 或 None。"""
    q = question or ""
    for crop in UNSUPPORTED_CROPS:
        if crop in q:
            return crop, UNSUPPORTED_CROPS[crop]
    return None


# 作物别名表（口语、俗称、错写都落到标准名）
_CROP_ALIAS = {
    "西红柿": "番茄", "番茄": "番茄", "番前": "番茄", "蕃茄": "番茄",
    "马铃薯": "土豆", "洋芋": "土豆", "土豆": "土豆", "地蛋": "土豆",
    "苞谷": "玉米", "包谷": "玉米", "玉米": "玉米", "棒子": "玉米",
    "橘子": "柑橘", "柑橘": "柑橘", "桔子": "柑橘",
    "青椒": "甜椒", "甜椒": "甜椒", "辣椒": "辣椒",
    "奇异果": "猕猴桃", "猕猴桃": "猕猴桃",
    "茶": "茶叶", "茶叶": "茶叶",
    "落花生": "花生", "花生": "花生",
    "稻子": "水稻", "水稻": "水稻",
    "黄豆": "大豆", "大豆": "大豆",
    "葡萄": "葡萄", "苹果": "苹果", "桃": "桃", "小麦": "小麦",
    "黄瓜": "黄瓜", "茄子": "茄子", "香蕉": "香蕉", "芒果": "芒果",
    "荔枝": "荔枝", "棉花": "棉花", "草莓": "草莓", "蓝莓": "蓝莓",
    "树莓": "树莓", "樱桃": "樱桃", "南瓜": "南瓜", "萝卜": "萝卜",
    "甘蔗": "甘蔗",
    "甜菜": "甜菜", "糖萝卜": "甜菜",
    "谷子": "粟", "小米": "粟",
}

# 知识库扩充后作物变多了，让每个支持作物都能被识别出作物名，
# 否则"火龙果软腐病"会抽不出作物，被当成通用病害答成别的作物。
for _c in SUPPORTED_CROPS:
    _CROP_ALIAS.setdefault(_c, _c)


def _extract_crop(question):
    """最长别名优先，避免「番茄」被短词抢占。"""
    best, best_len = None, 0
    for alias, std in _CROP_ALIAS.items():
        if alias in question and len(alias) > best_len:
            best, best_len = std, len(alias)
    return best


# 害虫统称 → 库里的具体种（用户常说"稻飞虱"，库里是"褐飞虱""灰飞虱"）
_PEST_ALIAS = {
    "稻飞虱": ["褐飞虱", "灰飞虱", "白背飞虱"],
    "飞虱": ["褐飞虱", "灰飞虱", "白背飞虱"],
    "稻螟": ["三化螟", "二化螟"],
    "蚜虫": ["蚜", "棉蚜", "麦蚜"],
    "红蜘蛛": ["叶螨", "二斑叶螨"],
}

_CROP_DISEASE_CACHE = {}
try:
    pass  # 缓存延迟构建，首次调用时填充
except Exception:
    pass


def _crop_disease_names(crop):
    """某个作物下的全部病害名。预计算一次，避免反复深拷贝知识条目。"""
    if not _CROP_DISEASE_CACHE:
        from collections import defaultdict as _dd
        m = _dd(set)
        for src in (knowledge.KNOWLEDGE, getattr(knowledge, "KG_KNOWLEDGE", {})):
            for info in src.values():
                c_ = info.get("crop_cn")
                d_ = info.get("disease_cn")
                if c_ and d_ and "健康" not in d_:
                    m[c_].add(d_)
        _CROP_DISEASE_CACHE.update({k: sorted(v, key=len, reverse=True)
                                    for k, v in m.items()})
    return _CROP_DISEASE_CACHE.get(crop, [])


_DISEASE_NAMES_CACHE = None


def _all_disease_names():
    global _DISEASE_NAMES_CACHE
    if _DISEASE_NAMES_CACHE is None:
        out = set()
        # 精编库 + 扩充库都算（扩充库里有"三化螟""稻飞虱"这类不含"病"字的名字）
        for src in (knowledge.KNOWLEDGE, getattr(knowledge, "KG_KNOWLEDGE", {})):
            for info in src.values():
                dn = info.get("disease_cn")
                if dn and "健康" not in dn and len(dn) >= 2:
                    out.add(dn)
        _DISEASE_NAMES_CACHE = sorted(out, key=len, reverse=True)
    return _DISEASE_NAMES_CACHE


def _extract_disease_name(question):
    """实体抽取：先精确命中知识库病害，再用编辑距离容错（覆盖错别字）。"""
    import re
    import difflib

    q = question or ""
    names = _all_disease_names()

    # 0) 害虫统称先展开（"稻飞虱" → "褐飞虱"）
    for alias, subs in _PEST_ALIAS.items():
        if alias in q:
            for sub in subs:
                if any(sub == n for n in names):
                    return sub

    # 1) 精确命中，长名优先（避免「疫病」抢「晚疫病」）
    #    同时比较去掉括号补注的形式（「黑星病（疮痂病）」应能被「黑星病」命中）
    def _norm(s):
        return re.sub(r"[（(].*?[）)]", "", s or "").strip()

    for d in names:
        if not d:
            continue
        if d in q or _norm(d) in q:
            return d

    # 2) 候选「XX病」→ 去作物前缀 → 编辑距离
    #    问题里点明了作物时，优先在同作物的病害里匹配（"番茄班枯病"不应匹配到"褐枯病"）
    asked_crop = _extract_crop(q)
    pool = names
    if asked_crop:
        same = _crop_disease_names(asked_crop)
        if same:
            pool = same

    # 排除这些词开头的片段（"危害的发病"这类明显不是病名）
    _BAD_HEAD = ("的", "了", "是", "有", "在", "和", "与", "及", "或",
                 "危害", "发生", "引起", "导致", "造成", "出现", "表现")

    best, best_ratio = None, 0.0
    for cand in re.findall(r"[\u4e00-\u9fa5]{1,8}病", q):
        if cand.startswith(_BAD_HEAD):
            continue
        if len(cand) > 6:            # 病名一般不超过 6 字
            continue
        frag = cand
        for alias in sorted(_CROP_ALIAS, key=len, reverse=True):
            if frag.startswith(alias) and len(frag) > len(alias):
                frag = frag[len(alias):]
                break
        m = difflib.get_close_matches(frag, pool, n=1, cutoff=0.5)
        if m:
            ratio = difflib.SequenceMatcher(None, frag, m[0]).ratio()
            if ratio > best_ratio:
                best, best_ratio = m[0], ratio
        elif best is None:
            best = frag
    return best


def _disease_in_kb(name):
    """判断这个病害名是否在知识库里（含扩充库）。"""
    if not name:
        return True
    _all = list(knowledge.KNOWLEDGE.values()) +         list(getattr(knowledge, "KG_KNOWLEDGE", {}).values())
    for info in _all:
        dn = info["disease_cn"]
        if name in dn or dn in name:
            return True
        # 去掉作物前缀再比一次
        short = name
        for crop in SUPPORTED_CROPS:
            if short.startswith(crop) and len(short) > len(crop) + 1:
                short = short[len(crop):]
        if short and (short in dn or dn in short):
            return True
    return False


def _disease_scope_message(crop, question):
    """作物在范围内，但问到该作物未收录的病害时，列出已收录的病害供选择。"""
    supported = [(i["disease_cn"], cls) for cls, i in knowledge.KNOWLEDGE.items()
                 if i["crop_cn"] == crop]
    if not supported:
        return None
    lines = [f"**「{crop}」的这条病害暂未收录**", ""]
    lines.append(f"系统目前收录了「{crop}」的以下 {len(supported)} 类：")
    lines.append("")
    for name, cls in supported:
        lines.append(f"- {crop}·{name}")
    lines.append("")
    lines.append("**建议**：")
    lines.append("1. 换一种上面的病害名称重新提问；")
    lines.append("2. 或用「智能诊断」上传叶片照片，让模型判断属于哪一类；")
    lines.append("3. 也可以联系当地植保站确认。")
    return "\n".join(lines)


def _scope_message(crop, family):
    """越界时的说明：讲清边界、给替代、给求助途径。"""
    supported = sorted(SUPPORTED_CROPS)
    lines = [
        f"**本系统暂不覆盖「{crop}」**",
        "",
        f"当前知识库覆盖 {len(supported)} 种作物、{len(knowledge.KNOWLEDGE)} 类病害：",
        "、".join(supported) + "。",
        "",
    ]

    alts = CROP_ALTERNATIVE.get(crop) or []
    if alts:
        lines.append(f"「{crop}」与系统内的 " + "、".join(alts) + f" 同属{family}，"
                     f"病害类型有相通之处，可以先参考这些作物的病害描述与防治思路：")
        lines.append("")
        for a in alts:
            for cls, info in knowledge.KNOWLEDGE.items():
                if info["crop_cn"] == a:
                    lines.append(f"- {a}·{info['disease_cn']}")
        lines.append("")
    else:
        lines.append(f"系统内暂无与「{crop}」同属{family}的作物，无法给出相近参考。")
        lines.append("")

    lines.append("**建议**：")
    lines.append("1. 联系当地植保站或农技推广部门，获取该作物的权威诊断与用药方案；")
    lines.append("2. 也可以换一种系统已覆盖的作物重新提问。")
    return "\n".join(lines)


_INTENT_RULES = [
    ("control", ["怎么防治", "怎么治", "用什么药", "打什么药", "药剂", "防治", "处理", "怎么办", "如何治"]),
    ("symptom", ["什么症状", "症状", "表现", "长什么样", "怎么判断", "识别", "特征"]),
    ("occurrence", ["什么条件", "发生规律", "什么时候", "什么季节", "气候", "温度", "湿度",
                    "为什么发生", "容易发生", "为什么危害", "危害这么高", "危害大",
                    "为什么严重", "为什么难防", "流行条件", "发病条件"]),
    ("compare", ["区别", "区分", "差别", "不同", "怎么分辨", "易混淆"]),
    ("system", ["系统", "模型", "准确率", "数据集", "导出", "报告", "怎么用", "使用", "功能"]),
]


# ---------------- 意图分类器（bge 嵌入 + 逻辑回归）----------------
# 关键词规则（约 43%）只作为兜底；正常情况下走分类器（严格泛化测试 84.9%）。
_INTENT_CLF = None
_INTENT_EMB = None
_INTENT_LOAD_FAILED = False


def _load_intent_clf():
    """懒加载分类器与嵌入模型；失败返回 None（调用方回退规则）。"""
    global _INTENT_CLF, _INTENT_EMB, _INTENT_LOAD_FAILED
    if _INTENT_CLF is not None or _INTENT_LOAD_FAILED:
        return _INTENT_CLF
    try:
        import os
        import pickle

        here = os.path.dirname(os.path.abspath(__file__))
        pkl = os.path.join(here, "intent_model", "intent_clf.pkl")
        if not os.path.exists(pkl):
            _INTENT_LOAD_FAILED = True
            return None
        with open(pkl, "rb") as f:
            _INTENT_CLF = pickle.load(f)

        # 复用项目里已有的嵌入模型（RAG 检索用的同一个）
        import torch
        from transformers import AutoTokenizer, AutoModel

        local = os.path.join(os.path.dirname(here), "assets", "embedding",
                             "bge-small-zh-v1.5")
        if not os.path.isdir(local):
            local = "BAAI/bge-small-zh-v1.5"
        tok = AutoTokenizer.from_pretrained(local)
        mdl = AutoModel.from_pretrained(local).eval()
        _INTENT_EMB = (tok, mdl)
        return _INTENT_CLF
    except Exception:
        _INTENT_LOAD_FAILED = True
        _INTENT_CLF = None
        return None


def _detect_intent_by_model(q):
    """模型预测意图；不可用或置信度太低时返回 None。"""
    if _load_intent_clf() is None:
        return None
    try:
        import numpy as np
        import torch

        tok, mdl = _INTENT_EMB
        with torch.no_grad():
            enc = tok([q], padding=True, truncation=True, max_length=64,
                      return_tensors="pt")
            h = mdl(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
            v = torch.nn.functional.normalize(v, dim=-1).numpy()
        proba = _INTENT_CLF["clf"].predict_proba(v)[0]
        idx = int(np.argmax(proba))
        if float(proba[idx]) < 0.30:      # 太不确定就交给规则
            return None
        return _INTENT_CLF["classes"][idx]
    except Exception:
        return None


def _detect_intent(q):
    # ① 模型优先
    try:
        _m = _detect_intent_by_model(q)
        if _m:
            return _m
    except Exception:
        pass
    # ② 关键词规则兜底
    for name, keys in _INTENT_RULES:
        if any(k in q for k in keys):
            return name
    return "general"


def _clean(text):
    """去掉知识库条目里的来源括注，让回答更干净。"""
    import re
    text = re.sub(r"（来源：[^）]*）", "", text)
    text = re.sub(r"（依据[^）]*）", "", text)
    return text.strip()


def compose_answer(question, hits, top_k=3):
    """不依赖大模型，直接从知识库组织出可读回答。"""
    if not hits:
        return None

    intent = _detect_intent(question)

    # 问题里提到的作物优先（避免"番茄晚疫病"被检索成"土豆晚疫病"）
    asked_crops = [c for c in SUPPORTED_CROPS if c in question]

    # 跳过"病害概览"这类没有 cls 的条目，取第一条真正的病害
    top_doc = None
    if asked_crops:
        for _d, _s in hits:
            if _doc_crop(_d) in asked_crops:
                top_doc = _d
                break
    if top_doc is None:
        # 扩充库条目没有 cls，但一样是有效结果，所以 cls 和 kg_key 都认
        for _d, _s in hits:
            if _d.get("cls") or _d.get("kg_key"):
                top_doc = _d
                break
    if top_doc is None:
        top_doc = hits[0][0]
    title = top_doc.get("title", "")

    # 从知识库取结构化字段
    info = None
    ctrl = {}
    cls = top_doc.get("cls")
    if cls:
        try:
            info = knowledge.get_info(cls)
            ctrl = knowledge.get_control(cls)
        except Exception:
            pass

    lines = []

    if intent == "symptom" and info:
        lines.append(f"**{title}的症状**")
        lines.append("")
        lines.append(info.get("symptoms", ""))
        lines.append("")
        lines.append(f"病原类型：{info.get('pathogen', '—')}；危害程度：{info.get('severity', '—')}")
        if ctrl.get("农业防治"):
            lines.append("")
            lines.append("**防治要点（摘要）**")
            for i, t in enumerate(ctrl["农业防治"][:2], 1):
                lines.append(f"{i}. {_clean(t)}")

    elif intent == "control" and (ctrl.get("农业防治") or ctrl.get("化学防治")):
        lines.append(f"**{title}的防治方案**")
        lines.append("")
        for grp, label in (("农业防治", "农业防治"), ("化学防治", "化学防治"), ("注意事项", "注意事项")):
            items = ctrl.get(grp) or []
            if not items:
                continue
            lines.append(f"**{label}**")
            for i, t in enumerate(items[:3], 1):
                lines.append(f"{i}. {_clean(t)}")
            lines.append("")

    elif intent == "occurrence" and info:
        lines.append(f"**{title}的发病条件**")
        lines.append("")
        if info.get("occurrence"):
            lines.append(info["occurrence"])
        else:
            # 没有专门的发生规律资料时，用已有的病原/症状/防治信息组织回答，
            # 而不是干巴巴一句"暂无"
            lines.append("该病的发生规律记载较少，以下是已有资料中与发病相关的信息：")
            lines.append("")
            pk = info.get("pathogen") or "—"
            sv = info.get("severity") or "—"
            lines.append(f"- **病原类型**：{pk}")
            lines.append(f"- **危害程度**：{sv}")
            sympt = (info.get("symptoms") or "").strip()
            if sympt:
                short = sympt[:90].rstrip("，。；") + ("…" if len(sympt) > 90 else "")
                lines.append(f"- **典型症状**：{short}")
            ctrl = info.get("control") or []
            if ctrl:
                lines.append(f"- **防治重点**：{_clean(ctrl[0])}")
            lines.append("")
            lines.append("如需更详细的发病规律，可参考当地植保部门的病虫情报。")
        lines.append("")
        lines.append(f"病原类型：{info.get('pathogen', '—')}；危害程度：{info.get('severity', '—')}")

    elif intent == "compare":
        lines.append(f"**关于「{question.strip('？?')}」**")
        lines.append("")
        lines.append("知识库里与这个问题相关的是：")
        lines.append("")
        for d, _ in hits[:3]:
            lines.append(f"- **{d['title']}**：{_clean(d['text'])[:120]}…")
        lines.append("")
        lines.append("如需逐项对比，可以分别查询这几类的症状与防治。")

    elif intent == "system":
        lines.append("**系统相关信息**")
        lines.append("")
        for d, _ in hits[:3]:
            lines.append(_clean(d["text"]))
            lines.append("")

    else:
        # 通用：给结论 + 相关条目
        if info:
            lines.append(f"**{title}**")
            lines.append("")
            lines.append(info.get("symptoms", ""))
            lines.append("")
            if ctrl.get("化学防治"):
                lines.append("**化学防治**")
                for i, t in enumerate(ctrl["化学防治"][:2], 1):
                    lines.append(f"{i}. {_clean(t)}")
        else:
            for d, _ in hits[:3]:
                lines.append(f"**{d['title']}**")
                lines.append(_clean(d["text"]))
                lines.append("")

    return "\n".join(lines).strip()


def answer(question: str, top_k: int = 4):
    """
    返回 dict:
        answer      回答文本
        sources     引用条目标题列表
        mode        'LLM' 或 '检索'
        elapsed     耗时秒
    """
    t0 = time.time()
    question = (question or "").strip()
    if not question:
        return {"answer": "请输入问题。", "sources": [], "mode": "-", "elapsed": 0}

    # 元问题："你是什么原理""你是谁"——解释系统本身
    if _is_meta_question(question):
        return {"answer": _meta_answer(), "sources": [],
                "mode": "系统说明", "top_cls": None,
                "elapsed": round(time.time() - t0, 2)}

    # 领域检查：跟植物病害、系统本身都无关的问题直接拒答
    if not _is_plant_domain(question):
        return {"answer": _off_domain_message(), "sources": [],
                "mode": "超出范围", "top_cls": None,
                "elapsed": round(time.time() - t0, 2)}

    # 越界检查：问到系统未收录的作物时，明确说明边界，不硬答
    oos = _detect_out_of_scope(question)
    if oos:
        crop, family = oos
        return {"answer": _scope_message(crop, family), "sources": [],
                "mode": "超出范围", "elapsed": round(time.time() - t0, 2)}

    # 病害名检查：提到的病害不在知识库里时，列出该作物已收录的病
    dn = _extract_disease_name(question)
    if dn and not _disease_in_kb(dn):
        for crop in SUPPORTED_CROPS:
            if crop in question:
                msg = _disease_scope_message(crop, question)
                if msg:
                    return {"answer": msg, "sources": [],
                            "mode": "未收录", "elapsed": round(time.time() - t0, 2)}
        return {"answer": f"**系统暂未收录「{dn}」**\n\n"
                          f"当前知识库覆盖 {len(SUPPORTED_CROPS)} 种作物、{len(knowledge.KNOWLEDGE)} 类病害。"
                          f"可以换一种已收录的病害名称提问，或用「智能诊断」上传照片让模型判断。",
                "sources": [], "mode": "未收录", "elapsed": round(time.time() - t0, 2)}

    hits = retrieve(question, top_k)

    # 如果问题里明确点出了某个已收录的病害名，把它提到第一位。
    # 否则会出现"问稻飞虱，答条纹叶枯病"这种语义相关但主体错位的情况。
    _dn_exact = _extract_disease_name(question)
    if _dn_exact:
        _exact_doc = None
        for _d, _s in hits:
            if _d.get("kg_key") or _d.get("cls"):
                _t = _d.get("title", "")
                if _t.endswith("·" + _dn_exact) or _t.endswith("·" + _dn_exact + "病"):
                    _exact_doc = _d
                    break
        if _exact_doc is None:
            # 检索结果里没有，就去全量库里精确找。
            # 但问题里点了作物时，只能在该作物里找——否则"高粱黑穗病"
            # 会被"甘蔗·黑穗病"（同样叫黑穗病）顶掉。
            _asked = _extract_crop(question)
            for _src in (knowledge.KNOWLEDGE, getattr(knowledge, "KG_KNOWLEDGE", {})):
                for _k, _info in _src.items():
                    if _info.get("disease_cn") == _dn_exact and (
                            not _asked or _info.get("crop_cn") == _asked):
                        # text 必须填上，否则后面组织回答时拿到空内容，
                        # 会退化成"标题对了内容却串到下一条"。
                        _parts = [f"{_info.get('crop_cn','')}{_info.get('disease_cn','')}。"]
                        if _info.get("symptoms"):
                            _parts.append(f"症状：{_info['symptoms']}。")
                        if _info.get("occurrence"):
                            _parts.append(f"发生规律：{_info['occurrence']}。")
                        if _info.get("control"):
                            _ctl = _info["control"]
                            _ctext = "；".join(_ctl) if isinstance(_ctl, list) else str(_ctl)
                            _parts.append(f"防治：{_ctext}。")
                        _exact_doc = {
                            "title": f"{_info['crop_cn']}·{_info['disease_cn']}",
                            "text": "".join(_parts),
                            "cls": _k if _k in knowledge.KNOWLEDGE else None,
                            "kg_key": _k if _k not in knowledge.KNOWLEDGE else None,
                        }
                        break
                if _exact_doc:
                    break
        if _exact_doc:
            hits = [(_exact_doc, 1.0)] + [(d, s) for d, s in hits
                                          if d.get("title") != _exact_doc["title"]]

    # 作物在范围内、但病害未收录：列出该作物已收录的病害
    if hits and hits[0][1] < 0.45:
        for crop in SUPPORTED_CROPS:
            if crop in question:
                msg = _disease_scope_message(crop, question)
                if msg:
                    return {"answer": msg, "sources": [],
                            "mode": "未收录", "elapsed": round(time.time() - t0, 2)}

    if not hits:
        # 兜底：用最相近的标题做模糊匹配
        titles = [d["title"] for d in DOCS]
        close = difflib.get_close_matches(question, titles, n=3, cutoff=0.3)
        if close:
            hits = [(d, 0.0) for d in DOCS if d["title"] in close]

    if not hits:
        return {
            "answer": "知识库里没有匹配到相关内容。可以换个问法，例如「番茄晚疫病怎么防治」"
                      "「三个模型的准确率分别是多少」「怎么导出诊断报告」。",
            "sources": [], "mode": "检索", "elapsed": round(time.time() - t0, 2),
        }

    # 信息不足时追问，而不是随便挑一条回答
    _want_crop = _extract_crop(question)
    _want_dis = _extract_disease_name(question)
    _intent_now = _detect_intent(question)

    # 病害类问题，但既没提作物也没提病害 → 追问
    if (_intent_now in ("control", "symptom", "occurrence")
            and not _want_crop and not _want_dis):
        return {"answer": _ask_more_message(question), "sources": [],
                "mode": "需要补充信息", "top_cls": None,
                "elapsed": round(time.time() - t0, 2)}

    # 提取到病害但知识库里没有 → 由后面的"未收录"分支处理
    context = "\n\n".join(f"【{d['title']}】{d['text']}" for d, _ in hits)
    sources = [d["title"] for d, _ in hits]
    # 最相关类别（供前端配图使用）：跳过"病害概览"这类没有 cls 的条目
    top_cls = None
    for _d, _s in hits:
        if _d.get("cls"):
            top_cls = _d["cls"]
            break

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{question}"},
    ]

    # 生成模型默认关闭：本系统用「向量检索 + 结构化组织」回答，不依赖任何大模型。
    # 如需接入本机 Ollama，设置环境变量 PLANT_USE_LLM=1 即可启用。
    if os.environ.get("PLANT_USE_LLM") == "1":
        text = _call_ollama(messages)
        if text:
            return {"answer": text, "sources": sources, "mode": "本地模型",
                    "top_cls": top_cls, "elapsed": round(time.time() - t0, 2)}

    # ② 其次云端（仅在用户显式开启时）
    if os.environ.get("PLANT_ALLOW_CLOUD_LLM") == "1" and _llm_available():
        text = _call_llm(messages)
        if text:
            return {"answer": text, "sources": sources, "mode": "云端",
                    "top_cls": top_cls, "elapsed": round(time.time() - t0, 2)}

    # 降级：结构化组织知识库内容（不依赖大模型）
    composed = compose_answer(question, hits)
    if composed:
        return {"answer": composed, "sources": sources, "mode": "知识库",
                "top_cls": top_cls, "elapsed": round(time.time() - t0, 2)}

    parts = ["根据知识库检索到的内容：", ""]
    for d, _ in hits[:3]:
        parts.append(f"**{d['title']}**")
        parts.append(d["text"])
        parts.append("")
    return {"answer": "\n".join(parts), "sources": sources, "mode": "检索",
            "elapsed": round(time.time() - t0, 2)}


def examples():
    return [
        "番茄晚疫病有什么症状，怎么防治？",
        "早疫病和斑枯病怎么区分？",
        "三个模型的准确率和速度差别大吗？",
        "土豆晚疫病为什么危害这么高？",
        "系统怎么导出诊断报告？",
        "葡萄黑腐病怎么处理？",
    ]


if __name__ == "__main__":
    print("文档数:", len(DOCS), " LLM 可用:", _llm_available())
    for q in examples()[:3]:
        r = answer(q)
        print("\nQ:", q)
        print("mode:", r["mode"], " sources:", r["sources"], " time:", r["elapsed"])
        print(r["answer"][:200])
