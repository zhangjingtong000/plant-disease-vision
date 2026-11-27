# -*- coding: utf-8 -*-
"""
knowledge.py — 38 类作物病害知识库

每条包含：作物中文名、病害中文名、病原类型、危害程度、症状描述、防治建议。
"""

CROP_CN = {
    "Apple": "苹果", "Blueberry": "蓝莓", "Cherry": "樱桃", "Corn": "玉米",
    "Grape": "葡萄", "Orange": "柑橘", "Peach": "桃", "Pepper,_bell": "甜椒",
    "Potato": "土豆", "Raspberry": "树莓", "Soybean": "大豆", "Squash": "南瓜",
    "Strawberry": "草莓", "Tomato": "番茄",
}

KNOWLEDGE = {
    "Apple__Apple_scab": {
        "crop_cn": "苹果", "disease_cn": "黑星病（疮痂病）", "pathogen": "真菌",
        "severity": "中",
        "symptoms": "叶片正面出现橄榄绿至黑褐色圆形斑点，边缘呈放射状；后期病斑木栓化、叶片畸形卷曲，果实表面形成黑褐色疮痂。",
        "control": ["清除并烧毁落叶、病枝，减少越冬菌源",
                    "萌芽前喷施石硫合剂，落花后 7~10 天开始喷保护性杀菌剂",
                    "选用抗病品种，改善果园通风透光"],
    },
    "Apple__Black_rot": {
        "crop_cn": "苹果", "disease_cn": "黑腐病", "pathogen": "真菌", "severity": "高",
        "symptoms": "叶片出现边缘清晰的紫褐色同心轮纹斑，中央灰白并散生黑色小点；果实由褐色软腐转为黑色僵果。",
        "control": ["及时清除病果、病枝并深埋",
                    "坐果后套袋保护",
                    "雨季前后喷施代森锰锌或苯醚甲环唑"],
    },
    "Apple__Cedar_apple_rust": {
        "crop_cn": "苹果", "disease_cn": "锈病", "pathogen": "真菌", "severity": "中",
        "symptoms": "叶片正面出现橙黄色近圆形斑点，边缘有黄色晕圈；后期叶背长出黄褐色毛状突起（锈子器）。",
        "control": ["砍除果园周边桧柏等转主寄主",
                    "展叶期至幼果期喷施三唑类杀菌剂",
                    "选用抗锈品种"],
    },
    "Apple___healthy": {
        "crop_cn": "苹果", "disease_cn": "健康", "pathogen": "无", "severity": "无",
        "symptoms": "叶片色泽均匀、无斑点与畸变，边缘完整。",
        "control": ["保持常规水肥管理与病虫害监测即可"],
    },
    "Blueberry___healthy": {
        "crop_cn": "蓝莓", "disease_cn": "健康", "pathogen": "无", "severity": "无",
        "symptoms": "叶片深绿有光泽，无斑点和失绿现象。",
        "control": ["维持土壤 pH 4.5~5.5，定期修剪通风"],
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "crop_cn": "樱桃", "disease_cn": "白粉病", "pathogen": "真菌", "severity": "中",
        "symptoms": "叶面出现白色粉状霉层，多从叶背开始；叶片扭曲、变褐，严重时早期脱落。",
        "control": ["减少密植，增强通风",
                    "发病初期喷施醚菌酯或粉锈宁",
                    "清除病叶与杂草"],
    },
    "Cherry_(including_sour)___healthy": {
        "crop_cn": "樱桃", "disease_cn": "健康", "pathogen": "无", "severity": "无",
        "symptoms": "叶片完整、色泽正常，无霉层与斑点。",
        "control": ["常规管理即可"],
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "crop_cn": "玉米", "disease_cn": "灰斑病", "pathogen": "真菌", "severity": "高",
        "symptoms": "叶片出现狭长矩形灰褐色斑，边缘深褐，沿叶脉平行延伸；后期病斑连片导致叶枯。",
        "control": ["选用抗病品种、合理轮作",
                    "清除田间病残体",
                    "心叶末期至抽雄期喷施苯醚甲环唑"],
    },
    "Corn_(maize)___Common_rust_": {
        "crop_cn": "玉米", "disease_cn": "普通锈病", "pathogen": "真菌", "severity": "中",
        "symptoms": "叶片两面散生褐色至红褐色隆起夏孢子堆，破裂后散出锈色粉末。",
        "control": ["种抗病品种", "发病初期喷施三唑酮", "增施钾肥提高抗性"],
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "crop_cn": "玉米", "disease_cn": "大斑病", "pathogen": "真菌", "severity": "高",
        "symptoms": "叶片先出现水渍状青灰斑，后扩展为长梭形褐色大斑，边缘有黄色晕圈，病斑可长达十几厘米。",
        "control": ["轮作与深翻减少菌源", "选用抗大斑病品种",
                    "发病初期喷施百菌清或苯醚甲环唑"],
    },
    "Corn_(maize)___healthy": {
        "crop_cn": "玉米", "disease_cn": "健康", "pathogen": "无", "severity": "无",
        "symptoms": "叶片翠绿、纹理清晰，无斑点与萎蔫。",
        "control": ["常规水肥管理"],
    },
    "Grape___Black_rot": {
        "crop_cn": "葡萄", "disease_cn": "黑腐病", "pathogen": "真菌", "severity": "高",
        "symptoms": "叶片出现褐色近圆形病斑，边缘暗褐、中央灰白，表面散生黑色小粒点；果实软腐后干缩成黑色僵果。",
        "control": ["清除病果、病叶与卷须",
                    "花前花后各喷一次杀菌剂",
                    "雨季加强排水与通风"],
    },
    "Grape___Esca_(Black_Measles)": {
        "crop_cn": "葡萄", "disease_cn": "埃斯卡病（黑麻疹）", "pathogen": "真菌", "severity": "高",
        "symptoms": "叶片脉间出现黄化条带，逐渐坏死呈虎斑状；果实表面出现暗色斑点，严重时整株枯死。",
        "control": ["避免大伤口、修剪后保护剪口",
                    "拔除重病株",
                    "减少土壤水分剧烈波动"],
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "crop_cn": "葡萄", "disease_cn": "叶枯病（褐斑病）", "pathogen": "真菌", "severity": "中",
        "symptoms": "叶片出现多角形褐色斑，边缘深褐、中央灰褐，可融合成大片，导致早期落叶。",
        "control": ["秋后清扫落叶", "发病期喷施波尔多液或代森锰锌", "控制负载量"],
    },
    "Grape___healthy": {
        "crop_cn": "葡萄", "disease_cn": "健康", "pathogen": "无", "severity": "无",
        "symptoms": "叶片平展、色泽均匀，无坏死斑。",
        "control": ["常规管理"],
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "crop_cn": "柑橘", "disease_cn": "黄龙病（柑橘绿化病）", "pathogen": "细菌", "severity": "极高",
        "symptoms": "叶片斑驳黄化、叶脉附近保留绿色形成不对称黄化；果实小而畸形、着色不均，味苦。属于毁灭性病害。",
        "control": ["使用无病苗木", "防控木虱媒介（这是唯一传播途径）",
                    "发现病株立即挖除，目前无有效治疗药剂"],
    },
    "Peach___Bacterial_spot": {
        "crop_cn": "桃", "disease_cn": "细菌性穿孔病", "pathogen": "细菌", "severity": "中",
        "symptoms": "叶片出现水渍状紫褐色小斑，随后病斑脱落形成穿孔；果实表面出现褐色凹陷斑并龟裂。",
        "control": ["冬季剪除病枝", "萌芽前喷石硫合剂",
                    "发病期喷施农用链霉素类药剂（注意合规）"],
    },
    "Peach___healthy": {
        "crop_cn": "桃", "disease_cn": "健康", "pathogen": "无", "severity": "无",
        "symptoms": "叶片狭长、色泽正常，无穿孔与斑点。",
        "control": ["常规管理"],
    },
    "Pepper,_bell___Bacterial_spot": {
        "crop_cn": "甜椒", "disease_cn": "细菌性斑点病", "pathogen": "细菌", "severity": "中",
        "symptoms": "叶片出现小型水渍状斑点，后变为褐色并带黄晕；果实上形成疮痂状凹陷斑。",
        "control": ["使用无菌种子与苗", "避免叶面长时间潮湿",
                    "发病初期喷施铜制剂"],
    },
    "Pepper,_bell___healthy": {
        "crop_cn": "甜椒", "disease_cn": "健康", "pathogen": "无", "severity": "无",
        "symptoms": "叶片浓绿平整，无斑点。",
        "control": ["常规管理"],
    },
    "Potato___Early_blight": {
        "crop_cn": "土豆", "disease_cn": "早疫病", "pathogen": "真菌", "severity": "中",
        "symptoms": "下部老叶先出现褐色同心轮纹斑，边缘有黄晕，形似靶心；严重时叶片干枯脱落。",
        "control": ["合理轮作、清除病残体", "增施钾肥",
                    "发病初期喷施代森锰锌或嘧菌酯"],
    },
    "Potato___Late_blight": {
        "crop_cn": "土豆", "disease_cn": "晚疫病", "pathogen": "卵菌", "severity": "极高",
        "symptoms": "叶尖和叶缘出现水渍状暗绿斑，迅速扩展为褐色大斑，湿度高时叶背产生白色霉层；可在一周内毁田。",
        "control": ["选用抗病品种", "高垄栽培、控制田间湿度",
                    "预警期提前喷施烯酰吗啉等药剂"],
    },
    "Potato___healthy": {
        "crop_cn": "土豆", "disease_cn": "健康", "pathogen": "无", "severity": "无",
        "symptoms": "叶片平展、深绿，无坏死斑与霉层。",
        "control": ["常规管理"],
    },
    "Raspberry___healthy": {
        "crop_cn": "树莓", "disease_cn": "健康", "pathogen": "无", "severity": "无",
        "symptoms": "叶片完整，无明显病斑。",
        "control": ["常规管理"],
    },
    "Soybean___healthy": {
        "crop_cn": "大豆", "disease_cn": "健康", "pathogen": "无", "severity": "无",
        "symptoms": "叶片翠绿，无病斑与卷曲。",
        "control": ["常规管理"],
    },
    "Squash___Powdery_mildew": {
        "crop_cn": "南瓜", "disease_cn": "白粉病", "pathogen": "真菌", "severity": "中",
        "symptoms": "叶片出现白色粉状霉斑，逐渐布满叶面；叶片变黄、干枯卷曲。",
        "control": ["通风降湿", "发病初期喷施粉锈宁或醚菌酯",
                    "避免偏施氮肥"],
    },
    "Strawberry___Leaf_scorch": {
        "crop_cn": "草莓", "disease_cn": "叶焦病（叶枯病）", "pathogen": "真菌", "severity": "中",
        "symptoms": "叶片出现紫红色小斑并扩大，中央变灰白，后期全叶焦枯呈褐色。",
        "control": ["清除老叶病叶", "控制湿度", "发病期喷施苯醚甲环唑"],
    },
    "Strawberry___healthy": {
        "crop_cn": "草莓", "disease_cn": "健康", "pathogen": "无", "severity": "无",
        "symptoms": "叶片鲜绿有光泽，无焦枯斑点。",
        "control": ["常规管理"],
    },
    "Tomato___Bacterial_spot": {
        "crop_cn": "番茄", "disease_cn": "细菌性斑点病", "pathogen": "细菌", "severity": "中",
        "symptoms": "叶片出现小型深褐斑、周围有黄晕，斑点密集时叶片黄化；果实出现疮痂状小斑。",
        "control": ["种子消毒", "避免喷灌、保持叶面干燥", "发病初期用铜制剂"],
    },
    "Tomato___Early_blight": {
        "crop_cn": "番茄", "disease_cn": "早疫病", "pathogen": "真菌", "severity": "中",
        "symptoms": "下部叶片出现褐色同心轮纹斑，边缘黄化，形似靶心；茎部与果实也可受害。",
        "control": ["轮作、清除病残体", "摘除下部老叶改善通风",
                    "喷施苯醚甲环唑或代森锰锌"],
    },
    "Tomato___Late_blight": {
        "crop_cn": "番茄", "disease_cn": "晚疫病", "pathogen": "卵菌", "severity": "极高",
        "symptoms": "叶缘出现水渍状暗绿斑并迅速扩大为褐色，湿度大时背面有白霉；果实出现油渍状褐斑，极快腐烂。",
        "control": ["控制棚内湿度、避免叶面结露", "发病前预防性喷药",
                    "及时清除病株，防止蔓延"],
    },
    "Tomato___Leaf_Mold": {
        "crop_cn": "番茄", "disease_cn": "叶霉病", "pathogen": "真菌", "severity": "中",
        "symptoms": "叶正面出现淡黄绿色斑，叶背对应处长出灰褐色至紫褐色绒毛状霉层。",
        "control": ["加强通风、降低湿度", "避免密植",
                    "发病初期喷施腐霉利或百菌清"],
    },
    "Tomato___Septoria_leaf_spot": {
        "crop_cn": "番茄", "disease_cn": "斑枯病", "pathogen": "真菌", "severity": "中",
        "symptoms": "叶片出现小型圆形褐斑，中央灰白并散生黑色小点，边缘深褐；下部叶片先发病并大量脱落。",
        "control": ["清除病叶并深埋", "避免土表浇水飞溅",
                    "发病期喷施代森锰锌"],
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "crop_cn": "番茄", "disease_cn": "二斑叶螨（红蜘蛛）", "pathogen": "虫害", "severity": "中",
        "symptoms": "叶片出现密集黄白色小点，逐渐失绿呈灰黄；叶背可见细小螨虫与丝网。",
        "control": ["保持适当湿度抑制螨繁殖", "释放捕食螨进行生物防治",
                    "轮换使用杀螨剂防止抗性"],
    },
    "Tomato___Target_Spot": {
        "crop_cn": "番茄", "disease_cn": "靶斑病", "pathogen": "真菌", "severity": "中",
        "symptoms": "叶片出现同心轮纹褐斑，中央灰白、边缘深褐，形似靶心；与早疫病外观相似，需仔细区分。",
        "control": ["轮作并清除病残体", "改善通风", "喷施苯醚甲环唑类药剂"],
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "crop_cn": "番茄", "disease_cn": "黄化曲叶病毒病", "pathogen": "病毒", "severity": "高",
        "symptoms": "新叶变小、边缘上卷、叶脉间黄化；植株矮化、落花落果严重。由烟粉虱传播。",
        "control": ["防治烟粉虱是核心", "使用抗病毒品种与无病苗",
                    "拔除病株，目前无特效药剂"],
    },
    "Tomato___Tomato_mosaic_virus": {
        "crop_cn": "番茄", "disease_cn": "花叶病毒病", "pathogen": "病毒", "severity": "中",
        "symptoms": "叶片出现黄绿相间的花叶斑块，叶面凹凸不平；果实出现褐色内部坏死。主要靠接触传播。",
        "control": ["种子与工具消毒", "操作时避免交叉传毒",
                    "拔除病株，防控蚜虫"],
    },
    "Tomato___healthy": {
        "crop_cn": "番茄", "disease_cn": "健康", "pathogen": "无", "severity": "无",
        "symptoms": "叶片舒展平展、颜色均匀，无斑点与卷曲。",
        "control": ["常规管理"],
    },
}


try:
    from knowledge_control import CONTROL as _CONTROL
except Exception:
    _CONTROL = {}

try:
    from knowledge_extra import EXTRA as _EXTRA
except Exception:
    _EXTRA = {}

try:
    from knowledge_rice import RICE as _RICE
except Exception:
    _RICE = {}

# 水稻类别并入库
KNOWLEDGE.update(_RICE)

try:
    from knowledge_multi import MULTI as _MULTI
except Exception:
    _MULTI = {}

# 茄子/萝卜/甘蔗/芒果类别并入库
KNOWLEDGE.update(_MULTI)

try:
    from knowledge_batch1 import BATCH1 as _B1
except Exception:
    _B1 = {}

# 茶叶/香蕉/辣椒/棉花/黄瓜/花生并入库
KNOWLEDGE.update(_B1)

try:
    from knowledge_batch2 import BATCH2 as _B2
except Exception:
    _B2 = {}

# 小麦/猕猴桃并入库
KNOWLEDGE.update(_B2)

try:
    from knowledge_litchi import LITCHI as _LIT
except Exception:
    _LIT = {}

# 荔枝并入库
KNOWLEDGE.update(_LIT)

# ---------- 扩充知识库（cropdpkg 原始数据，2536 条）----------
# 这批条目只用于问答，没有 cls（不参与图像分类与配图），
# 所以单独命名空间存放，避免污染依赖 cls 的逻辑。
try:
    from knowledge_kg import KG as _KG
    KG_KNOWLEDGE = dict(_KG)
except Exception:
    KG_KNOWLEDGE = {}

# 问答检索时合并查找（先查精编的 132 条，再查扩充库）
ALL_KNOWLEDGE = dict(KG_KNOWLEDGE)
ALL_KNOWLEDGE.update(KNOWLEDGE)

# 手工修补：展开"同XX""参见XX"这类未写完的引用
try:
    from knowledge_fix import PATCH as _KB_PATCH
    for _k, _fields in _KB_PATCH.items():
        # KNOWLEDGE 也要更新：诊断页查的就是它
        for _target in (KNOWLEDGE, KG_KNOWLEDGE, ALL_KNOWLEDGE):
            if _k in _target:
                _target[_k].update(_fields)
except Exception:
    pass


def get_control(class_name):
    """取该类别的结构化防治方案（农业防治 / 化学防治 / 注意事项）。"""
    return _CONTROL.get(class_name, {
        "农业防治": [], "化学防治": [], "注意事项": [], "sources": [],
    })


try:
    from knowledge_extra import EXTRA as _EXTRA_ALL
except Exception:
    _EXTRA_ALL = {}

# 原 38 类（手工标注，不覆盖其 severity）
_ORIGINAL_CLASSES = {k for k in KNOWLEDGE if "___" in k and k.count("___") == 1
                     and not k.startswith(("水稻", "甘蔗", "芒果", "茄子", "萝卜",
                                           "棉花", "花生", "茶叶", "辣椒", "香蕉",
                                           "黄瓜", "小麦", "猕猴桃", "荔枝"))}

_PATHOGEN_RULES = [
    (["病毒", "花叶", "黄化曲叶", "东格鲁", "褪绿", "白条"], "病毒"),
    (["细菌", "溃疡", "疮痂", "软腐", "角斑", "青枯", "条斑"], "细菌"),
    (["线虫"], "线虫"),
    (["螨", "蚜", "蓟马", "象甲", "甲虫", "螟", "虫", "盲蝽", "瘿蚊", "跳甲", "铁甲"], "虫害"),
    (["缺钾", "缺氮", "缺素", "营养缺乏", "缺镁", "缺铁"], "生理性"),
    (["锈", "霉", "斑", "腐", "枯", "瘟", "疫", "黑星", "白粉", "炭疽", "赤霉",
      "黑穗", "灰斑", "大斑", "靶斑", "轮纹", "褐斑", "黑斑", "霜霉", "蔓枯"], "真菌"),
]

_SEVERITY_HIGH = ["瘟", "疫", "黄龙", "枯萎", "黄萎", "软腐", "黑穗", "赤霉",
                  "巴拿马", "黑星", "溃疡"]
_SEVERITY_NONE = ["健康", "缺钾", "缺素", "营养缺乏"]


def _guess_pathogen(name):
    for keys, val in _PATHOGEN_RULES:
        for k in keys:
            if k in name:
                return val
    return "真菌"


def _guess_severity(name):
    for k in _SEVERITY_NONE:
        if k in name:
            return "无"
    for k in _SEVERITY_HIGH:
        if k in name:
            return "高"
    return "中"


def _clean_text(t):
    """清理知识库文本里的 Excel/Word 转义与多余空白。"""
    if not isinstance(t, str):
        return t
    t = t.replace("_x000D_", "").replace("_x000A_", "\n").replace("\r", "")
    t = t.replace("\u000b", "").replace("\u000c", "")
    # 去掉紧跟在括号前的残留空格
    import re
    t = re.sub(r"[ \t]+([（(])", r"\1", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


def get_info(class_name: str) -> dict:
    """按类别名取知识条目；缺失时返回兜底内容。"""
    info = KNOWLEDGE.get(class_name)
    if info is None:
        info = KG_KNOWLEDGE.get(class_name)
    if info is None:
        return {
            "crop_cn": "未知", "disease_cn": "未知", "pathogen": "未知", "severity": "未知",
            "symptoms": "暂无该类别资料。", "control": ["请参考农业植保部门建议"],
            "cls": class_name,
        }
    out = dict(info)
    out["cls"] = class_name
    extra = _EXTRA.get(class_name)
    if extra:
        out["symptoms_detail"] = extra.get("symptoms_detail", "")
        out["occurrence"] = extra.get("occurrence", "")
        out["control_detail"] = extra.get("control_detail", "")
        out["source"] = extra.get("source", "")
        # 若有详细症状，界面优先展示详细版
        if extra.get("symptoms_detail"):
            out["symptoms_brief"] = out.get("symptoms", "")
            out["symptoms"] = extra["symptoms_detail"]

    # 补齐缺失的病原类型 / 危害程度
    if out.get("pathogen") in ("—", "", None):
        out["pathogen"] = _guess_pathogen(out.get("disease_cn") or class_name)
    # 新扩充的类别（非原 38 类）危害程度也按规则推断，原类别保留手工值
    if class_name not in _ORIGINAL_CLASSES:
        if out.get("severity") in ("", None, "中"):
            out["severity"] = _guess_severity(out.get("disease_cn") or class_name)
    elif out.get("severity") in ("", None):
        out["severity"] = _guess_severity(out.get("disease_cn") or class_name)

    # 清理文本字段里的转义字符
    for k in ("symptoms", "occurrence", "control_detail", "symptoms_detail",
              "symptoms_brief"):
        if isinstance(out.get(k), str):
            out[k] = _clean_text(out[k])
    if isinstance(out.get("control"), list):
        out["control"] = [_clean_text(x) for x in out["control"]]
    return out


def format_report(class_name: str, topk: list) -> str:
    """生成一段可用于界面展示的诊断说明文本。"""
    info = get_info(class_name)
    lines = [
        f"### 诊断结果",
        f"- 作物：{info['crop_cn']}",
        f"- 病害：{info['disease_cn']}",
        f"- 病原类型：{info['pathogen']}",
        f"- 危害程度：{info['severity']}",
        "",
        f"**症状描述**：{info['symptoms']}",
        "",
        "**防治建议**：",
    ]
    for i, c in enumerate(info["control"], 1):
        lines.append(f"{i}. {c}")
    if len(topk) > 1:
        lines.append("")
        lines.append("**其他可能**：" + "；".join(f"{n}（{p*100:.1f}%）" for n, p in topk[1:4]))
    return "\n".join(lines)
