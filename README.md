# 作物叶片病害识别与本地检索问答

> **分类器在 PlantVillage 上 95.67%，到真实田间只有 12.29%。**
> 本仓库的重点不是再训一个分类器，而是把这个落差量化出来，并给出可验证的补救链。

一句话：作物叶片病害识别 + 本地检索问答；技术重心是**跨域泛化的量化、评估协议修正与补救**。

---

## 我做了什么

把 PlantVillage 分类器刷到 95.67% 不难，难的是它换到真实照片上会掉到 12.29%。这个项目做了四件事：

1. **把落差称出来**：实验室白底数据训练，直接测互联网真实场景的 PlantDoc，零样本 12.29%。
2. **把评估协议讲清楚**：初版拿全类 argmax 对 PlantDoc 标签，会系统性低估；改成按 1:1 映射把概率折叠到 27 类再 argmax，得 38.98%（Wilson 95% CI [33.0, 45.3]）。同时论证「受限协议」（只在 27 个候选类里取 argmax）与「折叠协议」在 1:1 映射下**数学等价**——实测 236 个样本上两者预测差异为 **0**，因此只报一个数，拒绝把同一个数当两个结果展示。
3. **把不确定性变成可用信号**：补 risk–coverage 曲线；做三模型一致性分层——三个模型结论一致时准确率 65%，不一致时只有 25.6%，一致性本身就是廉价的拒识器。
4. **给补救链，而不是只报坏消息**：AdaBN（只重估 BN 统计量，不动权重）→ 16.53%；域随机化（按偏移谱把尺度增强当主力）；混合域微调（用目标域少量标注）→ 65.68%。

## 跨域实验（核心）

| 设置 | 准确率 | 说明 |
|---|---|---|
| 同分布测试集，类均衡 4,986 张 | **95.67%** | Top-3 99.28%，RTX 4060 单张 9.2 ms |
| PlantDoc 零样本 | **12.29%** | 不接触目标域任何数据 |
| PlantDoc 折叠协议 | **38.98%** | Wilson 95% CI [33.0, 45.3]，n=236 |
| AdaBN 单轮 | 16.53% | 无监督，只用无标签图重估 BN |
| 混合域微调 | 65.68% | 用了目标域标注，衡量「适配」而非「泛化」 |

- **协议等价性**：全类 → PlantDoc 27 类的映射是 1:1（不存在多个源类并到同一目标类）。此时「折叠后取 argmax」与「只在可映射类中取最大概率」必然给出同一预测；实测 236 张里差异数 0。所以报告里只有两种协议（严格 / 折叠），不把受限协议单列成一个「更好的结果」。
- **risk–coverage**：按置信度排序、只回答最有把握的部分，覆盖率下降时准确率上升——低置信度可以安全弃答，而不是硬报。
- **一致性分层**：一致组 65% vs 不一致组 25.6%。无需额外训练，三个骨干的分歧就能当拒识信号。
- **偏移谱**：对同分布测试图逐级施加背景、光照、模糊、目标缩小、旋转透视偏移，定位主因是**目标尺度**而非背景颜色（`src/offset_spectrum.py`）。
- **复现命令**：`python src/cross_domain_protocol.py`（三种协议）、`cross_domain_diag.py`（诊断图）、`bn_adapt.py`、`domain_rand_train.py`、`mixed_finetune.py`。

## 系统组成

- **单文件 Gradio，五个页签**（`app/web_agent.py`，约 5.8K 行）：单张诊断 / 病斑定位 / 批量诊断 / 模型评估 / 问答与历史。
- **检索层**（`src/ai_qa.py`，约 1.3K 行）：本地 bge-small-zh 向量 + jieba/BM25 混合召回；独立意图分类器（450 样本 / 5 类，bge 嵌入 + 逻辑回归）；离域检测；4 个知识库覆盖度审计。
- **降级设计**：检索必有、生成可选。没有 API Key 或断网时直接返回检索到的知识原文，问答仍然可用，而不是报错。
- **知识库**：`src/knowledge_kg.py` 42,489 行，是自动生成的数据字典（CropDP-KG 2,536 条），与精编条目合并后覆盖 **127 类 / 28 作物**。

## 快速开始

**数据与权重不在仓库里。** 需要自备下列数据，按目录放好后代码才能跑：

```
plant/
├── data/
│   ├── parquet/                  PlantVillage parquet：train-00000/00001、test-00000
│   ├── images/<类别>/            prepare_data.py 生成的配平图片
│   ├── splits.json               类别名、作物映射、样本清单
│   └── plantdoc_extract/PlantDoc-Dataset-master/test/   PlantDoc 测试图
├── assets/embedding/bge-small-zh-v1.5/                  检索向量模型
└── outputs/checkpoints/                                  训练产物
```

| 数据 | 来源 |
|---|---|
| PlantVillage | https://github.com/spMohanty/PlantVillage-Dataset |
| PlantDoc | https://github.com/pratikkayal/PlantDoc-Dataset |
| CropDP-KG（127 类知识库来源） | https://github.com/dadadaray/CropDP-KG |
| bge-small-zh-v1.5 | https://huggingface.co/BAAI/bge-small-zh-v1.5 |

```bash
pip install -r requirements.txt
python scripts/prepare_data.py              # parquet → 配平图片 + splits.json
python src/train.py --model resnet50 --epochs 25
python src/cross_domain_protocol.py         # 跨域三种协议
python app/web_agent.py                     # http://127.0.0.1:7860
```

Windows 用户可直接双击 `start_local.bat`，或 `pwsh ./run_all.ps1` 跑完整实验流程。打包版（嵌入式运行时）未包含在本仓库。

## 已知限制

- **本地 LLM 路线未闭环**：`assets/llm/qwen3-4b-instruct/` 是 0 字节权重残留，生成层实际走「检索降级」或外部 API。
- 生成层是可选项，不承诺离线大模型的回答质量。
- `knowledge_kg.py` 是自动生成的数据字典，不是手写逻辑；仓库共 44 个 Python 文件、62,215 行，其中真实逻辑（训练/评估/跨域/检索/前端）约 1.4 万行。
- 类别空间为 127 类 / 28 作物，之外的病害无法识别（早期版本只有 PlantVillage 的 38 类、14 作物，原说明未更新）。
- 自动化测试覆盖规则、端到端、质量属性与恢复性（`tests/`），但训练与跨域实验没有自动化单测——它们依赖数据集和 GPU，只能按脚本手动复现。
- PlantDoc 测试集仅 236 张，置信区间较宽；且只覆盖一个目标域。

## 技术栈

PyTorch 2.6 / torchvision、ResNet50 · EfficientNet-B0 · MobileNetV3-Large、Grad-CAM、Gradio 6、jieba + rank-bm25 + bge-small-zh、Transformers、ReportLab、PyArrow。
