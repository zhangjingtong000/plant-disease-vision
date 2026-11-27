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
| 同分布测试集，类均衡 4,986 张 | **95.67%** | Top-3 99.28%，RTX 4060 单张 9.16 ms（`results/benchmark.json`） |
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
├── assets/
│   ├── samples/                  4 张示例叶片图（仓库自带，可直接拖进界面）
│   └── embedding/bge-small-zh-v1.5/                     检索向量模型
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
python src/predict.py --split test          # 在测试集上算 top-1/top-3
python src/cross_domain_protocol.py         # 跨域三种协议
python app/web_agent.py                     # http://127.0.0.1:7860
```

`assets/samples/` 里有几张示例图（苹果黑星病、玉米锈病、马铃薯健康、番茄晚疫病，共 4 张、256×256、约 136 KB），可直接拖进界面看效果（需先有模型权重；按「最小复现（实测）」跑一遍就有）。没有权重时界面仍能打开，顶部会标「权重未找到」，诊断页在推理时给出缺失提示。

Windows 用户可直接双击 `start_local.bat`，或 `pwsh ./run_all.ps1` 跑完整实验流程。打包版（嵌入式运行时）未包含在本仓库。

## 最小复现（实测）

上面那套命令跑的是完整实验。要回答「这台机器上这条链路通不通」，一个子集就够。
下面是一次真实跑完的数字，原始输出在 `results/smoke_minimal.json`。

**这是链路冒烟，不是 95.67%。** 8 个类别、320 张图、3 轮训练，只能说明「数据进得来、
训得动、推得出」，不能与完整数据集上的指标并列比较。

### 1. 数据：8 类 × 40 张，6 秒

不需要重新下载 16 GB 的 PlantVillage。`scripts/make_smoke_subset.py` 从已有的完整
parquet 里按类别抽样，输出格式与官方完全一致（`struct<bytes,path>` + label 索引），
所以下一步 `prepare_data.py` 不需要任何改动：

```bash
python scripts/make_smoke_subset.py --src-parquet <完整 PlantVillage parquet 目录>   # 2.9 s
python scripts/prepare_data.py                                                       # 3.1 s
```

| 项 | 实测值 |
|---|---|
| 抽样 | 8 类 × 40 张 = 320 张，parquet 4.7 MB |
| 类别 | 8 / 38 类有图（其余 30 类的目录为空，`splits.json` 仍按 38 类写）|
| train / val / test | 224 / 32 / 64（每类 28 / 4 / 8）|
| 落盘 | 320 张 jpg，9.1 MB |

每类总量不足 200 张时，`prepare_data.py` 会走「按 70/10/20 等比缩减」的分支。完整
数据集永远走不到这条路径，正好在这里被验证一次。

### 2. 训练：ResNet50，3 轮，57 秒

```bash
python src/train.py --model resnet50 --epochs 3 --batch 16 --workers 2
```

| 配置 | 实测值 |
|---|---|
| backbone / 预训练 | ResNet50 / ImageNet1K-V2，23.59 M 参数 |
| batch / epochs / lr | 16 / 3 / 1e-4，AdamW |
| 每轮耗时 | 20.0 / 19.1 / 17.3 s（合计 57.1 s）|
| train_loss | **3.3411 → 2.2120** |
| train_acc | 30.4% → 79.9% |
| val_acc | 62.5% → 78.1% |

loss 三轮单调下降，权重写到 `outputs/checkpoints/resnet50_best.pth`（90.3 MB）。

### 3. 推理：4 张示例图 top-1 全对，约 11 ms/张

`src/predict.py` 调的是界面（`app/web.py::infer`）的同一条推理路径，不是另写一套：

| 示例图 | 预测 | 置信度 |
|---|---|---|
| `apple_scab.jpg` | Apple / Apple scab | 9.19% |
| `corn_rust.jpg` | Corn (maize) / Common rust | 19.39% |
| `potato_healthy.jpg` | Potato / healthy | 11.23% |
| `tomato_late_blight.jpg` | Tomato / Late blight | 11.44% |

4 张 top-1 全部命中，但置信度只有 9%–19%：3 轮、320 张的模型在 38 类 softmax 上本来
就是这个量级，界面会把低于阈值的判为「建议人工复核」，这是预期行为。在完整划分上：
test 64 张 top-1 84.4% / top-3 100%，val 32 张 78.1% / 100%。

```bash
python src/predict.py                # 4 张示例图 + Top-3 + 单张耗时
python src/predict.py --split test   # 在 splits.json 的测试集上算 top-1 / top-3
```

预热后单张 11.2 ms（batch=1，3 组各 20 次取中位；同一张图重复测量在 11–16 ms 之间波动）。
这和 `results/benchmark.json` 里完整权重上的 9.16 ms 不是一个口径：那边是三个骨干在
完整训练权重上的正式测量，这里是 3 轮小模型的链路验证，量级一致即说明链路通。

### 4. 测试

接入数据后 `python -m pytest -q` 从 6 passed 变成 **35 passed, 3 skipped**。
`tests/conftest.py` 按环境分三档判断：有没有数据、数据够不够完整（是否每类都有图）、
界面能不能起（7860 有服务 + 系统装了 Chrome）。只有需要「每类都有图 + 三骨干都有权重」
的用例在冒烟子集上跳过，其余照常跑；三个 skip 分别是全类目录完整性、三骨干延迟对比、
跨骨干降级。新增的 `tests/test_smoke_data.py` 专门验证子集链路：划分比例、无跨划分泄漏、
记录指向真实文件、checkpoint 能推出合法 Top-5。

## 数据与证据

`results/` 下是本仓库引用的全部实验数字的原始输出（共 8 KB，均为实测 JSON）：

| 文件 | 内容 |
|---|---|
| `results/smoke_minimal.json` | 最小复现实测：8 类 320 张、训练 3 轮 loss 3.3411→2.2120、test top-1 0.8438、推理 4 张全对、单张 11.2 ms |
| `results/cross_domain_diag.json` | 跨域诊断：`acc_strict=0.1229`、`acc_fold=0.3898`、Wilson CI [0.3298, 0.4533]、risk–coverage |
| `results/cross_domain_protocol.json` | 两种评估协议对比：`acc_restrict` 与 `acc_fold` 完全相等，即 236 个样本上 1:1 映射等价 |
| `results/cross_domain_resnet50.json` | ResNet50 在 PlantDoc 上的逐类准确率与域内准确率 0.9567 |
| `results/benchmark.json` | 三种骨干在 RTX 4060 上的延迟/吞吐/模型体积（**延迟/吞吐的唯一口径**：ResNet50 9.16 ms、EfficientNet-B0 12.5 ms、MobileNetV3-Large 10.06 ms） |
| `results/domain_classifier.json` | 域判别器准确率 0.9481（5 折 CV） |

训练数据与权重不在仓库中，获取方式见「快速开始」。示例图片在 `assets/samples/`（4 张，约 136 KB），可直接拖进界面。

## 已知限制

- **生成层需外部 API 或自行放置模型权重**：本仓库不附带任何大模型权重（`assets/llm/` 不入库）。没有 API Key 时问答走「检索原文直出」的降级路径，仍然可用；要接本地大模型，自行把权重放进 `assets/llm/<模型名>/`，并在 `src/ai_qa.py` 里指过去。生成层是可选项，不承诺离线大模型的回答质量。
- **无权重时界面照常启动**：顶部会标「权重未找到」，单张诊断 / 病斑定位 / 批量诊断在推理时提示缺少权重（`resnet50_best.pth` 等）。缺 `data/splits.json` 时启动会直接给出放置说明并退出（码 2），不再是裸 traceback；检查逻辑在 `src/preflight.py`，不生成也不伪造任何权重或数据。
- `knowledge_kg.py` 是自动生成的数据字典，不是手写逻辑；仓库共 46 个 Python 文件、62,360 行，其中真实逻辑（训练/评估/跨域/检索/前端）约 1.4 万行。
- **类别空间是 127 类 / 28 作物**，之外的病害无法识别（会走「超出范围 / 未收录」提示）。PlantVillage 原版只有 38 类 / 14 作物，那是更早的版本，本文按当前 127 类写。
- 自动化测试覆盖规则、端到端、质量属性与恢复性（`tests/`）。训练链路有 1-step CPU smoke test（`python -m pytest tests/test_train_smoke.py -q`）：用 `assets/samples/` 的示例图 + 随机初始化权重走一遍 forward → backward → `optimizer.step`，断言 loss 是有限数、参数有梯度、权重被更新，不读数据集、不读 checkpoint、不需要 GPU。完整训练与跨域实验仍需 GPU 与数据集，只能按脚本手动复现。

  **没有数据时**，`python -m pytest -q` 是 **6 passed**：`tests/conftest.py` 会在缺 `data/splits.json` 时跳过 `test_rules / test_quality / test_recovery / test_smoke_data / test_e2e` 五个依赖真实图片、数据规模与界面的文件。按「快速开始」生成完整数据后它们自动恢复；只接入小样本子集时是 **35 passed, 3 skipped**，跳过的三个用例分别要求「每类都有图」「三骨干都有权重」。筛选逻辑全在 `tests/conftest.py`，不需要改代码。缺数据直接启动界面会在 import 之后、`launch()` 之前明确报错退出，不会抛裸 traceback。
- PlantDoc 测试集仅 236 张，置信区间较宽；且只覆盖一个目标域。

## 技术栈

PyTorch 2.6 / torchvision、ResNet50 · EfficientNet-B0 · MobileNetV3-Large、Grad-CAM、Gradio 6、jieba + rank-bm25 + bge-small-zh、Transformers、ReportLab、PyArrow。
