# 作物叶片病害识别与本地检索问答

> **分类器在 PlantVillage 上 95.67%，到真实田间只有 12.29%。**
> 本仓库的重点不是再训一个分类器，而是把这个落差量化出来，并给出可验证的补救链。

一句话：作物叶片病害识别 + 本地检索问答；技术重心是**跨域泛化的量化、评估协议修正与补救**。

---

## 我做了什么

把 PlantVillage 分类器刷到 95.67% 不难，难的是它换到真实照片上会掉到 12.29%。这个项目做了四件事：

1. **把落差称出来**：实验室白底数据训练，直接测互联网真实场景的 PlantDoc，零样本 12.29%。
2. **把评估协议讲清楚**：初版拿全类 argmax 对 PlantDoc 标签，会系统性低估；改成按 1:1 映射把概率折叠到 27 类再 argmax，得 38.98%（Wilson 95% CI [33.0, 45.3]）。同时论证「受限协议」（只在 27 个候选类里取 argmax）与「折叠协议」在 1:1 映射下**数学等价**——实测 236 个样本上两者预测差异为 **0**，因此只报一个数，拒绝把同一个数当两个结果展示。
3. **把不确定性变成可用信号**：补 risk–coverage 曲线；做多模型一致性分层——分组内结论一致时准确率 65%（n=80），不一致时只有 25.6%（n=156），一致性本身就是廉价的拒识器。
4. **给补救链，而不是只报坏消息**：AdaBN（只重估 BN 统计量，不动权重）、域随机化（按偏移谱把尺度增强当主力）、混合域微调（用目标域少量标注）。三条补救路径的脚本都在 `src/` 里，但它们的度量产物**未随仓库提供**（`outputs/metrics/` 只有质量报告与一次训练记录），所以本 README 不报这三步的准确率。

## 跨域实验（核心）

| 设置 | 准确率 | 说明 |
|---|---|---|
| 同分布测试集 | **95.67%** | 完整划分未随仓库提供；出处 `results/cross_domain_resnet50.json` 的 `in_domain_accuracy` |
| PlantDoc 零样本 | **12.29%** | 不接触目标域任何数据；出处 `results/cross_domain_diag.json`（`correct 29 / n 236`） |
| PlantDoc 折叠协议 | **38.98%** | Wilson 95% CI [33.0, 45.3]，n=236；出处同上（`acc_fold`、`ci_low`/`ci_high`） |
| AdaBN 单轮 | — | 脚本 `src/bn_adapt.py` 可跑，度量产物未随仓库提供，不报数字 |
| 混合域微调 | — | 脚本 `src/mixed_finetune.py` 可跑，度量产物未随仓库提供，不报数字 |

RTX 4060 上 ResNet50 单张 9.16 ms 的出处是 `results/benchmark.json` 的 `latency_ms`；Top-3 指标不在该文件里，本 README 不再报同分布集上的 Top-3。AdaBN 与混合域微调的产物路径是 `outputs/metrics/bn_adapt_*.json` 与 `outputs/metrics/mixed_*.json`，仓库里没有这两个文件——原始实验数据未随仓库提供，需要 GPU 与目标域标注自行补跑。

以上 95.67% / 12.29% / 38.98% / Wilson 区间 / n=236 五个值都能由 `results/` 下的 JSON 直接复算：29/236=12.29%、92/236=38.98%、0.9567 为 `in_domain_accuracy`，CI 用 Wilson 公式代入 p=92/236、n=236、z=1.96 即得 [0.3298, 0.4533]。

- **样本量先声明**：PlantDoc 测试集只有 236 张，且只覆盖一个目标域（互联网真实场景）。
  38.98% 的 Wilson 95% CI 是 [33.0, 45.3]，宽约 12 个百分点；所以跨域数字应当读作量级与
  方向，不是精确点估计。本仓库所有跨域结论都在这个样本量条件下报告。
- **协议等价性**：全类 → PlantDoc 27 类的映射是 1:1（不存在多个源类并到同一目标类）。此时「折叠后取 argmax」与「只在可映射类中取最大概率」必然给出同一预测；实测 236 张里差异数 0。所以报告里只有两种协议（严格 / 折叠），不把受限协议单列成一个「更好的结果」。
- **risk–coverage**：按置信度排序、只回答最有把握的部分，覆盖率下降时准确率上升——低置信度可以安全弃答，而不是硬报。
- **一致性分层**：一致组 65%（n=80）vs 不一致组 25.6%（n=156），出处 `results/cross_domain_diag.json` 的 `consistency`。无需额外训练，骨干之间的分歧就能当拒识信号。注意 JSON 里的键名写作「三模型一致」，而 `src/cross_domain_diag.py` 在参与模型数 ≥2 时即按一致性分组，产物本身不记录模型名单——所以这里只把它读作「多模型一致性分层」，不要当成「恰好三个骨干」的证据。
- **偏移谱**：对同分布测试图逐级施加背景、光照、模糊、目标缩小、旋转透视偏移，定位主因是**目标尺度**而非背景颜色（`src/offset_spectrum.py`）。
- **复现命令**：`python src/cross_domain_protocol.py`（三种协议）、`cross_domain_diag.py`（诊断图）、`bn_adapt.py`、`domain_rand_train.py`、`mixed_finetune.py`。后三个补救脚本需要 GPU 与目标域数据，产物不在本仓库内。

## 系统组成

- **单文件 Gradio，五个页签**（`app/web_agent.py`，5,857 行）：智能诊断 / 批量诊断 / 模型评估 / 知识问答 / 更多。其中「病斑定位」不是页签，是智能诊断页右栏的一个面板（`app/web_agent.py:5205`），批量诊断的详情区也有同类面板；「模型评估」页下再分同分布指标 / 评估图表 / 三模型对比 / 跨域泛化 / 数据资产五个子页签。
- **检索层**（`src/ai_qa.py`，1,296 行）：本地 bge-small-zh 向量 + jieba/BM25 混合召回；离域检测（作物白名单 + 别名剔除的规则实现，不是分类器）。
- **意图分类**：`src/intent_model/` 下有训练样本（450 条 / 5 类：occurrence 101、control 136、symptom 94、compare 86、system 33）、训练脚本 `train_intent_clf.py` 和**产物 `intent_clf.pkl`（34 KB，随仓库提供）**。分类器是「bge 句向量 + StandardScaler + 逻辑回归」，5 折交叉验证准确率 **0.9711 ± 0.0166**（这个数字可以从 pkl 的 `meta` 里直接读到，也可以重跑脚本复现）。运行时还缺一样东西：bge-small-zh-v1.5 模型本体（`assets/embedding/`，不随仓库提供）——模型缺失时 `_load_intent_clf()` 直接返回 None，自动回退到关键词规则，问答仍然可用。这里只认本地目录，不会退回 HuggingFace 模型 ID 去联网下载：断网环境下 `pytest -q` 不会被下载重试拖慢。重训：

  ```bash
  python src/intent_model/train_intent_clf.py \
      --model assets/embedding/bge-small-zh-v1.5 \
      --out src/intent_model/intent_clf.pkl
  ```
- **降级设计**：检索必有、生成可选。没有 API Key 或断网时直接返回检索到的知识原文，问答仍然可用，而不是报错。
- **知识库**（实测口径，见下「能力范围」）：`src/knowledge_kg.py` 42,487 行，是自动生成的数据字典（CropDP-KG 2,536 条 / 61 作物）。

## 能力范围

知识条目数以 `src/knowledge.py` 合并后的实测值为准（可以自己导入模块数一遍）：

| 口径 | 条目数 | 作物数 | 说明 |
|---|---|---|---|
| `KNOWLEDGE`（精编库） | 132 | 28 | 其中 12 条是「健康」类，非病害条目 |
| `KG_KNOWLEDGE`（CropDP-KG 扩充字典） | 2,536 | 61 | `src/knowledge_kg.py` 自动生成，来源 `data/cropdpkg/` 不随仓库提供 |
| `ALL_KNOWLEDGE`（合并后实际使用） | 2,645 | 66 | 检索问答走这一层 |

- **不硬套答案**：上表条目空间之外的病害不硬套答案，走「超出范围 / 未收录」提示，而不是用最接近的类糊一个结论。
- PlantVillage 原版是 38 类 / 14 作物（`data/splits.json` 的 `classes` 为 38、作物映射 14 个）。分类模型按 38 类训练、知识问答按上表条目覆盖，两者刻意不强行对齐：知识层比分类层宽，用来回答「这是什么病、怎么防」，而不只是「这张图属于哪一类」。
- 早先版本写的「127 类 / 28 作物」在精编库（132 条）和合并库（2,645 条）两个口径上都对不上，已按实测值改正。

## 代码结构

`app/`（Gradio 界面）→ `src/`（训练、评估、跨域实验、检索、知识库）→ `scripts/`（数据准备与抽样）→ `tests/` → `results/`（引用的全部实测 JSON）。

共 53 个 Python 文件、63,183 行（含空行）。这个总数会被自动生成的数据字典放大：
`src/knowledge_kg.py` 42,489 行（CropDP-KG 2,536 条），`src/knowledge_fix.py`
4,532 行（字段修补字典）。扣掉这两块，训练 / 评估 / 跨域 / 检索 / 前端的真实逻辑
约 1.6 万行——读代码时先看这些，不必被总行数吓住。

## 前置要求

- **数据**：PlantVillage parquet、`data/splits.json`、`data/images/`、PlantDoc 测试图都不在仓库里，需按下节目录树自备；来源见下节表格。
- **分类器权重**：`outputs/checkpoints/<backbone>_best.pth` 需自行训练，或按「最小复现（实测）」跑一遍就有。
- **检索向量模型**：`assets/embedding/bge-small-zh-v1.5/` 需自行下载（HuggingFace 链接见下节）。
- **生成层（可选）**：不附带任何大模型权重（`assets/llm/` 不入库），也不内置 API Key。没有 Key 时问答直接返回检索到的知识原文，仍然可用；要接本地模型，把权重放进 `assets/llm/<模型名>/`，并在 `src/ai_qa.py` 里把路径指过去。这一层是可选项，不承诺离线大模型的回答质量。

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
| CropDP-KG（扩充知识库来源，2,536 条） | https://github.com/dadadaray/CropDP-KG |
| bge-small-zh-v1.5 | https://huggingface.co/BAAI/bge-small-zh-v1.5 |

```bash
pip install -r requirements.txt
python scripts/prepare_data.py              # parquet → 配平图片 + splits.json
python src/train.py --model resnet50 --epochs 25
python src/predict.py --split test          # 在测试集上算 top-1/top-3
python src/cross_domain_protocol.py         # 跨域三种协议
python app/web_agent.py                     # http://127.0.0.1:7860
```

`assets/samples/` 里有几张示例图（苹果黑星病、玉米锈病、马铃薯健康、番茄晚疫病，共 4 张、256×256、约 136 KB），可直接拖进界面看效果（需先有模型权重；按「最小复现（实测）」跑一遍就有）。

**没有权重也能启动。** 界面顶部标「权重未找到」，智能诊断（含右上角病斑定位面板）/ 批量诊断在推理时提示缺少 `resnet50_best.pth` 等文件；检索问答与知识库照常可用。「模型评估」页的跨域泛化面板直接读 `results/cross_domain_protocol.json`（随仓库提供），所以 Top-1 38.98%、Top-5 66.95%、Wilson 区间这些格子有数字；AdaBN、混合微调、偏移谱三项的产物不在仓库里，对应格子显示「—」而不是 0.00%。这是刻意设计：权重缺失只关掉推理路径，不连累整个界面。缺 `data/splits.json` 才是硬错误——启动时打印放置说明并以码 2 退出，而不是抛裸 traceback，检查逻辑在 `src/preflight.py`（只检查、不生成也不伪造任何数据或权重）。

Windows 用户可直接双击 `start_local.bat`，或 `pwsh ./run_all.ps1` 按顺序跑五步：训练三骨干 → 评估出图 → 性能基准 → 退化鲁棒 → 遮挡。跨域三协议、AdaBN、域随机化、混合域微调**不在**这个脚本里，需要按「跨域实验」小节的命令单独跑。打包版（嵌入式运行时）未包含在本仓库。

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

三档环境各跑一次 `python -m pytest -q`（同一台机器）：无数据 9 passed / 1 skipped；
接入这份 320 张子集但没有权重 31 passed / 11 skipped；再接上 `resnet50_best.pth`
38 passed / 4 skipped。三档都没有失败用例，判据与跳过项见下一节 `## 测试`。

## 测试

**先看实测，再看判据。** `tests/conftest.py` 在收集阶段依次判断「有没有数据 → 数据够不够
完整 → 有没有权重 → 界面能不能起」，四件事决定收集哪些文件、跳过哪些用例：

| 环境 | `pytest -q` 实测 |
|---|---|
| 没有 `data/splits.json` | **9 passed, 1 skipped**（约 10 s） |
| 接数据（320 张冒烟子集）、没有权重 | **31 passed, 11 skipped**（约 18 s） |
| 再接 `outputs/checkpoints/resnet50_best.pth` | **38 passed, 4 skipped**（约 25 s） |

三档都没有 failed。完整 38 类配平数据集 + 三骨干权重的组合会再多通过「每类都有图」
「三骨干延迟对比」「跨骨干降级」这几个用例；那需要完整数据集与三份权重，本机没有实测，
这里不写没量过的数字。缺数据直接启动界面会在 `launch()` 之前明确报错退出（码 2），
不抛裸 traceback。

判据细节：

1. **有没有数据**：缺 `data/splits.json` 时 `test_rules / test_quality / test_recovery /
   test_smoke_data` 四个文件不收集（它们都要读真实图片与数据规模）。
2. **数据够不够完整**：断言「每类都有图片」的用例只对发布级配平数据集成立，冒烟子集上跳过。
3. **有没有权重**：推理、病斑定位、PDF 导出、批量吞吐这些用例要
   `outputs/checkpoints/resnet50_best.pth`，缺权重时跳过而不是报 fail。这一档以前是
   **5 failed**——权重缺失被当成代码 bug。另外，`web.batch_diagnose` 会逐行吞掉异常、
   照样返回行数，所以 `TestThroughput` 除了 skipif 还先调 `web.get_model()`，让加载失败
   当场抛出来，而不是量一段假吞吐。
4. **界面能不能起**：`test_e2e.py` 需要 7860 有服务 + 系统装了 Chrome，否则不收集。

`tests/test_smoke_data.py` 专门验证子集链路：划分比例、无跨划分泄漏、记录指向真实文件、
checkpoint 能推出合法 Top-5。pytest、playwright 与 pymupdf（`test_rules.py` 用 `fitz`
核对 PDF）都写在 `requirements.txt` 里。

**训练链路有 1-step smoke test**：`python -m pytest tests/test_train_smoke.py -q` 用
`assets/samples/` 的示例图 + 随机初始化权重走一遍 forward → backward → `optimizer.step`，
断言 loss 是有限数、参数有梯度、权重被更新；不读数据集、不读 checkpoint、不需要 GPU。
完整训练与跨域实验（三协议、AdaBN、域随机化、混合域微调）要 GPU + 数据集，只能按脚本
手动复现，不在自动化测试范围内。

## 数据与证据

`results/` 下是本仓库引用的实验数字的原始输出（6 个文件，均为实测 JSON）。README 里出现的每个跨域数字都能在这里找到出处；找不到出处的数字已从 README 删除，或标注为「未随仓库提供」：

| 文件 | 内容 |
|---|---|
| `results/smoke_minimal.json` | 最小复现实测：8 类 320 张、训练 3 轮 loss 3.3411→2.2120、test top-1 0.8438、推理 4 张全对、单张 11.2 ms |
| `results/cross_domain_diag.json` | 跨域诊断：`acc_strict=0.1229`、`acc_fold=0.3898`、Wilson CI [0.3298, 0.4533]、risk–coverage |
| `results/cross_domain_protocol.json` | 两种评估协议对比：`acc_restrict` 与 `acc_fold` 完全相等，即 236 个样本上 1:1 映射等价 |
| `results/cross_domain_resnet50.json` | ResNet50 在 PlantDoc 上的逐类准确率与域内准确率 0.9567 |
| `results/benchmark.json` | 三种骨干在 RTX 4060 上的延迟/吞吐/模型体积（**延迟/吞吐的唯一口径**：ResNet50 9.16 ms、EfficientNet-B0 12.5 ms、MobileNetV3-Large 10.06 ms） |
| `results/domain_classifier.json` | 域判别器准确率 0.9481（5 折 CV） |

这份目录里**没有** AdaBN（预期路径 `outputs/metrics/bn_adapt_*.json`）、混合域微调（`outputs/metrics/mixed_*.json`）的产物，也没有同分布集上的 Top-3 指标：前者只在 `src/bn_adapt.py`、`src/mixed_finetune.py` 里有脚本，后者不在 `results/benchmark.json` 的字段里。这三项因此不在 README 中报数字。

界面与这份目录的关系：跨域泛化面板按 `outputs/metrics/ → results/` 的顺序找指标文件，
所以 clone 下来（只有 `results/`、没有 `outputs/`）也能显示 Top-1 38.98%、Top-5 66.95%、
Wilson 区间这些数；AdaBN、混合微调、偏移谱三项缺产物时对应格子是「—」，不是 0.00%。
面板里的结论句（比如偏移谱「S3→S4 掉多少点」）由 JSON 现算，缺文件就换成「产物未随仓库
提供」，不会出现表格空白、旁边却写着一串硬编码数字的情况。

训练数据与权重不在仓库中，获取方式见「快速开始」。示例图片在 `assets/samples/`（4 张，约 136 KB），可直接拖进界面。

## 技术栈

PyTorch 2.6 / torchvision、ResNet50 · EfficientNet-B0 · MobileNetV3-Large、Grad-CAM、Gradio 6、jieba + rank-bm25 + bge-small-zh、Transformers、ReportLab、PyArrow。
