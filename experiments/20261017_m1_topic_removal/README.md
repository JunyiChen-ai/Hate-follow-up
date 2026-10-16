# 候选 46：读窗前去掉"话题"方向（topic-direction removal at read time）

声明日期 2026-10-17（NZDT），主机 sc474397（uoa-lab1）。自主迭代第 2 个候选（第 1 个候选 45 已归档）。
本文件 §2 的常数、方向构造、门全部在任何运行之前写下。**状态：规则 4 审查 PASS（2026-10-17，`docs/reviews/20261017_m1_topic_removal_novelty.md`）；按其建议改了拟合集、对照与词表（上面已注明）；跑方向提取中。**

## 1. 机制与来源

- **问题。** `research-wiki/DIRECTIONS.md` §D 规律 5：所有"这一段是否违规"的问法测到的主要是"这一段在说目标群体"
  （话题），不是"这一段在攻击"（行为）。在一个仇恨视频里几乎每个窗都在说同一个群体，所以话题分不开窗；铺垫、转述、
  中性提及的窗因为提到群体拿到高分（2026-10-14 诊断里的语音假高分），而用代词攻击的窗分数偏低。
- **已经试过、都不行的办法。** 在分数上减掉话题（TAD 减话题：减的比例越大掉得越多）；问言语行为、问起止（TAD 五选一、
  BND，塌回话题）；把群体名换成泛称的输入编辑（PCE，只提案）；用合成句子学一个方向去**读**隐状态（08-10 言语行为方向
  不迁移；08-09 的 24 个无监督轴都读不出）。共同点：要么在模型外面减分数，要么换一个问题，要么把隐状态当成读数。
- **机制假设。** 话题在残差流里是一个线性方向（"正在指称一个受保护群体"）。读窗时把这个方向从残差流里投掉
  （directional ablation：h ← h − (h·û)û，每层一个 û，作用在本窗分支的全部位置），模型仍然回答原来的问题，但不能再
  依赖"提到了群体"这个特征，只能靠剩下的：脏话、攻击动词、去人化的比喻、命令句、嘲弄的框架。前缀缓存（规则、帧、整段
  转录、整片裁定）不动，所以模型仍知道视频在说谁；视频间排序由 r6 的视频键负责（与候选 45 相同，键不变）。
  这不是减分数（分数相减会把"话题预测仇恨"的那部分一起减掉），不是换问题，不是读隐状态；是改变模型做判断时可用的特征。
- **来源。** Arditi 等 2024《Refusal in language models is mediated by a single direction》(NeurIPS 2024) 的 directional
  ablation：用两组提示的均值差定义方向，在每一层把它投掉，模型行为随之改变；Zou 等 2023 的 representation engineering
  用同样的均值差读和控概念。本候选把被投掉的概念从"拒绝"换成"指称受保护群体"，用于 hateful video 的窗级定位；
  规则 4 审查须检索是否有人在 hateful video detection / localization 里做过概念方向消除。
- **最接近的本仓库尝试与区别。** reinforcer（候选 13）与 projector：按 VISTA 往残差里**加**"有图 − 无图"的方向，目的是
  增强视觉证据；contraster（候选 7）：按 DoLa 对比层深度读答案。都没有去掉一个概念方向。08-10 的 illocution 方向是用来
  读的，而且概念是"断言 vs 转述"；本候选的概念是"话题"，用法是消除。

## 2. 方法与预检（常数先声明，两语料相同）

### 输入（沿用 r6）
读取器 `Qwen/Qwen3-VL-8B-Instruct`；前缀、立场缓存、20 帧、整段转录、窗文本（按比例切片）、语音分支提问
（`src/mllm_judge.yesno_question(..., "speech")`）全部与 r6 相同；画面读数沿用 r6 存档。r6 读数
`runs/20260926_glr/base_gridA/predictions.jsonl`；r6_bma 指标 `runs/20260926_twolevel/r6_bma/metrics.json`；
GT `data/gt_4fps/<语料>.npz` 只在预检分析与评测里读。

### 话题方向的构造（无标签；合成句子库 `topic_bank.py`，先写后跑）
- 载体句 40 条，每条一个槽 `{G}`，覆盖中性报道 8、称赞 4、攻击 8、提问 4、转述 4、反驳 4、日常 8（§3 文件里逐条列出）。
- 填充词两组各 16 个：**受保护群体**（种族、宗教、民族、国籍、性别、性取向、残障：Black people、white people、Jews、
  Muslims、Christians、Asians、Mexicans、immigrants、Arabs、gay people、trans people、women、disabled people、Indians、
  Africans、Hispanics）与**非受保护群体**（accountants、tourists、commuters、landlords、vegans、gamers、politicians、
  car dealers、delivery drivers、lawyers、influencers、neighbours、students、bankers、football fans、office workers）。
  40 × 32 = 1280 句。（规则 4 审查建议后、运行前改动：teenagers 换成 commuters，年龄在多数政策里也是受保护属性；
  非受保护组加入四个两词短语，与受保护组的五个 "X people" 平衡。）
- 每句放进与读窗相同的提问框架：文字前缀（系统消息、规则、阅读指令、转录 = `[0.0s-8.0s] 句子`，无帧、无立场轮）+
  `yesno_question(0, 1, 0.0, 8.0, 句子, "speech")`，一次前向。在嵌入输出和 36 个解码层输出（共 37 个位置）记录
  **句子 token 位置的平均残差**（主，`body`）和**答案位置的残差**（只报告，`last`）。
- 方向 û_l = normalize(mean_受保护 − mean_非受保护)，每层一个，用**偶数序号、非攻击的载体句**（16 条 × 32 = 512 句）算；
  8 条攻击载体句（序号 12–19）不进拟合：它们换上受保护群体就从"敌意"变成"违规"，会把"违规"混进话题方向（规则 4 审查
  建议后、运行前定下）。**奇数序号载体句**（20 条 × 32，含 4 条攻击句）留作检验。
  存 `data/topic_direction/qwen3vl8b/{directions_body.pt, directions_last.pt, directions_body_all_carriers.pt,
  directions_shuf0..4.pt, bank.json, PROVENANCE.md}`（37 × 4096 fp32 各 0.6 MB，是方法常数，不是审计张量）。
- **对照方向（匹配构造）**：把每条拟合载体句的 32 个填充词的受保护标签随机打乱（种子 0–4），用同样的均值差公式得到
  5 组"打乱标签"方向，再逐层减去它们在话题方向上的分量、重新归一化（第一次方向提取显示，只打乱标签时每一侧仍随机
  留下约一半受保护词，得到的方向与话题方向的余弦可达 ±.7；正交化后余弦为 0，构造噪声保留、话题成分为零。2026-10-17
  读窗前定下）。另加 1 组单位高斯随机方向（种子 0）只作参考：4096 维里随机方向几乎不改变任何东西，不是有力的对照。

### 读窗时的消除
- 对语音分支的前向（窗头 + 本窗文本 + 提问 + 助手回合头，即缓存之后新进的全部 token）在嵌入输出和每个解码层输出上
  挂 hook：h ← h − (h·û_l)û_l，作用于该前向的全部位置、全部 37 层。前缀缓存不改；画面分支不改；整片裁定不改。
  读数 = 消除后的 Yes/No 对数几率 `z_speech_abl`。读完 `cache.crop` 回滚、rope 偏移恢复（同 `src/stance_cache.margin`）。
- 同一窗另读 5 次打乱标签方向消除 `z_shuf[k]`（对照）和 1 次高斯随机方向消除 `z_rand[0]`（参考）。前 2 个视频 / 语料
  另回放一次无消除的 r6 语音读数作核对。

### 接入方法的方式（预检通过后才做）
- `z_speech_abl` 取代语音链的发射（`z_speech`）；`z_visual` 不变；视频键 K = z_video + 窗 `z` 均值**保持 r6 的值**
  （与候选 45 相同，`z` 原样取 r6）。M2 参数与 r6_bma 完全相同（`twolevel_r2.py --noleak --transform nscore --key calib
  --duration bma --bma-prior length --min-windows 2 --bma-grid 6 --arm m2`）；M3 不变。
- 原始 `score_curve` = max(z_visual, z_speech_abl)（两者本来同量纲，不需要仿射），只供原始读数表。
- 消融：语音链换回 r6 的判断读数（= r6_bma）；打乱标签方向消除（= 对照）。
- 新视频代价：与 r6 相同的调用次数（语音分支多了 37 个投影，可忽略）；方向是一次性常数。

### 预检（规则 10：读 test GT 做分析；一次 GPU 运行同时供预检与完整解码）
- 窗集合、标签、窗级 within 的定义与候选 45 相同：within 可计视频（84 / 99），有文本的窗，正窗 = 仇恨帧占比 ≥ .5，
  负窗 = 0，其余不计；每个同时有正负窗的视频一个窗级 ROC-AUC，再对视频平均。所有分数在同一批窗上算。
- **D1 方向成立**（不读 GT）：奇数载体句上，`body` 投影在最佳层把受保护 / 非受保护分开的 AUC ≥ .95；逐层 AUC 全部报告。
  另报告（不设门）：只用一半填充词拟合、在另一半填充词 × 检验载体句上的 AUC（方向是否只记住了词）；攻击载体句与
  非攻击载体句分开的检验 AUC；含攻击句拟合的方向与主方向的逐层余弦；打乱标签方向的检验 AUC（应接近 .5）。
  不过 → 方向不成立，不跑读窗。
- **D2 判断还在**：消除后语音读数在全部有干净标签的文本窗上的**合并**窗级 AUC（所有视频，不分视频）
  ≥ r6 `z_speech` 的合并 AUC − .05，两语料。不过 → 去掉话题后模型判不了，归档。
- **D3 主门**：`z_speech_abl` 的窗级 within ≥ r6 `z_speech` 的窗级 within + .02，两语料；且 5 组打乱标签方向的窗级 within
  均值 ≤ `z_speech_abl` − .02，两语料。高斯随机方向的 within 只报告。
- 报告（不设门）：`z_speech_abl` 与 `z_speech` 的 Spearman；窗文本的群体词命中数（受保护填充词表）与 `body` 投影
  幅度的 Spearman；按正负窗分的读数均值；`last` 方向版本的 D1。
- **决策**：D1、D2、D3 两语料都过 → 接入，固定 r6_bma 解码，按规则 8 对 r6_bma 判六项（任一项不下降超过噪声下限，
  至少一项两语料 ≥ +.01；机制主张要求那一项是 within）；任一不过 → 记录数字，归档为"预检未过"，不做修订。

### 代价
- 方向：1280 次短文本前向，约 2 分钟。读窗：每视频一次前缀预填充（同 r6），每个有文本的窗 1 次消除读 + 5 次打乱标签
  对照读 + 1 次高斯参考读（约 6.6 千窗 × 7 × 0.05 s ≈ 40 分钟）+ 回放核对；两语料一个作业约 1 小时，一张 RTX 5090。

## 3. 怎么跑
- 环境 `.cache/envs/HateVLM`。方向（GPU，Slurm 内）：`sbatch experiments/20261017_m1_topic_removal/launch/lab1_directions.sbatch`
  → `data/topic_direction/qwen3vl8b/`，日志 `runs/20261017_m1_topic_removal/directions/run.log`，标记 `C46_DIR_DONE`，
  D1 结果 `runs/20261017_m1_topic_removal/directions/validity.json`。
- 读窗（GPU，Slurm 内）：`sbatch experiments/20261017_m1_topic_removal/launch/lab1_measure.sbatch`（`SMOKE=1` 冒烟）
  → `runs/20261017_m1_topic_removal/main/{records/<语料>/<vid>.json, run.log, config.json}`，标记 `C46_MEASURE_DONE`。
- 预检与解码（CPU，conda `HateVideo`）：`python experiments/20261017_m1_topic_removal/analyze.py --stage precheck`
  → `main_precheck/`；`--stage predictions --arm abl|rand0..rand4`、`--stage decode --arm …`、`--stage report`
  → `main/<arm>/`、`main_raw/`、`main_decoded/`、`main_report/`。
- 合成句子库：`experiments/20261017_m1_topic_removal/topic_bank.py`（载体句与填充词逐条可见）。

## 4. Test-read log（规则 10）与冒烟记录
- （运行后填写。）

## 5. 结果与去向
- （待填。）
