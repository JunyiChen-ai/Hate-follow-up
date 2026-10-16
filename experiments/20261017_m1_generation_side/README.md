# 候选 45：生成侧读数（generation-side reading）— 复述阻力与最小合规改写

声明日期 2026-10-17（NZDT），主机 sc474397（uoa-lab1）。用户 2026-10-17 指示进入自主迭代，目标是 M1 的机制级 novelty；
M2 / M3 只允许为配合 M1 做小幅修改。本文件 §2 的常数、分数定义、门全部在任何运行之前写下。
**状态：规则 4 审查 PASS（2026-10-17），实现中；冒烟后跑完整 333。**

## 1. 机制与来源

- **M1 现在的样子。** 每个 8 秒窗对着缓存的整段上下文（规则 + 20 帧 + 整段转录 + 视频级裁定）各读一次画面、一次
  语音，读数是"这一窗是否违规"这个问题的 Yes/No 对数几率。`research-wiki/DIRECTIONS.md` §D 的规律 1 和 5：换问法、
  换读法都停在同一水平；这类问法测到的主要是"这段在说目标群体"，不是"这段在攻击"。追加的第二个问题与第一个问题的
  读数 Spearman .92–.98（08-09），所以再问一个问题得不到新信息。
- **机制假设。** 不再问模型"这段违规吗"，而是让模型**去产出这段内容**，读它产出时的行为：
  (A) **复述阻力**：让模型以说话人身份逐字复述本窗原话，用 teacher forcing 读它给原话每个 token 的概率。原话就在
  上下文里，复述对模型是零难度的复制任务，所以概率明显偏低的地方不是"猜不到"，而是模型不愿意说出口。安全对齐让模型
  在被要求产出攻击性内容时迟疑，而在产出铺垫、转述、中性提及时不迟疑。这个信号绑在要产出的具体词上，不随整段视频
  的印象走。
  (B) **最小合规改写**：让模型把本窗原话改成不违规的版本，要求改动尽量少、不违规就原样输出；读改了多少词。
  改动量是模型外数出来的，不是模型口头给的分数。
  两种都不出现"违规吗"这个判断问题；都是同一个冻结模型、每窗一次额外调用、无标签。
- **针对的误差。** 2026-10-14 的 within 诊断：错排配对里负窗读数来自语音分支的比例 .71 / .54；语音假高分集中在
  铺垫、转述、提到群体但没有攻击的窗。这些窗没有需要迟疑、需要改掉的词。无语音的正窗不在本候选范围内（没有话可复述），
  那部分仍由画面分支负责。
- **来源。** Arditi 等 2024《Refusal in language models is mediated by a single direction》：拒绝行为是对齐模型内
  一个稳定的内部量；Qi 等 2024《Safety alignment should be made more than just a few tokens deep》：对齐主要作用在
  回答的前几个 token，所以第一个 token 的概率最能反映拒绝倾向。改写方向来自文本去毒化 / 最小编辑改写一族
  （ParaDetox 等）。据我所知，没有工作把"模型复述或改写一段话时的阻力"用作 hateful video 的时间定位信号；规则 4
  审查须实际检索并记录。
- **最接近的已试（`research-wiki/DIRECTIONS.md`）与区别。**
  - GLR（09-26）：原话**不在**上下文里，读两种说话人条件下原话的似然比，差只有约 .01 nats/token，因为口语本身不可
    预测。本候选原话在上下文里，复制任务的基线损失接近 0，剩下的损失才是阻力。
  - 答案对比、第二次提问（08-09 / 08-10）：换一个判断问题，读数是 z 换名。本候选不问判断问题；G1 门直接用
    Spearman .80 排除"z 换名"。
  - HVL 引用匹配、候选 41 的帧引用：让模型口头指出证据在哪，模型选错证据。本候选不让模型指位置。
  - TAD 五选一、BND：问言语行为或起止，塌回话题。本候选不问。
  - 候选 43（音频）：换输入，不换读法。

## 2. 方法与预检（常数先声明，两语料相同）

### 输入（全部沿用 r6，不新增抽取）
- 读取器 `Qwen/Qwen3-VL-8B-Instruct`，前缀与 r6 相同：20 帧（`data/frames_k20`）、整段 Whisper 转录
  （`data/asr_whisper_large_v3`，`fill_untimed=True`）、规则与阅读指令（`src/mllm_judge.py` 冻结文本）、视频级裁定
  （`src/stance_cache.build`，与 r6 相同的立场条件化）。
- 每窗文本 = r6 语音分支的同一段按比例切片（`src/video_inputs.window_text`）。没有文本的窗不做生成侧读（r6 也没有
  `z_speech`）。
- r6 读数 `runs/20260926_glr/base_gridA/predictions.jsonl`（`z_visual`、`z_speech`）；r6_bma 指标
  `runs/20260926_twolevel/r6_bma/metrics.json`。GT `data/gt_4fps/<语料>.npz` 只在预检分析与评测里读。

### 两条读法的提问（逐字；`{body}` 为本窗文本，其余同 r6 的窗头）
- **A 复述**（teacher forcing，不生成）：
  `Consider only window {i+1} of {n}, from {t1:.1f}s to {t2:.1f}s of this video. The speech in this window is: {body}`
  `\n\nRepeat this speech verbatim, word for word, speaking in the first person as the speaker. Output only the repeated speech: no preamble, no quotation marks, no commentary.`
  助手回合强制为 `{body}` 原文。`{body}` = `window_text` 输出把连续空白压成一个空格、去首尾空白后的同一个字符串
  （`common.norm_body`），问题里和强制回答里逐字节相同。压空白的原因见 §4 冒烟记录：Whisper 文本里的双空格本身在
  teacher forcing 下要付 6–9 nats，与内容无关。
  计分的 token 序列 = 该字符串用 tokenizer 单独编码（不加特殊 token）得到的全部 token，不含助手回合头、不含结束 token、
  不设长度上限（8 秒窗的文本很短）；运行时断言它与聊天模板渲染出的回答 token 前缀逐个相同。
  读每个 token 的 log p（fp32 lm_head，全词表 softmax）。
  分数：`copy_mean` = 全部 token 的 −log p 平均（nats/token，**主分数**）；`copy_max` = 最大单 token −log p；
  `first_loss` = 第 1 个 token 的 −log p；`refuse_mass` = 第 1 个位置上 {"I","I'm","I’m","Sorry","As","Unfortunately","Apolog"}
  这些起始 token（各取有无前导空格两种写法的首 token）的概率总和的对数，**排除本窗文本自己的第 1 个 token**（否则
  以 "I" 开头的原话会把自己算成拒绝）；只报告。越大越像仇恨。
- **B 最小合规改写**（贪心生成）：
  `Consider only window {i+1} of {n}, from {t1:.1f}s to {t2:.1f}s of this video. The speech in this window is: {body}`
  `\n\nRewrite this speech so that it no longer violates any of the above rules, changing as few words as possible and keeping every other word exactly as it is. If nothing in it violates the rules, output it unchanged. Output only the rewritten speech: no preamble, no quotation marks, no commentary.`
  `max_new_tokens` = min(256, 2 × body 的 token 数 + 16)。词 = 小写后按正则 `[^\W_]+(?:'[^\W_]+)?` 切出的字母数字串
  （撇号连接的缩写算一个词，标点丢弃；`common.norm_words`），两边同一规则。
  **主分数 `edit_rate`** = 原话里被删掉或换掉的词占比 = 1 − LCS(原话词, 输出词) ÷ 原话词数，LCS 为最长公共子序列；
  插入的词（模型附加的解释、把字幕接着往下写）不计。副分数 `edit_lev` = 词级 Levenshtein 距离 ÷ 原话词数，截到 [0, 1]。
  两边词序列相同记 0；原话没有词（只有符号）记 0；输出为空记 1。拒绝文本（"I can't …"）几乎不保留原话的词，自然得到
  接近 1 的改动率，不另设规则。越大越像仇恨。
  （2026-10-17 冒烟后、完整运行前把主分数从 Levenshtein 改为 LCS 版，原因见 §4：冒烟里只有一个词的字幕窗被模型接着
  写成整句字幕，Levenshtein 给 1.0，与内容无关；冒烟没有读任何标签。）
- 两条读法都在 r6 的立场缓存上做独立分支（读完 `cache.crop` 回滚，rope 偏移恢复，与 `src/stance_cache.margin` 同一做法）。
  温度无（贪心 / teacher forcing），种子 0。

### 接入方法的方式（预检通过后才做）
- 通过的那条读法的主分数 s 取代每窗的 `z_speech`，`z_visual` 不变。M2 用与 r6_bma 完全相同的参数
  （`twolevel_r2.py --noleak --transform nscore --key calib --duration bma --bma-prior length --min-windows 2 --bma-grid 6 --arm m2`）；
  nscore 变换按语料对读数做秩正态化，所以 s 的量纲不影响时序层。M3 不变。
- 时序层的视频键 K = z_video + 窗 `z` 的均值（`twolevel_r2.py` 的 intercept）**与 r6_bma 完全相同**：预测文件里每窗的 `z`
  原样取 r6 的值（= max(r6 画面读数, r6 语音判断读数)），所以解码结果与 r6_bma 的差别只来自语音链的发射换成了 s。
  这意味着方法仍保留 r6 的语音判断读数给视频键用（视频间排序），与 §2 代价里"在 r6 之上多一次前向 + 一次短生成"一致。
  原始 `score_curve` 只供原始读数表：max(z_visual, s_aff)，s_aff = s 在本语料内标准化后乘以 r6 `z_speech` 的语料标准差、
  加其均值（无标签仿射，只为了和 `z_visual` 同量纲），不进时序层。权威数字是解码后的 `metrics.json`。
  （2026-10-17 代码审查指出原实现把 max(z_visual, s_aff) 写进了 `z`，会经视频键改变 pooled；改为上面这样，运行前写入。）
- 若 A、B 都过预检，取两语料窗级 within 平均值更高的一条；另一条只报告。不合并两条（不加模块）。这个选择用了
  test 标签，按规则 10 标 development-selected。
- 消融（若进入完整方法）：`z_speech` 换回 r6 的判断读数（= r6_bma）。

### 预检（规则 10：读 test GT 做分析；一次 GPU 运行同时供预检和完整解码使用）
- 视频：within 可计的视频（HateMM 84、HateClipSeg 99）。窗：有文本的窗；正窗 = 窗内仇恨帧占比 ≥ .5，
  负窗 = 占比 0，其余不计；帧映射同候选 44（[round(4·t1), round(4·t2))）。
- **窗级 within** = 每个同时有正负窗的视频算一个窗级 ROC-AUC，再对视频取平均。
- 比较对象（同一语料内所有分数与对照都在**同一批窗**上算）：r6 `z_speech`；A 的 `copy_mean`（另报 `copy_max`、
  `first_loss`、`refuse_mass`）；B 的 `edit_rate`。
  不用模型的对照：`lex_profanity` = 窗文本命中 LDNOOBW 词表的次数；`lex_hate` = 命中 Davidson 2017 精炼 n-gram
  词表的次数（`data/lexicons/`，见其 PROVENANCE）；`word_count` = 窗词数；`random` = 10 次均匀随机（种子 0）的平均。
- **门（每条读法的主分数单独判，两语料都要过）：**
  - G1 不是 z 换名：主分数与 `z_speech` 在全部计入窗上的 Spearman < .80（阈值来自 08-09 的 .92–.98）。
  - G2 比判断读数好：窗级 within ≥ `z_speech` 的窗级 within + .02。
  - G3 不是词表：窗级 within ≥ max(`lex_profanity`, `lex_hate`, `word_count`) 的窗级 within + .02。
- **决策**：某条读法 G1–G3 两语料全过 → 用它取代语音链的发射，固定 r6_bma 解码，按规则 8 对 r6_bma 判六项
  （任一项不下降超过噪声下限，且至少一项两语料 ≥ +.01；机制主张要求那一项是 within）；都不过 → 记录数字，归档为
  "预检未过"，不做修订。
- 预检只决定是否接入；接入后的六项才是方法结果。两个阶段用同一次 GPU 运行的记录。

### 代价
- 前缀预填充每视频一次（与 r6 同，333 个约 20 分钟）；A 每个有文本的窗一次前向（约 4.7 千窗，合计约 5 分钟）；
  B 每个有文本的窗一次贪心生成（预计 40–80 token，合计 1.5–2 小时）。一张 RTX 5090，两语料一个作业约 2.5 小时。
- 新视频在 r6 之上每个有文本的窗多 1 次前向 + 1 次短生成；不新增抽取与缓存。

## 3. 怎么跑
- 环境 `.cache/envs/HateVLM`（torch 2.11.0+cu128，transformers 5.15.1，与候选 43 的 Qwen3-VL 对照相同）。
- 读取（GPU，只在 Slurm 内）：`sbatch experiments/20261017_m1_generation_side/launch/lab1_measure.sbatch`
  （`SMOKE=1` 为 5 个冒烟视频）→ `runs/20261017_m1_generation_side/main/{records/<语料>/<vid>.json, run.log, config.json}`，
  完成标记 `C45_MEASURE_DONE`。
- 预检与解码（CPU，conda `HateVideo`）：`python experiments/20261017_m1_generation_side/analyze.py --stage precheck`
  → `runs/20261017_m1_generation_side/main_precheck/{summary.json,per_video.json,table.txt}`；
  `--stage predictions --arm <A|B>` → `main/<arm>/{predictions.jsonl,checks.json}`（覆盖、回放差、立场一致性）；
  `--stage decode --arm <A|B>` → `main_decoded/<arm>/metrics.json`、`main_raw/<arm>/metrics.json`；
  `--stage report` → `main_report/summary.json`（六项对 r6_bma 与规则 8 的门）。
- 规则 6 代码审查：`docs/reviews/20261017_m1_generation_side_code.md`（2026-10-17，一个阻断项：s_aff 经视频键进入解码；已按上面改法修复，复核见同文件）。
- 规则 4 审查：`docs/reviews/20261017_m1_generation_side_novelty.md`（2026-10-17 PASS；最近的文献是 MULI 2405.18822 的
  首 token 拒绝 logit 做文本提示的毒性检测、ParaDetox 一族的最小编辑去毒、LELA 2602.09637 的 training-free 仇恨视频定位；
  都没有把生成侧行为当定位分数）。

## 4. Test-read log（规则 10）与冒烟记录
- 2026-10-17 冒烟（Slurm 482，5 个视频，不读标签）：回放的 r6 语音读数与存档逐值相同（最大差 0.0），立场 5/5 一致。
  看了 `main_smoke/records/` 里每窗的原话、token 损失、改写。发现：(1) A 的复述对脏话、攻击句（"fuck you"、"piece of
  shit"、"Retard"、"I fuck your mother"）损失全为 0，模型照抄不误；非零损失只出现在双空格 token（6–9 nats）、被切断
  处的句号、重复串和芬兰语窗。(2) B 的改写是局部的词替换（"fuck you"→"thank you"、"Retard"→"Person with developmental
  disability"、"hang Mike Pence"→"remove Mike Pence"、"dictator"→"leader"），也会改政治敏感但不仇恨的词；一个词的字幕窗
  被接着写成整句（"ALEKSANDRA"→"ALEKSANDRA KUZNETSKAYA © BF-WATCH TV 2021"），Levenshtein 版给 1.0。
  由此在完整运行前改了两处定义：body 压空白；B 的主分数改为 LCS 版（见 §2）。A 的假设在冒烟里已经看起来不成立，
  仍按声明跑完整 333 并过门，作为正式记录。
- （预检运行后填写：读了哪些文件、发现了什么、影响了哪个决定。）

## 5. 结果与去向
- （待填。）
