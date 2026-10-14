# 候选 44：按说话人读窗（speaker-bound reading）— 预检阶段

声明日期 2026-10-14（NZDT），主机 sc474397（uoa-lab1）。用户 2026-10-14 同意先做预检。
**状态：预检中。方法未实现，r6_bma 不变。** 预检的常数与门在本文件 §2 先写后跑。

## 1. 机制与来源

- **M1 现在的样子。** 每个 8 秒窗对着一份缓存的整段上下文（规则 + 20 帧 + 整段转录）和一个**视频级**裁定读，
  画面 / 语音两个隔离分支。整个视频只有一个立场，所以同一视频里被攻击者的反驳、主播的转述、被引用的原话都
  继承同一个"违规"立场。2026-10-14 的 within 诊断里，错排的负窗里 .71 / .54 的读数来自语音分支。
- **机制假设。** 立场属于说话人，不属于视频；仇恨在一个说话人的轮次里持续，不在原始时间上持续。
- **若预检通过的改法（保留"每窗对着上下文读"这个机制）。**
  1. 说话人分割作为输入（规则 5）：转录带说话人标签进前缀。
  2. 说话人级证据**不问模型**："这个说话人是否认同仇恨"这类问句已三次证伪：DVD 的 E 条件在视频级去掉从不掉
     .01（`experiments/20260927_dvd/README.md` §9.4）；TAD 五选一在窗级 79.6 % / 56.0 % 答"攻击"，豁免项合计
     1.9 % / 9.2 %（`experiments/20260912_tad/README.md` §5c）；GLR 的违规 / 合规说话人条件只差约 .01 nats /
     token。说话人级证据由该说话人各窗的读数在模型外汇总，与视频裁定同理、粒度更细。
  3. 每窗语音分支对着本窗说话人的证据读（具体形式预检后声明）；M3 的语音链沿该说话人的轮次串联，画面链不变。
  4. 消融 / 对照：说话人证据换回视频裁定（= r6）；说话人链换回时间链；说话人标签在视频内打乱（错误绑定）。
- **最接近的已试。** DVD E（视频级认同问句）；TAD 五选一；GLR；QuoteGraph（候选 24，说话人归属由模型猜，无真实
  分割）；`research-wiki/DIRECTIONS.md` 末尾"说话人–目标的 diarization 绑定"只提案未测；Codex 2026-09-28 第 1 轮
  指出"更细的停顿和说话人轮次未测"（`experiments/20260928_headroom/README.md` 发现 1 的注）。

## 2. 预检（规则 10：读 test GT；只做分析，不改方法，不花读取 GPU）

### 输入
- 说话人分割：`pyannote/speaker-diarization-3.1`，默认超参，不给说话人数；输入 `data/audio16k/<语料>/<video_id>.wav`
  （16 kHz 单声道，候选 43 的产物）；输出 `data/speaker_turns/<语料>/turns.jsonl` + `PROVENANCE.md`。
  备选（HF gate 未开时）：wespeaker 的 VAD + 嵌入 + 谱聚类流水线，换了要在此注明。
- r6 读数 `runs/20260926_glr/base_gridA/predictions.jsonl`（每窗 `z`、`z_visual`、`z_speech`、起止秒）；
  r6_bma 解码 `runs/20260926_twolevel/r6_bma/predictions.jsonl`（4 fps `score_curve`）；GT `data/gt_4fps/<语料>.npz` test。

### 常数（先声明，两语料相同）
- 窗归属：某说话人的轮次在窗内重叠时长最大且 ≥ 1.0 秒，该窗归它；否则"无说话人"。
- 有效说话人：占该视频语音总时长 ≥ 10 %，且主导 ≥ 2 个窗。**多说话人视频** = 有效说话人 ≥ 2。
- 正窗：窗内仇恨帧占比 ≥ .5；负窗：仇恨帧占比 = 0。
- 帧映射：窗 [t1, t2) 对应 4 fps 帧 [round(4·t1), round(4·t2))，截断到 GT 长度。
- 只看 within 可计的视频（GT 同时含正负帧：HateMM 84，HateClipSeg 99）。

### 统计量与门
- **P1 多说话人比例**：多说话人视频 / within 可计视频。**门：两语料都 ≥ .30。**
  理由：机制只在多说话人视频里起作用；平均 within 要涨 .01，三成视频里每个要涨约 .033，再少就不够。
- **P2 说话人身份上限（读 GT）**：多说话人视频里，每个说话人 h_k = 其主导窗内的仇恨帧占比；窗分 = h_{spk(w)}，
  无说话人窗 = 该视频整体仇恨帧占比；按窗展开成帧曲线算 within，与 r6_bma 在同一子集上的 within 比较。
  **门：上限 − r6_bma ≥ .05，两语料。**理由：实际方法只能拿到上限的一部分。
  对照（只报告）：同一上限在视频内把说话人标签在有说话人的窗之间打乱（10 次，种子 0）后的 within；这个数衡量
  "按窗分组再用 GT 算组内占比"本身带来的泄漏，真实标签的上限要明显高于它才说明说话人身份带信息。
  干跑（用 Whisper 句段交替编两个假说话人）里打乱后的上限约 .64 / .62，低于 r6_bma。
- **P3 语音假高分窗口属于谁**：负窗中 r6_bma 窗分（score_curve 窗内均值）高于该视频正窗中位数者；统计其说话人
  ≠ 主仇恨说话人（仇恨帧最多的说话人）的比例，与全部有说话人的负窗里的该比例（基线）并列。只报告，不设门。
- **P4 无标签指标**：多说话人子集上，窗读数 z 换成同说话人各窗 z 的均值后的 within，对原始 z 的 within。只报告。
- **决策**：P1、P2 两语料都过 → 写完整提案（规则 4 审查、规则 6 代码审查）后实现；任一不过 → 记录数字，
  本候选归档为"预检未过"，不花读取 GPU。

### 代价
- 分割一次：16.0 小时音频（HateMM 8.13 h / 215 个，HateClipSeg 7.85 h / 118 个），一张 RTX 5090 预计 20–40 分钟；
  新视频每分钟音频约 2–4 秒。预检分析纯 CPU，一分钟内。

## 3. 怎么跑
- 环境 `.cache/envs/diar`（python 3.11 venv，torch 2.7.1+cu128，pyannote.audio 3.3.2；安装日志 `runs/_setup_lab1/diar/`）。
  `pyannote/speaker-diarization-3.1` 与 `pyannote/segmentation-3.0` 是 gated 模型，须用本机 HF token 的账号在网页上接受条款。
- 分割（GPU，只在 Slurm 内）：`sbatch experiments/20261014_m1_speaker_bound/launch/lab1_diarize.sbatch`
  → `data/speaker_turns/`，日志 `runs/20261014_m1_speaker_bound/diarize/run.log`（首行主机名），完成标记 `C44_DIAR_DONE`。
- 预检（CPU，conda `HateVideo`）：`python experiments/20261014_m1_speaker_bound/precheck.py`
  → `runs/20261014_m1_speaker_bound/precheck/{summary.json,per_video.json,table.txt}`。

## 4. Test-read log（规则 10）
- 预检读取 `data/gt_4fps/{HateMM,HateClipSeg}.npz` test 的帧标签，用于 P2 上限、P3 的正负窗定义和 within。
  结论与影响的设计决策在 §5 填写。

## 5. 结果与去向
（预检跑完后填写。）
