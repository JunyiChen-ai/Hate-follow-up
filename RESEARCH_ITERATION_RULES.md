# Research Iteration Rules

**2026-09-10 立。** 从 `Retrieval-hate/RESEARCH_ITERATION_RULES.md`（2026-09-02 版）改写，适配本仓库：label-free、无训练、4 fps 协议、三指标并列。原版围绕弱监督训练与 Optuna 搜索的条目（训练、checkpoint、seed 确认）在这里没有对应物，已替换为无训练方法的运行、常数与晋级规则。

## 目标

做出一个 label-free hateful video temporal localization 方法，同时满足：

- **零标签**：训练、适配、常数选择、阈值选择都不使用任何仇恨标注；方法对一个新视频的处理流程里不出现任何 split 的标签。
- **SOTA**：按第 8 条定义，在 HateMM 和 HateClipSeg 的 test 上，pooled frame ROC-AUC、pooled frame PR-AUC、within-video macro ROC-AUC 三项超过固定对照表（2026-09-10 裁定三项并列）。
- **Novel**：按第 4 条定义。迁移自其他任务、没在 hateful video detection / localization 用过、且有效的方法即算 novel。

## 前一阶段的教训（只记结论）

- 2026-08 idea discovery 没有晋级门和消融门，两周产出 260 多个脚本、v1–v6 六个版本，只保留了 v6。2026-09-09 补做消融才发现：pooled 两项全部来自整视频 MLLM logit；Möbius 交互项、置换校准次数、占比截距三个部件去掉后数字不变。没有消融门的部件会一直留在方法里。
- 同一方法的 within 在置换种子改变后只动第四位小数，pooled 不动；单次运行 .002 以内的 pooled 差异和 .005 以内的 within 差异不构成结论。
- STATUS 里 T3AL 一行的数字曾误抄自另一个变体的 metrics 文件（2026-09-10 发现）。权威数字只认 `runs/` 里评测器输出，表格引用必须带文件路径。
- Retrieval-hate 的教训同样适用：门槛取自方法自身或不可复现的对照会让所有候选"失败"；实现前的"理论上可能退化"审稿对结果没有预测力；每轮加规则只会让规则互相矛盾。

## 规则

1. **数据集固定**：主数据集只有 HateMM 和 HateClipSeg（2026-09-09 裁定）。MHC-EN、MHC-ZH 不跑、不作门、不进论文主表；旧文档里的 MHC 数字只作历史记录。新数据集只做 external validation，加入前须用户同意。方法不读任何 split 的标签，因此不存在 train / validation 的使用问题；test split 的用途只有评测和第 10 条的 error analysis。

2. **指标固定**：`CLAUDE.md` 裁定的三项（pooled frame ROC-AUC、pooled frame PR-AUC、within-video macro ROC-AUC），4 fps，test 集，三项并列作比较（2026-09-10 裁定）。评测器全仓库只有一份（`src/eval/evaluate.py` + `src/eval/evaluate_four_datasets.py`），不得复制或改写。within 只在正负帧都有的视频上计算（HateMM 84 / 215，HateClipSeg 99 / 118），报数时带上视频数。

3. **禁止 ensemble 与后处理。**
   - 不得组合多个独立模型对同一信号的 prediction、feature、posterior 或 decision（例如两个 VLM 的判断取平均、两个视觉编码器的曲线相加）。一个模态一个模型；不同模态各用一个冻结模型（视觉、语音文本、音频）不算 ensemble，整视频判断与逐段判断由同一 MLLM 给出也不算。
   - 不得做按语料路由、按语料常数、任何用到标签的校准或阈值。无标签的、由视频自身数据按固定规则得到的量（秩变换、去均值、置换空分布）不算后处理。
   - 平滑、区间解码等只作为方法的显式部件，且必须进消融。

4. **Novelty 判定（proposal review，一名独立 agent，一次）**。候选来源两种都行：从 test error analysis 出发自己设计的机制；或从其他任务迁移的方法。只在以下四种情况 STOP，否则放行：
   - 来源方法已用于 hateful video detection / localization（审稿必须实际检索并记录）；
   - 纯 ensemble；
   - 纯 calibration / 后处理 / 平滑；
   - 纯工程技巧而非完整科研方法（只改 prompt 措辞、只换编码器、只调常数、只改分辨率）。

   不得以"可识别性""可能退化为常数""可能存在 shortcut"等实现前推理 STOP；这些交给 test 结果判断。

5. **方法可以带自己需要的输入，但必须报计算代价。** 新编码器、词级 ASR 时间戳、更高帧率、VLM 逐段打分、人脸/说话人轨迹等直接抽取，不单独过审。新缓存放 `data/<类型>/` 并写 `PROVENANCE.md`。提案同时写明：哪些缓存可复用、每个视频新增多少次模型调用、预计 GPU 时间（`CLAUDE.md` 方法计算成本条）。换输入本身不构成 novelty（第 4 条）。

6. **Code review（一名独立 agent，一次）**。只查会改变实验观察或结论的 bug：机制是否实际进入最终分数、打分路径里是否读到任何标签或 GT 数组、特征 / 时间戳 / 4 fps 网格 / video_id 对齐、缓存是否对应当前输入版本、是否调用统一评测器。不审风格、重构、健壮性、理论完备性。发现 bug 只确认修复，不重开泛化 review。

7. **运行与常数（替代原版训练与 Optuna 条）**。方法无训练，一次运行 = 两个语料各自完整跑一遍并出三项指标，整体在一台机器上完成（`CLAUDE.md` 派发规则）。
   - **常数先声明后运行**：方法里所有数值常数（阈值、窗长、块长规则、prompt、锚点词、置换次数、温度等）在运行前写进实验 README，两语料共用同一组。
   - **常数扫描属于开发期证据**：允许扫描，扫描网格在扫描前写进 README，全部结果写进 `runs/`，不得只记最好的。最终方法只取一组常数，两语料相同；报数时同时给出扫描范围内的最差值，作为敏感性说明。用 test 指标选出的常数按第 10 条标 development-selected。
   - **随机性**：方法里的随机数种子固定为常数（禁用内容哈希派生种子，`CLAUDE.md` 哈希限制）；VLM / MLLM 推理用贪心解码。不跑多 seed。
   - **噪声下限**：pooled 差异 .005 以内、within 差异 .01 以内视为噪声，既不算赢也不算输。依据：置换种子改变引起的 within 变化 .0003、pooled 变化 0。
   - **运行记录**：`runs/<exp_id>/<run_name>/` 含 config 快照、代码版本说明（路径 + 日期 + commit）、`run.log`（首行主机名）、`run.pid`、`metrics.json`、预测文件。

8. **SOTA 与晋级定义**。对照表 = `research-wiki/STATUS.md` 结果表中用本仓库评测器在 4 fps test 上跑出的行，不重训、不重跑对照。
   - **对照门（label-free 同类方法）**：两语料三项指标都超过表中最强 label-free 对照（当前 T3AL 重跑：HateMM `.6091 / .3096 / .5068`，HateClipSeg `.6246 / .5645 / .5003`）。新增 label-free 对照（LAVAD、ZS-CLIP、ZS-ImageBind 等）须在本评测器下重跑后再进表。
   - **参照（弱监督）**：MultiHateLoc 重跑（HateMM `.7618 / .5188 / .6108`，HateClipSeg `.5056 / .4885 / .4996`）并列报出，标明用了视频级标签；不作门。
   - **晋级门（取代当前方法）**：候选相对当前方法（OMSL-v6：HateMM `.8507 / .5781 / .6494`，HateClipSeg `.6692 / .6622 / .5473`），两语料任一指标都不下降超过噪声下限，且至少一项指标在两语料上都提升 ≥ .01。过门的候选成为新的当前方法，更新 STATUS 与 `CLAUDE.md` 项目条。
   - 三项指标都要看；不挑指标、不挑语料。

9. **分流**。
   - 过晋级门：按第 14 条核对后向用户汇报；随后补消融与独立 novelty 复查。
   - 未过门，但某语料某项指标提升 ≥ .01：方法保留，用 test error analysis 找原因后修改再跑，同一方法最多 3 轮修改；仍未过则归档，写明最好数字。
   - 没有 .01 以上提升：归档，写一行负结果，换候选。
   - 实现不可靠：修复重跑，不评价 idea。

10. **Test error analysis 合规使用**（`CLAUDE.md` 对着 test 开发条）。允许读取 test 预测与 test GT 做 error analysis 并据此改设计；每次记录看了哪些文件、发现什么、影响了哪个设计决策，写进实验 README。由此得到的数字是开发期证据，STATUS 和论文里标 development-selected。test 标签不得进入任何打分、拟合、常数或阈值计算路径。不要求另设确认集（用户裁定 2026-09-09）。

11. **不设自动停机与 process review。** 每归档 5 个候选，主 agent 在 `research-wiki/STATUS.md` 写一段不超过 10 行的小结（试了什么、最好数字、下一步）给用户看，然后继续。流程规则只由用户改。

12. **归档**。每轮一个 `experiments/<YYYYMMDD>_<slug>/`，含 README（机制、来源、需要的输入与代价、常数、怎么跑、结果、去向）；输出在 `runs/<exp_id>/<run_name>/`。淘汰后整目录移入 `archive/experiments/`，README 顶部一行原因。权威数字只认 `runs/` 里评测器输出。每轮结束更新 `research-wiki/STATUS.md`。

13. **方法必须统一（反模式：按语料换流程）**。一个方法 = 一套输入、一套 prompt、一套常数、一套推理流程，两个语料完全相同。以下按语料改动一律视为不同方法，不得合并成一个方法 claim：
   - 常数、prompt、锚点词不同；
   - 模块开关不同；
   - 输出读出形式不同（残差、直接、只正证据等）；
   - 按语料的后处理或校准；
   - 按语料挑不同分支报数。

   允许的语料间差异只有：由视频自身数据按同一固定规则算出的量（例如每个视频的块长由其自相关决定）。

14. **"做完了"的声明门槛（防走捷径）**。向用户汇报 SOTA 或晋级前必须同时满足下列各项，任一缺失只能汇报为"进行中"并写明缺哪项：
   - (a) 数字来自最终代码的一次完整运行（两语料、`runs/` 里的 `metrics.json`），不是开发期零散跑出来的历史数字；
   - (b) 领先幅度小于噪声下限时不得写"超过"；
   - (c) 第 13 条统一性满足；
   - (d) 所有常数在运行前写进 README；由 test 选出的常数列出扫描过的候选值与对应数字，并标 development-selected；
   - (e) 无 ensemble、无后处理、无按语料分支；
   - (f) 用了比对照更强的编码器或 MLLM 时，报"最强 label-free 对照 + 同样模型"的数字，或说明为何不可行；
   - (g) 消融在 test 上显示每个作为 novelty 主张的部件去掉后，至少一项主指标在两语料上都下降 ≥ .01；下降不足 .01 的部件不能作为 novelty 主张，且应从方法里删除或降级为实现细节（2026-09-09 消融：整视频 MLLM 截距、三路主效应、视觉主导字典序过此门；Möbius 交互项、置换次数、占比截距不过）；
   - (h) 评测器、split、GT、4 fps 协议未改动；
   - (i) HateMM 与 HateClipSeg 两语料全部三项指标都报，不挑语料、不挑指标；
   - (j) 报计算代价：每个视频的模型调用次数、缓存复用情况、单语料 GPU 时间。

## 标准流程

1. 主 agent 提出方法：来自 test error analysis 的自研机制，或从其他任务迁移。写一页 README：机制、来源、需要的输入与代价、常数、预期改善哪项指标。
2. 独立 agent 按第 4 条做 proposal review。放行即实现。
3. 实现，抽取需要的输入（默认在实验室机器，`CLAUDE.md` 派发规则）。
4. 独立 agent 按第 6 条做一次 code review。
5. 一次完整运行：HateMM、HateClipSeg 各自全跑，出三项指标。
6. 按第 8 条对照门与晋级门判定。
7. 按第 9 条分流：过则补消融，按第 14 条核对后汇报，成为当前方法；有提升则修改继续；无提升则归档换候选。
