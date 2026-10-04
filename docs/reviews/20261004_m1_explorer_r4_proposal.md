# Explorer R4 proposal delta 独立审查

日期：2026-10-04。审查实例：`/root/explorer_review`。结论：**PASS，可以进入实现和独立窄代码审查。** R4 是当前 Explorer 的第三次、最后一次修订，不能据此宣称通过性能门、消融门或证明 self-conditioning 偏差。

本轮读取 `RESEARCH_ITERATION_RULES.md`、`experiments/20261003_m1_explorer/README.md` 的原始决策门与 R3/R4 声明、`docs/reviews/20261003_m1_explorer_proposal.md`、`experiments/20261002_verdict_analysis/README.md`，以及本机 `runs/20261003_m1_explorer/r3_main_decoded/explore/metrics.json`。为核对“原问题可在无裁定上下文中使用”，只读了共享 `yesno_question` 与现有 expanded-read 的上下文接口；没有进行 R4 实现审查。未直接读取 GT 数组、未启动实验、未修改生产代码、未计算内容哈希。

## 合理性与规则边界

R3 原始指标与 README 转录一致：HateMM/HateClipSeg 的有效 within 视频数为 84/99，主门仍未过；HCS 有合格的开发期改善，允许按规则 9 继续最后一次修改。R2、R3、R4 分别计为修改 1、2、3，不把 R4 改名成新方法来重置预算。

R4 保留同一模型、原生全局/语音、窗口和候选定义、选择公式及 .3-nat 继续规则，仅在新增局部图片读取时去掉模型先前完整的 global question/answer 对话。新增读取仍是完整获取方法中的部件，不是独立以删 prompt/上下文为 novelty 的新候选；没有 ensemble、分数后处理、按语料路由或标签校准。因此不触发原 proposal 的四项 STOP。本轮不重开已完成的来源检索，也不声称找到了新的文献 novelty。原 proposal 对 MATCH 近邻和不可夸大的贡献范围继续有效。

原 visual question 本身要求按规则判断指定窗口画面，不依赖“上述回答 Yes/No”一类缺失指代；保留 rules、原始帧和完整 ASR 后该问题仍有定义。但是 full ASR、全片原始图片和保留的 native speech/global 仍在系统中，不能把 R4 称为纯局部信息方法或完全消除了全局影响。

## 主门不放宽

README 原有主门比通用规则 8 更严格：**最终 within 在两个语料上都相对 native/current 提升至少 .01，pooled 任一项不下降超过 .005，三项全部报告**。本次 PASS 沿用该预声明；不能用“两语料任一共同指标 +.01”替代原 within 主门。消融的部件移除门则仍为两语料同一个主指标损失至少 .01，二者不要混用。对照基准继续是当前 native/r6，不换成较弱的 R3。

若 R4 最终未过预声明性能或必要机制门，按 README 记录最好证据并归档本机制。一次有效完整 R4 主结果加事前定义的机制对照不属于新增参数扫描；看到结果后继续改上下文、阈值或选赢家，则不能仍算同一个 R4。

## GT 与归因

已有 test error analysis 用于提出这个修改，符合项目允许的开发协议，所有结论必须保持 development-selected。README 已记录读取的 R3 预测、GT、指标、trace 和 case 文件以及设计影响。新推理只根据视频、模型读数和事前固定规则操作，不允许把 global 正误、GT-negative 尾部、视频身份或旧结果好坏变成运行开关。

删除的是整个自裁定对话，不是孤立 Yes/No token；同时改变序列长度、位置和对话结构。旧 Reader 分析只是动机，不能充当最新 Reader 的配对消融，也不能证明新增图片误报由自裁定造成。含真实仇恨图像但位于 GT-negative 区域的窗口仍须保留语义与标注的区分。

还需准确区分“同一获取规则”和“同一实际获取轨迹”：去掉对话会改变首轮 expanded margin、熵与 attention prior，进而可能改变是否第二轮、第二轮选帧和成本。因此 R3 与 R4 的全量差值是修改后策略的总效应，不能全归因于固定图片条件下的上下文处理。README 已提出在 R4 自己的 acquired sets/counts 上恢复原对话的 replay，方向正确；该 replay 在使用前需声明确切轮次/候选/输入顺序，固定 R4 轨迹，不重新由对照自身读数决定预算，并沿用独立 r6 corpus fit。它只估计条件于 R4 轨迹的对话处理效果，不是任意轨迹上的平均效应。若不执行，就保持上下文动机为假设而不声称因果验证。

四个获取对照必须使用 **R4 的 conditioning、实际 traces 和独立完整语料输出**，不能复用 R3 控制读数。uniform/distance 固定 main 数量重放不证明熵门贡献；fixed4 是简单输入增强基线；mismatch 干预内容与时间对应，仍须报告 matched/unmatched 覆盖。上下文固定后，这些控制才能检验选择和实际图片内容是否在新增机制中发挥必要作用。零获取恢复 native 是接口一致性检查，因其上下文也回到 native，不能单独隔离“新增像素”与“删除对话”的效果。context 本身继续不作独立 novelty；若简单输入增强解释全部收益，不能把 context 选择当作补救 novelty 的依据。

## 成本与实现验证边界

声明的调用数 `3+B+sum(R_w)` 和每窗最多两次 expanded read、四张新帧、第二轮累计重编码共最多六张次，与复用一次 observational prefix 的方案一致。真正独立的 prefix KV 副本需要额外显存和拷贝时间；不能因为不增加模型 forward 就记为免费。35–40 GPU 分钟只是估计，既有 18 GiB 左右峰值不保证复制后仍相同。固定五视频 smoke 应重新测峰值、耗时、forward 与编码张次，再给完整语料预算；上下文改变后 R_w 本身也可能变。原20帧和 ASR 的新视频预处理成本仍需保留披露。

后续窄代码审查应确认 pre-verdict KV 真正来自原始 observational prefix，缓存张量与 native 路径无可变别名，crop/restore 不残留问题或答案；full token render、image grids、mRoPE、attention mask 和 KV 长度一致，不能只检查打印出的 prompt 文本。原生 global、speech、initial visual/prior 与 post-acquisition native replay 必须保持精确一致；新增窗口和轮次相互隔离。使用 fresh observational-prefix render 的独立 oracle 核对缓存位置及实际读取，而不是把旧 verdict 后缀索引平移当作依据。保留候选不足/无候选、18原始帧、缺 speech 的原有边界。评分仍为 canonical 4 fps，r6 数值参数不改。

以上是本次 PASS 的已有协议和结论解释边界，不新增实现前性能保证。未发现必须否决 R4 的提案漏洞；进入实现后由另一实例审查新增缓存与版本路径，再做固定 smoke 和完整双语料运行。
