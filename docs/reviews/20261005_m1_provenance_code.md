# Candidate25 provenance graph：独立代码审查

2026-10-05；独立实例 `/root/latents_code`，gpt-6-astra，与作者不同实例、同模型；**same-family provisional**。按 RESEARCH_ITERATION_RULES 第6条完成一次 observation-only review，发现的具体问题由作者修复后仅窄复测。**最终 PASS，无遗留观察阻塞。** 此结论允许后续固定五视频实现检查，不是实际8B GPU等价、来源关系语义正确或性能/novelty认证。

范围：实验 README、graph/inputs/extract/reader/measure/analyze/selfcheck 与 launch，及新共享 `src/actual_video_frames.py`、`src/source_generation.py`；核对共享 native stance helper/renderer。读取既有独立 proposal PASS，不重审提案。未改生产代码；未读取 GT、实际预测或主实验 summary；未运行 GPU、计算内容哈希或记录 run commit ID。所有新测试和日志位于 `runs/20261005_m1_provenance/code_review/`。实际 acquisition source/ASR/JPEG 与 CPU source audit 可读；受控 readout 只在内存，不落盘伪实际预测。

## 发现与修复确认

1. **重复合法边与预声明不一致。** 原 parse_links 在同一调用出现重复 tuple 时整调用 UNKNOWN，README 明确去重；独立反例已复现。现在在端点/类型/anchor验证后按规范化 tuple 去重，不丢整块。实际非法边仍使整个调用 UNKNOWN；same_entity 对称规范化保持。
2. **实际图片来源绑定不足及跨主机路径问题。** 原 validate_frames 只检查 PNG 格式/尺寸；独立生成12帧64×64视频，正确应选 index4/8，把4的PNG替换为8仍通过。现以当前主机 resolve 的真实视频完整重解码，对照全部 PTS/index/timebase/时间/尺寸/origin/帧数，并对所选PNG实际逐像素比较；错帧反例拒绝。生成主机路径保留为可读出处，不要求等于回传后的本机路径；相同实际来源的不同生成主机目录可以通过。decode_seconds覆盖 acquisition 内这次来源验证。
3. **截断状态绑定不足。** 原 validator 可接受恰达 token cap 却标 truncated=False 的记录，可能改变整调用 UNKNOWN。现严格绑定 `truncated == (len(tokens) == max_tokens)`，与生成循环实际结束语义一致。真实小模型 cap 反例由接受变为拒绝。
4. **成本可与实际 trace 脱离。** 作者新增 native/repeat 各次时间与 prefix/input-binding 独立记录，检查新/原生/diagnostic求和、有限非负和 peak不小于输入峰值。独立内存 fixture 的 new_visual加100并同步改standalone、负峰值、负diagnostic、无限input-binding全部拒绝。调用数与视觉encoder hook/输入生成计数也绑定。

修前证据保留在 `oracles_before_fix.log`、`tiny_generation_before_fix.log`；修后对应脚本/日志均通过。成本复测运行时作者的修复已落位，因此不声称保存了修前成本执行日志。

## 独立实际检查

- `oracles.py/.log`：真实受控视频编码/PTS选择/PNG像素 oracle；同尺寸错帧拒绝、跨主机可读路径兼容；手算零跳实体链、两跳可达/第三跳不可达、远端window不反向展开成员、重叠重复literal整ledger拒绝。
- `source_revalidation.py/.log`：独立执行新来源守门，固定五个真实视频全部PTS与所选PNG像素精确通过。依次 decoded帧数2845/967/5439/7758/15075，selected图片数24/10/58/66/158。这是来源数据检查，没有模型预测。
- `actual_inputs.py/.log`：实际Qwen tokenizer/CPU processor、固定五个真实native JPEG/fullASR和实际本地PNG，分别重渲染visual/speech。P/N依次2343/2375、2748/2780、3022/3054、3083/3115、5829/5861；全部新分支前N个expanded IDs精确等于native原文+原始global question+指定ownstance的ID序列。每分支真实图像路径数22，本地两张确实追加；40项saved token/image顺序/role/source-record腐坏拒绝。CPU使用指定No stance验证渲染绑定，不冒称完成真实8B自身stance数值测试。
- `packet_oracle.py/.log`：手工有效实体/动作/文本mention/quotation及跨窗same_entity，图实际决定远端与本地endpoint路径顺序。visual新增字段只保留frame实体/动作；speech无action描述且包含实际body；context/local标签、路径关系、完整原始对应question均保持。结构合法不代表生成的身份/归属语义真实。
- `tiny_generation.py/.log`：真实随机初始化36层BF16 Qwen text model与实际Qwen tokenizer，CPU标准SDPA；独立FP32 linear argmax逐token核对，fresh重复tokens精确；cap3实际4forwards，立即EOS实际1forward。检查每步确有36层KV，并验证修后false cap flag拒绝。显式CPU位置包装用于本地HF接口，不将其当目标HF5.15/8B multimodal GPU parity。
- `orchestration.py/.log`：明确标注CPU model stub，实际调用 read_video/validate_bundle；main/smoke × 无speech/首窗speech/中窗speech，实际读调用数9/11/11、10/13/13，空speech缺失与每种可用分支首个repeat正确。新分支在原生读完释放native KV后各自fresh full forward，不使用上一个窗口/分支的答案或KV。
- 同一脚本拦截 evaluate 的4次subprocess，不执行GT或评分：两个stream各调用canonical four-dataset evaluator及既有固定r6入口，noleak/nscore/length prior/grid6/m2参数保持。report导入canonical within helper，检查完整333、native六指标精确及84/99；没有复制统一指标实现。shell语法检查通过，lab2 Slurm分区/GPU1/CPU4/32G及项目内日志路径正确。

## 计算及缓存路径结论

来源只来自当前manifest/native frame_paths/fullASR/真实raw视频。Acquisition ledger中只有native overview、实际local来源和中立schema；overview不能合法充当local frame。严格JSON重复key/类型/唯一字符串/上限/source约束；无retry/salvage。每8窗link读取完整实际节点表，实际token数在prefill前记录，没有截断table以适配机器。实体组件保留每个真实occurrence，0/1权重图执行2跳检索，远端仅按预声明每侧最近两窗选择。

缓存复用重建当前来源、生成prompt/expanded IDs/grids、解析结果、图和检索包，并绑定版本/模型/常数/ASR/窗口和生成次数。当前measure不跳过旧per-video评分记录；重跑评分，因此无混入旧分支的resume读取路径。prepare要求完整paired记录、当前source/trace重放和旧native allraw精确，之后才运行评测。source generation/reader没有GT/标签或旧预测计算入口；actual acquisition及full多模态重编码成本显式保留，新V/S没有复用paired reference分支来少算成本。

实际GPU阶段仍须通过独立native allraw exact、每种可用分支的fresh repeat精确、两语料真实非空ledger和非本地检索exercise守门。未通过应视为实现/输入检查失败，不据CPU PASS推断科学收益。当前没有实际候选25GPU或性能结论。
