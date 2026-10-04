# Candidate25 source-handle interface B：独立窄代码审查

2026-10-05；独立实例 `/root/latents_code`，gpt-6-astra，与作者不同实例、同模型；**same-family provisional**。按规则6仅审新增B source interface/decoder/绑定与执行接线，不重开proposal或A方法泛化审查。**最终 PASS。** 一项位置绑定问题修复并独立复测；冻结前另窄确认GPU前声明的compact source表示。无遗留观察阻断；不是8B GPU parity、事实语义或性能认证。

范围：README B事前声明、A实际接口诊断记录，`src/structured_source_generation.py`，实验 handle_interface/inputs_handles/extract_handles/measure_handles/analyze_handles 与三个handles launch脚本。未改生产代码，未读GT、实际预测或主实验summary，未运行GPU、计算哈希或另派agent。所有证据在 `runs/20261005_m1_provenance/handle_code_review/`。实际源ASR/JPEG/PTS/PNG可读；合成选择与readout只在内存，不写伪数据集预测/cache。

## 唯一观察问题及窄修复

原 validate_generation 只检查保存的rope_delta为整数、positions依赖该保存值递增，没有绑定当前expanded IDs/image grids。独立真实36层BF16 CPU小模型生成后，将delta和所有positions同时+1，重放仍接受，可能复用错误位置下的观测。作者新增 image_rope_delta，从当前静态图像expanded IDs、grid及merge size计算应有压缩delta；prefill实际delta与其相等，缓存重放也须相等，纯文本必须0。原反例现在拒绝。

独立只读SSH取得目标HF5.15实际 get_rope_index/get_vision_position_ids源码，保存为 target_rope_source.txt/target_vision_positions.txt；在CPU执行这些真实函数，同时对照本机HF4.57函数。固定五视频22图像的delta分别−1716/−1760/−1716/−1716/−1694，均与新增计算精确一致。该测试核对位置计算，不把CPU函数执行当真实8B视觉embedding/logits parity。

## 独立可重放证据

- `tiny_generation.py/.log`：真实随机初始化36层BF16 Qwen text model、实际Qwen tokenizer，独立FP32 linear allowed-token argmax逐步核对；每个forced/choice token均实际forward，显式三维位置等于当前KV长度加delta；fresh重复tokens/events精确；cap1实际2forwards且整体不完整。错误delta+positions修前接受、修后拒绝，修前日志另存 tiny_generation_before_fix.log。该数值测试为CPU文本小模型，非目标8B多模态验证。
- `interface_oracle.py/.log`：实际tokenizer执行合法选择与原Stream replay；独立枚举含中文、重音字符、重复词的全部唯一连续span，检查真实char边界、实体≤4词/引语≤16词；frame与text handle分别精确展开原schema。实际strict parser接收后产生类型正确节点/关系，same_entity真正使另一窗口可检索。whole capped ledger只保留确定utterance，STOP终止但不成为边；空观察/UNKNOWN仍可表达，没有强制实体/边以越过guard。
- `limits_and_wrappers.py/.log`：实际tokenizer验证64 content tokens和24 words截断闭合语义；手算并枚举58个合法type/direction/time/anchor endpoint组合，used tuple排除一致。A/B引用同一production read/validate_trace函数；两个stream的4个canonical evaluator/fixed-r6 subprocess仅拦截、逐参数与A相同，未执行GT或评分。
- `actual_inputs.py/.log`：最终compact输入的固定五视频真实overview/local PNG/full source body CPU processor重放；expanded长度3135/3098/3255/3273/3097，22图像各自真实当前paths；40项prompt/input tokens/image顺序/delta+positions/choice/compiled selection/capflag/forward-count腐坏全部拒绝。所选事实为明确脚本fixture，只存内存，不能作为GPU感知能力证据。
- 六个新Python源码编译与三个handles shell语法通过。计数/成本路径沿用A已审实现：所有forced和chosen token计入1+output_tokens；每ledger一次vision prefill、text-only link零vision；source总生成/解码成本进入新方法standalone，最终reader仍独立fresh V/S及对应repeat。

## Compact source表示的限定确认

冻结前作者实际全333纯输入预检发现exhaustive坐标表过长，README新增GPU前声明。当前 `t<startword:04d>_<exclusive_endword:04d>` handle与一份word字符边界表替代逐span坐标序列；不是删候选、截原文或改变4/16词常数。新的handle字串和prompt改变token序列，不声称与旧表示的模型输出数值等价。B尚无GPU生成，旧prompt若被复用会被当前严格重放拒绝。

`compact_source.py/.log` 独立读取已知最大来源 HateClipSeg/yt_VWWnMsLjCdY window29：原文2225字符，重建旧算法与当前算法的全部6550个span，start/end/entity eligibility按序精确相等；每个新handle映射正确word与char边界。独立实际解码该视频、验证PTS/selected PNG，再构造本窗口真实22图像输入：旧catalog本身164242 tokens，当前boundary表3980 tokens，完整新prompt plain5721、actual processor-expanded7701 tokens。仅验证该实际最大旧catalog来源的现输入；不将它冒称未来完整生成link表的最大尺寸或GPU容量证明。固定五视频的最终compact输入/choice replay/腐坏检查已重跑；exhaustive阶段日志另存 actual_inputs_exhaustive.log。

## 最终接线、来源和结果边界

B生成的是预声明support/span/choices接口；确定compiler只取当前真实frame IDs或唯一原文字串，随后走原strict parser，没有补齐A输出或近似字符串匹配。Link只在现有typed nodes、实际anchor/time约束里选，STOP和整体cap不产生伪边。同一个graph/components/retrieve实际控制原reader的endpoint pixels/body；native fullASR与当前k20实际18–20张、ownstance/G、fresh V/S/max、4fps及固定r6未改。源码读取及运行入口没有方法内GT/旧预测路径。

B source cache独立为 data/temporal_entity_discourse_graph_handles，所有新run位于 runs/20261005_m1_provenance/r1_handles_*；A缓存和输出不改用为B模型观测。输入版本/常数/模型、实际来源像素、prompt/expanded IDs/grids、完整choice/forced events、编译结果、图/packet、调用与成本均重放绑定。新获取metadata仍atomic落盘；评分重跑fresh读数，不用历史分数拼接。lab2 Slurm GPU1/CPU4/32G、项目内日志与HateVLM环境符合既定路径。

实际固定五视频仍必须通过native allraw exact、当前fresh repeat和每语料非空ledger与非本地检索guard；CPU语法/来源合法不能证明身份、归属或描述事实正确。空/UNKNOWN输出仍可能使guard失败，不能强制成功。只有实际检查通过后才允许完整主实验；本PASS不评价B收益或消耗结果修订预算。
