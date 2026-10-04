# Program23 source-handle interface B：独立窄代码/接口审查

2026-10-05；独立实例 `/root/latents_code`，gpt-6-astra，与主writer不同实例，same-family provisional。**PASS。发现的一处新增smoke计数绑定缺口已修复并实际确认，无未解决的观察有效性blocker。** B是已声明的新source/decoder接口，不是对A原始输出的自动修复。A固定5失败结论保留；本报告不认证B真实8B执行或性能。

## 范围与约束

读取README末节、原program/inputs及A窄诊断，审查全部七个handle Python文件和三个launch入口，核对最终planner_content布局。没有改生产代码、读GT/实际预测/main summary、GPU或下载模型、计算哈希。实际固定5 ASR/JPEG仅供CPU来源与tokenizer检查；新fixture元数据、模型读数与评分records只存在内存，没有伪实际缓存/预测落盘。

## 核对结果

1. **来源与编译。** S/F句柄映射真实整数segment/frame，不生成坐标。speech候选为原local算子的完整包含字符单元；frame按nominal point半开窗筛选；context仅取时间上不相交的真实segment。选中source仍编译到原span/local/context/scope/action/join/emit解释器，最多12操作、2modules、2contexts；join必须满足真实local span与frame时间重叠。local span和module输出进入各自visual/speech问题，contexts随scope仅作解释来源，不独立当本窗事实。
2. **最终输入布局。** 原source inventory中的每segment原文一次呈现，明确S/F handle与numeric ID；各窗口remote contexts仅重复handle/ref/time，不重复全文。speech clip仍是原文真实子串。最终布局已用于本次真实来源/编码/重放检查，未沿用旧prompt。
3. **受限greedy。** choice trie使用实际Qwen tokenizer；在合法下一token集合中以FP32 head logits做argmax，shared前缀不跳过token。固定标点、JSON key、window ID、支持source ID、强制结束引号均真实forward、保存、计数。EOS不在合法选择中，不能提前终止语法；语法完成结束，未采用外部修复。每次planner/module新cache、rope重置，计数为prefix一次加所有输出token forward。
4. **描述/目标边界。** scope target仅UNKNOWN或真实local文字中1–4个连续whitespace词的Unicode字符边界。action字段仅允许单token及累积decode均合法的JSON内容，排除quote/backslash/control/replacement/special token；关闭引号可由模型选中，在12words或24content tokens强制关闭并记录reason。2048/96为全部输出tokens的硬cap；不补齐被截断结构、不重试，planner整chunk UNKNOWN，module由原解释器判无效。字段cap与整个生成cap分开保存。
5. **机制与成本。** 必有候选时选择真实speech/frame是B新声明；模块仍需真实同模型fresh读取，语法有效不证明描述或归属正确。原native20/fullASR/G/own stance不改；带执行record的独立V/S问题及原max/fixed-r6保留。原获取、planner/module token-forward成本计入新视频；clone与reference成本分列。B专用cache/version/run路径不会复用A结果。
6. **绑定与守门。** 当前source、plan selection、编译ops、实际执行packet/result、prompt/images/grid/input tokens、每个生成token/event/choice/field-stop、cap标记与forward数可重放；旧版本和输入变更拒绝。测量记录与执行record、问题/suffix token、native P/N及当前CPU conversation绑定。原每语料module与meaningful guard保持，并新增structurally-valid module guard；仍须实际固定5才允许main。统一canonical evaluator与fixed-r6参数未改变，GT不进入获取/reader。

## 唯一发现与修复

新增smoke guard读取`perception_nonUNKNOWN_calls`，初版validate_records只复核总module_calls，没有重算新增nonUNKNOWN/UNKNOWN计数。checks中的该值被改变时可能使guard错误通过。主writer现从当前execution.calls的result.kind逐项重算两值，同时绑定planner/valid/nonempty计数、source preprocessing/forward成本及两份standalone分量。独立修改这些字段的fixture均被拒绝；正式解码/选择/评分算法未因此改变。

## 独立执行证据

`runs/20261005_m1_program/handle_code_review/`：

| 证据 | 实际结果 |
|---|---|
| `decoder_oracle.py` / `.log` | 真实离线Qwen tokenizer，独立FP32合法token argmax oracle；EOS即使最高logit也不能越过grammar，forced token逐次调用_step；12word/24token字段cap与96/2048整体cap检查。真实5视频158窗的每字符单元独立枚举和current grammar replay通过，受控logits fixture触发141scope/76action/51join，planner共5564tokens。仅合规可执行性fixture，非GPU感知/效果。 |
| `tiny_decoder.py` / `.log` | 真实随机初始化两层BF16 Qwen3VLTextModel、真实Qwen tokenizer，合法选择逐步FP32 argmax核对；14个forced/choice tokens均执行真实模型forward，实际KV长度增加14，保存tokens等于全部step输入。CPU数值，不冒称目标8B parity。 |
| `binding.py` / `.log` | 真实第一视频ASR/JPEG/CPU processor，stub logits实际调用完整acquire→source+generation replay→paired reader。2planner/24module，scope/action record进入最终问题；native P/N/current conversation一致。14类source/current-ASR/version/token/event/selection/感知计数/input calls/time/standalone/question/suffix/record corruption拒绝。真实generate达到96cap后保存与replay一致。所有fixture仅内存。 |
| `eval_and_unicode.py` / `.log` | 非ASCII/emoji字符单元、原local target offsets手算核对；真实tokenizer生成24个`中`后field cap正确且JSON内容合法。拦截4次canonical/fixed-r6 subprocess参数，不执行评测或访问GT。PASS。 |

七个handle文件编译和三个shell入口语法检查通过。独立测试使用最终单次source-inventory布局。CPU实际小模型环境为本机HateVideo（Torch2.7.1 / Transformers4.57.6），目标lab3 HateVLM/HF5.15尚无B实际运行；前次A诊断对目标增量cache/mRoPE源码的结论可沿用，未将它当B输出质量证据。

**后续仍需实际固定5。** sc474398 Slurm运行后必须重新检查两语料真实模块执行、structural有效结果、native allraw exact、cloned margins、source/call绑定与成本；不允许用上述CPU fixture数代替。PASS仅结束本次独立代码/接口审查，不弱化guard，不授权跳过固定5，也不触碰正在运行的QuoteGraph任务。
