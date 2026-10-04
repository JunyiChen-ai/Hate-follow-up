# Program23 固定5获取接口失败：独立窄诊断

2026-10-05；独立实例 `/root/latents_code`，gpt-6-astra，same-family provisional。

**结论：本次实际输入重放和目标接口检查未发现导致零 module call 的代码缺陷。观察到的是Qwen原始输出不满足当前typed程序API，严格解析/执行正确拒绝。R1 smoke仍失败，不得进入main，不得降低mechanism guard。后续需要显式声明新的source/decoder接口设计，不能把自动类型转换、JSON修复或放宽cap当作透明实现修复。**

## 范围与证据

仅审查Slurm130回传的五份 `data/temporal_evidence_program/<dataset>/<video>.json` 获取metadata：source/chunks/实际raw generation/tokens/plans/execution，以及原`program.py`、`inputs.py`、`extract.py`、共享Judge._step/model_inputs和实际生成主机sc474398的HF5.15.1源码。读取prepare日志末尾的guard异常，没有读取任何实际预测、GT、main metrics或main summary；未GPU、未下载权重、未修改生产代码、未计算哈希或记录run commit ID。

证据目录 `runs/20261005_m1_program/interface_diagnosis/`：

- `audit.py`、`audit.log`、`acquisition_audit.json`：逐真实缓存重建source与执行结果，全部相等；真实离线CPU processor重建所有22个planner prompts、图片grid和input token IDs，输出token decode/raw逐条相等、调用数相等。JSON记录包含实际raw生成和source坐标/字符长度，便于复查；不是评分输出。
- `target_position_source.txt`、`target_model_source.txt`：只读SSH取得实际lab3 HateVLM HF源码，不含权重。
- `positive_and_position.py` / `.log`：使用真实source的显式合规手写fixture，scope/action各实际调用一次；该fixture仅内存，不替换/修复任何生成输出。执行捕获的实际HF position函数，验证multimodal prefix建立delta后，无cache_position的逐token增量读取使用past KV长度+delta。

## 实际失败来源

总计22个planner chunks、158个固定窗口。51窗仅通过顶层JSON/window/ops列表解析；这不等于内部操作合法或实际机制执行。其余91窗缺失、8窗invalid_array、8窗因整chunk截断而UNKNOWN。五视频module调用均0。

| 视频 | 窗口/chunks | 实际原始输出与拒绝依据 |
|---|---:|---|
| HateMM `hate_video_1` | 12/2 | 原始span为`["span","s0","s0",0,94.94]`等：segment传字符串`s0`，end_char误用94.94秒。真实source segment为整数0、42字符、结束时间94.94。后续部分窗口还引用当前窗口未定义变量。12个span全部invalid_span，local/context/scope继承UNKNOWN；没有合法perception source。 |
| HateMM `non_hate_video_4` | 5/1 | 只输出window0，20个frame操作加emit共21操作；前12frame执行，其余9操作正确受cap拒绝，缺emit，其余4窗缺失。原输出没有action/scope调用。 |
| HCS `bit_0EHvMSiEHVoc` | 29/4 | 首chunk含未加引号的变量` s0/l0/c0 `，不是JSON，8窗invalid_array。其它chunk缺11窗，span将时间字符串/小数用作字符offset、local arity错误、join缺合法local witnesses。内部错误来自原raw，不是token decode后改写。 |
| HCS `bit_0nXuyV2rypaf` | 33/5 | 17窗缺失；一chunk2048 tokens达到上限，8窗整块UNKNOWN；另有20frame枚举/超cap、时间作字符offset和错误local参数。无合法perception执行。 |
| HateMM `hate_video_114` | 79/10 | 59窗缺失；多数chunk只返回首窗的frame枚举。另一chunk的span少参数、scope context给对象而非先前变量名，local所选frame在当前窗之外；部分emit为空。无module调用。 |

所有实际chunks未发现重复JSON键。以上分类按原生成、原解析与原执行逐条复现，没有重新解释model意图。实际guard异常为 `mechanism not exercised in this corpus smoke`；它正确阻止把无perception机制的输出当正式实验。

## generation/cache/interface核查

`extract.generate`每次重新render/encode真实输入、重置rope_deltas、建立fresh cache，FP32输出head贪心argmax；保存非EOS tokens，达到上限且未遇EOS标truncated。每个已保存token后实际执行一次_step，所以缓存记录的forward数为1+token数，全部实际记录吻合。没有在生成中把整数变字符串、把字符offset变时间、插入错误op、混用上一个chunk cache或偷偷裁剪/修复JSON的代码。

实际HF5.15.1 `compute_3d_position_ids`在初始带图输入用真实grid/mm_token_type_ids计算mRoPE；后续_step无图且past KV>0时使用past_seq_len+rope_delta。源码forward将它传入language_model。Judge.model_inputs传递实际processor的mm_token_type_ids并有明确fallback；_step不显式传cache_position符合该目标接口。本次CPU执行实际函数验证了该分支，但没有冒称重新验证8B每一步logit或排除所有模型数值因素。

解析和执行API明确整数segment/frame/字符offset、窗口内变量引用、固定arity、12操作/2perception上限。实际输出违背这些明确要求；拒绝与声明一致。独立合规fixture在同一真实source上能通过span/local和frame/local分别触发scope/action，说明调用入口并未被代码永久阻断。

## 后续边界

当前“自由greedy JSON生成→严格typed解释器”接口在实际固定5上未可靠执行，这是已观测实现可靠性失败，不是方法性能结论。若继续候选23，应预声明显式source/decoder接口变更与新版本/cache绑定，再按固定5无GT验证机制确实执行；不自动将`s0`转0、不将时间猜成字符位置、不修补缺窗/错误arity、不提高cap或2048上限来让本批输出通过。既有main停止与mechanism guard应保持。本报告不修改算法、不授权main或任何新GPU任务。
