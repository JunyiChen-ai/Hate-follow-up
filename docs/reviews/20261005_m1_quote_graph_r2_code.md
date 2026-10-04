# QuoteGraph24 R2：独立窄代码审查

2026-10-05；独立实例 `/root/latents_code`，gpt-6-astra，同模型不同作者实例，**same-family provisional**。依规则6仅审 R2 empty-packet native fallback 与新增绑定/成本/编排，不重新审 proposal 或 R1 完整方法。**最终 PASS，已发现的两项问题修复后窄复测通过，无遗留阻断。** 不代表实际8B GPU parity、语义或性能结果。

审查对象：`measure_r2.py`、`analyze_r2.py`、三个R2 launch脚本，对照原 measure/reader/graph 和 README R2预声明。没有修改生产代码，没有读取GT、实际预测或实际主实验summary，没有计算性能、运行GPU、计算哈希或新派agent。全部测试证据在 `runs/20261005_m1_quote_graph/r2_code_review/`；模型/来源图模拟明确限于内存，未落盘伪数据集预测或cache。

## 具体问题与修复

1. 新 `reference_speech_window_seconds` 是list，但原标量时间检查按字段名 `endswith('seconds')` 纳入它，`np.isfinite(v) and ...` 对多窗口数组产生 ambiguous truth value，实际 validate_bundle 会失败。作者已从标量循环明确排除此list，保留逐项finite/nonnegative及reference总和检查。固定五视频的多窗口CPU fixture现在全部通过。
2. 实际读取已把fallback原生时间正确加进new_speech_seconds，但validator原先不能把该总量绑定到各个新分支；修改总量并同步改optimized standalone即可隐去fallback成本。作者现保存每个available trace.seconds：fallback严格等于同一次当前视频的reference时间，graph记录该次structural时间，重算new_speech总和，且空speech reference时间必须0。独立“new S成本置0并同步standalone”和“fallback trace时间置0”反例均拒绝。未要求或修改R1代码。

## 独立执行证据

`binding.py/.log` 使用实际Qwen tokenizer/CPU processor、固定五个真实native图像/fullASR；生成图和模型数值为明确CPU stub，全部在内存。执行当前R2 read_video/validate_bundle以及原reader.structural/compile_branch和共享native margin，不是仅复制判断式。实际expanded native IDs为2375/2780/3054/3115/5861，普通渲染575/1000/1254/1315/4121；新增原生上下文保存/重建一致。构造首窗fallback与其余有graph context的组合，验证当前native S精确复用、非空分支仍实际进入原structural、真实BF16 mask dtype与crop路径、source/prompt/token重放。八项腐坏全部拒绝：fallback flag、suffix、margin、native IDs、graph调用数、source packet、new S成本、fallback时间。空speech main/smoke没有额外S/diagnostic。

`layouts.py/.log` 独立补充3窗口布局，main/smoke各覆盖无speech、仅首窗fallback、首个可用speech在中窗且有graph、fallback+graph混合。main actual forwards分别6/7/10/11，smoke6/10/13/14。fake clock的standalone S时间分别0/1/2/3秒，包含fallback成本；这是计数/归因oracle，不是性能时间。两个clone分支均执行：首可用为fallback时clone原生问句，首可用为graph时clone structural。contextless的两个额外forward仍只检查原R1空scaffold masked与普通serial数值近似，不被当作R2 fallback与原生等价性证据。

`wrapper.py/.log` 实际断言R1/R2引用同一production structural与compile_branch函数对象；拦截而不执行两个stream的4个canonical evaluator/fixed-r6 subprocess，全部参数与R1相同。代码编译与三个R2 shell语法通过。R2 run路径独立，Slurm lab2 local-sc474399、GPU1/CPU4/32G；runner仅调用新的measure_r2，来源获取不变，没有从R1预测计算新曲线。analyze prepare从当前来源及bundle重放并校验native allraw；report仍要求native六指标exact、完整333与84/99，统一评测器未复制/修改。

## 结论边界

fallback条件精确为当前已验证packet.selected为空，与标签/旧分数无关；无speech仍缺失。空包复用当前fresh native测量，其S时间计入新方法standalone。非空包保留原R1全部token、mask、positions、FP32与缓存恢复。G/V仍同一次native测量，shared prefix与视觉成本只在各standalone归因一次；physical paired计数为native calls+实际graph S calls+diagnostics，来源处理成本另外明确计入新方法。

R2新缓存按独立VERSION/current source/native input/trace/计数与成本验证后才resume，逐视频atomic记录，不读历史预测拼接。实际8B固定五视频的native allraw、clone与contextless守门尚需GPU完成；CPU stub和R1既有证据不能代称该验证。此PASS仅结束本次窄代码审查。
