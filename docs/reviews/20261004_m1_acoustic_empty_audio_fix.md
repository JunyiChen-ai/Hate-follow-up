# Candidate20 空音频裁剪修复：独立窄范围确认

2026-10-04。**PASS（same-family provisional）**。仅复核 Slurm99 在进入 Qwen/GT 前遇到 ASR 名义时间超出真实音频的具体输入边界修复，不重开方法或其它泛化审查。

两个 sample 端点均裁到 `[0,len(audio)]`。空交集在任何 feature/encoder/decoder 调用之前被跳过，记录 block、名义/实际时间、原字符items和原因。没有补零伪造可对齐音频，也没有把其它词音频当作该词证据。`encoder_calls=len(traces)` 与实际执行块一致，跳过块不计模型调用。

跳过块中每个非space字符按原 `weight` 将其自身词的 proportional 支持分别加入 soft/hard 累加量，并增加该词总字符计数。已有声学字符的单位质量不被覆盖。最终按词归一化等于对“真实字符occupancy + 缺失字符比例回退”按原字符权重平均，既不遗漏部分词的缺失字符，也不会把多token字符重复按token数计权。重复长单字符的分数权重同样保留。

新增 `acoustic_counts` 仅在真实模型对齐后增加。标点词 donor 从 `acoustic_counts>0` 的同segment词中选，因此纯回退词不会被冒称声学 donor。部分对齐词仍可成为 donor，其支持本身是已声明的声学/比例混合；本修复不声称 donor 的每个字符都具有声学证据。

可读 CACHE_VERSION 已增加 `with empty-audio guard`，原缓存会被当前 validator 拒绝；旧五份缓存保留后重新抽取是正确迁移方式。原五份正常输入的精确支持一致性应在新抽取回传后实际比较，本审查没有提前宣称比较已通过。

独立 CPU 证据：

- `runs/20261004_m1_acoustic/code_review/empty_audio_cpu_checks.log`：完整五项现有测试全部通过，新增 test_empty_audio 的 aligner 没有任何 model/processor，所以意外进入模型路径会立即失败；结果验证零调用、支持保留和纯回退词不作 donor。
- `runs/20261004_m1_acoustic/code_review/empty_audio_partial_weights.log`：抽取修复代码的实际空块字符累加循环，用预先已有两单位声学字符质量、`.25+.75`单位缺失字符和space构造部分词 fixture，确认声学质量保留、缺失质量恰加一次、space忽略，且 acoustic_counts 不因回退增加。

未读取GT、未运行GPU、未修改生产代码。该修复不改变已有非空音频块的 DP、attention、reader 或常数；没有性能结果，不据此评价或修订方法。可同步后按新缓存版本重新执行完整抽取与既定评分，真实输入覆盖仍由运行确认。
