# Tree R3 controls：独立代码审查

2026-10-05；独立实例 `/root/latents_code`，gpt-6-astra，与实现作者不同实例；**same-family provisional**。结论：**PASS（已准备的 R3 控制代码与 CPU 范围）**。本轮明确问题均已窄修复确认；没有控制 GPU 执行或机制有效性结论。

范围为 README 最后 R3 full controls specification、control_inputs/control_extract/control_measure/control_analyze、measure 的可选 observation 与 binding 参数、BindingRenderer、两个控制 launch。依赖此前已通过的纯几何与 R3 主 reader 审查，不重开 proposal/理论/性能。仅写独立 CPU fixtures/证据和本报告，未改生产、未读 GT 或已保存预测、未运行 GPU。

## 发现及已确认修复

1. control_measure 未初始化 `j.forward_calls`，实际入口会报 AttributeError；现初始化为0，hook/read 共用同一计数器。
2. 初版只有两个 temporal 臂承担全部额外 witness 成本，no_depth 需要新增帧时却标记0。现记录真实 source-index 首次额外 witness 的时间，按每臂实际 required frame 集合归因；temporal 不重复计入已包含于 tree_seconds 的新 caption 帧解码，再分别加 temporal tree 与 fresh priority 时间。物理 joint acquisition/read 与各臂复用归因分列，不把七次共享成本加总成物理工作。
3. 新 caption 仅检查 token 长度，未绑定文本，且仅 local pool 图像被打开校验。现 mandatory tokenizer.decode(tokens).strip()==text；全部新 caption 代表帧 witness 都打开并核尺寸，即使未被 local packet 选中。
4. 初版报告只有分数变化窗口，缺已声明的 wrong-link 实际干预覆盖。现从验证后的 current packets 逐video/window保存 ancestor IDs/context 实际改变、原/供体祖先、local pixels 保持及 singleton/no leaf/no ancestor/same-chain 原因，另行聚合；不把分数改变当绑定改变。
5. 控制 Slurm 入口新增已完成 R3 summary 的 performance_pass==True 及 eligible84/99 硬门，位于任何控制抽取/reader之前。summary 不进入方法打分或阈值计算。

## 完整路径确认

- temporal 使用已审过的真实连续时间成员、匹配拓扑/成员数、中位真实代表帧与 feature mean；caption 仅精确同 frame 复用，新帧调用真实 Witness/caption。Witness 新解码检查原 absolute time、PTS 和尺寸，写独立control目录；原 witnesses 仅相对链接，main输入不修改。抽取中无CPU placeholder路径。
- fresh_priority 在相同 temporal root observations 上调用原 relevance reader，将新值传播到原后代，只影响固定拓扑上的局部优先级，不重新扩树。保存root observation、三项logprobs和argmax值，重建时核对。flat、wrong_links、no_depth、no_added_pixels 的完整共享 observation 与 local packet 分别符合声明；wrong_links 保留正确全局 inventory，不能声称破坏全部关联。
- native G/own stance/reference V/S只构建和测量一次。七臂每次clone恢复后的native cache，各自追加对应factual turn+固定ack；V/S在本臂扩展cache上独立crop/delta恢复。每臂生产4+W+S；joint为native reference calls加七臂各自`1+W+S`与diagnostics。smoke每臂检查extension fresh、首V full render、首V和首可用S clone exact；空speech无S也无S诊断。
- 每video原子bundle保存全部七臂、native、源segments、extensions/traces/cost；完整bundle通过验证才替换完成文件，最终各臂predictions由验证后的bundle导出。resume/current prepare 重新解析main及control inputs，重建每臂 observation/packets，核token、native会话P/N、trace实际source路径、当前cost和所有计数。输入变更不能仅靠旧版本字段跳过。
- BindingRenderer只缓存同一个当前native msgs/files的CPU编码结果，使用直接列表比较和保存副本；任何会话或源文件列表改变即重新编码。它不缓存模型/分数，不共享七臂GPU推理。
- prepare要求完整expected ID集合和bundle/export一致，并逐video精确复现原R3 native/main raw曲线、窗口、G/stance和prefix。之后调用原canonical evaluator与固定r6；report核native/main六项指标，所有差值和paired bootstrap均为main-minus-arm。单部件门要求同一指标双语料≥.01，完整acquisition门要求两个temporal对照都满足同一指标双语料≥.01。mechanism_supported始终false并要求最终独立机制审查。GT只由postscore评测/报告访问。

## 审稿实际 CPU 证据

全部在 `runs/20261004_m1_tree/r3_controls_code_review/`：

- `reader_oracle.py/log`：实际调用生产read_controls、extension、standard及bundle validator；main/smoke × 空speech/仅中间窗speech，七臂cache独立/native不变、计数及24次token/global/packet/path/count/cost损坏拒绝PASS。context、image/model forward和binding renderer是明确CPU stub，非真实模型parity。
- `extract_oracle.py/log`：runs下显式合成PIL source fixtures，直接调用validate_control，核caption/source/priority/topology/cost；不一致caption text/token、priority、members、constants及缺失caption witness六类拒绝PASS。未写data。
- `analysis_oracle.py/log`：拦截而不执行全部8输出臂的16条subprocess，逐项核canonical evaluator/fixedr6参数；7次相同prefix memo命中，以及当前文字/源路径/返回旧prefix重新编码PASS。
- `launch_gate.log`：仅合成summary测试嵌入式Slurm门；true+84/99放行，false或任一eligible不符均拒绝。未读取真实结果summary、未提交作业。

**真实控制caption/GPU parity/成本和科学效果均未验证。** 主作者通知R3完整主门未过，因此这些R3控制本轮不启动；本审查未复核结果数字，也不把代码PASS当运行授权。未来若改变主reader/revision，对应控制上下文、exact复现和gate路径仍需窄审，不能直接沿用此R3代码结论。
