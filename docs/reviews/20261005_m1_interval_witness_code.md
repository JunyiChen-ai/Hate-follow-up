# Candidate26 interval witness：独立初版代码审查

2026-10-05；独立实例 `/root/latents_code`，gpt-6-astra，同模型不同作者实例，**same-family provisional**。依据规则6完成一次完整 observation-only review；不重开已通过的proposal/spec审查。**最终 PASS。** 两项实际科学输入问题及一项实际运行缓存问题已修复并窄复测，无遗留代码阻断。科学8B GPU、完整来源容量与性能仍未验证。

范围：本实验 interface/inputs/extract/reader/measure/analyze/selfcheck、spec.json、README和launch；必要核对共享实际帧、结构生成、native Judge/stance cache/renderer。未改生产或评测器、未读GT/真实预测/主实验summary、未GPU、未计算哈希、未新派agent。证据全部在 `runs/20261005_m1_interval_witness/code_review/`。真实来源可读；合成观察只在内存，真实随机小模型结果也仅为CPU fixture，不落盘假数据集预测/cache。

## 实际问题与修复

1. **重叠字符串唯一性错误。** catalog/resolve_handle原用str.count，它不计重叠。`a a a`中的`a a`有两个出现位置却count=1，违反唯一原文跨度约束。作者改为当前位置必须是首次find，且start+1后无第二次find。独立catalog和resolve反例均拒绝；实际source/choice重放重新通过。
2. **内部路径泄漏到模型文本。** source_table完整frame字典、repair literal sources及reader source-path正文原含路径，例如HateMM的`hate_video_...`/`non_hate_video_...`，向模型提供了带类别语义的内部文件名。作者在模型可见序列化处递归移除path；真实catalog、repair-plan审计字段、trace.source_path及image_paths保留原路径，用于实际加载/来源验证。独立检查leaf/parent/repair/local模型文本全部不含该路径；真实五视频再次检查global头和V/S文本不含真实图像路径，图像顺序及审计来源仍精确。该修复尚在科学GPU前，不对旧输出追溯改写。
3. **指定离线HF_HOME不可加载模型输入组件。** 新环境版本本身正确，但项目HF_HOME下初始Qwen目录只有processor部分文件，真实CPU AutoProcessor无法解析tokenizer/config；之前静态基础设施PASS不等于缓存完整。作者将项目路径连接到已有完整本地缓存，保留原目录。独立使用与Slurm相同的显式项目HF_HOME、隔离HateVLM解释器完成真实processor、来源输入和tiny模型测试，现通过。首次失败栈仍保留tiny_reader.log；没有修改Judge或下载安装不同权重来改变基线。

## 实际执行的独立证据

**production acquisition与确定性逻辑**：`acquisition.py/.log`独立编码32秒64×64真实CPU视频，走实际PTS选帧/PNG、实际processor、production extract.acquire及inputs.validate。仅生成的事实选择采用明确脚本fixture，仍经过真实Stream合法grammar replay和strict compile。执行4leaf、完整3parent postorder、2repair；6次vision source prefill，所有forced/choice tokens计数绑定。构造深层与根重复目标，确认deepest优先、每叶至多一次、冻结一轮plan；repair不含旧leaf标记/child table/parent status，保留规定上下文。修后完整重组7节点、冲突resolved；原记录仍保存。144个独立ANY/coverage-ALL/UNKNOWN组合oracle通过；8项source body/spec/choice/repair context/target/final status/count/position腐坏拒绝。

**真实模型代码路径**：`tiny_reader.py`最终证据为`tiny_reader_final.log`，在隔离Torch2.11.0+cu128/HF5.15.1下用CPU随机初始化BF16 Qwen3VL（36文本层、1真实vision层）实际执行共享native build/margin、production build_tree的新G与其自身hard stance、独立full multimodal V/S、fresh repeat及validate_bundle。不是任意stub logits：实际14 forwards、7vision forwards，重复精确；tree/text/token/image/role/current-tree/cost/encoder count八种腐坏拒绝。来源metadata validator在此小模型fixture隔离，来源本身由独立acquisition测试覆盖。

同一个真实CPU多模态模型进一步实际调用共享structured generation与本实验write_record/compile_record；最终完整记录105输出tokens、106实际forwards、1vision prefill，实际图像mRoPE delta−56。cap12实际13forwards、整体UNKNOWN，选择/forced events和位置重放通过。这是小模型可执行性与数值路径证明，不是实际8B模型事实判断或GPU数值等价证明。

**真实五视频输入与路径屏蔽**：`actual_inputs.py/.log`在修复后的显式项目HF_HOME/HF5.15.1下，独立重验证全部源PTS与所选PNG，重渲染实际fixed5 leaf输入和原生20/fullASR+新树/ownstance+V/S消息。最终leaf expanded IDs长度3202/3157/3457/3483/3180，各22图像；与移除内部路径前的作者尺寸不同是预期输入修复。测试树是明确内存fixture，含真实frame witness，以确保source-path文本确实执行。新branch前缀IDs、global overview顺序、追加local两图及原问题一致；源审计路径仍存在，global与局部模型正文没有内部文件路径。`path_visibility.py/.log`另外明确核对leaf/parent/repair/reader可见文本为false、审计metadata路径保留。

**完整树及评测编排**：`wrappers.py/.log`独立验证1至149个叶子的完整2W−1节点、左右floor midpoint、chronological postorder和深度。拦截4个subprocess而不执行GT：两个stream分别调用唯一canonical evaluator及既有fixed-r6，HateMM/HateClipSeg、noleak/nscore/length prior/grid6/m2保持。report导入canonical within helper，完整333、native六指标exact、84/99数量与同指标双语料门保留；未复制或更改评测器。七个Python源码编译及scientific shell语法通过。

## 方法接线与成本结论

所有原始leaf/parent/repair来源、完整spec/常数、模型、ASR/window、实际源像素、prompt/tokens/grids/choice/position/编译记录/树/plan在复用前重建绑定。一个parent调用实际看到两个已完成child与所有合法descendant sources，只有坐标而无descendant pixels；不会被描述为父节点视觉验证。present需真实owned witness，absence缺任一预期模态转UNKNOWN；最终父状态由ANY/coverage-ALL重组，不把证书类别直接当数值margin。修复上下文保持解释角色，旧叶答案不直接作为输入字段传回。

新global确实将完整final tree加到当前native20/fullASR并重新测量G，ownstance来自新G；不复用native G作新G。每个V/可用S均是同新会话的fresh full multimodal forward，局部像素是真实本窗口；remote witness保持owner/context身份。数值为原FP32 Yes/No head，max和4fps/r6不变。原生参考同视频fresh重算，prepare必须与权威旧native allraw精确，代码没有在方法中读取旧预测或GT。

paired actual counts为base.calls+new.calls+diagnostics，两套global setup分别计费；vision为两个prefix加实际新分支/diagnostics。新standalone含完整source acquisition+new prefix+new V/S；不混入paired native分支成本或把来源缓存当免费。逐branch/native/repeat时间、计数、finite/peak和来源成本绑定沿用并实际通过CPU检查。零repair允许并诚实汇总，不强迫smoke产生语义冲突；完整实验有无修复是后续机制事实。

## 尚未实测的边界

候选26科学8B GPU尚未运行。修复后的原生固定五视频运行时精确复现、科学fixed5 native allraw/fresh repeat及完整333六指标仍须实际作业完成，不能由本CPU PASS代替。当前完整7359窗span预检不是全部生成parent输入、新global树或每次full branch的实际最大token/显存证明；源缓存与全333实际容量仍pending，未编造GPU测量。预计耗时、身份/描述事实正确性、修复收益和论文贡献不是本次代码PASS的结论。
