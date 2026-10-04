# 候选23 executable temporal evidence program：独立 code review

日期：2026-10-05。独立实例：`/root/latents_code`，gpt-6-astra，与实现作者不同实例、相同模型；**same-family provisional**。结论：**PASS（规则6，代码与 CPU 范围）**。本轮未发现需修复的观察有效性阻塞项。

审查范围为 `experiments/20261005_m1_program/` 的 README、program.py、inputs.py、extract.py、measure.py、analyze.py、selfcheck.py 和全部 launch 文件；结合已通过的 proposal、共享原生 `src/stance_cache.py`、`src/mllm_judge.py` 和 `src/video_inputs.py` 检查输入与接口。以仓库 CLAUDE/RESEARCH_ITERATION_RULES 为准，没有重开来源、理论收益或成本否决审查。未修改生产实现，未读取 GT 或预测文件，未运行 GPU、下载模型或计算哈希。

## 观察路径核对

- **程序实际执行。** 原始 ASR Unicode 字符由原 segment、offset 取出；local 按完整字符单元裁剪，保留原始引用及比例时间。frame 使用 native 文件名 nominal 时间点，窗口与 join 均为半开范围。scope 必须接收 local span 及显式 context；action 必须接收 local frame，并由对应原始 frame index 选出实际 JPEG。不存在用另一个代表帧替换所选帧的路径。未知或错误来源不调用模型，也不改写成不存在。
- **模块及最终输入。** fresh factual generation 的系统指令、源包、所选图像、input IDs/grid、输出 IDs 和文本都保存。support/target 绑定该次实际源包。visual 最终使用 action/join，speech 使用 span/scope/context；join 的 visual 记录只传 frame 与 span 引用/时间，不传 transcript 内容。README 允许直接 emit 既有 typed values，不要求每个 raw span 都经过 local；因此没有把该明确 API 行为误判为实现漏裁剪。planner 截断一律 UNKNOWN；事实模块按其 JSON/schema 检查，README 的截断必 UNKNOWN 规则只针对 planner。
- **真实重新测量。** executed record 的 canonical JSON 进入该窗口的新 question，随后接原生 yes/no question；独立 FP32 margin 进入 new visual/new speech，raw max 后沿固定 r6。没有把 planner/module 输出作为数值分类分数，也没有混合多次候选答案。原生 body/全局解释 prefix 保留，未声称信息硬隔离。
- **缓存与调用。** planner/module 每次 fresh cache 且先清 rope_deltas；原生 cache 只构建一次，新旧分支每次恢复到同一 stance cache 长度与 delta。production shared margin 的 finally 同时恢复长度和 delta。生成计数包含 prefix、每个已追加 token 的 forward，包含最后一次追加；重复 smoke 回放单独计 diagnostic。成本分别保存 acquisition、公共 prefix、reference/new local 及 diagnostic，不把已有输入缓存写成新视频免费处理。
- **保存与重用。** readable cache version、常数、model、manifest identity、原始 ASR/nominal frames、窗口 inventory 均核对；从原 planner 文本重解析并以保存 raw 模块输出重新执行所有依赖，逐项核对 trace/result/packet；再以当前 renderer 重建 literal prompt、图像 grid 和 input IDs。paired 输出覆盖/唯一 ID、窗口、branch availability、曲线及 score/record/question/suffix 绑定有校验。续跑重渲染保存的 native conversation；分析 prepare 还从当前原生 inputs 和已保存 stance 重建整个 conversation/长度并精确比较，此检查位于启动脚本的 GT 评测之前。
- **评测与数据边界。** extract/reader 输入仅为原始派生媒体/ASR及当前执行记录；原生已保存预测仅在分析 prepare 中作 exact 复现核对。评分路径不加载 GT。4 fps 索引与 native 固定8秒窗口一致。分析只以 subprocess 调用 canonical evaluator 与既定 r6 参数，within-video 调用 canonical helper，无复制评测逻辑。六项 native metrics 必须精确一致；主门同时要求同一指标两语料 .01 并限制三项退化，mechanism_supported 初始保持 false。

## 独立验证与边界

审稿自写并实际执行 `runs/20261005_m1_program/code_review/independent_checks.py`，完整输出在同目录 `independent_checks.log`，结果 `REVIEW_CPU_PASS`：

1. 1,500 个随机 Unicode span/window 案例，以逐字符 cell 枚举而非复制 ceil/floor 实现作为 oracle。
2. 修改 source frame index 后检查真实 packet、窗口拒绝和实际调用数；第三 perception 被 cap 拒绝；visual/speech typed 输出分离。
3. 调用生产 generate 的 CPU stub 验证旧 delta 清零、每次 cache 独立、FP32 readout、EOS/token cap 与真实 forward 计数。
4. 调用生产 shared margin 的 CPU stub 验证多次独立分支及注入异常后 KV/delta 恢复。

另实际重跑实现者 selfcheck，23 个字符 cell、quotation/context、point join、UNKNOWN、replay、literal prompt 均 PASS；核读固定五视频 `cpu_source_preflight/summary.json` 的实际原生 JPEG/source inventory 与 CPU renderer 证据。这些不等于生成内容的事实正确性或模型数值 parity。

**仍须固定五视频 noGT Slurm smoke 实测真实模型 native exact、两语料非空实际执行、clone exact、输入/调用覆盖与时间。** 当前 PASS 不宣称该 smoke 已完成，不启动 GPU，不替换仍在前序队列的 Tree21/Lattice22，不构成性能、晋级或机制有效性认证。
