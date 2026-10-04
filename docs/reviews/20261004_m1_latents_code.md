# Candidate19 M1 Latents 独立代码审查

日期：2026-10-04。独立实例；same-family provisional。最终结论：**PASS，可进入五视频 Slurm smoke**。两项已定位问题经修复确认，无剩余代码 blocker；不作性能判断。审查对象是 README 声明的论文公式适配，不要求改成官方 contextual Stage I。

## 范围与证据

阅读 CLAUDE.md、RESEARCH_ITERATION_RULES.md、独立 proposal review、实验 README、latents.py、measure.py、test_latents.py、analyze.py、launch 脚本、src/mllm_judge.py、统一 within 接口与既定 r6 的参数/评测调用，以及本机 HateVideo 环境的 Qwen3-VL 和 SDPA 接口源码。另检查既有 base_gridA 首条 prediction 的字段和 r6_bma/config.json 的既定配置，仅用于接口一致性。未读取 GT，未运行 GPU、下载模型或计算哈希，未改生产代码。

独立 CPU 检查脚本及输出位于 `runs/20261004_m1_latents/code_review/check_cpu.py`、`cpu_checks.log`。通过：bool mask/causal 与 GQA/普通头四组合的 full-key attention 对照、原 SDPA 输出逐元素相同、相关度行与图像列选择、全相等相关度的正负组互斥与稳定顺序、仅 detached 四向量的 Adam 更新、NES 实际评价状态选择与较早平局状态保留、全词表 top20 内归一化熵进展计算。

作者 CPU 测试的首次独立执行记录为 `author_cpu_checks.log`：attention、优化及 zero-delta 缓存对照通过；nonzero-delta 用例在断言之前遇到 expanded tensor 原地修改错误，需修复并复跑。

修复后的生产测试脚本已独立执行，`author_cpu_checks_fixed.log` 全部通过，包括真实 tiny Qwen3-VL 的 cached/fresh/full 对照、zero/nonzero rotary delta、槽干预改变 margin。`resume_checks.log` 另验证完整 fixture 可通过，而 duplicate/unexpected IDs 与配对 global/stance 不一致会被拒绝。

## 已定位问题

1. **Resume 可能混用不一致结果。** 首次审查的 measure.py 先覆盖 config，再以 ID 集合相等判断已有配对有效；无法拒绝旧配置、重复记录、缺少 details 或同 ID 的配对元数据不一致。需要在覆盖前比较语义配置，并验证唯一且属于当前名单的完整配对记录；遇到部分写入直接拒绝即可。
2. **CPU 非零 rotary delta 用例不能运行。** test_latents.py 对 expand 得到的 full_positions 切片直接 `-=3`，多个元素共享内存。应 clone 后再修改，重新执行原用例。

**修复确认：** measure.py 现在在写配置前核对已有语义配置（仅日期可不同），每个 arm 配置一致，existing_records 拒绝重复或名单外 ID；两份预测与 checks 必须覆盖相同集合，details 必须存在，validate_pair 核对全局、stance、speech、窗口、最终 trace margin、帧曲线形状与 merger shape。缺记录/不匹配直接失败，不静默混用。test_latents.py 已 clone 后修改位置，完整测试通过。以上仅确认原问题修复，未重开泛化审查。

## 核心机制核对

SDPA hook 使用实际调用的 post-RoPE Q/K、真实 mask、scale 与 GQA 参数，softmax 分母保留全部 keys，再选择图像列；返回值仍由原 kernel 计算。question_rows 由原问题字符区间与 tokenizer offset 定位，并排除模板 special token。主 merger 输出与 Qwen get_image_features 顺序一致，image token 总数与分帧 patch 数有断言。

Stage I 的支持与前缀 detach，唯一 optimizer 参数是四个 FP32 输入向量。Stage II 只用单状态 reward，不使用标签、Yes/No reward 或预测平均；17 次后缀调用包括 1 初始、15 扰动、1 选中状态重算。best 状态保存实际被评估的候选，严格更优更新。最终 margin 通过共享 margins_fp32。

LatentReader 明确传入缓存长度加 rotary delta 的三轴位置；每次调用恢复长度与 delta。视觉分支结束后恢复全局 stance 缓存，speech 走原路径；缺语音保持 None。全局分支发生在状态干预之前。smoke 加做原生回放、复制缓存对照和槽顺序干预。原生 Judge._step 的既有位置行为没有被本候选改写。

部署调用数与诊断调用数分开；attention 获取计入时间，smoke 的两次额外 latent 评价与一次原生回放计入诊断。配对 base 时间也含 attention 获取，因此是带测量开销的配对原生时间，config 已注明。

launch 采用目标 local 分区、1 GPU、4 CPU、32G；模型离线加载，运行与日志路径在仓库内。最终补读 analyze.py 与 run_analysis.sh：prepare 检查预定 5/333 覆盖、原生逐窗/曲线完全一致、真实调用数和状态轨迹；smoke 禁止进入 evaluate/report，不读 GT。完整阶段 subprocess 调用唯一 evaluate_four_datasets 和既定 twolevel_r2 的固定 r6 参数，不互相 import 实验实现；per-video 分析调用 src.eval.evaluate.within_video_macro，不复制 ROC 算法。r6 native 六指标要求与原权威输出逐项精确相同，主门按两个语料同一指标 +.01 且其它指标不越噪声下限计算。机制支持没有被性能门自动宣告，仍需既定消融。

尚未有真实 8B GPU smoke 证据。CPU 小模型/张量检查不替代五视频 smoke 的真实 merger/输入对齐、资源开销和完整 native 六指标验证。
