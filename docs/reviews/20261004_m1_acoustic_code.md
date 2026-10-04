# Candidate20 acoustic：独立代码审查

2026-10-04。**PASS（same-family provisional，独立实例）**。按 RESEARCH_ITERATION_RULES 第6条检查观察有效性；下述三个具体问题已修复并确认，无剩余代码 blocker，可进入五视频 Slurm smoke。该结论不是性能、机制成立或晋级结论，不代替真实 8B/Whisper GPU 验证。

## 范围和实际验证

完整读取实验 README、独立 proposal review、alignment.py、extract.py、reader.py、measure.py、analyze.py、test_acoustic.py 及全部 launch 脚本；遵循已读 CLAUDE/RESEARCH_ITERATION_RULES。核对共享 ASR/transcript 接口、本机安装的 HF Whisper/Qwen/SDPA 接口。未运行 GPU、未下载模型、未读取 GT、未执行完整评测或修改生产文件。请求原文及证据在 `runs/20261004_m1_acoustic/code_review/`。

- `author_cpu_tests_fixed.log`：独立执行修复后的四项作者测试，全部通过。K/T=1..3 的全部九种格点尺寸将 forward/backward occupancy、logZ 与 Viterbi 对照显式路径枚举；另检查40×300数值有限。实际 tiny Whisper 验证 hook 前后原输出逐元素相同；实际 tiny Qwen 验证 neutral prior 等于原生、非平凡 prior 改变 margin、复制缓存一致，以及 hook prior 与直接传入 additive causal mask 一致。
- `scaling_edge_checks.log`：另用真实 tiny Whisper 各 cross-attention 模块的实际输入，独立通过 q_proj/k_proj 构造期望矩阵；验证仅一次 head_dim 缩放、字符自身输入行、valid audio 帧裁剪、median3、top10 顺序及先平均再列归一化。非空 padding 帧存在于模型输入，比较使用较短 valid 帧，避免只测无 padding 情况。另实测零词与80秒纯标点词的无模型 fallback、归一化和空支持序列化恢复。
- `token_mapping_checks.log`：实际离线 Qwen tokenizer/processor、内存图片及自造重复词前缀，验证 expanded-image token offsets 与编码完全相同；prefix/body 中相同词面保留不同 word ID 和不同支持；未绑定 scaffold 支持为1。初始测试错误要求 BPE token 完全落在词内，已修正为方案声明的字符重叠规则；没有因此改变实现。

读取父 agent 的无 GT 全333 block planner 结果 `runs/20261004_m1_acoustic/cpu_checks/block_plan.json`：HateMM215视频/3472块/333303字符token，HateClipSeg118视频/2767块/273961字符token，实际数据无 repeated_single_character。这里只核对预处理覆盖和资源规则，没有自行访问 GT 或据此评价性能。

## 已修复问题

1. 超22秒且标点移除后为空的词原会进入空字符递归并除零。build_blocks 现先排除无有效字符项，后续 inheritance 明确使用同segment donor 或 proportional fallback。独立空字符实测通过。
2. 零词视频 JSON support 为 `[]`，与 arm_support 的 `(0,V)` 数组直接比较原会形状失败。prepare 现按 p.shape 恢复后比较，独立序列化回读用例通过。
3. 派生缓存原未核验可读 source version，已完成视频续跑原跳过当前缓存/ASR核验。现 CACHE_VERSION 明确写入并检查 cache、extract/measure config；验证原视频路径、原segment/word/time、算法、模型和支持形状。续跑在跳过已完成 ID 前，重新验证当前 ASR/cache，并核对 details 中 words 和当前 arm support 精确一致。后续改变对齐来源必须递增此可读版本；没有引入哈希或 Git 内容标识。

## 观察路径核对

DP 的 alpha 包含当前 cell，beta 从终点0开始只累加下一 cell，occupancy 没有双计。每次转移均为-log3，起点不加转移；Viterbi 使用同能量/转移势。字符多token先平均、跨块同词按字符贡献累加，单字符极长跨度的重复项按1/parts计权，最后按词归一化。删除ASCII标点不删除原词ID，空标点词继承有明示记录。20ms帧中心按原音频裁剪起点加回，全窗覆盖，越界尾帧归最后窗不丢质量。

安装版 HF Whisper 的 q_proj 已乘 head_dim^-0.5，而 SDPA 调用 scale=1.0；capture 遵循实际传入 scale，没有重复缩放。选择字符自身输入行，排除 special rows；只对有效音频帧做softmax，实际head评分/排序与README定义一致。原模型输出始终由原SDPA返回。论文/官方与本轮行范围、列归一化及Gibbs扩展的差异已明确声明。

Qwen prior 是每个语言 self_attn 的 additive attention_mask，即 softmax 前的 log(max(p,1e-6))，不是输出乘权。构造完整 cached-prefix+query keys 和逐行 causal 上三角屏蔽，所有 branch query rows均受同一支持约束。hook visited 要求经过全部语言层；分析实际8B要求36层。未绑定global问答、policy、视觉和scaffold保持logprior=0。原生global/visual在prior关闭时产生，所有变更仅发生于新speech分支。每次分支恢复cache长度和rope delta，关闭active prior。

MAP正文按原word顺序和ID构造，重复词不去重；无MAP但非零支持的窗仍进行一次读取，完全零质量保持None。`unweighted` 保持新的声学MAP正文，不能把它误称原生speech；neutral parity单独使用原native正文检查。最终只有一个新speech margin与原生visual取max，global/r6不变，无候选预测平均。

extract/measure计算路径不加载任何GT。分析的prepare也不读GT；完整evaluate/report才调用唯一canonical evaluator和固定twolevel_r2参数，未复制指标算法或修改共享评测器。raw speech配对仅在两arm都有读数的共同帧上计算，并单列eligible数量；native六指标要求精确一致，主门三指标并列且同一指标双语料+.01。机制标记仍False，完整控制只在主门之后。

Slurm脚本使用目标local分区、1GPU/4CPU/32G；先抽取Whisper后退出进程，再加载Qwen，所有输出在仓库内。新视频音频decode/align时间和encoder/decoder调用与Qwen测量一起记入部署成本，原生配对speech/额外诊断分开。真实峰值、完整333时间、原生六指标一致性及全部真实模型层hook行为仍由下一步smoke/full验证，不以CPU通过替代。
