# Interval26 parent prefill MLP 内存修复：独立窄确认

2026-10-05，**PASS，未发现本次修复的实质 bug**。复用独立审稿实例 `/root/ideation_jury_2`，与实现作者不同，同模型、same-family provisional。本次只确认 OOM 修复的计算、缓存、调用与来源语义，不重开方案、通用代码审查或性能门。

## 实际范围与判断

阅读 `src/qwen3_mlp_memory.py`、`experiments/20261005_m1_interval_witness/extract.py` 的新增 wrapper 使用及原 source 保存/复用入口，追踪 `src/structured_source_generation.py` 的完整 prefill、grammar stream、cache/position/call 断言，以及实际安装 Qwen3VLTextMLP。阅读保存的 `initial_151_oom.log`，失败定位在 parent generate 的 prefill MLP gate/up 逐元素乘积，日志在第116个视频完成后失败。

Qwen 当前 MLP 为 tokenwise gate/up 投影、SiLU/乘积、down 投影，没有跨 token 归一或混合。wrapper 沿倒数第二维切分行、调用原 bound forward、按原顺序拼接；weights、完整 attention、输入 token、位置和 KV 协议不改。长度≤4096直接调用原 forward。只包裹 parent generate，leaf/repair 不进入该 context；decode 的单 token 直接走原 forward。context 的 finally 恢复每层 forward，包括异常退出。

模型级 forward hook 仍只见原来的完整模型调用，MLP 内部分块不会计为新增模型/vision 调用。完整 parent source catalog、prompt、writer、grammar、限制、source compile/validation、atomic metadata 保存入口保持原样；未添加标签路径或复用部分失败 metadata 的路径。生成时间在原 generate 内计量，覆盖分块工作。

## 独立 CPU 实测

证据：`runs/20261005_m1_interval_witness/prefill_mlp_fix/independent/{check.py,check.log,summary.json}`。独立创建当前实际 Transformers 随机缩小 Qwen3VL：36层、32 query/8 KV头、head_dim128，FP32/BF16；无预训练权重、无 CUDA。文本 parent 场景17 token prefill，强制 chunk=7，随后4步贪心 decode；另在全部36层 MLP 上逐层测试实际默认边界4096和4097行。

| 检查 | FP32 | BF16 |
|---|---:|---:|
| 强制分块完整模型 hidden 最大绝对差 | 1.3113021850585938e-6 | 0 |
| 全层 KV 最大绝对差 | 2.562999725341797e-6 | 0 |
| 每条路径模型调用 | 5 | 5 |
| 4步贪心 token 一致 | 是 | 是 |
| 4096行全部36个MLP输出差 | 0 | 0 |
| 4097行全部36个MLP最大输出差 | 6.51925802230835e-9 | 0 |

attention 各次完整输入形状一致，所有参数逐值未变，normal/exception 两种退出都恢复原 forward。文本模型原生 rope_deltas 为 None，两条路径一致；本次没有多图片输入，正好对应 parent generate 的空 image paths。初次审稿 fixture 错把这个合法 None 当 tensor clone，修正仅限审稿脚本，失败日志保留为 `initial_fixture_none_rope.log`。

预置检查容差：完整模型 hidden/KV FP32 1e-4、BF16 .05；逐层 MLP FP32 1e-4、BF16 .003。实际差异如表，未用“算法等价”掩盖 FP32 舍入变化。4096行路径另要求逐值相同。

## 限制

没有读取 GT、预测文件、metrics 或预训练权重，没有访问 GPU，没有修改作者代码或共享文档。CPU 结果确认本次计算语义及实际边界，但不保证实际8B GPU分块与未分块的每一个浮点/token逐位相同；矩阵批形改变可有正常舍入差异。尚未验证原失败长 parent 的实际显存峰值或 OOM 已消失；运行时结果需由后续 Slurm 给出。这不新增性能门，也不重新评价科研方法。

本报告由主 agent 落位 `docs/reviews/20261005_m1_interval_mlp_memory_fix.md`。
