# Candidate21 纯文本 relevance rotary 状态修复

2026-10-04。**PASS（same-family provisional）**，仅确认 Slurm103 的 `rope_deltas=None` 崩溃修复，不重开整体审查。

已读本机回传 traceback：错误发生在纯文本 relevance prefix 之后对 None 调用 `.clone()`。修复对 snapshot、每节点开始、finally 恢复统一使用 `None if delta is None else delta.clone()`；每个节点仍裁回同一 prefix cache 长度，digit1/2/3读出及方法常数未改，也未修改共享 Judge。

只读 SSH 核对目标 HF5.15.1：multimodal wrapper 在该纯文本路径保留 position_ids=None，由语言模型根据 `past_key_values.get_seq_length()` 生成 ordinary position IDs；因此保留 None 是正确语义，不需要制造 multimodal delta。目标源码证据为 `runs/20261004_m1_tree/code_review/text_rope_target_source.txt`。

CPU stub 调用实际修复后的 relevance 函数，验证 None/tensor 两种 snapshot、连续节点相同cache起点、模拟正常/异常返回后的 crop与状态恢复；全部通过。证据：`runs/20261004_m1_tree/code_review/text_rope_fix_checks.log`。这是状态管理验证，不是模型数值parity证明。

没有GPU运行、GT读取或生产代码修改。可同步后继续原定固定五视频 smoke；真实模型新图像路径及完整运行结论仍待实际验证。
