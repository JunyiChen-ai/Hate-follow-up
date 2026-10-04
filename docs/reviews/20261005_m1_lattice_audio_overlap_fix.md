# Candidate22 音频PTS重叠处理：独立窄范围确认

2026-10-05。**PASS（same-family provisional）**。仅确认实际媒体预检暴露的重采样块重叠修复，不重开整体代码审查或评价方法。

`audio.place_block` 使用每块原PTS对应的 start_sample，仅在 `[0,total_samples)` 范围内填充尚未observed的样本。已占位置保留解码顺序中的首次观测，后来的重叠部分不平均、不覆盖、不顺移。负起点、倒退块、越界和空块均保留原坐标裁剪；没有通过拼接压缩时间。每块保存实际PTS/timebase/start/count及 placed_intervals、overlap_discarded_intervals，原gap仍显式保留。

策略已写入 CONSTANTS、可读 CACHE_VERSION、timeline与README，缓存验证要求相同 overlap_policy；新旧输入规则不会静默混用。此次未改变beam5、词槽构建、结构attention、reader或评测路径，也未产生模型分数。

独立证据 `runs/20261004_m1_lattice/code_review/audio_overlap_fix_checks.log`：

- 复跑手算重叠/倒退/裁剪用例，通过。
- 以独立逐样本循环作为oracle测试100个随机块，包含负起点、倒退、越界、零长和重叠；最终样本值、observed mask和逐块放置/丢弃区间全部一致。
- 解析已有真实五视频预检记录，逐块重建observed并检查当时overlap恰等于既有覆盖、placed恰等于尚未覆盖部分。五例全部通过；hate_video_114重建observed为9,878,398/10,060,448样本，累计重叠丢弃182,741样本，其余四例重叠为0。来源为 `cpu_audio_preflight_fixed/summary.json`，没有重新解码媒体或读取GT。

未运行GPU、未修改生产代码或计算哈希。此PASS确认确定性输入规则与记录正确，不声称恢复了损坏PTS的真实原始波形，也不代替未来Whisper/Qwen smoke。候选22仍受Tree结果分流约束。
