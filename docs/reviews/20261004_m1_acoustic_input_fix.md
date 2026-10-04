# Candidate20 smoke 输入路径修复：独立窄范围确认

2026-10-04。**PASS（same-family provisional）**。仅检查 Slurm96 在 Qwen 读取前遇到 HateClipSeg manifest 旧媒体路径后的输入修复；不重开方法、常数或核心实现审查。

`extract.resolve_video` 先使用仍存在的 manifest 路径，否则按同一 dataset/video_id 在运行机 `~/data/<dataset>/video/`、`videos/` 查找；没有跨数据集路由或改写媒体源。记录将 manifest 字面路径与实际打开的 input_video 分开保存。实际媒体仍由原 ffmpeg 读取，提取失败仍报错。

`validate_cache` 检查原 manifest 路径、实际 input_video basename 的 video_id，以及既有 dataset、版本、ASR词/时间和支持验证。它不要求本机重新解析出与生成主机相同的媒体目录，因此 lab2 的 `videos/` 缓存回传后可在本机 `video/` 环境验证。旧的两条缓存没有 manifest_video_path 时，只有 input_video 本身等于原 manifest 字面路径才兼容；没有放宽成任意旧路径均可复用。

独立 CPU fixture 证据：`runs/20261004_m1_acoustic/code_review/input_fix_checks.log`。实际调用修复函数验证 manifest 优先、失效路径回退到 videos、缺输入拒绝、生成主机路径不同的回读、旧格式精确兼容；错误 ID、错误 manifest、旧格式路径不一致均被拒绝。fixture 全部位于 code_review 内，未修改生产文件、未解码真实媒体、未读 GT、未运行 GPU或计算哈希。

本修复不改变字符对齐、Gibbs DP、attention prior、常数或评分。可提交并同步后重跑原定同五视频 smoke；已完成兼容缓存可按原验证复用。本结论不声称 smoke 已通过，真实解码与后续 Qwen 检查仍由该次重跑确认。
