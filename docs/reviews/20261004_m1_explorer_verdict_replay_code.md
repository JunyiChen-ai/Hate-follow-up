# R4 verdict_replay 独立窄代码审查

日期：2026-10-04。审查实例：`/root/explorer_review`；主机：`sc474397`。结论：**PASS，无生产代码阻塞项。** GPU 控制仍须等 R4 完整主门通过，再做目标环境固定五视频 smoke；本结论不表示控制已实际运行或上下文机制成立。

范围为 `experiments/20261003_m1_explorer/{controls,analyze_controls}.py` 的新增 verdict_replay 分支，以及随后补入的 `launch/run_control_analysis.sh`。没有修改生产代码、启动 GPU、读取真实 R4 性能/预测/GT 或计算内容哈希。新脚本、合成输入和证据全部位于 `runs/20261003_m1_explorer/verdict_replay_review/`。

## 实际 CPU 模型与固定轨迹

`check_model.py` / `check_model.json`、`model.log`：重新执行真实 CPU Qwen3VLModel，36层、32/8 heads、head_dim128、hidden_size64、DeepStack `[0,1]`，FP32/BF16 × 20/18 原始帧共四组。运行库为 **torch2.7.1+cu128 / transformers4.57.6**；复用旧 harness 的架构与合成媒体/tokenizer 输入定义，但没有复用旧结果代替执行。

本轮实际运行生产 R4 reader 生成 trace，再运行生产 `read_control(..., arm='verdict_replay')` 与 `read_images`。构造三个窗口、仅首窗有 speech，最后一窗为明确的合成零轮，用于测试 native fallback。对前两窗，逐轮 added、累计 acquired、suffix_ids、image_grid_thw、image_counts 与 main 完全一致；共4次获取读取，部署11次 outer forwards、smoke14次，计数符合 `3+B+R` 和额外 V 次 replay。

逐次断言 native cache 的完整 Q/answer history 与实际 native stance 一致，读取前后全层 KV exact；没有 observational copy、重新选帧或 uniform 调用（测试中将这些入口替换为一旦调用便失败的函数）。global、speech及缺speech状态、每窗 native replay exact。零轮 visual 返回 native 值。另在仍执行实际模型的读数之后注入 +8 的确定高置信 margin，第二轮仍按 trace 完成，证明不会因 replay 自身低熵提前退出。

顺带执行四个原控制臂的 R4 路由，确认仍是 observation-only、仍创建其独立 observational cache、native replay exact。本次仅查新增分支兼容性，不重审原选择算法或 clone 机制。

## 旧 CPU 库接口差异及额外 oracle

额外 full-native fresh-forward 对比最初暴露的是测试运行库差异：transformers4.57.6 的 Qwen3VL.forward 在已有 cache 且未提供 cache_position 时，其文本 position 分支使用从0开始的位置；共享 Judge._step 不显式传该参数。因此本机旧库上的 global Q/answer KV 不能直接当成 fresh 完整序列 oracle。这不是本次 verdict_replay 新增的输入或缓存修改。

只在派生 CPU Judge._step 中补上 `cache_position=arange(past_length,past_length+len(ids))`，生产 Judge 未改。之后实际新控制路径继续调用生产 read_control/read_images。完整 native prefix 和 expanded suffix 的位置与 fresh oracle 逐元素相同；fresh/cached margin 的最大绝对差为 FP32 两组各 `1.1921e-7`，BF16 20/18帧分别 `.0032981` / `.0041667`，通过事先固定的 `2e-5` / `.06` 容差。这里没有声称 BF16 hidden exact，也没有以旧库额外 oracle 重新否决已通过的目标 native 路径。

此接口适配及小模型/合成 processor 范围不能冒充目标 transformers5.15.1、实际8B权重或真实 processor 验证。主代理已另行核对目标库使用 past_length 推进增量文本位置；本轮目标 smoke 仍应按原计划验证真实固定trace replay和native parity。

## 完整合成333与报告

`check_analysis.py` / `check_analysis.json`、`analysis.log`：使用本轮模型产物构建215/118共333条合成记录，并在隔离 synthetic_root 内生成合成 GT 和指标。实际运行 verdict_replay 及原四臂 prepare/report，全部通过。

新增 raw visual 和 raw max 配对 within 以 canonical within 函数独立核对；每语料配对数、均值以及 **R4 main − verdict_replay** 方向正确。visual 按4fps中心映射窗口，raw-max直接用相同范围曲线；报告明确处理是完整 Q/answer、位置与对话结构，final还含独立 corpus r6拟合，不声称孤立 answer-token 或任意策略效果。

破坏 added、acquired、suffix IDs、image grids/counts、prefix长度、context或轮数共8例均被 prepare 拒绝；对 R1/R2/R3 的 prepare/evaluate/report 共9例均拒绝，总17类。完整native长度强制等于 R4 observational P 加被移除的对话长度，且不存在 expanded_prefix clone记录。捕获五臂共10条评测命令，见 `captured_commands.json`：全部调用共享评测器及原 r6，`noleak/nscore/calib/bma/length/min-windows=2/bma-grid=6/m2` 参数不变；命令未执行真实评测。

`check_entry.py` / `check_entry.json`：生产 controls.main AST 配显式模型/reader替身，验证正式及smoke入口均将 verdict_replay 路由到 `expanded_without_verdict=False`；config是native_verdict，读取R4对应trace，写入独立 `r4_controls[_smoke]/verdict_replay`，details.main_trace正确。非法 reader版本R1/R2/R3在模型和产物访问前被拒绝。

## CPU 分析 launcher

`check_launcher.py` / `check_launcher.json`：`bash -n` 通过。在本轮独立镜像目录中执行脚本，仅将固定 Python 路径换成记录参数的测试替身；其余shell逻辑原样执行。原四臂×四版本及verdict_replay/r4共17种合法组合均按 prepare→evaluate→report串行转发正确参数，log/PID保存在对应仓库内控制目录、首行为host。5种非法arm/version组合退出码2且未调用解释器。故意令evaluate退出7时，report没有执行，退出码保留且日志写入 `CONTROL_ANALYSIS_FAILED`。

**最终 PASS。** 新分支固定R4的完整轨迹，仅恢复native完整对话缓存；没有re-gate、另选answer、读取R3分数或标签路径。上下文对照不能补救获取novelty，实际控制提交仍以完整R4主门和目标smoke为前提。
