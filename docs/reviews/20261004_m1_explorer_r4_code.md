# Explorer R4 独立窄代码审查

日期：2026-10-04。审查实例：`/root/explorer_review`；主机：`sc474397`。结论：**PASS，无生产代码阻塞项；可进入目标环境固定五视频 smoke。** 本结论不替代目标 8B 模型/运行库验证，也不表示 R4 性能或机制成立。

范围：`experiments/20261003_m1_explorer/{measure,controls,analyze,analyze_controls,case_analysis}.py` 和 `launch/run_analysis.sh` 的本轮 R4 diff。未修改生产代码、未运行 GPU、未读取真实 R4 预测或 GT、未执行真实评测、未计算内容哈希。全部测试及合成证据在 `runs/20261003_m1_explorer/r4_review/`。此前 proposal-delta 审阅由本实例完成，但本实例未参与生产实现。

## 实际模型验证

入口：`check_model.py`；结果：`check_model.json`；日志：`model.log`。使用本机 `/home/jehc223/miniconda3/envs/HateVideo/bin/python`，实际版本 **torch 2.7.1+cu128 / transformers 4.57.6**。运行 CPU Qwen3VLModel，36 语言层、32 query heads / 8 KV heads、head_dim 128、mRoPE `[24,20,20]`，视觉 DeepStack `[0,1]`，残差 hidden_size 64。FP32/BF16 × 20/18 原始帧共四组均实际 forward，包含视觉编码、语言层、attention capture 和共享 12-token FP32 margin head；不是仅 import 检查。

媒体和 tokenizer/processor 输入使用明确的合成替身，GPU timer 替换为 CPU 可用实现。原始 fixture 来自此前独立 harness 定义，但本轮重新执行生产函数，新增证据全部写本轮目录。没有复用前次 PASS 结果冒充本次执行。

- 实际执行 `observational_cache`：从完整 native verdict cache deepcopy 后 crop 到 P。逐层确认 local/native keys 和 values 不共用数据地址；主动修改 local 一个元素时 native 全层 snapshot 不变，随后恢复 local。长度分别保持 P 和完整 native 长度。
- 实际执行 `observational_native`：prefix forward 返回的 rope_deltas 与重新调用 get_rope_index 得到的 delta 相等；ids、grids、positions 使用原 observational prefix，history 为空，不保留 global Q/answer。prefix 位置与完整 native 的前 P 个位置一致。
- 对累计 1、2、3、4、再 2 张新图调用生产 `read_images`，检查独立 local cache 恢复 exact，native 全层 KV 始终 exact。缓存后缀位置与 fresh observational-prefix 完整输入的相应位置逐元素相等。实际 fresh Qwen forward 的末 token hidden 与 cached forward 比较通过：FP32 最大绝对差为 20帧 `1.1921e-6`、18帧 `1.3113e-6`；BF16 分别为 `.03125`、`.0234375`。使用事先固定 `atol=rtol=2e-5` / `.06`，因此这里是数值容差一致，**不是 BF16 hidden 精确相等**。
- 每次 expanded read 后重设原 native rope 再做 native 问题，margin 精确恢复；首次每种累计图数触发生产 fresh-render seam 分支，重复 2 图不再重复验证，检查集合为 `[1,2,3,4]`。seam 分支在本轮使用合成 processor，真实目标 tokenizer/processor 的 seam 仍由目标 smoke 验证。
- 实际运行完整 R3 和 R4 `read_video`，三个窗口，其中只有首窗有 speech；R4 native global、stance、全部 native 曲线、speech 缺失状态、initial visual/prior 和首轮 selected entries 与 R3 相同。原生每窗 replay exact，新增读取进入最终 visual/max/4 fps 曲线，forward 数满足部署 `3+B+R`、smoke 额外 V 次。
- 同一四组真实小模型上实际运行 uniform、distance、fixed4、mismatch 的 R4 `read_control`。四臂均使用 observational cache，native global/speech 与 main 保持相同，smoke replay exact；控制部署/smoke forward 数和 fresh-render `[2,4]` 记录一致。没有重新审判原有选择算法的科研价值。

CPU 4.57.6 验证不能冒充目标 torch 2.11 / transformers 5.15 或 8B 权重验证。本机真实 AutoProcessor 的额外离线加载尝试因 tokenizer `vocab_file=None` 失败，因此未把真实 tokenizer 加载或像素处理通过写作本轮证据；这不是生产评分失败。目标 smoke 必须使用其真实 processor、完整模型及生产图片完成 seam 和 native parity 检查。

## 决策和边界

`check_decisions.py` / `check_decisions.json`：显式分数替身隔离调度，共 32 组，包括有/无 nominal support、initial margin -8/0/+8、0/1/3/4 候选、首轮高低熵，以及 .3-nat 继续阈值两侧；每组另执行四臂控制。全部通过。

R4 仍强制有合法候选时首轮获取，之后按自身 expanded margin 的熵决定第二轮；替身指定相同读数时其轨迹与 R3 相同。无候选明确 exhaustion，返回 native visual，不创建 local cache，copy seconds 为零。候选不足时 1、2+1 不补假帧，四臂按原规则处理。没有 speech 时不新增 speech 分支；18原始帧已在真实模型测试覆盖。

## 完整合成语料与路径

`check_analysis.py` / `check_analysis.json`、`analysis.log`：由本轮真实小模型读数构建完整 333 条合成记录（215/118，含18原始帧示例），并建立本轮隔离 synthetic_root 下的合成 GT、配置和指标。实际执行 R4 main prepare/report、四臂 control prepare/report、case_analysis 和固定五视频 smoke prepare，全部通过。GT 只在这个隔离的合成报告阶段读取，没有访问真实 R4 GT/分数。

prepare 验证 native 对齐及 999 个合成首轮集合与合成 R3 一致。独立破坏 context、prefix token 数、copy/position 标志、step context/token 数、R3 initial z/prior/first set、版本/强制首轮配置、smoke fresh-render 记录，共 13 类均拒绝；改变 R3 后续 expanded margin 和第二轮数量则允许，避免把 R4 错误约束为重现 R3 后续读数。额外合成指标反例确认：即使 pooled 两项提升 .02，只要双语料 within 仅提升 .005，原严格主门仍失败。

捕获 12 条 main/control evaluator/r6 命令，见 `captured_commands.json`。raw 均调用共享 `src.eval.evaluate_four_datasets`；decoded 均调用原 r6 入口，保持 `noleak/nscore/calib/bma/length/min-windows=2/bma-grid=6/m2`。命令只捕获，不执行，R4 raw、decoded、analysis 和 control 目录一致隔离。

`check_routes.py` / `check_routes.json`、`routes.log`：实际执行生产 main 函数 AST，以模型/reader/媒体替身验证调度和写入，共 measure/controls × R1–R4 × 正式/smoke 16 组；默认 R1 行为保留，R4 唯一开启 expanded_without_verdict，两入口 config 和 main_trace_root 正确。另执行 analyze.main 的 12 组版本/stage 分派，以及 analyze_controls/case_analysis 的 R4 CLI 分派。`bash -n experiments/20261003_m1_explorer/launch/run_analysis.sh` 通过，脚本将 R4 原样转发到 prepare、两臂 evaluate 和 report。

## 结论与成本范围

实现只在首次实际获取时创建独立 observational cache，原生完整 verdict cache 不裁剪；每轮 expanded cache 裁剪和 rotary 状态恢复不污染下一个 native 查询。完整 self-verdict 对话被移出 expanded input，规则、原20帧、全ASR和原视觉问题不改。没有 GT 进入 reader、拟合或阈值路径，协议和 r6 参数未改。

copy 时间已经在 expanded 计时范围，并额外记录 observational_copy_seconds；没有额外模型 forward。smoke 的全输入 processor 诊断有额外 CPU/像素处理时间，README 已说明计费；不能用该诊断的耗时声称正式新视频会有同样额外 forward。实际目标显存、运行时间、真实 processor seam 和完整模型 native parity 尚待预定 smoke，不是本轮 CPU 检查可以代替的结论。

**最终 PASS。** 本轮未发现需要主代理修复的生产 bug，可执行目标环境固定五视频 smoke；通过后才进行完整 R4 双语料运行和既定门判定。
