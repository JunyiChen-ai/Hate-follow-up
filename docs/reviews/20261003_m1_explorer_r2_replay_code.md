# Explorer R2 缓存回放与最小 reader 修订独立审查

日期：2026-10-03。审查实例：`/root/m1_grounder_code_review`。结论：**PASS**，没有需要修复的评分/回放实现错误。

范围为新增 `experiments/20261003_m1_explorer/replay_r2.py`、`launch/run_replay_r2.sh`，以及本次随后加入的 `measure.py` 停止条件、R1/R2 路径和配置、`analyze.prepare` 的 R2 parity 检查及对应 launcher 参数。没有重审已通过的 R1 模型路径，没有修改生产代码或启动 GPU。读取了 README 中已公开的 R1 修订依据和 R2 预声明，没有读取真实 GT、R2 性能或真实预测文件。

独立产物全部位于 `runs/20261003_m1_explorer/r2_replay_review/`：`check_replay.py/.json/.out` 与 `check_reader_revision.py/.json/.out`。测试使用完整333条合成记录；读写均在本次目录的 synthetic fixture 内。

## 回放语义与证据边界

R2 唯一评分变动是取消初始阶段的 no-nominal-frame override：所有窗口初始二元熵 `< .1` 即停止；`>= .1` 的窗口保留完整 R1 获取与停止流程。没有改变后续 `.3`、每轮两张、最多两轮、prior、selector、native context/global/speech、12-token head 或 r6。

这允许从 R1 保存的读数回放：每窗模型读取均从同一个恢复后的 native cache 开始，R2 只删除整窗的获取过程，不改变被保留窗口的任何输入或计算。这个前提复用已经独立验证的 R1 每窗 crop/RoPE/native replay 证据；本次没有另跑模型来假装 fresh GPU confirmation。缓存回放自身是零次新 MLLM forward，不是一次新的部署计时。

`replay_r2.py` 对每窗重新从 native visual margin 计算熵，并检查与保存的 initial entropy 一致。保留窗直接使用 R1 expanded 窗口，跳过窗使用 native 窗口，所以 speech、visual/speech max、时间范围和窗口索引均来自原完整记录。保留 global 与原 native answer，不从窗口结果重新判断全局。输出4fps曲线按原8秒窗口和 manifest duration重新展开。

## 完整333合成回放检查

`check_replay.py` 独立构造215/118条记录，每条含八个窗口；最后一窗截断至56.05秒，曲线225个4fps位置。覆盖：

- 无 nominal support、initial margin `+8` 或 `-8`，R1 曾强制获取；R2 恢复 native margin并移除全部对应读取。
- 有 support 的 confident 窗口，R1/R2 都没有获取。
- 有/无 support 的 uncertain 窗口，以及首轮停止和两轮停止，R2 保留原最后读数。
- 恵值恰为 `.1` 与相邻浮点值 `.09999999999999998`，前者保留、后者跳过，严格符合 `< .1`。
- 候选耗尽、原本没有 acquisition 的窗口，不制造获取或调用。
- speech 可用/缺失交错，全部保持原状态；curve等于对应窗口 max。

全333输出逐窗和曲线与独立预期一致，原 source 文件未改变。fixture 每视频 B=12：原生15次调用，R1有8次获取共23次；R2保留3次获取、跳过5次，共18次部署调用，删除3个原有获取窗口。`replay_checks.json` 的窗口/读取计数与此一致。缺失一个 source 视频会拒绝，不产生缩小语料的结果。

回放期间将 array/GT loader 设为禁止调用，实际运行未触发它。`calls` 明确表示未来部署逻辑调用数，config/checks 的 `actual_new_model_calls` 为0。原 `standalone_seconds` 被删除，记录增加 `timing_status` 声明尚未测量实际 R2 GPU 时间；不把原R1时间复用为R2实测。新 record `code_path` 指向 replay 脚本，`source_prediction` 保留原R1来源路径；最后这项 provenance 修正经静态核对，不改变任何分数。

## 新 reader 条件与版本路径

`read_video(..., support_override=True)` 保留 R1 默认。停止条件新增 `or not support_override`：

- R1时，round0仍允许无支持窗口越过初始熵停止条件。
- R2时，round0总是执行 `.1` 检查；round1原本已满足 `round_index>0`，因此后续 `.3` 检查完全不变。

`check_reader_revision.py` 以明确分数/输入替身实际执行这条生产控制流，覆盖九组 support/confidence/候选不足情形。默认参数与显式 `support_override=True` 的窗口、分数、曲线和调用计数相同。R2的 retained 窗口与 R1相同，initial-confident窗恢复 native。该测试是停止逻辑的定向证据，不声称重新测试8B或量化真实运行时间。

main 的 `--version r1/r2` 显式进入 config 的 revision/support_override，并分别写 `r1_*` / `r2_*`；Slurm的第二位置参数、run_lab转发以及CPU分析第一位置参数均接线正确。旧调用默认R1，不会隐式改成R2。`r2_cache`、`r2_main` 与两者decoded/analysis目录分开，缓存结果不会冒充 fresh结果。

`analyze.prepare(version='r2')` 检查版本与 override配置，R2任何实际获取都必须符合初始熵门，并将对应的窗口字典、调用数、global与整条曲线和 `r2_cache` exact比较。另一个333条合成fixture通过该新检查；分别改变 replay窗口值、calls、global、curve后均被拒绝，错误版本也被拒绝。smoke仍使用既定5视频子集，与完整cache按key比较；full继续要求333完整覆盖。

## 评测与报告

回放仅产出分数；GT只在独立 evaluate/report阶段进入。四条捕获而未执行的命令确认两臂raw调用canonical `src.eval.evaluate_four_datasets`，decoded调用既定r6入口和原有 `nscore/calib/bma/length/min-windows=2/bma-grid=6/m2/noleak`，输入输出均指向R2 cache目录。没有复制评测器。

生产 report在完整合成GT/metrics fixture上实际运行，within调用共享canonical函数；paired bootstrap按视频、2000次、seed0。base三指标必须与current逐项exact，因此对base的delta同时是对current的delta；两语料within至少`.01`及其余指标下降界的主门与声明相同，正/负门fixture均通过。`mechanism_supported=False`，报告范围明确是开发期cache replay，不能直接宣称GPU确认或机制成立。

向decoded结果注入错误rate、duration、短曲线、NaN或global，均在GT loader调用前拒绝。launcher先replay、分别执行两臂canonical/r6评测，逐一检查退出码后才report；shell语法通过。

**最终 PASS。** 可以使用R2完整CPU回放作本次修订的开发期筛选。若过门，仍按已声明流程运行真实R2全333，核验缓存分数parity并测量实际成本；本审查没有把尚未发生的GPU确认或机制对照算作完成。
