# Explorer 控制版本路径独立代码审查

日期：2026-10-04。审查实例：`/root/explorer_review`；运行主机：`sc474397`。结论：**PASS，无阻塞项**。仅审查当前未提交的版本路径扩展，不重开已在 `docs/reviews/20261003_m1_explorer_controls_code.md` 通过的四臂计算审查，不代表 R3 性能或机制通过。

范围为 `experiments/20261003_m1_explorer/controls.py`、`analyze_controls.py`，以及主代理随后交付的 `case_analysis.py` 版本参数扩展。核对 README 的 Autonomous continuation 对应说明。未修改生产代码、未提交 git、未启动 GPU、未读取真实预测/指标/GT，也未计算内容哈希。新增合成输入、脚本及输出全部位于 `runs/20261003_m1_explorer/version_review/`。

## 独立验证

运行环境：本机 `/home/jehc223/miniconda3/envs/HateVideo/bin/python`。本次复用上次独立审查的合成 333 视频 fixture，先复制到本轮目录，再仅补入新要求的 `main_trace_root` 配置；R1 特意不加 `revision`，验证旧配置兼容。R2/R3 使用各自独立复制的合成输入，并添加对应 revision。没有写回旧 fixture。

- `check_versions.py` / `check_versions.json`：三版本 × 四臂，共 12 组实际生产 prepare/report 通过。report 使用版本不同的合成指标替身，验证选中的 main/control 指标来源与差值；输出里的 `revision`、`metric_source`、`main_metric_source` 均指向对应版本。GT loader 替换为合成数组，within 仍调用共享实现。共 24 次注入错误 revision 或 main_trace_root，全部被拒绝。
- 同一脚本执行生产 `controls.main` 的 AST 函数体，以显式模型、媒体和 reader 替身隔离本轮路径调度：三版本 × 四臂 × 正式/smoke，共 24 组通过。实际读取对应版本的合成 main trace，写入对应 control 目录；config 中 revision/main_trace_root 和逐视频 details.main_trace 正确。R1 故意省略 CLI version。此测试验证入口调度，不声称验证实际模型推理。
- 捕获 24 条评测/解码命令，见 `captured_commands.json`；没有执行真实评测。12 条 raw 命令均调用 `src.eval.evaluate_four_datasets`，12 条 decoded 命令均调用原 `experiments/20260926_twolevel/twolevel_r2.py`，保持 `--noleak --transform nscore --key calib --duration bma --bma-prior length --min-windows 2 --bma-grid 6 --arm m2`。run、out-root、tag 按版本和臂隔离，没有新增评测公式。
- `check_cases.py` / `check_cases.json`：实际执行 case_analysis 的默认 R1 及显式 R2/R3，每次使用完整合成 333 条输入，四个 raw/decoded predictions 来源和 analysis 输出目录均严格匹配对应版本。GT 为合成替身；预先建立各 analysis 目录，符合既有调用前提。
- `check_cli.py` / `check_cli.json`：实际执行两个分析脚本的 CLI AST 分派，覆盖省略 version、显式 r1/r2/r3、非法版本，共 20 例。合法值正确传递给各 stage，非法值由 argparse 拒绝。

日志为 `run.log`、`cases.log`。对应运行命令是上述 Python 路径加本轮目录下各检查脚本；入口模拟会保留输出，重复运行时需使用新的本轮派生入口目录，不能覆盖既有 predictions。

## 路径与兼容性结论

R1 继续使用 `r1_main` / `r1_smoke`、`controls` / `controls_smoke`、`controls_decoded`；R2/R3 分别使用 `<version>_main` / `<version>_smoke`、`<version>_controls` / `<version>_controls_smoke`、`<version>_controls_decoded`。报告读取同版本 `<version>_main_decoded/explore`；case_analysis 同步选择同版本 raw、decoded 和 `<version>_main_analysis`。

`read_control` 内既有 details.main_trace 默认字符串仍写 R1，但生产 main 在落盘前总是用实际 source 覆盖，本轮 24 组入口测试验证了这一点，因此不构成 R3 产物污染。prepare 的旧 R1 配置兼容限于已含原有 main_trace_root 的真实配置；只缺 revision 时按 R1 处理，符合此前生产配置结构。

四个控制算法、预算、问题文本及 r6 参数未因该 diff 改变。可以将本次通过作为版本路径扩展的代码门；真实 R3 对照是否运行仍由主代理按已声明的性能门决定。
