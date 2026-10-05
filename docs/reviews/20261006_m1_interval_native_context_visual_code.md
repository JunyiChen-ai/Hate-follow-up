# Candidate26 R2 native-context visual：独立窄代码确认

结论：**PASS（已实现 reader/analyzer 范围）**。与作者不同的复用 gpt-6-astra 审查实例，**same-family provisional**。本次只确认已声明 R2 的实现，不重开方案/性能审查，不裁定真实性能。

实际独立验证：`runs/20261005_m1_interval_witness/r2_code_confirmation/oracle.py` 在本机 HateVLM CPU（torch2.11、transformers5.15.1）实例化随机权重、36文本层 Qwen3-VL。覆盖 FP32/BF16 × 18/20 原始帧 × full/smoke 共8组，使用生产 `measure_r2.read_video`、`validate_bundle`、共享 stance build/margin 及原 `reader.read/encoding`。测试adapter采用小宽度16、2 heads、单视觉层；来源树和媒体为明确合成fixture，source metadata上游validator在此模型fixture中替换为空操作，未冒称完整实际数据source审计。

- 独立重建原 native cache 后重算 G/自身hard stance、逐窗原V/S，与R2 base全部exact。optimized G/stance仍相同，有speech窗口的S exact复用、无speech窗口保持缺失；没有tree-global header或第二次S读数。
- 每个新V真正执行完整multimodal forward；原 native conversation+自身stance token前缀exact，实际source图像/路径经原encoding进入前向。另一次显式 `use_cache=False` full forward 与记录V及optimized窗口V完全一致。原 max与4fps曲线由生产validator检查。
- 两窗、一窗有speech的fixture：full为8 language/3 vision，smoke为9 language/4 vision；smoke只增加一次视觉repeat且exact。optimized声明调用为6 reader+10 source，全部2秒source acquisition哨兵成本保留，native S计费一次，新S成本0。诊断另计。
- 每组独立破坏token、S、global、source成本、实际调用数、repeat计数，生产validator均拒绝。全8组通过，证据 `run.log`、`summary.json`。
- `delegation.py` 在隔离输出根捕获实际R2 CLI wrapper：smoke传入5个repeat要求，full传None；使用原prepare/evaluate/report及R2 validator，R2 raw/decoded/analysis路径独立。原prepare的默认 `expected_smoke_repeats=None, global_exact=False` 不变，原R1 repeat公式保留。canonical evaluator/fixed r6未复制或修改；捕获没有执行GT评测。

代码检查确认：main逐video先调用原source validate再读/复用bundle；native缓存完成原V/S后释放；新V不读取R1margins。原 reader/inputs/interface/source缓存与生成逻辑没有此修订改动。新增input_binding_seconds继续单独记录，模型诊断与主要读数成本分开；原来源成本没有因已有缓存免除。

范围限制：审查时R2两个launcher尚未落盘，故本报告不包含它们。真实fixed5来源重放的作者证据不代替本次独立模型实验，也不在此宣称重新跑了真实source全量审计。未读取真实GT、预测分数或指标；未启动CUDA、加载预训练模型权重、修改生产代码或计算内容哈希/Git标识。PASS不代表实际8B/GPU native数值或R2性能已经通过，后续实际执行仍须原严格prepare。


## 新增 launcher 窄确认（2026-10-06）

**PASS**。本附录补足上文“launcher尚未落盘”的范围限制；未重复reader审查。

`lab1_r2.sbatch` 为 local-sc474397、1 GPU、4 CPU、32G，进入本仓库并激活HateVLM；仅运行 `measure_r2.py`，没有source重抽。默认SCOPE=smoke传 `--smoke`，SCOPE=main不传；R2读数目录由已审查入口隔离，Slurm日志在仓库runs内。

`run_analysis_r2.sh` 是full-main分析入口，日志/PID在r2_full_main_analysis；先以HateVLM执行严格prepare，通过后才激活HateVideo依次执行canonical base/optimized evaluation与report。smoke分析须直接使用已确认的 `analyze_r2.py --stage prepare --smoke`；此main分析脚本本身不提供smoke切换。两脚本均 `set -euo pipefail`。

独立 `bash -n` 通过；派生副本仅替换工作根/激活路径，使用命令stub捕获执行，不运行GPU或GT：默认smoke/main各只有一次measure_r2；分析成功序列环境和四条命令exact；prepare注入exit7后只有prepare，evaluate注入exit7后不继续optimized/report。证据 `runs/20261005_m1_interval_witness/r2_code_confirmation/launcher/{check.py,summary.json,*_failure.txt,analysis.txt,gpu_*.txt}`。未观察到launcher实现bug，same-family provisional限制不变。
