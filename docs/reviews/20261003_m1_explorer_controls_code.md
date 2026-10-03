# Explorer 机制对照独立代码审查

日期：2026-10-03。审查实例：`/root/m1_grounder_code_review`。最终结论：**PASS**。新增评分路径未发现需要修复的生产计算错误；新增报告路径的两项缺口已由主代理修复，并经本实例定向复验通过。此结论只表示实现可用于已声明的对照，不表示 R1 通过性能门或机制成立。

范围是 `experiments/20261003_m1_explorer/controls.py`、随后新增的 `analyze_controls.py` / `launch/control_lab.sbatch`，以及 `case_analysis.py`、`trace_diagnostics.py` 的必要对齐核对。不重开已经 PASS 的 R1 审查。没有修改生产代码、启动 GPU/真实评测、读取 R1/Preserver 真实性能或真实 GT，也没有计算内容哈希。所有新测试和输出位于 `runs/20261003_m1_explorer/control_review/`。

## 对照计算与真实模型测试

独立入口 `control_review/check_controls.py`，结果 `check_controls.json`。复用前次独立检查的合成输入 harness 定义，重新执行本次生产 `read_control`；不执行或借用主代理自检作为独立证据。环境为本机 CPU torch 2.14.0+cu130 / transformers 5.16.1。实际 Qwen3VLModel 有 36 语言层、32 query heads / 8 KV heads、head_dim 128、mRoPE `[24,20,20]`、视觉 DeepStack `[0,1]`；残差 hidden_size 64，明确为小模型，不是目标 8B 实机验证。

FP32/BF16 × 18/20 原始图片均实际执行四臂。只替换媒体/tokenizer 输入和 GPU timer，真实 attention capture、图片视觉编码、语言层、共享 12-token FP32 head、KV/mRoPE、控制轮次和最终打分全部运行。对照的独立检查包括：

- **uniform**：按最终 main 帧数建立窗口等分中心，依顺序选择实际 PTS 最近且尚未选中的候选，等距选较早 PTS；然后按 main 每轮实际新增数切片。独立排序 oracle 与生产选择一致。
- **distance**：将 attention 权重设为常数，只按到原图与本臂已获取图片的最短时间距离贪心选择；批内选中后更新距离，跨轮重新使用本臂历史。独立距离 oracle 一致。没有误用 main 的已获取历史或新 prior 权重决定排序。
- **fixed4**：忽略 main 是否触发，按 `min(4,eligible_candidates)` 等分中心选图，先至多两张，再读取剩余图片。0/1/3/4 个候选对应 0、1、2+1、2+2，候选不足不借窗或重复图片。
- **mismatch**：按每窗 main 最终帧数分组，窗口排序后用 `ceil(k/2)` 轮转 donor；独立验证 k=3 的轮转、singleton 保持自身、零获取窗 unmatched。按 receiver 最终时间槽与 donor 排序后图片逐一对应，保留 receiver 原每轮新增集合与时间戳。真实编码 fixture 检查传入 pixels 的身份来自 donor，同时生成的 timestamp 文本仍来自 receiver；记录的 donor_entries 与实际图片一致。

每臂都返回最后一次读取 margin，零轮返回原生 margin。另以明确的分数替身隔离生产调度：native margin 为 -8、控制首轮变为 +8，仍严格重放 main 轮数，不因为控制自身低熵提前停止。这个替身只验证 replay 语义，不作为模型性能证据。

真实模型每次 `read_images` 前后逐层比对 KV snapshot，全部 exact；rope_deltas 恢复 exact；smoke 每窗普通原生重读 exact。控制全局 margin、native answer、speech 及 speech 缺失状态与对应原生输出相同。原问题字符串逐字相同。最终 visual/max/4 fps 曲线与最后读数一致。原输入、模型参数及 head 权重未变。

实测 fixture 有 V=3、B=4，main 的两个窗口各两轮、第三窗口零轮：

| 控制 | 获取读取 R | 部署 outer forwards | smoke 总 forwards | 新图编码张次 |
| --- | ---: | ---: | ---: | ---: |
| uniform | 4 | 11 | 14 | 12 |
| distance | 4 | 11 | 14 | 12 |
| mismatch | 4 | 11 | 14 | 12 |
| fixed4 | 6 | 13 | 16 | 18 |

计数分别符合 `3+B+R` 与额外 V 次 smoke replay。第二轮重新编码累计图片，编码张次按 `2+4` 计，不按最终四张计。`standalone_seconds` 包含控制本身的原 prefix、原生查询/prior、源索引、decode/PNG I/O、选帧及全部获取读取，减去明确的 smoke replay 时间；单独执行一个控制并未把这些计算误写为免费复用。数据留在原只读位置，图片见证写控制对应 `runs/` 子目录。

## 报告修复与复验

初稿报告有两项影响解释完整性的缺口，已反馈主代理并完成窄修复：

1. **decoded 对齐缺少拒绝条件**：只有 keys 相同不足以保证在同样帧集合比较。修复后，读取 GT 前检查 paired raw/decoded 的 rate=4、duration、曲线长度/finite、native global 与存在时的 stance。独立注入 rate=1、错误 duration、短曲线、NaN、改变 global，均在调用 GT loader 前被拒绝。canonical shared-min 不再替这些损坏控制输出悄悄截取比较范围。
2. **mismatch 缺少有效错配子集报告**：prepare 现由 main 轮次的最终帧数独立重建 matched flags，未错配窗口 visual 必须等于 main 最后 margin；report 分别给 matched/unmatched 窗口数、有/无 matched 窗口的视频数，并对窗口掩码子集和整视频组分别报告 visual/raw-max/final canonical within 与按视频配对区间。保留说明：完整语料 r6 重拟合可能改变未错配窗口的 final 分数，因此不能把该子集的 final 变化解释为新增时间证据。

复验入口 `control_review/check_analysis.py`，结果 `check_analysis.json`，全部输入位于 `control_review/synthetic_root/`。使用完整 333 条合成记录（215/118，含一条18帧视频和一个完全无有效错配的视频），未读取任何真实预测、GT 或 metrics。生产 prepare/report 实际运行，只有 GT/指标来源及命令执行使用显式合成替身；within 计算仍直接调用 canonical 实现。

四臂均通过 prepare/report。独立验证 mismatch 的 matched/unmatched 窗口及视频计数，未错配 raw visual 子集的差值为零；空组保留零 eligible videos/null 效果。统一差值方向是 `main - control`，视频是 bootstrap 单位，2000次、seed0。两语料同一主指标的移除损失至少 `.01` 才进入 `common_metrics_with_removal_drop_at_least_01`；该字段是数值门，不自动宣称因果贡献。单个视频的 bootstrap 仍是退化区间，读取报告时不能把它当作有总体统计精度的证据。

prepare 报告逐轮图片数与 image token 数是否匹配 main。数量匹配并不自动代表时间或 token 匹配；fixed4 明确不要求 main 数量匹配。matched-budget 机制解释必须结合这些字段和实际时间，不能只看 `.01` 差值门。本次测试所用混合 grids 也会在后验 trace 报告中正确标记 `same_grid_within_video=false`，没有把固定张数误写成相同尺寸。

捕获四臂共八条 evaluator/r6 命令，确认 raw 调用 `src.eval.evaluate_four_datasets`，decoded 使用原 r6 入口及 `nscore/calib/bma/length/min-windows=2/bma-grid=6/m2/noleak`；各臂输出目录和 tag 独立，没有复制评测公式。配置 smoke 路径使用 `r1_smoke` traces，正式控制使用 `r1_main`；Slurm 入口接收 `--arm` 和可选 `--smoke`，没有自行绕过 Slurm 运行 GPU。脚本语法/编译检查通过。控制是否启动的主门由主代理调度，reader 本身不读取性能或 GT 来选控制。

## 两个 posthoc 工具

在同一完整333合成 fixture 上实际运行了 `case_analysis.main()` 与 `trace_diagnostics.run()`：

- `case_analysis` 按 canonical shared extent 截取 GT/预测，以 4 fps 中心映射原窗口，支持/获取掩码在同一帧集合取子集。窗口正帧占比、visual/raw/final、global 正误和 gain/loss 排序均从同一 video key 对齐，没有发现可改变观察的错位。它是 GT 进入的 post-scoring 描述路径，不被任何 reader import 或反馈到 R1 常数。
- `trace_diagnostics` 从保存的实际 PTS、legacy exclusion 和已用 index 重建候选，验证 main 真正选择，再比较同一 observed history 下的 distance-only 选择。合成 fixture 的逐轮选择全部通过，读取次数/新增帧/重复编码张次与独立预期一致。其文字明确是 conditional trace diagnostic，不冒充具有独立历史的性能控制。

**最终 PASS。** 两项报告问题已修复且复验，不需改评分方法。未运行真实控制，未读取 R1 性能；只有主门通过后，才按原声明运行这些控制并据结果决定机制结论。
