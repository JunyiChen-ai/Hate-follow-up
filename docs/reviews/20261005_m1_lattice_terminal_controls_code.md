# Lattice R3 七臂控制：独立窄代码审查

2026-10-05。独立实例 `/root/latents_code`，gpt-6-astra，与主writer不同实例，same-family provisional。**PASS，无未解决的观察有效性bug。** 本次只认证已准备控制代码的CPU检查，不认证真实8B控制parity、性能或机制。控制GPU仍须完整R3主门PASS。

## 范围

阅读R3控制预声明、`terminal_control_reader.py`、核对`terminal_control_measure.py` / `terminal_control_analyze.py`相对于已审查R2控制的完整差异及其保留路径，检查`lab2_terminal_controls.sbatch` / `run_terminal_control_analysis.sh`。不重审proposal或一般方法理论。没有生产代码修改、GPU、模型下载、GT/真实预测/main summary读取、假实际预测/cache落盘、哈希或run commit ID。实际固定5缓存beam/ASR/JPEG只用于CPU输入绑定；stub记录与summary全部只在内存。

## 核对结论

- **匹配的allword成立。** full直接调用正式`terminal_reader.structural`。allword与full保持完全相同的scaffold/head/tail、完整lexical+newline tokens、合并质量、物理库存和shared logical positions；各路径自己的causal history及跨路径排斥也相同。仅query/tail行改为对全部lexical/terminal keys加入logmass，而非只看terminal keys。实际矩阵差异集合恰好等于query行×lexical keys，没有拿历史R2不同token结果充作消融。
- **其它六臂。** onebest为原最高beam完整text加R3真实newline，普通serial；flat保留全部R3 token并改serial causal/positions；binary保持endpoint-only mask，只令finite bias为0。wrongmass固定token/order/positions，合并path质量含epsilon旋转；zero组与singleton不动，float与BF16 logmass变化分列（是path参数变化计数，epsilon本身无KV）。wrongsource沿available窗口映射，保留真实donor nominal/actual坐标，并在改变来源时追加destination指令；报告保留非text-matched限制。
- **状态与成本。** 七臂从同一native G/own stance cache独立读取，V/reference S只获取一次。每个available窗每臂一个新S，smoke每臂首available clone一次，每视频共7额外forward；main无诊断。KV与rope正常/异常均恢复。实际joint调用为native.calls+7×available+diagnostics，standalone各自包含原实际ASR获取+prefix+V+该臂S；reference/diagnostic与物理共享成本分列，不把多份standalone求和当joint。
- **来源与续跑。** 当前ASR/实际beam、CPU processor-expanded native IDs/image counts/files、P与stance-cache N、当前graph/token/质量/位置/dtype/arm/operator/terminal/coverage与成本逐项绑定；atomic bundle为输出源，旧版本及输入变更拒绝。实际prepare先验证所有bundle及导出，再要求native/full分别与独立R3 reference精确一致。新路径使用r3_controls和r3_full，不覆写R1/R2。
- **评测与门。** 八流均使用原canonical evaluator/fixed-r6；report方向为full-minus-arm，全部六指标及84/99保持。主结构门是同一指标双语料对onebest和flat均≥.01，末端部件门对matched allword也是双语料≥.01，并额外报告交集；mechanism_supported仍False，仍需最终独立机制审查。launcher在GPU检查/reader之前要求R3完整performance True、84/99和333行；summary不进入模型。分析顺序prepare→八流evaluate→report，GT仅postscore report。

## 独立实际执行证据

目录：`runs/20261004_m1_lattice/terminal_controls_code_review/`。

| 脚本与日志 | 结果 |
|---|---|
| `tiny_controls.py` / `.log` | 真实随机初始化36层Qwen3VLTextModel，BF16 SDPA/GQA。七臂各72次layer forward/clone检查，共504次；六个结构臂的实际每层bias正确，onebest走真实普通serial kernel。七clone exact，full=正式R3 exact；每臂注入第二层异常后所有36层KV内容/rope恢复。allword所有IDs/位置/路径history与full exact且mask差异仅query lexical访问；epsilon质量旋转、zero整组/singleton不动、真实donor检查PASS。 |
| `binding.py` / `.log` | 实际固定5 beam/ASR/JPEG及离线真实CPU processor，main/smoke均执行，134个available窗；模型/位置为显式stub。expanded stance token数2375/2780/3054/3115/5861，每视频7clone。19类source/native/token/image/ASR/beam/mass/arm/version/position/P-vs-N/cost/count/coverage/operator/terminal变更均拒绝；无available无新增S/诊断。PASS。日志末尾沿用了旧review harness的R2文字标签，脚本实际导入及上述所有结果均为本次terminal_control模块，不将该标签当版本依据。 |
| `prepare_parity.py` / `.log` | 独立R3 production编排与七臂编排在相同真实输入上的stub重读，native/full exact；实际调用prepare，8次reference读取全为R3路径且全部拦截为内存fixture。native曲线、full global、full窗口值改动均拒绝。summary仅内存，未读取真实预测。PASS。 |
| `evaluation_gate.py` / `.log` | 截获八流16次canonical/fixed-r6 subprocess，逐项参数核对；执行实际R3 launcher guard代码的内存fixture，只有True+84/99+333通过，False/83/332均拒绝。无真实summary/预测/GT读取或子进程启动。PASS。 |

三个Python文件编译与两个shell入口语法检查通过。实际tiny模型使用本机HateVideo、Torch2.7.1 / Transformers4.57.6；目标HateVLM / HF5.15 / 冻结8B控制尚未验证。后续仅在完整R3主门PASS后，按声明固定5核对真实七臂clone、native/full exact、mask及实际成本，再运行完整控制。无本次代码blocker，也无提前机制成立结论。
