# Lattice R2 六臂控制：独立窄代码审查

2026-10-05，独立实例 `/root/latents_code`，gpt-6-astra，与主 writer 不同实例，same-family provisional。**PASS：未发现本次新增控制路径中改变观察或结论的未解决 bug。** 当前仅 prepared；控制 GPU 必须等待完整 R2 主门通过。本报告不认证实际8B parity、效果或机制成立。

## 范围与约束

阅读 README 的 R2 控制预声明、`path_control_reader.py`、`path_control_measure.py`、`path_control_analyze.py` 和两个控制launch文件，核对它们复用的同实验R2生产reader、native/context、获取、canonical evaluate与统计helpers。不重开R1/R2 proposal或一般方法审查。未修改生产代码，未读取GT、实际预测或实际main summary，未运行GPU、下载模型、计算哈希或记录run commit ID。实际固定5的beam/ASR/JPEG只用于CPU绑定；stub读数、参考records、summary和guard fixture均仅在内存，无假实际predictions/cache。

## 窄结论

- `full` 确实直接调用生产 `path_reader.structural`，且编译graph/head/tail/tokens一致；prepare要求native与R2 base、full与R2 optimized的global/stance/windows/curve/calls/关键prefix字段精确复现。
- `onebest`取原最高实际beam的完整text，同scaffold/question，普通serial cached forward；`flat`保留full全部tokens，改为serial位置和causal矩阵；`binary`保留原排斥与位置，仅将finite bias改0。每臂只有单个新S，不平均候选预测。
- `wrong_mass`在合并path层旋转floor(K/2)，不改token/order/position；epsilon质量参与，存在zero则整组不动，singleton不动；浮点mass变化与BF16 logmass变化分别保存。它们是path参数变化计数，epsilon没有新增KV token。`wrong_audio_window`只在available窗之间按原映射旋转，保存真实donor坐标/actual crop，changed donor时追加destination指令；README与report明确它不是text-matched对照。
- 共用原native prefix/G/own hard stance/V/reference S，六个新S各自crop/restore，不积累前臂输出。native P、stance-cache N及logical start明确分开，BF16 4D bias/serial或共享logical位置与对应臂一致。smoke每臂仅首available额外clone，共6次；无available没有新S/clone。
- 当前实际ASR/beam获取metadata、CPU processor-expanded native IDs/files/image counts、arm/source/graph/path质量与token/位置/dtype/coverage/版本，均在resume与prepare重建核对。完整atomic bundle是写出源，七份predictions须逐条等于bundle；source获取成本不因物理共享而免除各方法standalone。native时间、每臂时间和input调用精确绑定，joint实际读取成本独立列出，包含reference与diagnostics，不将六份standalone相加充当joint。
- 七流由原canonical evaluator/fixed-r6函数执行，report使用canonical within helper和配对bootstrap，方向为full-minus-arm，onebest/flat须同一个主指标在两语料均≥.01；native/full六指标分别精确对照，eligible断言84/99。`mechanism_supported=False`，保留最终独立机制审查。
- launcher在nvidia-smi和reader之前读取R2完整main门，要求performance_pass严格True、84/99及333输出行；不把该summary传入方法。分析入口先prepare、七流evaluate、最后report；评分路径无GT依赖。

## 实际证据

脚本及日志：`runs/20261004_m1_lattice/path_controls_code_review/`。

| 证据 | 实际执行结果 |
|---|---|
| `tiny_controls.py` / `.log` | 真实随机初始化36层Qwen3VLTextModel、BF16 SDPA/GQA。六臂各72次decoder层forward/clone检查；五个结构臂逐层收到预期bias，onebest执行真实普通serial kernel；六臂clone exact，注入第二层异常后所有36层KV内容及rope恢复。full与生产R2 exact。独立手算wrong-mass含epsilon后的质量[.2,.1,.45,.25]，zero整组/单path不动；真实donor标签和available映射。PASS。 |
| `binding.py` / `.log` | 实际固定5缓存beam/ASR/JPEG+离线真实CPU processor，main与smoke均执行；模型与位置显式stub。available=6/1/17/33/77，展开stance tokens=2375/2780/3054/3115/5861；每视频六clone。17类native token/image/file、当前ASR/beam、source/mass/arm、count/inputcost/inputcalls/standalone、version/position/P-vs-N/path token/coverage corruption均拒绝。无available不新增S/诊断。PASS。 |
| `prepare_parity.py` / `.log` | 实际第一视频输入上的独立R2生产编排与六臂编排stub重读，native/full exact；实际调用prepare并拦截8次reference读取，均为R2路径。正常配对接受，native curve/full global/full window改动拒绝。所有record/summary fixture仅内存，不读取真实预测。PASS。 |
| `evaluation_gate.py` / `.log` | 截获七流共14次canonical/fixed-r6 subprocess，逐项核对路径与flags；直接执行launcher的实际guard代码，内存mock中仅True+84/99+333接受，performance False/83/332均拒绝。没有启动子进程或访问真实summary/GT/predictions。PASS。 |

三个新Python文件编译与两个shell入口语法检查通过。本机真实tiny模型环境为HateVideo、Torch2.7.1 / Transformers4.57.6；目标HateVLM / HF5.15 / 冻结8B控制运行尚未验证。本次检查不替代主门通过后固定5的六臂实际clone、native/full精确复现和真实成本核对，也不允许在主门失败时启动控制。
