# Lattice R3 terminal-state reader：独立窄代码审查

2026-10-05。独立实例 `/root/latents_code`，gpt-6-astra，与主 writer 不同实例；same-family provisional。**PASS：发现的两处 unit 诊断 blocker 均已修复并实际复核，无未解决的观察有效性问题。** 本结论不代表实际8B GPU parity、性能或机制成立。

## 范围与约束

依规则6，核对 R3 README 预声明、`terminal_graph.py`、`terminal_reader.py`、`terminal_measure.py`、`terminal_analyze.py` 与两个 terminal launch 文件。复用已审查同实验获取/native/完整路径/canonical基础设施，不重开proposal或R1/R2一般方法审查。只读真实固定5 ASR/beam/JPEG并运行CPU验证；没有GT、实际预测或main summary读取，没有GPU、下载模型、哈希或run commit ID。未修改生产代码；fixture读数仅内存，不写假实际prediction/cache。

## 机制与绑定核对

- 相同5个真实beam及FP32 softmax质量、exact-text合并、排序和完整句tokenization沿用R2。每个非空正质量路径在完整lexical tokens后追加恰好一个单独编码的字面newline token；真实Qwen tokenizer通过single-token断言。epsilon/zero项不增KV，不重归一化。
- 每条分支含自己的完整lexical history和末端，跨分支−inf。所有query/tail行对新增graph只能读取各terminal key，权重为log merged path mass，全部lexical key直接访问为−inf；native prefix/scaffold仍可见。position从每条路径共享起点开始，query在含terminal的最长路径后开始，物理KV仍按路径依次存储。只产生一个FP32最终margin替换S。
- BF16 4D bias和显式3D位置实际进入36层decoder。正常、clone及异常后crop所有层并恢复native rope。unit reference使用独立手工构造的同token、同末端可见性矩阵；它不要求等同普通all-word causal，不能用它替代native exact。空graph独立验证为scaffold/query causal。
- native20/full ASR/G/own stance/V和reference S保持原流程；新S进入原max与4fps曲线。当前CPU processor-expanded prefix IDs、P与含global-Q/stance的N、ASR/beam/graph/terminal IDs与indices/位置/版本/trace/margin/calls/成本重新绑定，atomic bundle用于resume与导出；旧reader版本拒绝。
- 部署成本仍为实际完整获取+prefix+V+new S，reference S和诊断分列。每available新S一次，有available的smoke首窗clone/unit/reference共3次；main无诊断，无available无新增S。各分量、input调用与metadata、实际调用和peak检查保持。
- R3 `r3_full_smoke/main`及其analysis/decoded路径独立，不覆盖R1/R2。prepare全覆盖并与原native global/windows/curve精确比较；evaluate/report直接复用原canonical函数对象，四次subprocess原fixed-r6 flags不变，report原双语料六指标和eligible数/paired within逻辑不变。extract/reader/measure没有GT路径，GT只在独立postscore report。

## 发现及修复确认

1. 原unit质量断言要求Python精确等于1。五个score=0的FP32 softmax权重经Python累加为`1.0000000149011612`，会令所有非空unit诊断在forward前退出。主writer改为单位质量近似断言，仅处理诊断的FP32舍入；正式mass/token/bias与`.01` margin容忍未改变。`unit_mass_repro.log`保留该数值复现。
2. 原unit要求terminal_indices为`[g-1]`，在最高beam为空的合法epsilon情形得到g=0、实际indices=[]，导致退出。现明确空单路径indices=[]，手工reference不设置terminal key，使用scaffold/query causal矩阵；非空保持原末端限制。

独立真实小模型已执行修复后两种unit，均与正式reader exact；并对正式/参考函数分别注入decoder异常验证恢复。没有通过改变正式算子或放宽margin容忍规避问题。

## 可复查证据

`runs/20261004_m1_lattice/terminal_code_review/`：

- `tiny_terminal.py` / `.log`：独立手列带两种长度路径、duplicate/epsilon/zero的矩阵与位置；恰1newline、epsilon/zero无KV、query全部lexical keys不可见。真实随机初始化 **36层Qwen3VLTextModel、BF16 SDPA/GQA**，72次层mask/dtype核对；clone exact；非空unit endpoint-reference exact、空unit causal-reference exact；正式及reference异常后全部36层KV内容/rope恢复。PASS。
- `binding.py` / `.log`：真实固定5缓存beam/ASR/JPEG及离线CPU processor，main/smoke两模式；134个available窗，模型/位置为显式stub。expanded stance tokens2375/2780/3054/3115/5861；18类version/native IDs/image/file/ASR/beam/path mass/token/range/position/cost/count/diagnostic/unit IDs/P-vs-N/terminal token/indices/flag变更均拒绝。无available无额外S/诊断。PASS。
- `eval_identity.py` / `.log`：evaluate/report与原函数对象相同；截获4次canonical/fixed-r6 subprocess核对参数，未执行评测或读GT。PASS。
- `unit_mass_repro.log`：原严格质量断言必失败的数值证据。

四个新Python文件编译、两个shell入口语法检查通过。实际tiny模型环境为本机HateVideo、Torch2.7.1 / Transformers4.57.6；目标HateVLM / HF5.15 / 冻结8B仍须Slurm固定5完成native全值exact、clone exact、独立unit-reference≤.01及真实成本核对。CPU PASS不替代该GPU步骤，也不提前认证完整333或机制控制。
