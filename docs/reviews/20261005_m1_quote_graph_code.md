# Candidate24 QuoteGraph R1：独立代码审查

2026-10-05。独立实例 `/root/latents_code`，gpt-6-astra，与主 writer 不同实例；same-family provisional。**PASS：本次完整新增代码审查未发现改变观察或结论的未解决 bug。** 这不是性能、归属语义正确性或目标 8B GPU parity 认证。

## 范围与限制

依照 CLAUDE.md / RESEARCH_ITERATION_RULES.md 规则6，阅读 `experiments/20261005_m1_quote_graph/README.md`、已通过 proposal、该目录全部六个 Python 文件及三个 launch 文件、新共享 `src/mllm_generate.py`，并核对现有 Judge、CPU renderer 与 stance-cache 接口。未修改生产代码，未启动 GPU、下载模型、读取 GT 或实际数据集预测，没有计算/记录内容哈希或 run commit ID。实验仍为备用；本结论不改变候选22/23的结果分流。

证据目录：`runs/20261005_m1_quote_graph/code_review/`。所有测试脚本可复查；实际数据集 fixture 的 graph/readout 仅存在内存，没有写成数据缓存或预测文件。

## 核对结果

- **来源和图。** 当前 shared window_text 原文、nominal 8s 窗口、字符 offset、合法 kind、typed endpoints 全部绑定。任一非法项使整 chunk UNKNOWN；不存在保留有效子集的路径。chunk8/stride6 的 overlap 只携带已接受且完整位于当前范围的节点/边，canonical node/edge 合并保留原来源。utterance→span 单向零跳，共指组件双向零跳，其余关系检索为一跳无向边；显示仍保留原方向。≤2 semantic hops，排序和最多8个 nonlocal span 与声明一致；traversed relation metadata 和 omitted count 保存。
- **实际读取算子。** suffix 各段分别 tokenize，保存 literal parts/ranges/IDs；source 标签使用真实窗口与原文切片。新增 context 各自只见 prefix/scaffold/本记录历史；local 可见全部新增 context；query 无法直接看新增 context。native prefix 仍可见，与限定声明一致。serial 3D positions 使用 stance-cache 的 N+delta，而不是把未含 global-Q/stance 的 P 当起点。BF16 bias 的0/-inf精确进入 decoder，最后隐藏状态交给原 FP32 Yes/No margin。
- **native 与恢复。** 共用原 native20/full ASR、原 global 问题及自身 hard stance；native V/reference S 各自 crop，只有新 S 进入 max(V,S)。新 branch 正常及异常均 crop KV 并恢复 rope；smoke 另计 clone 与 contextless 两次诊断。共享 native 输入 head/token/image expansion 与当前 CPU processor 的真正编码一致。prepare 再将 native global/windows/curve 与既有 native 原值精确核对，未改变 canonical evaluator。
- **生成、调用和成本。** 每 chunk 单次 fresh text-only greedy；全词表 FP32 输出矩阵，EOS 不进入 JSON text，max-token 耗尽标 truncated，无补救生成。当前实现每个非 EOS token 后实际再做一步，因此真实 forward 数为1+保存 token 数，包含末尾产生后执行的那一步，记录与实现一致。输入/生成 token、无效/截断、耗时、peak及 extraction 调用保留。native 读数成本、生产新 S 成本与诊断成本分列；新视频 standalone 包括图获取，未把 reference S 或诊断算入方法 standalone。
- **续跑与评测。** 当前版本/常量、ASR、source windows、overlap prompt、生成 token解码、graph/packet、native files/image counts/IDs、suffix和位置、margin/max/curve、availability、calls/cost 均重新绑定；按视频原子 records 写出后导出成对 predictions。完整333身份覆盖检查，三指标双语料及84/99有效视频断言。统一4fps evaluator与固定r6均原样 subprocess；bootstrap调用 canonical within helper。GT仅在独立完成打分后的 report 路径；extract/reader/measure不使用GT。机制结论保持 false。

## 实际执行的独立 CPU 证据

| 文件 | 实际检查及结果 |
|---|---|
| `tiny_graph.py` / `.log` | 独立零代价闭包+两次语义扩张 oracle，14个起点一致，8-span截断及遗漏计数一致。实际随机初始化 **36层 BF16 Qwen3VLTextModel + GQA + SDPA**，逐层检查72次attention的mask/query dtype和可达矩阵；clone读数exact；无context与同token普通causal读取exact；注入第二层异常后全部36层KV内容及rope恢复。PASS。 |
| `source_schema.py` / `.log` | 独立Unicode offset、overlap canonical endpoint、真实source label；11项整chunk拒绝，包括越界、错误type/window、范围外endpoint、节点/边上限、truncated。PASS。 |
| `binding.py` / `.log` | 实际固定5 native ASR/JPEG与离线CPU processor，真实展开token数2375/2780/3054/3115/5861，与原head575/1000/1254/1315/4121区分；134个非空窗走实际graph编译/reader编排，model与位置为显式stub。17项source/token/image/P-vs-N/range/position/dtype/count/cost corruption被拒；空speech main/smoke不新增S或诊断调用。PASS。 |
| `generation_eval.py` / `.log` | 独立确定性logit stub：首EOS、普通EOS、长度截断三种情况，calls分别1/3/3且FP32 greedy一致。截获4次subprocess检查canonical evaluator与全部固定r6参数，不执行评测、不访问GT。PASS。 |
| `author_selfcheck.log` | 另运行作者图/整chunk/mask CPU自检，PASS；不将其计作独立oracle。 |

Python编译与三个shell入口语法检查通过。实际CPU环境为本机 HateVideo（Torch2.7.1、Transformers4.57.6）；目标运行环境是 HateVLM / HF5.15。CPU tiny 模型只证明真实36层attention/KV路径与此算子可执行，不能替代目标冻结8B权重、实际JSON输出及图像mRoPE的Slurm固定5检查。

**后续边界。** code-review PASS 后仍须按已声明顺序完成实际固定5：真实生成/来源、每层mask、native exact、clone与contextless≤.01，以及实际调用/成本；再按候选分流运行完整333。没有本次代码 blocker，也没有提前的性能或机制成立结论。
