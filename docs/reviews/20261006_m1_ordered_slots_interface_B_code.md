# Candidate34 interface B：窄修复确认

结论：**PASS**。接续原 Rule6，仅确认显式长度提示及路径隔离；未观察到改变实验观察的实现 bug。复用的独立审查实例与作者不同，模型同为 gpt-6-astra，属于 **same-family provisional**。

- 独立比较 A/B spec：差异仅 version/interface 和两个 system 指令的追加文本。caption 最多12词、condition 最多6词且立即关闭字符串，实际进入 source renderer 的模型可见 system；原16/8词、32/16字段tokens、96/768整次tokens、封顶转UNKNOWN均保持。模型、seed、调用流程及成本公式未改。
- 修改后重新运行 A 全5真实 source 的生产 `extract.validate`，逐帧/像素、PTS、当前ASR、原token/prompt/字段与来源选择只读重放通过。仍为158 captions不可用、127可用slot embeddings、0 remote；原A失败保留。
- 在环境 `SOURCE_INTERFACE=B` 下，独立脚本重新执行80秒合成视频、10窗口的生产 acquire→validate。脚本化生成/向量配合真实 tokenizer/processor，10 caption和2 slot批次，实际PTS、原始像素、逐序上下文、embedding token offsets、NONE/有序选择、current固定及最终memory只含原始证据均通过；篡改来源文本/offset/绑定的派生副本被拒绝。此为软件fixture，不是预训练模型结果。
- 默认未设变量选择A；B选择独立 spec/cache。在独立进程内执行原CLI AST中的路径赋值，覆盖A/B×smoke/main×extract/measure/analyze。分别隔离 source输出、reader输出、decoded、analysis及 `data/temporal_ordered_slots_B`，未启动生产CLI模型或评测。所有模块使用同一所选SPEC/CACHE。
- 源采集、embedding、reader/native G/ownstance/V/S、共享模型分支与评测函数没有算法改动。原remote/newV/newS执行guard和canonical evaluator/fixed r6调用仍在；没有降低执行门或新增模型调用。此次不重复已通过的36层模型审查。

证据：`runs/20261006_m1_ordered_slots/interface_B_cpu_checks/independent/` 中 `a_replay.py`、`b_fixture.py`、`routes.py`、对应日志、A_replay/B_fixture摘要、routes_A/B.json及summary.json。实际执行环境为本机 `.cache/envs/HateVLM/bin/python` CPU，torch2.11/transformers5.15.1。

限制：未读真实GT、预测分数或指标，未运行CUDA/预训练权重；未修改生产代码。PASS只允许检验B实际接口执行，不保证真实8B闭合成功或科学性能。提示明确化不是科研贡献或性能版修订；原失败与预算0保持。未计算、记录或依赖内容哈希/Git标识。
