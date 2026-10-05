# Candidate35 来源接口B：独立窄确认

**PASS：未观察到实现bug。** 本次接续原Rule6，只检查显式字段长度提示与隔离/启动，不重复proposal或核心模型审查。审查者为与作者不同的复用gpt-6-astra实例，**same-family provisional**。

- A/B spec独立比较只有version、interface、query/planner/feedback system变化；三个旧system均作为B前缀保留，分别追加query≤6词、reason/feedback≤12词及立即关闭字符串提示。原8/16词、16/32字段tokens、whole caps、UNKNOWN转换、工具状态及原执行guard不变；没有新增模型调用或修改最终reader。
- 修改后独立重跑A实际固定5 `extract.validate`，原始PTS/pixels、当前ASR、模型可见prompt、token grammar及每个description span只读重放通过。A仍是10 query字段wordcap、5 planner字段wordcap、5 feedback字段wordcap，0可用query/clip prefix/tool；没有覆盖或挽救A。
- B独立脚本在合成80秒视频执行生产acquire→validate，实际tokenizer/processor配合脚本化生成/相关性provider。覆盖PROGRESS_BAR→HIGHLIGHT→CUT实际状态/媒体更新及TERMINATE/UNKNOWN合法退出；真实1fps/PTS、模型可见history、原始像素、最终不传reason/feedback/score、token/工具/source文本篡改拒绝均通过。该fixture仅软件执行检查，不是模型准确率；processor使用已明确的小尺寸CPU适配。
- 独立进程验证默认A和环境B；执行生产CLI AST的路径赋值覆盖A/B×smoke/main×extract/measure/analyze。源缓存 `data/temporal_time_tools_B`、source/reader输出、decoded与analysis全部与A隔离。collector_selfcheck同样采用OUTPUT_SUFFIX，旧A路径保持。
- `lab1_B.sbatch` 符合local-sc474397、1GPU/4CPU/32G及HateVLM，显式SOURCE_INTERFACE=B；默认smoke或SCOPE=main依次执行extract与measure。分析shell仅full-main，先HateVLM严格prepare，再HateVideo调用原canonical base/optimized evaluate和report，输出main_B_analysis。smoke prepare通过原analyze CLI的 `--smoke`。两shell语法通过；派生副本仅替换路径并stub命令，捕获B环境、smoke/main参数与完整顺序；prepare/evaluate注入exit7后立即停止后续命令。

证据：`runs/20261006_m1_vtimecot/interface_B_cpu_checks/independent/` 下 `a_replay.py`、`b_fixture.py`、`routes.py`、`launcher/check.py`、对应日志与summary.json。实际使用本机HateVLM CPU（torch2.11/transformers5.15.1），没有CUDA、预训练权重推理、真实GT/预测分数/metrics访问；未改生产代码或计算/记录内容哈希与Git标识。

PASS只确认实现与隔离，不保证真实8B接口成功或科学性能。长度提示属于接口执行修复，不是科研贡献或新方法，预算0/3不重置；A失败及所有原caps/guard保留。
