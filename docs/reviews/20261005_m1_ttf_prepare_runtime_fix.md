# TTF prepare CPU runtime 修正：独立窄确认

2026-10-05，**PASS**。复用独立审稿实例 `/root/ideation_jury_2`，不是实现作者；same-family provisional。只确认 `launch/run_analysis.sh` 的 prepare Python 入口调整，不重开既有代码/提案审查。

prepare 现由 `.cache/envs/HateVLM/bin/python` 执行，与实际生成缓存的 Torch/HF 后端一致；evaluate/base、evaluate/optimized 与 report 仍使用激活后的 HateVideo Python。没有改变 select、阈值、source plan、严格 equality 或统一评测/r6。实际阅读 `measure.validate_bundle` 仍执行 `assert plan==b['plan']`；`analyze.prepare` 在写出成功摘要前对每个样本调用该验证及位置/layout重放。shell 语法检查通过。

实际读取的失败证据 `runs/20261005_m1_ttf/r1_full_smoke_analysis/older_torch_anchor_float_replay.log` 定位到严格 plan equality。作者/README报告旧 Torch 的 frame_cosines 差为 5.96e-8–1.79e-7，离散选择不变；这些逐浮点差异本次没有独立重算，不将其作为自行观察结果。改用匹配后端符合保存精确浮点轨迹的重放约束，不是放宽数值容差。

实际读取本机 `r1_full_smoke_analysis/plumbing_summary.json` 与 prepare 日志，coverage=5、GT_read=false、native_allraw_exact/global_speech_exact/source_layout_replayed 均 true，cloned_cache_checks=5。两个语料压缩视频分别3和2，摘要记录 mechanism_supported=false。结合现存源码，可确认该次 prepare 已完成原严格检查；本审稿人没有自行重新运行 prepare，因为其内部会读取预测，而本次审稿限制禁止读取预测。

未打开 GT、prediction 文件或 metrics；没有 GPU、预训练权重运行或生产代码修改。该结论只确认启动后端与完成的源重放验证一致，不证明性能、完整333通过或机制成功。README末尾“Local complete plumbing PASS still pending”在本次检查时已落后于实际成功摘要，应由作者同步活文档。
