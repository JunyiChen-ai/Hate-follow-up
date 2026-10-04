# Program B：allocator-only 重试窄确认

2026-10-05；独立实例 `/root/latents_code`，gpt-6-astra，same-family provisional。**PASS：允许在原固定5和原验证条件下重试；不代表OOM已经解决或smoke通过。**

实际读取 `launch/run_lab_handles.sh`、README新增执行记录、Slurm133日志和获取缓存复用分支。脚本新增 `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 位于conda激活之后、nvidia-smi及两个Python进程之前，两个子进程均继承该设置。shell语法检查通过。smoke仍传同一个`--smoke`，获取/评分入口、来源、token/model/ops/2048与96 cap、方法版本、解析及全部guard没有在该修改中变化。

日志确认四视频获取完成，module calls依次24/5/38/53；第五视频在`handle_decoder.generate`首次model forward的MLP中OOM，申请902.00MiB、GPU free769.25MiB、PyTorch reserved但未分配3.25GiB。该证据支持尝试减少allocator碎片，但不证明长prefill本体一定装得下。

[PyTorch 2.11官方CUDA内存文档](https://docs.pytorch.org/docs/2.11/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)将`expandable_segments`定义为可增长分配段，旨在减少变化大小分配造成的不可复用碎片；这是allocator策略，不是减小输入或改变模型计算的接口。

已有四份成功获取记录可复用：`handle_extract.main`对每份已有文件仍先执行`validate(metadata,row,current_segments)`及`validate_generations(j,metadata)`，通过当前source/grammar/prompt/image/token/forward绑定后才continue；第五份只有完整获取并验证后才从.partial原子替换。allocator设置不改变这些绑定的语义，不要求重新生成已精确绑定来源。原四份记录的生成主机、日期与成本仍保留；失败尝试的实际成本由原日志保留。

本次没有重新运行GPU、读取GT或实际评分、计算哈希、修改生产文件，未覆盖既有冻结审查。实际同5重试仍须完成paired native/clone/source/exercise/factual guards，之后才能判断是否允许main。若仍OOM，应按新实际错误诊断prefill所需内存，不能将本次PASS解释为允许改输入、截断或放宽guard。
