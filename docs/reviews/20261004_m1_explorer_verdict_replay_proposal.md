# R4 verdict_replay 对照方案独立审查

日期：2026-10-04。审查实例：`/root/explorer_review`。结论：**PASS，可以实现；仅在 R4 原主门通过后执行固定五视频 smoke 和完整333控制。** 本轮只审阅 README 的 `Conditional fixed-trajectory context control` 事前声明，沿用 R4 proposal 中的协议边界。未读取 R4 性能、预测或 GT，未修改生产代码、启动 GPU 或计算内容哈希。

该对照固定 R4 每窗的全部 round counts、每轮 added source entries、累计时间顺序、actual PTS/time strings 和原视觉问题，只恢复完整 native global question/answer 对话。不得根据本臂 margin/entropy 重定轮次，不重新选帧，不替换回答，也不读取 R3 scores。最终使用最后一轮 replay margin；零轮返回 native visual。原 global/speech 保持一致。因此它是对 R4 已实现轨迹的条件性上下文对照，不是第四次修订或另一个可独立部署的获取策略。

可识别范围声明准确：它估计 **条件于 R4 所选轨迹，恢复整个 global Q/answer 对话的效果**，同时包含位置和对话结构变化。不是孤立 Yes/No token 效应，也不是两种各自重新获取策略的总效应。R4 轨迹本身受 observation-only 读数影响，这种条件化不被消除；不能外推到任意轨迹或把固定重放描述为重新执行原 R3。

raw visual/max 的对比可以描述相同新增观测下的读取变化。final 指标还包含各臂完整语料独立 r6 拟合产生的差异，不能全归因于 hidden representation 改变。零轮窗口的 raw visual 保持不变，也不意味着其 decoded 分数必然不变。报告保留三项主指标、按视频配对 within 和开发期标记；原双语料 within ≥ +.01、pooled 不降 >.005 的主门不能由此对照改变。

模型调用预算 `3+B+sum(R_w)`、累计图像编码张次与 R4 trace 一致；每轮仍重新编码累计图片。没有第二份 observational KV 副本，但仍有 native KV、suffix 激活、source indexing/图片解码与读取成本。声明已收费，35–40 GPU 分钟仅为估计。相同 forward/图片数不等于相同耗时或内存，因为上下文长度、选择开销和缓存复制不同；应实测，不能声称严格 compute-matched。

后续窄代码审查落实以下已声明的干预边界即可：逐轮重放而非只固定最终图片数；R4 原始输入、原生回答、speech 与固定的图片身份/时间保持一致；恢复完整对话以外的 acquired 输入 tokens/grids 不变；缓存恢复与 native replay 精确；拒绝 R1/R2/R3，使用独立 `r4_controls[_smoke]/verdict_replay` 和对应 decoded 路径；共享评测器和原 r6 参数不改。仍需目标固定五视频 smoke，proposal PASS 不替代实现验证。

**最终 PASS。** 该对照可以检验上下文假设，但不能补救获取机制 novelty。uniform、distance、fixed4、mismatch 仍使用 R4 observation-only conditioning 和 R4 traces；其原有机制门和归因限制不变。没有需要修改提案才能进入实现的阻塞项。
