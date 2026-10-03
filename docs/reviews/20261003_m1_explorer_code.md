# Explorer 独立代码审查

日期：2026-10-03。审查实例：`/root/m1_grounder_code_review`。结论：**PASS**。未发现需要修复、会改变当前实验观察的生产实现错误。可以进入 README 已声明的五视频 8B smoke；本报告不代替真实环境的显存、吞吐、原生 baseline 精确复现和完整结果核对，也不表示机制有效。

本轮按 `RESEARCH_ITERATION_RULES.md` 第 6 条审查 `experiments/20261003_m1_explorer/{README.md,explorer.py,measure.py,analyze.py,selfcheck.py,launch/}`，读取已通过的 proposal 和共享 Judge/输入/评测接口。未修改生产代码，未运行 GPU，未读取真实 GT、Explorer 性能或 Preserver 性能，没有计算内容哈希。独立测试和媒体仅写入本候选 `runs/20261003_m1_explorer/independent_review/`。

## 独立实际模型证据

入口 `runs/20261003_m1_explorer/independent_review/check_explorer.py`，输出同目录 `check_explorer.json` / `.out`。本机现有 memora 环境，torch 2.14.0+cu130、transformers 5.16.1，`CUDA_VISIBLE_DEVICES=''`，CPU 单线程。测试实际 `Qwen3VLModel`：36 个语言层、32 query heads、8 KV heads、head_dim 128、mRoPE `[24,20,20]`，三层视觉编码器与 DeepStack `[0,1]`。残差 hidden_size 为 64，是实际小模型 plumbing 检查，不是 8B 模型实测。目标部署 5.15.1 的实际运行由主代理负责，本报告不将本机版本写成目标版本。

原始输入有 20/18 张非连续图像块，交替每图 4/6 个 visual tokens，即 100/90 个 tokens。输入张量复用此前独立检查中经实际 Qwen 图像 processor 生成的确定性合成图片张量，复制到本候选目录；没有 import 其他实验实现。新增图片的 processor/media I/O 使用确定性输入替身，生成不同图像像素及匹配 grids；语言层、视觉层、DeepStack、KV、mRoPE、生产追加与获取逻辑、共享 12-token FP32 Yes/No head 均实际执行。这个替身验证追加调用及 tensor 对齐，不声称验证真实 8B processor 在所有媒体尺寸下的资源上限。

| 检查 | 独立结果 |
| --- | --- |
| pre-RoPE prior | 从实际每层 q_norm/k_norm 输出独立捕获，float64 逐 head 使用 `floor(h/4)` 的 KV head、逐图 token 均值及归一化；与生产 prior 最大绝对差 `< 9.64e-9`。覆盖原始图像以及 2/4 张追加图像。 |
| 原生读数 | 只读捕获开启与普通共享 Judge 读取的 margin exact；没有改变模型 attention 或输出。 |
| 完整位置 | 原 prefix + global Q + native answer + appended block 完整计算位置；缓存 prefix 位置 exact，实际传入 suffix 位置与完整拼接位置 exact。 |
| 追加与 fresh 全序列 | 18/20 旧图分别追加 2/4 新图，FP32 margin 最大差 `2.3842e-7`，BF16 最大差 `.00507582`。BF16 不声称 fresh/append bitwise 相同；没有将这种 kernel/运算分组差异当作缓存污染。 |
| 缓存/状态恢复 | 每次追加后所有 36 层原 KV exact、长度 exact、原生 rope_deltas exact；后续原生 query margin exact。连续 2/4 图读取与多个窗口均通过。 |
| 输入/参数 | 原始输入张量、模型所有参数、head 权重均 exact 未变。 |
| 完整生产 read_video | FP32/BF16 × 18/20 帧均实际执行。B=4、V=3、R=6；base 7 次、Explorer 13 次，smoke 总 16 次 outer model forwards。新增图片编码共 18 张次，即三个窗口各 `2+4`。 |
| 输出通路 | 每窗最后一次 expanded margin 写入 Explorer visual 读数，确实改变最终曲线；native global/answer/speech 保持；没有语音的窗口不制造 speech 读数；4 fps max 曲线与逐窗值一致。 |

## 获取规则与源帧时间

`FrameAttention` hook 的位置是实际 `q_norm` / `k_norm` 输出，位于 transpose/RoPE 之前。长期保存原 prefix 的 8 个 KV heads，只在 prior 计算时扩展至 32 query heads；query 只取最终 token。每层 softmax 的分母仅包含原始与本轮全部追加 image keys，不包含文本 keys。36 层/32 heads 平均以后，按各图实际 token 数求均值，再归一化。故这是声明的 pre-RoPE 选帧信号，不是原模型完整 post-RoPE attention。

`choose_frames` 的最近帧单元先合并重合时间的均值 prior，时间排序使等距匹配较早帧；候选按实际 PTS 排序，最大分数相同时取较早候选。批内 prior 固定，选中一帧后更新所有候选的最近距离。代码省略共同正因子 `1/video_duration`，不改变同一视频窗口内的 argmax。下一次模型读取后才刷新 prior，顺序与原图 + 按时间排序的新增图一致。

独立 `check_decisions.py` / `.json` 使用明确的分数与媒体替身隔离生产 `read_video` 控制流，不冒充模型效果证据。通过九种情况：

- 有 nominal support 且 initial margin 为 `+8` 或 `-8` 时零获取。
- 无 nominal support 时，即使同样 confident 仍触发第一次获取；首轮低熵后停止。
- initial margin 0、首轮 margin 3 时只获取一轮；首轮仍为 0 时进入第二轮，返回第二轮读数而非历史最大/平均值。
- 只有 0、1、3 个合法候选时，分别执行零轮、一张一轮、两张再一张；没有重复旧的新增源 index 或无限补帧。
- 缺语音保持缺失，native global/answer 不随这些控制流改变。

独立 `check_source.py` / `.json` 使用 PyAV 17.0.0 实际写出 run 内 FFV1 视频，带非零 presentation origin 和不等间隔 PTS，然后调用生产 `FrameSource`。用另一次完整顺序解码作为 oracle：

- normalized PTS/index 与独立顺序解码一致；首个不早于 4 fps target 的映射、半开窗口限制、覆盖不足、used/excluded index 去重均一致。
- 从 legacy 文件名 index/duration 重建初次和提前 `.5 s` 的候选排除，与独立计算一致。仅排除实际存在的 legacy 文件对应 index，没有强求 20 张文件齐全。
- 逆序 seek 所有 source frames，PNG 像素与完整顺序解码 RGB exact；再次读取已存在 witness PNG 也 exact。
- 原媒体只读，索引/图片见证在 `runs/`，不写 `data/`。

仍须保留 README 的边界：旧 JPEG 没有保存原 PTS，source-index 排除只是 conservative legacy exclusion，不能据此声称证明新图和所有旧 JPEG 像素身份完全不同。nominal support 也不等于旧 JPEG 的已验证真实 PTS 支持。合成 seek 检查不代替所有真实视频的解码覆盖；实际输入无法满足单调 PTS/精确 seek 时当前实现明确失败，不悄悄替换邻窗图片。

## 追加接口与成本

新增 user turn 中的 question 直接使用 native visual query 的原字符串；只有前面的新图片及实际 PTS 文本是新增内容，没有追加前次窗口答案。编码所有已获取图片，所以第二轮重新编码第一轮图片，不能只报四张次。`read_images` 用完整 token 序列与全部 grids 重新计算 mRoPE，再切 suffix；完整 attention mask 长度包含原 KV。传给视觉编码器的 pixels/grids 仅是当前追加的图片，与 suffix 的 expanded image tokens 对齐；原图缓存不重新编码。每次读取后恢复 native rope 并裁剪共享缓存，下一窗口不继承已获取图片。

原 prefix/global/answer 各一次，native B 次分支，加实际 acquisition reads R，故 Explorer 为 `3+B+R`。paired 收集可共享原生视觉读取，不另加 baseline forward；smoke 每视觉窗口额外一次原生重读，为 `3+B+R+V`。实测调用计数与此一致。初次 PTS indexing、源图像 decode/PNG I/O、selector、每轮全部图片的重新视觉编码和模型 forward 均在 `extra_seconds` 内。报告 baseline 明确是带只读 capture/prior 的 instrumented baseline，不能据此称 attention extraction 免费，也不能用该比值当作无插桩原生 baseline 的真实部署比。CPU 测试没有测有效 GPU 峰值或目标时间。

## 分析、覆盖与评测接线

完整 333 条合成 fixture（215 HateMM / 118 HateClipSeg，包含一条 18 帧记录）实际通过生产 `prepare` 和 `report`。fixture 的预测、GT 和 metrics 全部合成，未读真实标签或性能文件。`read`/metrics/GT I/O 用 fixture 替身，within 计算调用共享 canonical `within_video_macro`，没有复制指标公式。

核对及实测内容：固定 constants/model/36 layers、paired/check/manifest 完整集合、旧 native baseline 的逐窗/曲线/global parity、18 帧 actual frame metadata、每轮合法新增/去重/源 index、最后 margin、global/speech 不变、4 fps 长度、调用统计均接线正确。删除视频或改变 round limit 会在 prepare 被拒绝。`report` 使用 `explore` / `explore_raw`，没有前候选字段残留；decoded global 保持原生 global。配对 bootstrap 的单位是视频，2000 次、seed 0。主门同时检查两语料 final within 对 native/current 至少 `.01`，并检查全部指标下降界；合成正/负门 fixture 均通过。`mechanism_supported` 始终 false，不凭主门自动宣称归因机制成立。

捕获两臂的实际 `evaluate` 命令，确认 raw 评测调用 `src.eval.evaluate_four_datasets`，decoded 继续调用既定 r6 入口及 `nscore/calib/bma/length/min-windows=2/bma-grid=6/m2/noleak` 参数，输出各臂独立。没有新评测公式或 GT 进入 reader/selector。`run_analysis.sh` 先 prepare，再分别执行两个 CPU 臂；逐一检查 wait 退出码后才 report。Slurm/launcher 路径和全部 shell 文件语法通过，未启动 GPU 或真实评测。

**最终 PASS。** 没有生产修复要求。后续能否运行完整333、是否有性能/机制增益、attention 相对同预算获取是否必要、真实成本和媒体兼容性，仍由已经声明的目标环境 smoke、完整评测及通过主门后的机制对照回答。
