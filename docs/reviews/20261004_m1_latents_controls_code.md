# Candidate19 预声明机制对照比较器：独立补充审查

日期：2026-10-04。same-family provisional；仅审查 `experiments/20261004_m1_latents/compare_controls.py` 新增比较逻辑，不重审已通过的核心读出实现。最终结论：**PASS**，具体比较完整性问题均已修复确认。只有完整 main 通过性能门、四个完整控制及其 canonical 评测结束后，才可执行该比较器；本次未执行。

阅读实验 README 的四个预声明完整控制、既有核心代码审查，以及比较器调用的 canonical within 和配对 bootstrap 接口。未读取 GT、未运行比较器、未运行 GPU，未修改生产代码。初次比较器相关代码摘录保存在 `runs/20261004_m1_latents/controls_code_review/initial_comparator_excerpt.txt`。

已确认正确：full/initial/warmup/search_only/wrong_support 五个 arm 均从各自 `r1_<arm>_main_decoded/optimized/metrics.json` 读取并输出两语料三指标，来源路径随结果保存。所有差分为 full 减 control；全部优化对应 initial，Stage I 必要性对应 search_only，Stage II 必要性对应 warmup。阈值要求同一个主指标在两个语料均达到 .01；不跨语料挑指标。`mechanism_supported=False` 不随阈值自动变成成立。

final/raw_max/raw_visual 每视频排序都调用 `src.eval.evaluate.within_video_macro`，未复制 ROC 实现。bootstrap 对同一 eligible 视频的 full-minus-control 差分抽样，seed=0、10000 次、均值百分位 95% 区间；没有将两组独立抽样冒充配对。六项 pooled/within 主结果与 within 视频级配对区间分开报告。

初次审查定位的具体缺口：

1. wrong_support 仅验证 target_frame，未核验绑定个数、顺序、source patch 与原支持索引对应，亦未验证非等长帧的 target_patch 整数映射；错误 patch 映射可通过该检查。应核验完整绑定，并报告逐窗口不可干预/未改变内容标记。现有 `actual_content_changed` 来自 positive 支持向量比较，汇总须明确这一范围，不夸称正负全部内容已经验证改变。
2. 333 条数量检查不保证 raw/decoded/checks 与既定 215+118 名单完全一致；trace 通过 zip 比较会在截断处静默停止。应检查完整 key 集合、trace/window 等长、各 arm 的 image_counts/image_positions/original_frame_times/merger_shape 对齐，确保同一支持索引指向同一原始帧与 patch。

修复确认：

- 比较器在读取任何 GT 前先检查完整 full 的 `summary.json` 中 `performance_pass=True`。五个 arm 的 raw/decoded/checks 都必须精确等于既定 333 名单（215 HateMM、118 HateClipSeg）；跨 arm 语义配置仅允许 arm/host/date 差异。
- 窗口和 trace 数必须等于固定窗口规则给出的数量，四项图像元数据跨 arm 相同；global/stance/speech、窗口和原 positive/negative 支持保持检查仍在。
- wrong_support 对正负支持均逐项检查 cardinality、源索引到 frame/patch 的解码、半视频 frame 偏移、非等长帧整数 patch 映射和边界。记录按视频/窗口列出 unchanged_positive_support 与 single_frame，单帧还要求实际未改变。覆盖明确只指存储的 positive embedding 内容变化；negative 仅声称绑定索引核验。

CPU 证据 `runs/20261004_m1_latents/controls_code_review/mapping_checks.log`：静态编译通过；仅抽取实际 comparator 的 binding 检查 AST，在自造两帧非等长 fixture 上执行，正确映射通过，错误 target_patch、缺少 binding、错序 binding 均被拒绝。未运行 main、未接触任何真实 GT 或 GPU。差分、配对区间、六指标来源和独立阈值方向在修复后保持正确，无剩余 blocker。

这次 PASS 只确认比较实现可用，不代表 main 通过、不代表控制已经完成，也不证明机制成立。其输出的 `mechanism_supported` 仍固定 False，原始排序与实际干预覆盖须结合六指标解释。
