# Final Proposal: Margin-Certified Poset T3AL Successor

## Problem Anchor

T3AL 在 label-free hateful video localization 中依赖单视频 pseudo-label/projection adaptation，但该更新容易因正负集合退化和粗粒度跨模态证据的尺度不匹配而破坏 dense frame ordering。

## Method Thesis

用带时间戳 transcript 的高置信相对顺序替代绝对 pseudo-label：先在 query space 做 ordinal preconditioning，再用 graph-poset constrained minimum-change projection 替换 T3AL projection-TTA。语言只约束可靠 coarse relations，视觉保留所有未约束排序自由度。

## Formal Core

对 chunk confidence `b_i`，仅建立 `E = {(i,j) | b_i - b_j >= 0.4}`。给定 dense visual score `s` 和 chunk-offset rasterizer `R`，求：

`delta* = argmin_delta 0.5 ||delta||_2^2`

subject to `mean_i(s + R delta) >= mean_j(s + R delta)` for every `(i,j) in E`.

没有进入 `E` 的 pair 不受语言强制排序。

## Dominant Contribution

一个用于 training-free temporal localization 的 margin-certified、non-compensatory cross-modal partial-order decoder，作为 T3AL projection-TTA 的稳定 successor/replacement。

## Supported Claims

- 正确 timestamp correspondence 有因果作用：held-256 显著胜端到端 shuffle/reverse。
- topology-equipped ordinal query 对 fixed 和 ordinal-only 均有显著增益。
- 所有 7,931 条 qualified constraints 零违反、零 solver failure。
- 方法改善 T3AL-derived visual localization controls。

## Unsupported Claims

- 不声称 held 上显著超过 direct transcript teacher。
- 不声称是 `T3AL + module`、首个 multimodal TTA 或首个 change-point TAL。
- 不把 frozen boundary snap 当核心贡献。

## Paper Hardening

补齐完整覆盖、LODO/非测试集调参、per-dataset CI、transcript corruption/missing fallback、计算成本与多 seed 稳定性。
