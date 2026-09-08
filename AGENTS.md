# AGENTS.md

规则全文在 `CLAUDE.md`，此处只列必须先知道的四条。

- **项目**：label-free hateful video temporal localization，零仇恨标注。主数据集 HateMM、HateClipSeg（MHC 停用）。当前方法 OMSL-v6（`experiments/20260829_omsl_v6/`）。
- **状态入口**：`research-wiki/STATUS.md`，每轮结束更新。权威数字只认 `runs/` 里评测器输出。
- **评测**：4 fps 协议，主指标 pooled frame ROC-AUC 与 PR-AUC，within 只报告；评测器只有 `src/eval/` 一份。
- **禁止哈希**：不计算、记录、比较任何哈希 / checksum；唯一例外是多机代码同步时比较 Git commit（`scripts/check_machines.sh`）。
