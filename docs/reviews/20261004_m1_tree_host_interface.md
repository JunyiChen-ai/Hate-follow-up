# Candidate21 lab2 主机切换：接口窄范围确认

2026-10-04。**PASS（same-family provisional）**，仅确认从缺少原视频的 lab3 改用 lab2 的运行接口与路由；不重开算法审查。

实际只读 SSH 检查两台 HateVLM 环境的 `transformers/__init__.py`，版本均为 **5.15.1**。lab2 的 `get_image_features` 将视觉 `pooler_output` 按 `image_grid_thw.prod()/merge_size²` 拆成逐图tuple；`compute_3d_position_ids` 的文本增量分支从实际 `past_key_values.get_seq_length()` 起算，并加已保存的 rope_delta。相关源码摘录与此前已核验的 lab3 摘录直接文本比较完全相同，未计算哈希。

证据：`runs/20261004_m1_tree/code_review/lab2_target_hf_interface.txt`。新增 lab2.sbatch 使用 `local-sc474399`、1 GPU、4 CPU、32G，仍调用相同 HateVLM/run_lab.sh。extract 记录实际 Python 命令与生成host；读取提供的无GT媒体预检文件确认333条输入记录，与所述切换原因一致。

没有GPU执行、GT读取或生产代码改动。可以按原固定五视频流程在 lab2 验证；此前 native 可复现不等于本候选 appended-image 新路径已经通过，后者仍须本次 smoke 实测。
