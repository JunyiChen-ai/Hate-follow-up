# Explorer 条件性控制部署脚本独立审查

日期：2026-10-04。审查实例：`/root/explorer_review`；审查主机：`sc474397`。结论：**PASS，无代码阻塞项**。范围仅为 `experiments/20261003_m1_explorer/launch/control_lab3.sbatch` 和 `control_server.sbatch`，以已审查的 `control_lab.sbatch` 为基准；本结论不表示目标机器已具备运行条件，也不替代性能门和目标机五视频 smoke。

逐字比较确认仅有预期替换：lab3 将分区/节点从 `sc474399` 改为 `sc474398`，账户保持 `jehc223`；lab-server 将分区/节点改为 `sc448960`，仓库和 Python 账户路径改为 `junyi`。未改变算法入口、环境变量或参数传递。

两个脚本均显式使用对应 `local-<hostname>` 分区和匹配 nodelist，申请一张 GPU、4 CPU、32G 内存，低于实验室 56000 MB 配置。没有 dependency、子任务提交或后台 GPU 启动。`nvidia-smi` 位于作业脚本内。`set -euo pipefail` 保持失败传播，最后 `exec .../HateVLM/bin/python -u experiments/20261003_m1_explorer/controls.py "$@"` 保持逐参数转发，因此 `--version r3`、`--arm` 和 `--smoke` 可原样传入。

脚本切换到目标账户的主仓库，HF_HOME 为仓库 `.cache/hf`，control 产物由已经审查的入口写仓库 `runs/`。Slurm output 为 `runs/20261003_m1_explorer/control_slurm_%j.out`；它是相对提交工作目录解析的路径，按仓库规定必须在目标仓库根目录提交并提前建好日志目录，不能依赖脚本中的 cd 来修正 Slurm 日志位置。

验证命令：`bash -n experiments/20261003_m1_explorer/launch/control_lab3.sbatch experiments/20261003_m1_explorer/launch/control_server.sbatch`，退出码 0；Python 独立文本比较和资源/转发断言通过，证据见 `runs/20261003_m1_explorer/version_review/check_launch.json`。无内容哈希。

README 声明这些部署是条件性的：独立完整 333 视频控制实验，不切分数据，先通过 main gate；每个目标运行环境都要新做 smoke 和 native parity。实际启动前仍需主代理完成实时资源、输入视频、权重与运行环境可用性检查，再进行规定的五视频 smoke。本实例没有提交作业、启动 GPU、读取真实性能/GT、修改生产代码或执行 git commit。
