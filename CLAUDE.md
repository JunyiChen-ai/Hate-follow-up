# CLAUDE.md

## 项目
**Label-free hateful video temporal localization**：训练、适配、阈值选择都不使用任何仇恨标注，输出帧级仇恨分数。当前方法 SPVL-r2（stance-conditioned evidence localization，`experiments/20260910_spvl/README.md`，2026-09-10 晋级；前一方法 OMSL-v6 在 `experiments/20260829_omsl_v6/`）。研究迭代流程与晋级标准见 `RESEARCH_ITERATION_RULES.md`。

- **主数据集**（2026-09-09 裁定）：HateMM、HateClipSeg。MHC-EN、MHC-ZH 停用：不跑、不作门、不进论文主表；旧文档里的 MHC 数字只作历史记录。新数据集只做 external validation，加入前须用户同意。
- **对照**：零标签 / training-free / test-time adaptation 方法（T3AL、LAVAD、Vad-R1、EventVAD、ZS-CLIP、ZS-ImageBind 等）为同类对照；弱监督方法（MultiHateLoc、MACIL-SD、DSANet 等）作参照并标明使用了视频级标签。
- **当前状态唯一入口：`research-wiki/STATUS.md`**（最新代码在哪、权威数字在哪、做到哪了）。每轮迭代结束必须更新它。

## 评测协议（2026-09-09 裁定）
- 网格：**4 fps**，本地协议。GT 数组 `data/gt_4fps/<dataset>.npz`（出处见同目录 `PROVENANCE.md`），test split。1 fps 时代的表（`docs/protocol_1fps_legacy/`）不与 4 fps 数字合表：视频集、GT 数组、上采样方式都不同。
- **主指标（2026-09-10 裁定）：pooled frame ROC-AUC、pooled frame PR-AUC、within-video macro ROC-AUC**，三者并列报告与比较。pooled 两项是文献通用指标，主要反映视频级排序；within 等于 arXiv 2608.21854 的 Macro-AUROC（每个含正负帧的视频算一个 ROC 再平均），反映视频内定位顺序。
- 评测器全仓库只有一份：`src/eval/evaluate.py` + `src/eval/evaluate_four_datasets.py`。所有方法与 baseline 调用同一份；任何目录不得复制或改写评测逻辑。改评测器等于全表数字失效，必须显式裁定。
- 权威数字只认 `runs/` 里评测器直接输出的 `metrics.json`；markdown 表格一律是转录，引用时注明来源文件路径。
- **对着 test 开发（沿用 Retrieval-hate 规则 10，2026-09-09 适配）**：允许读取 test 预测和 test GT 做 error analysis，并据此改方法；允许用 test 指标比较设计版本。每次记录看了哪些文件、发现了什么、改了哪个设计（写进实验 README）。由此得到的数字是开发期证据，STATUS 和论文里必须标"development-selected"，不能写成未揭盲的确认结果。test 标签不得进入任何训练、拟合或阈值选择的计算路径（方法本身零标签）。不要求另设确认集（用户裁定 2026-09-09）。

## 哈希限制（沿用 Retrieval-hate 2026-09-05 裁定）
- 禁止计算、记录、比较或依赖哈希、checksum、digest（SHA、MD5 等），包括文件、媒体、特征、缓存、模型、配置、代码、文档和结果的内容哈希，也包括用内容哈希派生随机种子。
- **唯一例外**：多机代码同步时允许读取、记录、比较 Git commit 标识，并检查未提交修改与未跟踪文件，确认各机代码一致（`scripts/check_machines.sh`）。此例外不扩展到 run / 结果溯源、数据、缓存、模型或其它文件校验。
- 现有代码里的哈希碰到时删除，不得继续扩展。溯源只记录可读的输入/输出路径、配置、模型名称与版本、代码版本说明、日期和生成命令。文件是否可用通过实际解析、覆盖率、shape、split isolation 确认。rsync 不开 checksum 比较。

## 方法计算成本
允许使用 VLM / MLLM。避免没有明确必要性却大幅增加预处理或新视频推理成本的方案，尤其每窗口多次重复调用。提案时同时给出性能假设与计算代价：哪些缓存可复用、新增调用次数、预计 GPU 时间。不能用"冻结""离线预处理"掩盖处理新视频仍需付出的成本。

## 环境
- 主环境 conda `HateVideo`（`environment_HateVideo.yml`）。OMSL-v6 推理只需 numpy / scipy / scikit-learn，CPU 一分钟内跑完；VLM / ASR / ImageBind 抽取需要 torch 2.7 + cu128。ImageBind 从 `third_party/lavad/libs/ImageBind` 导入，权重在 `data/assets/imagebind/`。
- 原始视频：实验室机器在 `~/data/<数据集>/`，校区在 `/data/jehc223/<数据集>/`。仓库内 `data/` 只放派生缓存。
- 长任务必须与 SSH 会话解耦：`setsid nohup` 后台运行，日志与 PID 写进该 run 的输出目录，随时可 `tail -f` 查进度。

## 长任务监控
- 一律用 harness 自带机制：等待单个事件用 Bash `run_in_background` 加 `until ...; do sleep 60; done`（本机或经 `ssh <别名>`），结束自动收到通知；需要逐条事件用 Monitor 工具。
- 不手写监控脚本或线程，不新建 `runs/*/monitor/`。
- 监控命令必须同时匹配完成与失败（进程消失、Slurm job 从 `squeue` 消失、`Traceback`、`FAILED`、DONE 标记），不能只等成功标记。

## 多机运行

**本机 = `uoa-lab1`，是唯一改代码的地方。** 其它机器只 `git pull`。

| 别名（`~/.ssh/config`） | 主机 | GPU | 运行方式 | 登录 | 仓库路径 |
|---|---|---|---|---|---|
| 本机 `uoa-lab1` | sc474397 | 1 × RTX 5090 32G | 直接跑 | 本机 | `~/Hate-follow-up` |
| `uoa-lab2` | sc474399 | 1 × 5090 | 直接跑 | key | `~/Hate-follow-up`（运行副本） |
| `uoa-lab3` | sc474398 | 1 × 5090 | 直接跑 | key | `~/Hate-follow-up`（首次用时 clone） |
| `lab-server` | sc448960，账号 `junyi` | 1 × 5090，与他人共用 | 直接跑 | key | `~/Hate-follow-up`（首次用时 clone） |
| `uoa-campus1` | foscsmlprd01 | 8 × A100 80G | **Slurm** | token | `/data/jehc223/Hate-follow-up` |
| `uoa-campus2` | foscsmlprd02 | 7 × A100 80G | **Slurm** | token | `/data/jehc223/Hate-follow-up` |
| `uoa-campus3` | foscsmlprd03 | 7 × H200 143G | **Slurm** | token | `/data/jehc223/Hate-follow-up` |

lab2 和 lab3 常被 Retrieval-hate 的搜索占用；机器是否空闲每次实时查（`bash scripts/check_machines.sh`），不沿用快照。

### 校区服务器的登录
- 三台只能 token 登录，key 无效。自动化全靠 ControlMaster socket（`~/.ssh/cm-jehc223@foscsmlprd0N…:22`，`ControlPersist yes`）。开工先 `ssh -O check uoa-campus1/2/3`；socket 死了（本机重启、master 进程退出）就让用户在终端 `ssh uoa-campusN true` 输一次 token，之后自动复用。
- 校区真正的 home 配额只有 100M，`$HOME` 指向 `/data/jehc223/home`；所有东西放 `/data/jehc223/`。

### 代码同步
1. 本机 commit 并 `git push origin follow-up`；远端 `git pull`。远程机上不改代码；如必须改，改完立即 commit 并 push，回本机 pull。提交只含当前任务已授权的改动，不用 `git add -A` 混入其它工作。
2. 各机代码一致性用 `scripts/check_machines.sh`：打印每台的 commit、脏文件数、未跟踪数、GPU 占用、磁盘和配额、越界文件。开跑前、汇报前各跑一次；commit 不一致或出现 STRAY 就先处理。
3. 首次在某台机器运行前：实验室机 `git clone https://github.com/JunyiChen-ai/Hate-follow-up.git ~/Hate-follow-up`；校区 `git clone … /data/jehc223/Hate-follow-up`。conda 按 `environment_HateVideo.yml` 建 `HateVideo`（campus1 已有；campus2 需先重装 miniconda 到 `/data/jehc223/miniconda3`；campus3 需新建）。安装日志记 `runs/_setup_<机器>/`。

### 校区 Slurm 规则
- 只用 `sbatch`，登录节点不跑 python。sbatch 文件进 git：`experiments/<id>/launch/campus_<corpus>.sbatch`，头部固定 `--gres=gpu:1 --cpus-per-task=8 --mem=64G --output=runs/<exp_id>/slurm_%j.out`，不写 `--time`，不写 `--dependency`。提交：`cd /data/jehc223/Hate-follow-up && sbatch experiments/<id>/launch/campus_<corpus>.sbatch`。
- sbatch 里 `source /data/jehc223/miniconda3/bin/activate HateVideo`，`cd` 到仓库根，`export HF_HOME=/data/jehc223/Hate-follow-up/.cache/hf`（机器上已有的权重用 symlink 接进 `.cache/`，不复制）。所有输出只写仓库内 `runs/` 和 `data/`。
- **并发预算：三台校区合计最多 2 个 job**（运行中或排队中都算）。提交前在三台各跑 `squeue -u jehc223 -h | wc -l` 求和，< 2 才提交。每台每人 16 CPU / 128G 是上限。
- **不 chain job**：不用 `--dependency`，job 里不 sbatch，不用 `&` 并发提交；下一个 job 由主 agent 在上一个结束后提交。
- 初始 `PENDING (JobHeldUser)` 属正常，等自动放行，不 `scontrol release`。
- 监控：`run_in_background` + `until` 循环经 ssh 查 `squeue -j <id>` 消失，再看 `slurm_<id>.out` 尾部有无 `Traceback` / `FAILED` / DONE。

### 派发规则
- **一个实验（一个语料的完整一轮）整体在一台机器上跑，不切片。** 任何需要整个 train / test 分布的步骤都在一台机器上看到全部数据。并行只发生在实验之间：HateMM 和 HateClipSeg 可以在两台机器上各跑各的，两个互不依赖的实验也可以。
- 选机：选当时最空、最快的一台。先看实验室 5090（`nvidia-smi` 占用 < 50% 且空闲显存 ≥ 16G，不排队，立刻起）；实验室没空或模型显存 > 32G 时用校区，校区里按空卡数和排队长度选最空的一台，H200（campus3）留给大模型。校区 2 个 job 的预算用完就等。
- 提交前查磁盘：校区 `df /data` 剩余 < 50G 或 `quota -v` 带星号就不提交，先清。

### 校区磁盘、原始视频、特征
- 校区常驻只放：HateMM、HateClipSeg 的 `video/`（合计约 11G）+ 当前实验需要的 `data/<类型>/` 子目录。仓库在校区的常驻占用（视频 + 当前特征）控制在 60G 以内。
- 原始视频缺失时从本机 `~/data/<数据集>/video/` rsync；本机也缺时 `rclone copy b2:junyi-data/RGCL_video/raw/<数据集> … --transfers 8`。rclone 配置含密钥，不写进任何文档或 commit。
- **特征抽取默认在实验室机器做**（磁盘 900G 到 1.6T），结果放本机 `data/<类型>/`，校区按实验需要 rsync 子目录过去。校区 GPU 留给需要 80G 以上显存或大吞吐的 VLM 打分。
- 校区必须抽特征时：输出写 `runs/<exp_id>/features/`，job 结束 rsync 回本机 `data/<类型>/`，再删校区副本。

### 结果回传（强制）
- `runs/` 不进 git。远程实验结束后立即 `rsync -a --info=progress2 <别名>:<仓库>/runs/<exp_id>/ ~/Hate-follow-up/runs/<exp_id>/`（不用删除目标文件的选项），回传完成后才允许更新 STATUS；STATUS 只引用本机路径。
- 远程新生成的派生缓存同样 rsync 回本机 `data/<类型>/`，并在 `PROVENANCE.md` 注明生成机器。
- 每个 run 的 `run.log` 首行与实验 README 写明运行主机名。

### 文件落位（所有机器一致）
- **本项目文件只能在本项目仓库下**：实验室 `~/Hate-follow-up`，校区 `/data/jehc223/Hate-follow-up`。不写别的项目目录，不写 `~` 或 `/data/jehc223` 根，不用 `--out_dir ~/xxx`。
- 启动脚本进 git：`experiments/<id>/launch/`，各机器用同一份（`git pull` 后运行）。实验室启动方式：`cd ~/Hate-follow-up && setsid nohup bash experiments/<id>/launch/run_<corpus>.sh > runs/<exp_id>/launch_<corpus>.out 2>&1 &`。
- 第二份 checkout 只允许命名 `~/Hate-follow-up-<分支名>`，输出仍写主仓库 `runs/`、`data/`；分支合入后立即删除。

## Agent 调用
- 所有通过 Agent 工具 spawn 的子 agent 一律使用与主会话相同的模型：主 agent 从自己的系统提示读取当前模型名并显式指定（不依赖默认继承），不得指定更弱的模型。
- 用户可能要求单独 spawn 一个 agent 并直接交代任务；主 agent 先 spawn 待命，再用 SendMessage 把用户的任务原文转给它。

## 目录规范（2026-09-09 立；新文件必须遵守，存量按"碰到才迁"）

| 目录 | 放什么 | git |
|---|---|---|
| `src/` | 稳定共享基础设施：评测器、数据加载、特征抽取、通用工具 | 提交 |
| `experiments/` | 迭代原型代码，每轮一个目录 `experiments/<YYYYMMDD>_<slug>/` | 提交 |
| `scripts/` | 数据准备、一次性工具、baseline 复现入口（含 1 fps 时代的 `scripts/duplex`、`scripts/reproduction_baselines`） | 提交 |
| `configs/` | 配置文件 | 提交 |
| `docs/` | 冻结文档：协议、预注册、最终报告、评审记录，只新增不改写 | 提交 |
| `research-wiki/` | 活文档：当前状态、权威结果表、方向索引，可原地更新 | 提交 |
| `data/` | 输入与派生缓存：特征、转录、GT 数组、模型权重，对实验代码只读 | 忽略 |
| `runs/` | 全部实验输出：分数、日志、指标；`runs/legacy_*` 是 2026-08 的旧输出 | 忽略 |
| `archive/` | 淘汰的实验、过时文档、历史根目录文件（`root-2026-09`、`base-paper`、`detection-2026-08`、`idea-stage-2026-08`、`experiments/idea_discovery-2026-08`） | 提交（仅文本） |
| `third_party/` | 外部代码原样克隆 | 忽略 |

- 根目录白名单：`CLAUDE.md`、`AGENTS.md`、`Readme.md`、`RESEARCH_ITERATION_RULES.md`、`LICENSE`、`environment_HateVideo.yml`、`.gitignore`。其余 markdown / JSON / txt 不得新增到根目录；报告进 `docs/`，状态进 `research-wiki/`。
- 每个实验目录含 `README.md`（机制假设、怎么跑、结论与去向）。实验目录之间不得互相 import；共享逻辑先升入 `src/`。实验淘汰后整目录移入 `archive/experiments/`，README 顶部补一行淘汰原因。
- 每次运行写 `runs/<exp_id>/<run_name>/`：config 快照、代码版本说明（路径 + 日期 + commit）、`run.log`、`run.pid`、`metrics.json`。
- 新建派生缓存放 `data/<类型>/`，同目录放 `PROVENANCE.md`（生成脚本、代码版本说明、日期、上游输入、生成机器）。没有出处的缓存视为不可信。大文件永不进 git。
- 同一事实不写第三份。两份文档数字冲突时以 `runs/` 原始输出为准，当场修正并注明。

## 汇报语言
标准技术词汇直说，不发明黑话、不打比喻、不起外号；先说跑了什么、出了什么数、对决策意味着什么。
