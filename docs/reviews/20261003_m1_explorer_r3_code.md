# Explorer R3 首轮获取修订独立代码审查

日期：2026-10-03。审查实例：`/root/m1_grounder_code_review`。最终结论：**PASS**。评分分支正确；发现的一项新增 prepare 拒绝缺口已由主代理补齐，并通过定向复验。

本次仅审 README 预声明的 R3 首轮强制获取分支、R1/R2 默认行为、版本路径/配置以及 R3 与 R1 重合窗口的 parity 检查，没有重新审查已通过的 attention、图片追加、KV/mRoPE 或完整评测管线。未修改生产代码、启动 GPU、读取真实 GT 或 R3 性能。没有运行或宣称新的实际模型测试；复用前次独立实际模型的缓存恢复与原生 parity 证据。

独立测试入口：`runs/20261003_m1_explorer/r3_review/check_r3.py`。证据为同目录 `check_r3.json` / `.out`，全部新 fixture 和输出仅在 `r3_review/`。

## 评分分支

`read_video(..., support_override=True, always_acquire_first=False)` 保留旧默认。R3传 `support_override=False, always_acquire_first=True`，仅在 round0 将 threshold 设为 None；round1仍用 `.3`，不受这两个初始开关影响。原 native visual query仍执行一次，得到 native读数与 prior，然后才获取；不是用均匀先验或省去原生读取。

新分支不改变候选、attention × distance选帧、每批两张、最多两轮、第二轮 prior刷新、累计图片重编码、原问题、global/answer/speech或最终margin写入。没有合法候选时继续按原逻辑返回最后可用读数并标记 candidate_exhausted。部署调用计数仍为 `3+B+实际获取读取次数`，smoke另加V次原生replay。

独立fixture使用明确的分数/输入替身实际调用生产 `read_video`，只隔离本次停止条件；没有将其冒充模型效果证据。32组输入，每组执行默认、显式R1、R2、R3四种调用，覆盖：

- initial margin为 `-8/0/+8`，nominal support有/无；R3不依赖任一因素决定首轮。
- 合法候选0/1/3/4个；分别允许0、1、2+1、2+2的实际新增数，首轮后低熵则停止。
- 第一轮margin为3时停止；第一轮margin为0时继续第二轮；第二轮最终输出与预期一致。
- 第二轮熵阈值两侧 `0.30000000000000004` 与 `0.29999999999999993`：前者继续、后者停止。
- native prior读取保持一次；global保持原值、speech缺失保持缺失；窗口输出、4fps曲线和calls符合实际轮数。

默认参数与显式R1输出一致。R1或R2已经实际获取的窗口，在R3中的selected entries、轮次、最后margin和calls一致；新分支只为过去初始阶段跳过且有候选的窗口增加读取。

## 发现并修复的 prepare 缺口

初稿只在逐轮检查中允许R3无视初始熵，没有验证“有候选就必须存在首轮”。这样将R1某个 confident、有支持的零轮窗口原样留下并标为R3，可能通过新检查；R1 overlap检查也会跳过该窗口。这会使R3实现与声明不符而没有被预评测检查拒绝。

主代理已补仅针对R3的校验：要求保存的source存在且含PTS序列，从同一4fps中心点、实际PTS映射、半开窗口及legacy exclusions重建初始候选集合；有候选时必须有首轮，且首轮图片数等于 `min(2,候选数)`；无候选时只能零轮，并且必须标candidate_exhausted。不重新解码媒体、不读取GT、不改评分。

独立复验确认：在其他输出/counts同步改为自洽之后，有候选却省略首轮仍会被拒绝；实际候选覆盖在最后窗口前结束时，零轮且标耗尽可以通过；同样输入未标耗尽会拒绝；R3 source为None会拒绝。

## 配置、重合窗口和覆盖

完整333条合成fixture（215/118，含18帧情况）通过生产 `prepare(..., version='r3')`。各视频三个窗口，其中两个设置为R1已扩展、第三个为R3新增；报告计数为666个R1 expanded窗口exact。另运行固定5视频smoke集合，包含两语料各前两个与 `HateMM/hate_video_114`，对应R1 smoke路径，报告10个重合窗口exact。fixture只借用前次独立小模型输出的合成trace，不读取真实实验预测。

分别篡改R1对应轮的selected entries、margin、轮数或最终window字典，新增parity检查均拒绝；篡改revision、always_acquire_first、support_override或初始threshold配置也拒绝。R3的主要常数明确为 `[None,.3]`，没有把未使用的初始`.1`写成启用状态。R1/R2旧配置的缺省always_acquire_first仍为False。

`--version r3` 在reader、prepare、CLI以及CPU分析shell中一致；Slurm第二位置参数经run_lab传入，默认仍为R1。输出分别写 `r3_smoke/r3_main/r3_main_decoded/r3_main_analysis`，不覆盖R1/R2或缓存回放目录。reader仍按完整manifest运行，smoke选择规则未改。canonical evaluator/r6调用函数和报告指标门没有因这次修订变化；shell语法通过。本次没有重新执行真实评测。

**最终 PASS。** 新增首轮强制获取逻辑和校验已满足声明，可进入原定五视频实机smoke。真实效果、目标机器显存/时间以及完整333结果仍待执行，本报告不将这些事项算作完成。
