# Tree R2 controls：纯输入几何窄审查

2026-10-05；独立实例 `/root/latents_code`，gpt-6-astra，与作者不同实例；same-family provisional。**PASS（仅当前纯输入变换与 CPU 几何预检）**。尚无 control acquisition/reader 实现，本报告不放行完整控制实验或机制结论。

实际审查 `control_inputs.py`、`control_preflight.py`、新增 `test_control_membership_and_true_coordinates`、README 最后 R2 controls scope，结合原 `window_packet`/节点定义。未改生产代码，未运行 GPU，未读取 GT/预测，未生成控制 caption 或进行评分。

本轮发现并已窄确认一项声明不一致：temporal 继承的主树 root_relevance 不仅固定诊断 depth/branch 预算，还进入 `window_packet` 的 `(-root_relevance, distance, ...)` 局部帧优先级。主作者明确这正是意图，并在尚未运行控制时修订 R2 规范：同时固定主 topology、各节点成员数与对应 main root priority，仅改变语义成员分配为连续时间成员；显式替代 R1 branching-only 文字。已实际重读修订，声明与当前代码一致。README 同时说明继承 priority 未重新判断 temporal roots，不能据此声称优于自行重新计算 priority/depth 的可部署 temporal 算法。实现未改，无剩余本轮阻塞项。

核对结果：

- temporal 按排序后的 root/child IDs 递归分配连续 pool-index 区间，长度精确等于主节点实际成员数，保留 parent/depth/topology。输入 pool 本身按真实 time/index 排序，因此连续索引具有声明的时间顺序。每层 partition coverage 有断言，不靠等份分组猜测分支数。
- representative 是分配组 `len(group)//2` 的真实 member；time 来自该 entry，center 为该组真实 feature 的 FP32 mean。caption 仅按相同 pool-index（同一 entries 表对应的真实 source frame）复用，否则调用传入 callback；同一帧跨节点只调用一次。没有把原节点 caption 当新代表帧 caption。实际模型 caption/token cap 尚须未来 acquisition 实现核查。
- flat 只替换 context 的显式 node/parent/depth 表达，保存同一 ancestor caption/time inventory、选帧与选择依赖；按时间重排符合声明，不声称消除语义选择。
- wrong_links 对全部终端 leaf IDs 排序，按 floor(L/2) 旋转 donor ancestor chain，以原顺序去重；保留原 local pixels/leaf IDs 和 donor 实际 time/text，不把 donor 时间伪装到当前窗。singleton/共享祖先可能无变化，不能把形式旋转当有效干预覆盖率。
- no_depth 使用原最终 roots、原 center/relevance/caption 成为终端节点，重新走同一 local 选择规则；no_added_pixels 仅清空 packet pool_members，保留其它 packet 字段与 context。
- control_preflight 的 placeholder 只在内存构造临时 temporal tree，输出目录是 runs，仅保存几何/机会计数，不写 data/cache、不调用模型、不评分。temporal changed_windows 只比较 selected indices/leaf/ancestor IDs，不包含未来 caption 内容变化，summary 明确这一限制。

独立证据：`runs/20261004_m1_tree/r2_control_geometry_review/oracle.py` 与 `oracle.log`，实际执行 **GEOMETRY_REVIEW_CPU_PASS**。手工构造不等成员数、不等深度树，用写死的预期连续区间、独立 ancestor 旋转和精确 callback 记录验证上述行为，并确认 main 输入未修改。另实际重跑新增 selfcheck，结果 PASS。核读已有 `runs/20261004_m1_tree/cpu_control_preflight/summary.json`：完整333（HateMM215、HateClipSeg118）的纯几何预检完成；该覆盖证据不是控制执行、事实正确性或性能证据。

后续实际 acquisition/reader、额外 witnesses、caption 成本、各臂 native/R2 exact、wrong-link 无变化机会报告与 canonical 评测仍需其完整实现的代码审查。控制 GPU 启动依然取决于完整 R2 主门，不由本几何 PASS 替代。
