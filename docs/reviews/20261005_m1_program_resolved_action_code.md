# Program23 R3：resolved-action 读取的独立窄确认

2026-10-05，**PASS，未发现本次修改引入的实质 bug**。复用独立实例 `/root/ideation_jury_2`，不是实现作者；same-family provisional。仅确认R3差异，不重开proposal或R1/R2完整审查，不作性能/机制有效性判断。

实际阅读范围：`handle_measure.py` 的predicate/question/revision/验证路径，`handle_analyze.py` 的revision与prepare绑定，新 `resolved_selfcheck.py`、两个resolved启动脚本，README中R3事前声明及相邻历史文字，既有R2独立审查。追踪既有 `program.normalize_module/branch_record` 确认来源/字段约束；没有打开GT数组、预测文件、metrics或error-analysis JSON。

`available_field` 精确执行左strip、case-insensitive `^UNKNOWN\b`；UNKNOWN后接标点、空白或串尾被识别，UNKNOWNish/UNKNOWN_thing等不匹配，符合声明。空字段本身不由该predicate拒绝，但既有normalize_module把空/全空白字段变成UNKNOWN；本轮完整实际来源未发现空字段穿过上游。不是事实真实性判断或语义纠错。

R3仅在visual action对象上deepcopy后归一actor/action/target；`branch_record`为每个evidence另建容器，故替换value不会改变原execution。sources/support等绑定原样保留。visual应用要求实际action可用或已有合法join，actor-only/target-only不应用；join合法性继续依赖原解释器与测量前source重放。本次未改变它们。speech路径没有归一操作，条件与R2相同。

不可用时返回字面native问题，read_video仍重新调用生产margin，随后以当前配对native作精确断言；没有历史分数拷贝或少计fallback调用。crop/rope/clone路径不变，原始source acquisition成本继续计入。R1默认、R1/R2原条件和格式保留；R3使用单独run目录、方法名与reader_revision，prepare将detail revision绑定config。统一评测/r6入口未改；新Slurm脚本传revision3，两个shell脚本语法检查通过。

## 实际独立执行证据

证据目录：`runs/20261005_m1_program/resolved_code_review/`。

- `tiny.py / tiny_summary.json / tiny.log`：复用原审查的随机小模型/真实tokenizer fixture，在当前真实Transformers 36层BF16 CPU路径上增加R3场景。没有预训练权重或GPU。2个窗、1个有speech窗，R2/R3的main均9次forward，smoke均11次；R3 UNKNOWN前缀action回退与当次native margin逐值相同，第二窗resolved action正常应用且actor/target归一。R3 speech整个分支记录与R2逐值相同，source对象不变。11个regex边界检查通过；predicate、question、token、fallback-margin、revision、normalized-record六类破坏在main/smoke各被拒绝，共12次。
- `source_oracle.py / source_summary.json / source_oracle.log`：实际读取完整333 B来源JSON，独立用字符边界规则而非生产regex构造归一结果，逐窗比较R3及来源不变；独立重建旧R1/R2条件和question拼接。HateMM3768窗，visual应用1788/回退1980，归一2616字段；HateClipSeg3591窗，应用1675/回退1916，归一2356字段。全部7359窗的R3 speech与R2 question/record一致。这里使用固定body检查算法兼容，包含原生无speech窗；不是实际历史query-token重放，计数不能与作者available-branch计数混用。
- 作者 `runs/20261005_m1_program/resolved_cpu_checks/summary.json` 已PASS：333视频、实际原R1/R2 source-token trace重放分别14414/13464次，原source不变；speech应用1/3、回退3438/3138。本审稿人读脚本与summary，没有另开其历史prediction文件。上述实际来源归一/visual计数与独立oracle完全一致。

新读取question的actual8B GPU parity、完整运行数值和性能仍待后续fixed5/full333。随机缩小CPU结构确认本次读取、fallback与缓存路径，不能替代真实模型结果；本次没有新增generation机制，因此未重跑既有源生成器审查。
