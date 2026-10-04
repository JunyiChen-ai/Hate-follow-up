# Program B prefill内存生命周期与capacity入口：独立窄确认

2026-10-05；独立实例 `/root/latents_code`，gpt-6-astra，same-family provisional。**PASS，无本次变更的未解决blocker。** 仅确认内存管理和source-only容量检查入口；实际53k-token容量仍须Slurm验证，不能据CPU结果直接进入full333。

## 实际核对范围

读取`handle_decoder.generate`、`handle_size_preflight.py`、`handle_extract --capacity-check`、`lab3_capacity.sbatch`及README对应预声明。未读取GT、实际预测或main summary，未运行GPU、计算哈希或修改生产代码。仅使用已有纯输入size审计和当前原始ASR/JPEG作CPU重放；所有测试产物位于`runs/20261005_m1_program/prefill_memory_review/`。

### 内存修改保持计算

每次fresh prefill前删除`j.generation_W32`，prefill结束后从同一冻结embedding/output权重重新执行`.float()`。prefill不使用该vocabulary副本，正式weights未改。clone最后hidden row保留其数值、切断对整段prefill输出storage的引用；传入Stream后删除局部hidden引用，防止其在后续step替换Stream.hidden后仍常驻。语法writer、tokens、positions、attention、FP32 head乘法、greedy、caps及model forward次数均未改。计时从生成开始至结束，包含head重建成本。

**实际独立测试：** `tiny_equivalence.py` / `.log` 用同一真实随机初始化两层BF16 Qwen3VLTextModel及真实Qwen tokenizer，分别执行修改前/后逻辑的两次连续fresh生成。两次输入长度不同，全部input IDs、tokens、events、selection、truncated、forward数及每次合法选择的完整FP32 logits逐元素exact。两次均15 forwards。旧版head在prefill存在标志为`[False, True]`，新版为`[False, False]`；prefill最后row持有storage从`[672,896]`字节变为`[32,32]`。这是CPU真实模型的等价/生命周期证据，不是8B GPU峰值预测。

### source-size与capacity边界

size脚本遍历固定333当前sources与全部planner chunks，使用当前literal renderer/tokenizer统计plain长度，再对plain最大的三个候选实际CPU image-encode；不读预测、GT或性能指标，不选择方法常数。最大plain候选是HateMM/non_hate_video_134 offset40。需准确区分该实现为“全部plain排名、三个候选expanded核对”，不是对每个chunk都做expanded编码。

`source_selection.py` / `.log` 独立核对审计覆盖恰好当前333 IDs，并使用当前该视频ASR/JPEG和CPU processor重建offset40：plain51537、image-expanded53137，与纯输入审计一致。第二视频最大plain为38571；容量对象由输入长度确定，不是根据得分或标签挑选。

`--capacity-check`与`--smoke`互斥，在全体manifest中精确选中该视频，处理完整视频全部chunks/窗口/module，调用相同acquire与source/generation验证。输出run为独立`r1_handles_extract_capacity`；科学source cache仍为原B目录，成功记录经相同验证及.partial原子替换，main可按相同输入绑定复用。入口没有measure/analyze、native/final分数或GT路径，不修改main覆盖集合、decoder或验证guard。容量获取记录保留自己的实际时间/主机，先前缓存保留原实际执行成本，不伪称旧缓存重新经历了新head重建。

`lab3_capacity.sbatch`使用sc474398明确local分区、1GPU/4CPU/32G，项目内日志、HateVLM，allocator设置先于Python；只执行source获取后打印CAPACITY_DONE，不内嵌下一任务。shell语法和三处Python编译检查通过。

## 后续结论边界

可按既定流程先同步并运行该source-only容量检查。必须以实际capacity DONE及原source绑定确认可执行，再考虑完整333；现有fixed5 native/clone/exercise/factual守门不得弱化。重建FP32 head会改变实际时间/分配轨迹，但不改变科学计算；新增容量/失败成本应保持独立可见。若仍OOM，应依据实际prefill错误继续诊断，不能把本次PASS当作输入截断或改变decoder的许可。
