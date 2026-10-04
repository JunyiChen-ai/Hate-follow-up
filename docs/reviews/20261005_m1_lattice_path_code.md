# Lattice R2 intact-path reader：独立窄代码审查

2026-10-05，独立实例 `/root/latents_code`，gpt-6-astra，与主 writer 不同实例，same-family provisional。**PASS。唯一发现的报告计数问题已修复并实际复核，无未解决的观察有效性 blocker。** 本结论不是实际8B GPU parity、性能或机制成立认证。

## 范围

核对 README 的完整 R2 预声明，以及 `path_graph.py`、`path_reader.py`、`path_measure.py`、`path_analyze.py` 和两个 path launch 入口。依赖既有同实验获取/当前 native context/record/evaluate/report helpers；不重新审查 R1 方法或 proposal。仅查看实际固定5 ASR/beam缓存/JPEG并执行CPU检查，未读取GT或实际预测，未启动GPU、下载权重、计算哈希或记录run commit ID。没有修改生产代码或生成假实际数据集预测/cache；fixture读数仅在内存。

## 核对结果

1. **整句相关性确实进入读取。** 原5 beam的FP32 softmax质量，完全相同decoded text合并质量，按mass/首次beam/text排序。非空正质量整句只调用一次tokenize，保留单个leading space；不再按槽重组。空text的epsilon质量和非空zero-mass项均保留在trace，不造token、不重归一化正质量。各path token只读取自己的causal history；其他path全部−inf。query对各path key加入该path合并质量的log，prefix/scaffold/历史query为0。只有一个最终margin，没有逐假设分类器或输出ensemble。
2. **几何与cache。** 物理KV按path序列存储，逻辑位置在每条完整路径起点复用，query从最长path结尾开始。native P与含global-Q/ownstance的N明确区分；起点为N+native delta。真实36层BF16 additive mask测试已通过。正常、clone和注入decoder异常后，所有层KV内容和rope恢复；没有跨窗口新答案积累。概率1路径及全epsilon路径均实际测试。
3. **配对与续跑。** 原native20/full-ASR/G/own-hard-stance/V/reference-S路径不变；新S仅在available窗读取，进入原max(V,S)。原获取CACHE_VERSION与新reader VERSION分开；原子per-video bundle重验当前ASR/beam/graph/token/native展开前缀/scaffold/位置/成本/诊断。source cache复用不等于免除新视频获取成本。实际reader调用数为native.calls+available+diagnostic；有可用speech的smoke只有首窗额外clone、unit graph、unit serial共3次。无available时不产生新S或诊断。
4. **评测与入口。** R2独立路径，完整333身份覆盖与native raw exact在prepare核对；evaluate/report直接引用原函数对象，未复制评测器。统一4fps evaluator及原fixed-r6参数保持。双语料全部三指标、eligible数量及paired within统计沿用原report。评分/验证不读GT；GT仅在独立postscore report。Slurm仅目标lab2运行，smoke/main参数清楚；分析先prepare再两臂evaluate再report。
5. **成本。** new standalone为原实际acquisition+native prefix+native V+new S；reference S和smoke诊断分列。当前metadata的preprocessing/encoder-decoder counts必须相等，peak下界和所有计时非负有限、standalone分量和实际forward计数重新核对。

## 唯一发现及窄修复

原 `path_analyze.prepare` 以 `not p['tokens']` 统计epsilon，会将非空但浮点underflow到zero-mass的路径也计为epsilon，与声明的“空text才是epsilon”不同。主 writer 已改为 `not p['text']`，另列 `zero_mass_paths`。独立执行生产源码的两个实际统计表达式，在“空text正质量、非空zero-mass、正常path”fixture上得到epsilon=1、zero_mass=1。该修复不改变token/mask/评分；既有R1真实zero-mass为0的说明由主writer提供，本次未重新读取完整运行结果验证。

## 可复查证据

证据目录 `runs/20261004_m1_lattice/path_code_review/`：

- `tiny_paths.py` / `.log`：独立手列access matrix，duplicate mass=.5、epsilon=.25、非空underflow=0，完整句tokenize调用与longest logical positions核对；**真实随机初始化36层Qwen3VLTextModel、BF16 SDPA/GQA**的72次层mask/dtype检查；clone和概率1普通serial读数exact，全epsilon finite，正常及异常后全部层KV内容/rope恢复。PASS。
- `binding.py` / `.log`：实际固定5的缓存beam、native ASR/JPEG与离线真实CPU processor；134个available窗分别走main/smoke编排。raw head575/1000/1254/1315/4121，真实expanded stance-cache2375/2780/3054/3115/5861。模型及位置为明确stub，读数只在内存。15类version/ASR/beam/native token/image/file/path mass/token/physical range/logical position/cost/count/diagnostic/unit IDs/P-vs-N变更均拒绝；empty available两模式不产生额外调用。PASS。
- `eval_identity.py` / `.log`：原evaluate/report函数对象identity一致；截获4次subprocess并核对canonical evaluator与fixed-r6 flags，不执行评测、不打开GT。PASS。
- `count_fix.py` / `.log`：直接执行最终生产统计表达式，确认空text与非空zero质量分别计数。PASS。

四个新Python文件编译和两个shell入口语法检查通过。CPU实际小模型使用本机HateVideo（Torch2.7.1 / Transformers4.57.6）；目标HateVLM / HF5.15 / 冻结8B仍须在Slurm固定5完成native exact、clone exact、概率1普通serial≤.01和实际cost验证。本次CPU PASS不冒充该GPU验证，也不授权提前控制实验或改变结果分流。
