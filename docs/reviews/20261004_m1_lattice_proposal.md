# Candidate22 frozen speech lattice：独立 proposal review

方案日期2026-10-04；审查完成2026-10-05（Pacific/Auckland）。**PASS（same-family provisional）**。

依据 `RESEARCH_ITERATION_RULES.md` 第4条，四种 STOP 均未成立。放行的是完整的 beam→独立词槽DAG→前驱概率attention与最长路径位置→单一speech margin，不是普通转录替换或separator prompt。候选22仍是备选；仅在当前Tree结果允许换方向后实施。本审查不评价Tree表现，不改其方案、不实施、不运行GPU、不读GT/预测、不外发消息。

## 实际阅读范围

完整读 `experiments/20261004_m1_lattice/README.md`，重读规则4与相邻ensemble条款，沿用本独立实例前次完整读取的权威 `CLAUDE.md`。读 `runs/20261004_m1_ideation/backup_source_reads/word_lattice_source_scope.json`，但不把主agent阅读当成本次独立核验。

实际在线打开 [Huang/Chen 原始PDF](https://www.csie.ntu.edu.tw/~yvchen/doc/ASRU19_LatticeSLU.pdf)，读§3.1–3.3、Eq10–13及监督实验设置；打开[官方仓库](https://github.com/MiuLab/Lattice-Transformer-SLU)，实际下载并检查 `src/lattice_utils.py` 的longest distances/conditional probabilities与 `src/modeling_openai.py` 的attention bias、position接口。原文是ASRU2019方法，后来arXiv上传日期不改变该会议来源身份。

另外重新读取 [2024 WCN方法§2.1–2.4](https://arxiv.org/html/2401.02921v1)与[MultiHateLoc §3.1](https://arxiv.org/html/2512.10408v1)。没有执行外部代码。源码文本及三轮原始web请求/响应保存在 `runs/20261004_m1_lattice/proposal_review/`。

## 实際目标任务检索

以下查询均实际执行；精确字符串和原始结果见 `web_request_01.json`、`02.json`，后续原文检查见03。

| 查询 | 结论 |
|---|---|
| `"hateful video" "lattice transformer"` | 未发现完整来源方法的目标任务使用。 |
| `"hate video" "word confusion" lattice` | 未找到对应目标工作；返回一般词汇交集不算证据。 |
| `"Lattice Transformer" "hate" Huang Chen` | 返回NER/一般lattice等非同一目标方法，不能混为已使用。 |
| `"HateMM" "ASR" "lattice"` | 未定位到本轮完整结构编码方法。 |
| `"hateful video" "speech recognition" uncertainty hypotheses`；`"hateful video" "confusion network"` | 找到转录噪声、视频推理相关工作，未见条件可达性+最长路径结构读数。 |
| `"Adapting Pretrained Transformer to Lattices" hate video` | 定位原方法及非目标任务引用，未发现迁移至hateful video的肯定证据。 |

[MultiHateLoc](https://arxiv.org/html/2512.10408v1)已有单转录Whisper句子时间戳、BERT表征与时间扩展，这约束“首次利用语音/时间文本”之类宽泛表述，但不是多词汇DAG编码。2024 [WCN论文](https://arxiv.org/html/2401.02921v1)直接把alternatives用separator表示并删除null，属于SLU背景动机；当前不借其结果证明冻结结构attention有效。检索支持“本次未发现”，不保证穷尽全部文献。

## 四项裁定

1. **来源已用于目标：未成立。** 可核对原文的来源任务是监督SLU；本次未发现完整结构方法已用于hateful video detection/localization。
2. **纯ensemble：未成立。** 同一Whisper的beam是输入假设，Qwen只执行一次结构化speech读数；不存在逐beam仇恨分数平均、投票或第二语义模型融合。原native视觉/全局按项目既有单模型流程保留。
3. **纯calibration/postprocessing：未成立。** mask、概率bias和logical positions改变Qwen内部输入测量，之后才产生margin；r6、4fps与统一评测固定。声学权重归一化不是用仇恨标签拟合分数。
4. **纯工程输入/prompt：未成立。** 核心是互斥可达性及子词最长路径结构，已提供onebest/flat控制。若实际只实现新ASR或separator文本而没有这些结构算子，则不属于此次PASS的方法。

## 来源与适配是否完整

来源定义前驱集合、条件路径概率log bias、自节点0、不可达负无穷，以及子词拆分后的最长路径位置；在监督微调GPT中验证。本方案保留这些核心算子，明确冻结Qwen与RoPE逻辑坐标、beam5独立词槽重组合、新窗口识别为适配，不承接源性能数字。[原论文](https://www.csie.ntu.edu.tw/~yvchen/doc/ASRU19_LatticeSLU.pdf)

在声明的独立词槽模型下，当前词槽query看到较早词槽某alternative的条件概率恰是该alternative质量；同一分支已发生的早期子词概率为1；同槽其它alternative为0。末尾必经question看到每个graph token的概率同样是其alternative质量。因此 README 的直接bias表不是任意用边概率冒充一般DAG条件概率，而是所声明因子化模型的特例。一般beam DAG不满足这个等式；当前明确承认重组合近似，不能转而宣传保持了原beam相关性。

epsilon没有文本KV节点是已声明设计，仍必须在每槽总质量中保留其质量；不能只对非空alternative重新归一化。后续问题中的非空节点贡献受到null质量影响，但不等于模型输出已对所有识别路径作精确贝叶斯边缘化，也不能宣称beam权重已校准。缺失音频、空识别文本、空native body的三种行为已分别定义。

当前没有阻止实施的源定义遗漏，也不需要额外新模型或标签。2019监督微调是否能迁移到冻结Qwen是需要实测的假设，不是规则4的STOP理由。

## 仅涉及观察有效性的实现核验

- 最小DAG枚举应独立核对条件概率、epsilon和各subword位置。逻辑位置与物理KV索引必须分开：同槽分支可共享位置，不可共享/覆盖物理cache entry；ordinary causal mask不能悄悄替代graph mask。
- 官方utils同时支持fwd/bwd方向，不能仅凭函数默认值复制成后继attention。本方案要的是query条件下的前驱概率；真正不可达值为负无穷。原源码另保留顺序causal mask，移植时要确认topological序列与graph bias一致。[官方代码](https://github.com/MiuLab/Lattice-Transformer-SLU/blob/master/src/lattice_utils.py)
- 独立branch分词与普通整句BPE不必相同，已声明。概率1单路退化核验必须使用同一手工token IDs/scaffold/positions的顺序forward，不能把分词差异误报为mask bug。onebest对照的具体token化方式事前固定并记录。
- HF实际generation接口应显式给出beam5/返回5/EOS及总448token上限、language/no-timestamp/抑制token设置；保存beam score语义，不能把默认截断或不同length normalization悄悄当成声明算法。重复文本合并后槽质量应守恒，零质量不得生NaN或被悄悄floor。
- 真实音频起点、截窗、结束与重采样映射需核对；实际30秒feature padding不冒充源音频。若解码存在间断sample区间，保留时间映射，不可无记录地拼接而压缩时间。
- 原native全局/视觉和reference speech须精确；新speech确实进入固定max/r6。prefix始终不可变，crop/rotary恢复独立。scaffold/query尚可见完整原ASR，README已承认不存在硬语境隔离。

这些属于后续一次规则6审查，尚未实施，不能标已通过。没有要求额外泛化审稿或运行前证明有效。

## 控制与成本

`onebest`在同样新ASR/crop上排除仅输入变好的解释，`flat`在相同token集合上质疑结构作用，二者双语料同主项≥.01是完整uncertainty机制的既定声明门。`binary`和`wrong_mass`检查质量解释，null必须参与错权；`wrong_audio_window`检查实际窗口对应，保留真实donor区间，且实际W的定义/空音频处理需在控制实现前写入配置。若只有新转录提升、结构不必要，不得称本候选机制成立。

成本已包括每8秒窗padded encoder、beam decoder、single Qwen reader、mask/positions及音频处理；不是5份语义预测。60–180GPU分钟只是待测预算。需用固定无GT smoke记录实际解码步数、graph物理token数、attention/memory与全部调用；不能因耗时高默默降beam或丢alternatives。完整结果仍按规则9分流，无任何主项+.01则归档；可能无效或可能shortcut不作为此时STOP。

最终：**PASS，作为候选21结果分流后的独立备用候选22。**
