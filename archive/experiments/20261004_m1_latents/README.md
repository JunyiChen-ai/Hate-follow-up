Archived 2026-10-04: full333 shows no main metric gain ≥ .01; visual ordering and final within decline. No revision/full controls.

# M1 连续视觉状态优化（候选19，R1负结果）

同一个冻结Qwen3-VL-8B先按当前原生流程读取20帧、完整ASR、全局Yes/No及独立8秒分支。
对于每个视觉分支，在原始问题的assistant header之后增加四个临时连续向量。
按问题关注的视觉patch对向量做五步对比优化，再做十五步不使用答案标签的连续状态搜索，
只用所选单个状态的最后隐藏向量读出Yes/No。原始全局和语音读数不变，r6及统一评测器固定。
不平均搜索过程中的预测，不更新模型参数。两个语料使用完全同一流程。

## 来源、假设与实际新视频成本

迁移完整两阶段方法：Visual Latents Know More Than They Say，
https://arxiv.org/html/2605.02735v1 ，已实际阅读3.2、3.3、Algorithm1、实现与消融。
来源为图像推理、非仇恨视频；目标任务使用情况由独立审查实际检索。
官方https://github.com/zhangxin-xd/Unsilencing-Latent-Reasoning已公开代码，
此前“没有实现”的初读记录已由独立方案审查纠正。其Stage I使用contextual hidden states、
反传冻结backbone并用endoftext初始化；Stage II主reward是全softmax top-k概率的负对数均值，
另有可选progression项，并保存更新后的搜索中心。官方MMVP脚本常数也与此方案不同。
本轮采用论文Eq2–6为主的独立Qwen3适配：直接优化输入embedding、使用纸面熵进展reward，
保留实际已评价的最佳candidate。它不是官方contextual优化的等价加速或精确复现。
论文未充分给全的常数均在下文事前声明，不根据运行结果追改。
独立方案审查PASS（same-family provisional）：docs/reviews/20261004_m1_latents_proposal.md。

假设：原有视觉证据可能没有被即时答案有效使用，优化中间状态可改善测量。
更大的注意力、reward、置信度本身都不证明定位改善；必须看原始排序和机制对照。
这不是Explorer第五次修订：不取得新帧、不使用采帧熵/候选选择，也不更换全局问答语境；
源算法的干预对象是每窗口的连续输入状态，不是对既有注意力或分数作重加权。

复用原有20帧、ASR和原生前缀；新视频仍需原生编码。
捕获同一Qwen视觉tower的主merger输出作为视觉输入embedding，无第二个编码器/特征模型。
每个视觉窗增加一次初始状态评价、十五次候选评价和一次最终读数，最多17次短后缀forward；
五步Stage I只在四个向量与所选patch上求导，不反传MLLM、不重复编码前缀。
Qwen问题forward正常计算实际attention，同时取得按问题平均的patch相关度。
完整333预算初估60–120 GPU分钟，峰值预期<32GiB；五视频实际smoke后替换估计。
smoke后更新：两个普通视频/语料外推合计46.3min，仅作粗略预算；另一个长视频用于资源验证。
实际峰值20.45GiB，完整成本仍等333结束后记录。五视频合计108.5秒，含诊断。
额外全词表lm_head FP32缓存约2.5GB、视觉embedding与分支缓存复制均计入内存/时间。
原生配对诊断forward单列，不把它们冒充部署成本。

## 运行前固定的算法与常数（R1，无扫描）

- 原生模型、图片像素上限、20帧、ASR、8秒窗和问题均从src共享入口取得，保持原值。
- K=4、positive=2/latent、negative=4/latent、warmup=5、search=15来自论文。
- 相关度为原生视觉问题**文本内容tokens**对全部原始视觉tokens的实际post-RoPE
  causal softmax attention，平均全部语言层/头/问题tokens。计算分母包含全部可见keys，
  不把图片间单独归一化的分布当作实际attention。排除chat header/EOS模板tokens。
- 稳定排序：相关度降序，同值按原始token顺序；top8顺序分成四组、每组2positive。
  bottom16按升序分成四组、每组4negative，与positive不重合；完全同值时先排除positive
  再按原始顺序取negative，避免稳定排序两端仍重合。
  视觉token数若不足24，明确报实现错误，不静默改K或常数。
- 四个初始向量分别为各positive组的主merger embedding均值，FP32。
  对比loss按论文Eq4，cosine similarity，tau=.1，beta=(2+4)/2=3。
  Adam优化仅这四个向量，lr=.02、betas=(.9,.999)、eps=1e-8、weight_decay=0，5步。
  每步相关度不重算：问题因果地位于latent之前，冻结前缀/问题不依赖后续latent。
  因此每步重复计算得到同一选择，复用避免无意义forward。
- 在原生视觉Q缓存之后直接插入四个向量作为连续inputs_embeds；不新建/训练special token、
  不添加说明文字或latent delimiter。位置为真实缓存长度加Qwen3的既定rotary delta，
  连续4个位置，缓存重复评价均恢复到同一问题长度及delta。
- Stage II：每个latent位置的全词表logits取top20，在这20项内softmax归一化并计算熵。
  reward=mean(max(0,E[k]-E[k+1]))，k=0,1,2；不是二元Yes/No置信度。
  seed=0，每个视频的随机流在进入视频时固定重置，按窗口顺序推进，绝不由内容派生。
  独立Stage II索引i=0,...,14，首个sigma=.01；sigma_i=.01*.95**i，alpha=.001。
  eps~N(0,sigma_i²I)，candidate=H_i+eps；
  H_next=H_i+alpha/sigma_i²*reward(candidate)*eps（论文NES更新）。
- 源Algorithm1比较candidate的reward却保存H_next，和其文字“保留最佳状态”不一致。
  明确采用文字定义：将Stage I初始状态与每个**实际已评价的candidate**比较，
  严格更高reward时保存该candidate，同reward保留较早状态。
  H_next只作下一步搜索中心；不把未经评价的H_next称为最佳。
- 17次短forward包含初始评价、15candidate评价、选中状态重新forward读数；
  原生视觉读数用于配对检查和注意力选择，不混入新最终读数。
  最后hidden按共享Judge.margins_fp32读出，语音缺失按原生None处理。

## 主门与机制对照（先声明，主门后运行完整对照）

R1完整HateMM215+HCS118。三指标并列：要求相同一项主指标在两语料均+.01，
其它pooled下降不超过.005、within不超过.01。不是把Explorer较严格的within门挪来这里，
也不按语料挑指标。所有选择与数字development-selected，GT只用于打分后的统一评测/分析。
完整native必须精确复现当前六个指标，不能通过修改评测器/GT/网格消除差异。

若主门通过：

1. initial：四个初始向量，不优化，其它位置/接口一致；区分状态优化与增加计算槽。
2. warmup：仅五步Stage I，不做Stage II；若Stage II贡献未在双语料过.01，不能单独主张必要性。
3. wrong_support：视觉patch按同视频frame顺序循环置换到半视频偏移处，保留相关度排序位置、
   positive/negative数量、优化预算和其余输入；单图视频保持原图并报告不可干预。
   frame偏移floor(N_frames/2)，非等长frame patch数将原下标j映射为
   floor(j * N_target / N_source)，明确记录源/目标frame和patch。
   对照检验正确视觉内容绑定，不声称纯时间因果效应；不得查GT选donor。
4. search_only：保留同一初始向量/15步搜索而不warmup，区分两阶段贡献。

主张的完整优化部件须满足full减initial在同一主指标双语料均≥.01；
wrong_support须显示所声称视觉绑定作用，报告全部六指标、配对视频区间、原始visual/max排序。
若只有最终r6交互而原始排序不改善，必须如实分析，不说“读到更多正确视觉证据”。
未过主门按规则9分流，有某项+.01最多三修订，无任何+.01归档继续其它候选。

## 实现验证与落位

src稳定设施只读；此目录自足原型，不import其它实验。
独立代码审查PASS（same-family provisional）：docs/reviews/20261004_m1_latents_code.md。
CPU检查通过实际SDPA原kernel输出保持、mask/GQA概率与支持内容映射、Adam/NES状态选择，
以及真实tiny Qwen3的缓存/完整forward与zero/nonzero rotary delta读数一致。
原用例的expand张量原地修改错误已修复，续跑现明确拒绝配置、重复ID或配对记录不一致。
证据runs/20261004_m1_latents/cpu_checks/与code_review/，不代替真实8B验证。
拟运行主机sc474399（uoa-lab2），Slurm五视频smoke只查实现，无GT/性能评测；
通过后完整333，同台不切片。运行命令如下：

```bash
mkdir -p runs/20261004_m1_latents
sbatch experiments/20261004_m1_latents/launch/lab2.sbatch smoke full
sbatch experiments/20261004_m1_latents/launch/lab2.sbatch main full
# 回传本机后，smoke只运行prepare；main才运行统一评测与固定r6。
python experiments/20261004_m1_latents/analyze.py --stage prepare --smoke --arm full
setsid nohup bash experiments/20261004_m1_latents/launch/run_analysis.sh full > runs/20261004_m1_latents/analysis_launcher.log 2>&1 < /dev/null &
```

这两个sbatch必须由主agent先完成smoke并检查，再单独提交main；不自动chain。
审核实际attention与SDPA的一致性、native缓存独立/positions/merger embedding对齐、
K=0退化原生精确、优化确实影响margin、无GT进入计算、统一r6/评测命令。
结果runs/20261004_m1_latents/<run>/，新输入若有才建data/出处；本版无新输入缓存。
GPU只Slurm，日志/PID/配置/可读版本说明入run。

## 实际运行记录

2026-10-04，主机sc474399，Slurm92：五视频smoke完成，回传后本机prepare通过。
覆盖5，原生global/逐窗两模态/曲线完全一致，K0重放精确；158个视觉窗改变读数，
158个槽干预改变读数，复制缓存评价与实际已评价最佳状态reward精确一致。
峰值20.45GiB，采集108.5s；预算/诊断来源
`runs/20261004_m1_latents/r1_full_smoke_analysis/plumbing_summary.json`与
`runs/20261004_m1_latents/r1_full_smoke/checks.jsonl`。
未读GT，不计算子集性能，机制尚未支持。完整333下一步提交，无性能结果。

完整采集已由主agent在smoke检查后单独提交：sc474399，Slurm93，
`sbatch experiments/20261004_m1_latents/launch/lab2.sbatch main full`。
所有333视频一起采集配对native/优化，核心计算代码和常数保持smoke已审查版本。
主门后比较器`compare_controls.py`独立补审PASS（same-family provisional），
`docs/reviews/20261004_m1_latents_controls_code.md`；实际未执行，也没有运行任何控制。
只有完整main过性能门且四控制各完整333+统一评测结束后才允许读取GT作此比较。
它检查正/负support的真实frame/patch映射，正支持embedding内容变化按窗口单列，
负支持只报告映射检查，不把未存储的内容相等检查冒充已有证据。

## 完整主实验结论与去向（2026-10-04）

主机sc474399，Slurm93完整333完成并已回传本机。
统一评测权威文件：`runs/20261004_m1_latents/r1_full_main_decoded/optimized/metrics.json`；
配对native：同目录`base/metrics.json`，精确复现当前r6全部六项。

| 语料 | ROC | PR | within | 相对当前r6变化ROC/PR/within | within有效视频 |
|---|---:|---:|---:|---|---:|
| HateMM | .894436599 | .691121134 | .727685515 | −.002682062 / −.003113469 / −.023096529 | 84 |
| HateClipSeg | .685950717 | .652591954 | .566665516 | −.030874146 / −.018480131 / −.070683598 | 99 |

没有任何主指标+.01，规则9要求直接归档换候选，不进入修订/完整控制。
全部development-selected；打分无GT。评测后分析读取两臂预测、固定r6输出与
`data/gt_4fps/{HateMM,HateClipSeg}.npz`，结果见
`runs/20261004_m1_latents/r1_full_main_analysis/{summary,per_video}.json`。
发现原始visual within两语料下降.133664/.040081，原始max下降.055583/.034882；
最终within配对95%区间HMM[−.053360,.007796]，HCS[−.110661,−.032770]。
7359个视觉窗读数确实改变，原生逐窗/global/曲线完全一致，但优化生效不等于有效定位。
分析影响的决策仅为归档此R1公式适配，继续不同候选；不回改常数或隐藏负结果。
它不是官方contextual-backbone反传算法的精确复现，本结果不排除该未测试实现。

实际新视频处理时间HateMM2026.215s（33.77min）、HCS1834.202s（30.57min），
合计64.34min，对应同次仪器化native10.24min，约6.28倍；
峰值20.17/19.89GiB，实际forward71908/68133。原生14938次，新增125103次，
即7359窗×17次；不存在新像素/新模型抽取。
来源`runs/20261004_m1_latents/r1_full_main_analysis/alignment.json`。
全采集墙钟3868.9s，诊断和输出不冒充方法核心耗时。
机制未支持，未运行任何完整控制，正式当前方法保持r6_bma。
