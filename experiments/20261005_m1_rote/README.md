# 候选29：区间 RoTE 语音来源绑定

运行主机待实际派发；当前无 GPU 或 GT 读数。该项是完整九候选池的第二顺位，
独立一次方案裁定 PASS：`docs/reviews/20261005_m1_ideation_jury.md`。
不重开泛化方案审查；实现后另一个 agent 做规则6代码审查。

来源为 [TIE 原论文](https://arxiv.org/html/2605.10543v1) §3.1–3.3、Eq4–13，
以及实际读取的官方 `pipeline/wan_video_tie.py::gen_physical_rope_for`
`sinc_form` 分支。官方代码已发布，旧候选页所记的 coming soon 是当时读取的项目页，
不能沿用为当前事实。代码来源与可读目录记录保存在
`runs/20261005_m1_rote/source_reading/`，不记录内容校验值。

原方法用于训练后的 Wan 视频生成。这里迁移完整区间积分算子到冻结 Qwen3
理解模型的原生时间子空间，不声称复现作者训练系统、模型配置或性能。
官方存在半径映射、L2归一变体；本项使用论文的未映射真实半径和平均 sinc
归一分支。Qwen 的频率取实际原生时间通道，而非另建生成模型频率。

## 事前固定的科学算法

输入仍为原生请求20帧（实际18–20）、完整原ASR、固定8秒窗；原G、自己生成
的硬Yes/No立场和视觉分支完全保持。模型、BF16/SDPA、fp32 Yes/No head、
seed0和原像素上限不变。两个语料共用 `spec.json` 的全部常数。

1. 对实际展开 image token 的原生 prefix 做 tokenizer offset 对齐。只给ASR
   原文token绑定其原segment ID和完整 `[start,end]`；时间戳文字、规则、图片、
   其它上下文不绑定。复制到 speech question body 的原词仍绑定同一原segment，
   不生成词时间。一个token若跨两个原文片段，使用覆盖到的原segment区间并集
   的最小起止，显式保存所有source IDs；不猜测具体词时刻。
2. 在原生prefix读取中，捕获每层 `k_norm` 后、RoPE前的ASR key时间通道。
   不修改原生输出。原G及立场的原生缓存继续正常建立；不重新裁定视频。
3. 从实际Qwen `apply_interleaved_mrope` 和 `[24,20,20]` 推导时间频率索引，
   并核对24个旋转对。只选这些对的第一、第二半通道，空间及其它通道保留。
   ASR `[s,e]` 的中心 `c=4(s+e)/2`、半径 `r=4(e-s)/2`；每对使用原频率theta。
   完整算子为 `sinc(theta*r) R(theta*c) / C_r`，其中
   `C_r=mean(sinc(theta*r))`，sinc采用 `sin(x)/x` 的零点极限。
   先检查完整333输入的归一项有限且正；没有clip、epsilon或隐式回退。
   复制body中的key也用这个区间编码。
4. 新独立speech读取中，每个query token的物理时间为本8秒窗中心乘4。
   从实际 `q_norm` 后、RoPE前的query得到点旋转表示。对每个ASR key，
   替换原生query/key时间通道点积为点query与完整RoTE key的点积；
   以两者之差构成加性attention bias，使用原head_dim缩放和原BF16查询dtype。
   所有36层、32个query头执行同一算子；保留原因果mask、原值、非ASR key、
   非时间通道。更高层表示会因正常因果传播改变，这不是冻结表示的假设。
5. 独立窗读取后恢复原cache长度、rope delta和source状态。无speech的窗
   仍为None。新S与原V按原max组合，进入完全不变的r6 M2–4和唯一评测器。
   不平均原新读数，不校准分数，不按语料路由；标签不进入任何打分路径。

完整频谱与来源区间绑定可能减少只因共享话题造成的错误本窗排序。
预期改善within，但必须由完整结果证明，不能把频谱变化或bias非零当成有效机制。

## 成本与验证

部署外层语言forward次数仍为 `3+W+Ws`；没有新编码器、生成调用或视频预处理。
配对实验额外Ws次原生S；每视频新增所有层的ASR时间key缓存与显式attention bias。
这可能失去融合kernel、增加显存和延迟，不能沿用论文的零开销结论。
333视频10–18GPU分钟仅原池未测估计；必须分别报告捕获、原生读取、新S、
新增bias内存和整个Slurm墙钟。原ASR/帧可复用，获取成本与当前方法相同。

作者CPU检查包括：闭式算子与实际数值积分、原生频率/旋转对、零半径极限、
非ASR bias逐值0、原因果不可见性、完整原文及复制body来源、空转写、跨片段
token、所有36层随机缩小真实Qwen模型的独立full-forward数值参考与cache恢复。
完整333纯输入数值域预检后，独立代码审查；真实8B固定五视频先验证原生全部
读数精确、G/V不变、实际新S进入读数、clone及来源映射，再完整333评测。

仅主门通过才运行完整333机制对照：区间半径零但保留物理中心；来源DoTE
边界编码；去掉时长归一；保留半径但所有中心置视频中点；保持原词和各segment
时长，循环半圈错绑segment中心、保存真实来源置换及重叠变化。
每个作为novelty的部件必须在两语料同主指标造成至少.01删除损失；否则删除或
降级为实现细节。正确来源也必须优于错绑定；没有控制读数不能宣称机制成立。

规则9：无任一主指标+.01则归档；有则做实际test error analysis后最多三次
修订。当前没有读取GT或预测来提出这个候选，后续全部结果标development-selected。

## 作者检查2026-10-05

完整333纯输入预检PASS：`runs/20261005_m1_rote/full_input_preflight/summary.json`。
6580 speech窗、最长原生prefix5829/最长speech query756，最小C_r .6102832631696301。
只读真实原ASR/JPEG/offset，没有模型权重/GPU/GT。原时间频率24对已逐值记录。
真实36层、32/8头、FP32/BF16和18/20图像的作者CPU检查PASS：
`runs/20261005_m1_rote/cpu_checks/summary.json`。闭式与数值积分误差<1e-6，
四种fixture均72次独立scalar bias oracle、最大bias差0、native全部KV/margin精确、
clone全部KV/margin精确、非ASR bias为0且原因果可见性保留，新margin实际改变。
随机fixture BF16外除原生FP32 rotary频率buffer；这恢复HF默认显式FP32频率，
不更改生产算法。初始custom attention未注册SDPA mask导致native缓存失败，已修复
为复用原SDPA mask函数；boolean mask断言及随机fixture rotary cast问题也已修。
全部初始失败日志保留，未放松科学数值门槛；尚未8B GPU执行。
唯一独立规则6代码审查已交给与作者不同的原jury实例，不重开proposal审查。

独立审查发现复制body的branch_ids/tokenization/offset来源绑定未计入standalone时间，已单列source_mapping并计入new_seconds及对应校验，base不包含该增量；审查正在窄确认。shared_prefix计时包含原ASR pre-RoPE key捕获和prefix来源绑定，部署需支付；native配对时间含此仪器开销，不能作为无仪器原生绝对延迟。actualSlurm墙钟另报，不计审计序列化为模型阶段秒数。

唯一独立规则6审查 `docs/reviews/20261005_m1_rote_code.md` PASS，source_mapping计时漏项已窄确认修复。独立实际36层CPU FP32/18帧和BF16/20帧，SDPA对手工masked-softmax最大差4.77e-7/.002061，bias对独立双精度算子4.31e-7/.003889；全KV/native恢复精确、新S变化、原因果保持。全333原ASR/7359窗复制body检查通过、6580speech；未GT/预测/metrics/CUDA/权重。证据 `runs/20261005_m1_rote/independent_code_review/`，真实8Bfixed5待派发。

真实8B fixed5在sc474399提交Slurm156；allfourlabs09913cfclean，外国STRAY原名未变未动，lab2idle569Gfree，151/155占QOS2所以正常排队。来源 `runs/20261005_m1_rote/machines_before_smoke{,_note}.txt`。尚未GPU结果/GT。

Slurm156于15:28:27完成fixed5，全部runs已回传本机。samebackend noGT prepare PASS：原生allraw/G/V逐值一致、134S变化、五次clone精确、全部来源映射重放通过。HMM/HCS新阶段9.9503/4.3633s，配对原生8.3985/4.0556s；peak18.33/17.36GiB，来源key最大73,986,048/19,049,472bytes。来源`runs/20261005_m1_rote/r1_full_smoke_analysis/plumbing_summary.json`。完整333待提交，无性能/机制结论。

Whole333 sameR1 dispatched onsc474399/Slurm157 after allfourlabs f966a2e clean, literalforeignSTRAY namesunchanged andlab2 idle569Gfree. Evidence `runs/20261005_m1_rote/machines_before_main{,_note}.txt`. No mainresult yet.

Slurm157完整333于15:52:37 DONE，全runs已回传本机 `return_main.log`，新缓存无。18:14本机samebackend严格来源/allraw prepare及统一评测已启动，主结果待定。
