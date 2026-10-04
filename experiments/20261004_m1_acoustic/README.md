# M1 声学路径分布条件化读取（候选20，R1方案）

2026-10-04事前声明；本机sc474397开发，拟在实时空闲的sc474399或sc474398 Slurm运行。
当前native Qwen3-VL-8B全局/视觉/语音与固定r6为配对基线。
只改变局部speech测量：同一Whisper-large-v3对既有ASR作字符教师强制，
用无监督head过滤形成单调路径分布，把词对窗口的支持作为Qwen attention先验。
不平均任何路径/模型的仇恨预测；每窗只产生一个新speech margin。

## 来源、边界和假设

Whisper Has an Internal Word Aligner，https://arxiv.org/html/2509.09987v1，
实际读II-A/B/C、IV-C及官方timing.py/retokenize.py：
https://github.com/30stomercury/whisper-char-alignment 。
完整来源为字符重编码+逐utterance无监督head过滤+DTW。
本轮迁移其表示和过滤，扩展DTW为明确的Gibbs路径分布/前向后向，
再进入单个语义reader；不是精确复现作者word-boundary结果，也不声称校准声学后验。
论文Eq4先平均再列norm，发布代码先每head列norm再平均，本轮使用论文定义。
官方数字num2words会改变字符/词映射，本轮不展开数字，仅移除ASCII标点但保留apostrophe，
保持原始词ID和原Qwen词文本。HF接口和下述长片切分为显式适配。
MultiHateLoc/CLARA已有语句时间对齐，单独换时间戳不构成本轮novelty。
新假设是时间不确定性应进入测量：比例切词把未确定的声学绑定当确定局部证据。
直接attention受到词支持约束，但缓存表示仍上下文相关；不主张完全信息隔离。

## R1固定算法/常数（两个语料共用，无扫描）

1. 共享load_asr(fill_untimed=True)，同native固定8秒窗、20帧、原prompt、global YesNo。
   原转录分块仅给声学局部范围，不读任何GT。按空白词保留segment+word ID/原字符串。
2. 每segment按词顺序递归中分，直至比例时间跨度≤22秒且字符重编码总token数≤400。
   空word标点移除后无字母/数字的，赋最近同segment有字词支持；若整个segment无字符则
   保留比例支持并明确记录，不能静默丢词。单词超400token则按字符token上限连续切片，
   所有片段的字符支持最终合回同一word ID；超22秒单词按字符递归中分及等比例时间切片。
   极端单字符跨度>22秒时复制同一字符到ceil(span/22)等时子块，贡献权重各1/块数，
   保留char_index并单列重复标记，总字符权重仍为1；这是明示资源处理，不丢词。
   块音频加前后4秒context并裁到[0,audio_duration]，最长30秒，16kHz单声道。
   有转录词时ffmpeg解码一次/video，音轨缺失报错误；零词视频不需音频对齐、保留原视觉。
3. 冻结现有openai/whisper-large-v3 HF模型，FP16 SDPA，复用模型权重，没有第二aligner。
   每视频首个非空块用同模型encoder+language-token logits贪心语言识别，随后统一语言。
   decoder input=[startoftranscript, language, transcribe, notimestamps]+char tokens+eot，
   chars独立encode(add_special_tokens=False)，内部space也独立token。总长度≤405<448。
   所有32×20 cross-head真实scaled QK，valid audio encoder帧数ceil(samples/320)，20ms。
   原QK沿audio轴reflect median3，然后valid帧softmax；不混入padding帧。
   head评分=sum行l2norm+sum列l2norm，只在字符rows算，逐块stable降序选top10。
   平均10个head后按列l2norm除，floor1e-12。记录实际头、每字符多token映射。
   对齐rows只使用decoder输入中字符本身的位置，不使用预测该字符的前一位置。
   官方force_align先对含special行的全矩阵评分/列norm，再取len(sot_sequence):-1，
   因而包含notimestamps行；本轮字符专属head评分/列norm/行切片为明确适配，非源码同一行约定。
4. C=平均后列norm矩阵，K×T；路径从(0,0)到(K−1,T−1)，步骤(1,0)/(0,1)/(1,1)。
   每个访问cell权重exp(C)，每次转移乘1/3，温度1。起点不乘转移。
   alpha/beta为float64 log-space DP，occupancy=exp(alpha+beta−logZ)。
   每字符token行occupancy归一化为audio-time分布，再平均同字符的多token行，
   最后平均词的非space字符；不是每帧取最大词，不使用训练/监督alignment。
   Gibbs posterior仅指该明确能量模型，不是Whisper声学概率校准保证。
   同模型energy+transition Viterbi路径另存，用作hard对照，不用它选择本轮参数。
5. 各audio帧时间为block_audio_start+(frame+.5)*.02；词对每8秒窗支持p为窗口内质量和。
   视频结束范围裁剪，再对word的窗支持归一化，保证sum_windows=1；若padding/尾帧超时
   明确归入最后窗口，不丢概率。字符切片合并按字符数加权；标点词继承规则记录。
6. 单Qwen读取：原生完整前缀/原global YesNo/原视觉不变。
   原生speech问题body改为声学support最大窗口的词，原词顺序，不改标准yesno_question。
   p的argmax同值选早窗。每窗有MAP词则读一次；无MAP词时body='(no speech)'，
   若该窗声学总词质量>1e-6仍进行一次新speech读取；完全零质量才None。
   在这一次branch所有语言层，所有因果可见key：原前缀ASR词及新body词的attention logit
   加log(max(p_window(word),1e-6))，其余key保持0。一个Qwen token跨多个词的，
   取按char offset重叠长度加权的p；空白/标点一并归所属词，跨语句词映射依字符串位置。
   原前缀ASR定位必须核对完整render+encode expanded image token offset，不靠字符串第一次出现。
   body词带原word ID，不因重复词文本去重；全局Q/A、policy、视觉、chat scaffolding不施加prior。
   所有branch query rows（含body/assistant header）应用此prior；causal mask保持真实。
   body保留原词语义，soft支持通过完整ASR key进入读取，不向Qwen输入人工概率文本。
   恢复SDPA/缓存crop/rope_delta后逐窗独立；正常Judge.margins_fp32为单个speech logit。
7. 最终z=max(native_visual,new_speech)，无speech则visual；原global不变，直接进入原固定r6。
   seed0、贪心；没有按语料分支、ensemble、拟合、标签阈值或新平滑。

## 成本与落位

复用既有20帧/ASR和两模型权重。新视频仍需解码原音频、Whisper每块1 encoder/decoder，
语言识别额外一个短decoder/video；Qwen新增一个speech forward/有声学支持窗。
原生Qwen测量可共享prefix/visual/global，但配对基线speech单列诊断成本。
先Whisper抽取完整语料、释放模型，再Qwen测量，避免同时加载两大模型。
预计两语料合计30–90min（未测）；记录decoder块数、字符tokens、audio秒、实际calls、GPU时间/峰值。
derived cache data/acoustic_path_support/{dataset}/，PROVENANCE记生成机器/命令/路径/版本说明，
不记录任何hash/Git内容pin。缓存完整test视频，每语料完整在一台机器上。
runs/20261004_m1_acoustic/存日志/PID/config/预测/统一指标；data只输入、不存评测GT衍生量。

## 门槛/事前消融

配对native必须完整333与当前r6六指标精确一致；smoke固定两个视频/语料+HMM114只验实现，
不读GT、不做子集指标。独立proposal/code审查后完整HMM215+HCS118，同组常数。
性能门：相同主指标双语料均≥.01，其余pooled loss≤.005/within loss≤.01。
失败按规则9：任何+.01允许最多三修订；无+.01直接归档；实现错误修复不评价方法。

主门通过后各完整333控制，流程/模型/attention/词映射相同：
- hard：Viterbi word分布归一化再取单一argmax窗(0/1)，body依据该hard支持；检查soft传播必要性。
- proportional：原segment按字词均分时间得到窗支持，body取同argmax，同logprior；检查真实声学作用。
- unweighted：本轮acoustic MAP body保持，attention prior全1；检查只是更准input slicing。
- shifted：每词support向窗轴固定循环floor(V/2)，词/熵/模型预算不变；body随shifted MAP改变。
  单窗无效干预单列。不能把保持全局上下文的错误时间绑定称完全因果替换。

整个posterior+reader部件须相对proportional/common native双语料同主项≥.01，
软不确定性若相对hard不达双语料同项.01则不得作novelty贡献；attention条件化若去掉不达
同项.01则不得以这一部件主张novelty，删/降级后仍须完整新方法评测与novelty判定。
报告raw speech/max within、final六项、bootstrap配对CI、shift错时及无MAP/空speech覆盖。
机制_supported只有性能+匹配消融+独立结果解释成立才可置true。
全部development-selected，GT只统一评测和打分后error analysis，输入计算无标签。

方案审查PASS（same-family provisional）：`docs/reviews/20261004_m1_acoustic_proposal.md`。
正式PASS后实施，极端单字符长跨度/零词音频规则已在首次GPU前明确补齐。

## 已做的实现检查（首个GPU前）

四项CPU验证PASS：小格路径逐条枚举与alpha/beta/logZ/Viterbi核对；
长词/重复词/标点/字符切片与token跨词映射；真实tiny Whisper捕获不改变原SDPA输出；
真实tiny Qwen neutral prior精确退化native、非均匀prior改变读数、完整4D mask/缓存重放一致。
证据`runs/20261004_m1_acoustic/cpu_checks/checks.log`。
无GT完整333 blockplan验证HMM3472块/333303char tokens，HCS2767块/273961char tokens，
所有块满足事前资源约束，实际输入无长单字符复制情况。
来源`runs/20261004_m1_acoustic/cpu_checks/block_plan.json`，不是性能评测。
独立审查指出长纯标点递归、零词序列形状和缓存续跑版本检查缺口，GPU前已修复。
CACHE_VERSION为可读来源版本，缓存/续跑核对算法、词/time身份及本轮实际support，不用hash。

运行命令：
```bash
mkdir -p runs/20261004_m1_acoustic
sbatch experiments/20261004_m1_acoustic/launch/lab2.sbatch smoke soft
# smoke回传输入与输出、本机prepare通过后才单独提交main，不自动chain。
sbatch experiments/20261004_m1_acoustic/launch/lab2.sbatch main soft
python experiments/20261004_m1_acoustic/analyze.py --stage prepare --smoke --arm soft
setsid nohup bash experiments/20261004_m1_acoustic/launch/run_analysis.sh soft > runs/20261004_m1_acoustic/analysis_launcher.log 2>&1 < /dev/null &
```

原始speech排序另报告两臂都存在语音读取的帧子集，使用统一within函数并单列有效视频数；
raw max/final仍为全部标准覆盖，不能把这个子集分析冒充标准主指标。

独立代码审查PASS（same-family provisional）：`docs/reviews/20261004_m1_acoustic_code.md`。
三项具体修复已复核，真实8B尚待Slurm smoke；不以CPU验证声称机制/性能成立。
