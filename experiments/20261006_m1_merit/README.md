# M1候选33：四键检索与查询条件邻域过滤

2026-10-06。九项池C7/rank6一次proposal PASS：
`docs/reviews/20261005_m1_ideation_jury.md`。原型未实现，未CPU/GPU/GT。
保持同一Qwen3-VL-8B-Instruct、原native20请求帧/实际18–20、完整ASR、
nativeG/自己hardstance、8秒独立V/S、max和固定r6；两语料完全相同。

## 实际来源和功能适配

已实际读[MERIT论文](https://arxiv.org/html/2608.07663v1)3.1–3.4、4.2、AppendixA/C，
以及[官方代码](https://github.com/choi-yeeun/MERIT)的MultiKeyMemory、
MeritMemory、neighbor_filter、Qwen3Embedding wrapper和key extraction。
副本与已过滤的API文件列表在`runs/20261006_m1_merit/source_reading/`，
只记录可读name/path/downloadURL，不记录内容摘要。

原实现从30秒caption生成四键，Qwen3-Embedding-4B MaxSim匹配，top-N锚点
展开±2邻居，solver过滤并生成相关信息文本；每轮另有search/answer及query生成，
最多5轮，最终重新读真实视频帧。我们保留四键/MaxSim/邻域过滤/不足续查/
实际来源回读链，但单Qwen承担caption/embedding/filter/reader，8秒单元、两轮、
±1邻居。过滤只输出实际ID而不生成证据内容；这与原作者distilled text不同。
不是完整原配置或性能复现，也不是目标任务首次普通检索/证据结构。

假设：本窗省略对象、跨窗指代或对白承接能被实际相关邻域补充，从而改善
原始V/S局部排序；不能只依靠视频级排序或r6细胞先验的偶然变化作机制结论。
它不构建实体/话语边、聚类树、证书或事实修订，不重开25/21/26/27预算。

## 冻结R1规格（科学运行前）

单Qwen、seed0、greedy、FP32 YesNo margin。每8秒窗两张实际PTS1/3与2/3
帧，半开本窗内首个不早于目标，缺尾取最后本窗帧；真实重复帧去重。

1. 每窗一次同Qwen读两帧和本窗原ASR，生成caption与event/action、dialogue/
   mention、object/state、summary四键。128生成token；caption≤16words/32tokens，
   每键≤8words/16tokens，全部都是中性观察/检索键，不判断仇恨/身份或群体归属。
   空、leadingUNKNOWN word或字段cap使该字段不可用；整体cap使本窗四键不可用。
2. 同Qwen固定text-only wrapper：system为“Represent the supplied observable
   video content for semantic retrieval. Do not classify hate or infer unprovided
   facts.”，user为原键/查询全文；无图、无G/stance。实际tokenizer字符offset
   只pool与user内容范围重叠的非special token，最后层hidden转FP32均值，再
   L2归一（epsilon1e-12）。不pool system/role/assistant/end标记，不裁剪文本。
   空/UNKNOWN/无内容token或零norm不可检索，不以默认文本补全。
3. 初始query是固定“Observable context of this window:”加本窗原ASR和caption，
   不含仇恨判断；两者均不可用则query不可用。所有键embedding只在同视频复用。
   非本窗候选的score为四个可用键余弦最大值，FP32；不混入最终hate margin。
   top2锚点，平局按最早窗口；key平局按event/dialogue/object/summary顺序。
   不按caption相同去重（真实时间发生保留），只按原窗口ID去重。
4. 每个锚点展开±1完整窗口，剔除本窗和已见候选；按锚点rank、距离0/1、
   最早时间择最多4个远程窗口，随后时间排序展示。原ASR不截字，caption仅
   检索用；所有候选都有真实原窗ID/边界/PTS，不能凭生成内容建立新源。
5. 一次text-only过滤读current ASR/caption、query和四个候选的caption/原ASR，
   96生成token，输出SUPPORTED/INSUFFICIENT/UNKNOWN、真实候选ID的唯一子集、
   可选新query≤12words/24tokens。UNKNOWN或cap拒绝本轮选择。仅INSUFFICIENT
   且新query可用允许第二次embedding/retrieve/filter；无其他重试、无跨视频检索。
   已选真实ID先保留，再按第二轮rank/时间填入，最终union最多4远程窗；达到
   容量仍可检索但不能无限扩张。两轮所有实际候选/拒绝/不足/空源保留。
6. 最终不把caption、键或filter生成文字当事实。实际回读最终ID的两张原PTS帧
   和原ASR，当前两张LOCAL帧/原ASR始终保留。V在原nativecache/自己的stance
   下读当前2+远程≤8图的真实新image suffix；S用当前原speech question加带
   实际时间/来源的远程原ASR，不含虚构filtered info。两个新分支均独立恢复
   cache/rope。无远程时S=fresh原生S；V保留当前实际LOCAL帧，来源不足不是
   当前窗口的不存在证明。G不变，新V/S→原max/r6。

当前帧新增带来的收益单独控制；不能把LOCAL新增输入本身当四键/过滤的贡献。
未来证据可以使用（本项目离线定位），但明确其真实时间，只是context，不能
声称发生在本窗。权威cache`data/temporal_multikey_memory/`含PROVENANCE/host/
原路径/实际输入tokens/角色/参数/成本/命令/日期，不用任何内容hash或派生seed。

## 成本、检查和分流

W=7359完整窗：W次caption+四键、W–2W过滤（无候选则0），4W键及W–2W查询
embedding前向（无内容跳过），最终V/S替代原branch；原生pair读数另计。
最坏7359×128+14718×96=2354880生成token预算、≤44154 text-only embedding
prefill，另真实新图编码≤10图/V。源解码、caption、嵌入、CPU MaxSim/选择、
filter、新媒体编码和reader全计新视频成本，不以缓存或冻结藏成本。
原池粗估20–60GPUmin/100窗，若线性外推完整333约24.5–73.6GPU小时；这是
保守未测预算，不称便宜。先实际fixed5测量再记录实际估计；可复用原native/
ASR/视频，新增调用只同一模型。已有更低新增调用候选先运行。

完整333输入预检、作者实际36层/真实token pooling/source/cachedvsfull检查、
唯一独立Rule6代码审查后fixed5。实际执行guard须两语料有真实远程source/
两branch变化且native allraw/G精确；UNKNOWN真实保留，不强迫supported。
完整主门过后才做single-summary键、去query过滤、只一轮、仅当前LOCAL输入
matched控制，以及匹配远程数量/帧数/ASR长度桶的真实错源ID置换。
每项novelty删除须同主指标双语料下降≥.01；否则删或降级。原始V/S/max排序
和falsifiable source干预共同解释涨点，不以source执行次数当机制成立。
规则9无任一+.01直接归档，有则真实test error analysis后最多三修订。
所有结果development-selected；性能和机制同时成立才结束当前用户目标。


原型开始：单视频MaxSim/稳定平局/邻域优先及cap4 union、同模型text-only exactuser-token pooling wrapper、source-ID filter有界接口已实现。已知向量/空键/UNKNOWN/tie/neighborhood/cap fixtures及真实native tokenizer user-content offset/转义与IDsubset JSON检查通过，仅软件/输入检查，没有模型embedding或性能结论。源caption、两轮collector、V/S和完整sciencechecks尚待接入。


完整collector/真实ID两轮/source回读新V/S/统一评测接口已实现。完整333真实raw header/JPEG/ASR/native三轴输入PASS，实际36层FP32/BF16×18/20source-image cached/fullreference/KV/clone检查PASS；实际native tokenizer+36层最后hidden user-content pool与独立完整前向逐值相同、rope恢复/不同文本vector变化PASS。生产read_video/validate_bundle8组actual36layerCPU no_remote/remote通过，13actualforward/4vision、双branch clone、G精确、无远程speech freshfallback、原source41秒/23forwards完整计入。都是randomweights/软件输入检查，不是预训练模型性能。来源见manifest所列runs，完整sourcecollector重放及一次独立Rule6审查进行中；无GPU/GT/性能结论。


MERIT33一次独立Rule6代码审查PASS，来源`docs/reviews/20261006_m1_merit_code.md`及`runs/20261006_m1_merit/independent_code_review/`。实际合成video PTS→128tokencaption/key→MaxSim/neighbors/insufficient第二轮→sourceID union→当前inputtoken/pixel只读重放通过；真实36层CPU生产reader/current+remote/V+S/clone/KV/模型计数和source成本通过，source不伪造filter文字为事实。same-family provisional，与作者不同，未真实GT/预测分数/指标；实际8Bfixed5 ready，未GPU/性能。
