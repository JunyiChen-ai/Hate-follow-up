# M1候选35：完整视觉时间工具交互

2026-10-06，R1在本候选实际8B/GT前冻结，主机sc474397实现；运行主机待派发。
原九池C1/rank8一次方案PASS：`docs/reviews/20261005_m1_ideation_jury.md`。
独立代码审查及实际GPU未进行，不存在本候选性能数字。

## 来源与定位假设

实际读[VTimeCoT v1](https://arxiv.org/html/2510.14672v1) §3.1–3.3、Algorithm1及§4实现细节。
[官方项目页](https://vtimecot.github.io/)当前仍标Code Coming soon，未声称核验官方工具代码。
原文把进度条、高亮、真实切片和更新后的视觉记忆用于多步视频推理。
本方法保留这条功能链，单Qwen片段相关性替代VideoCLIP-XL cosine，typed工具替代自由Python，
native20替代原32初始帧，最后为逐窗V margin而非时间区间/QA答案；不是原模型数值复现。
主张限于真实媒体驱动的时间工具交互，不能只做overlay或生成动作文字就称完整迁移。
原源阅读记录`runs/20261006_m1_vtimecot/source_reading/`。

假设：当前窗和跨片段事件在同一真实时间坐标上可见，检索→高亮→真实切片→再次观察
减少整视频事件的错误本窗归属，改善原始V/max及最终within排序。保持native G/ownstance/S、max/r6。
MATCH/MAESTRO等目标任务已有检索/工具推理，不能声称首次检索、首次agent或首次grounding。

## 冻结输入、检索和工具

完整数值常数及literal prompts在`spec.json`，两语料同一流程。
唯一冻结Qwen/Qwen3-VL-8B-Instruct，greedy FP32 head、seed0；训练/适配/常数/阈值不使用标签。
最终评测仍为4fps原GT/原split，1fps只作为新增真实来源采样，绝非旧1fps协议。
对k秒目标取首个不早于目标且小于真实duration的原始PTS帧，去重；按原8s半开窗口分组，
每窗最多8帧，若实际PTS空缺导致某窗>8则按时间均匀取8真实序号；空窗明确缺失，不插值。
保存全解码PTS/shape和原RGB PNG，返回后只读重解码逐像素验证；原时间原点为container.start_time
或第一帧实际时间，与共享actual_video_frames规则一致。工具切片只取已逐像素验证的原1fps来源，
不重新编码视频、不虚构新帧；该缓存等价于对指定真实区间取原采样帧，成本完整保留。

查询：native20 overview/full ASR生成固定两字段，每字段8words/16tokens/整条96tokens，
UNKNOWN/空/字段cap不可用、重复条件去重、整条cap全拒绝，实际Q≤2；无重试/强迫非空。
每个真实8帧片段与本窗ASR构造固定source prefix，加固定assistant `Source supplied.`作为
明确协议字串（不是模型观察/答案）。视觉prefill每片段一次，两个实际查询独立suffix和
16token有界整数0–10/UNKNOWN生成，复用前缀并完整恢复KV/rope。所有条件只参与检索。
按有效整数降序、时间ID升序取最多8窗，不设正分阈值，0也仅代表最近候选、不能证明事件存在。
连续窗合成真实半开区间，UNKNOWN片段不补分数。整数从不进入最终仇恨分数。

初始memory为新增1fps真实来源中最多20帧按时间均匀选择，无重复；原生G输入不改变。
最多3个有条件动作：progress_bar（可TERMINATE/UNKNOWN）→highlight（选实际query ID）→cut
（选已高亮query的实际连续区间ID）。每步均由当前实际memory+之前实际工具结果及reason驱动，
不是后台自动执行完整三个动作；拒绝/UNKNOWN/整条或字段cap停止，不补动作。动作中性reason
16words/32tokens，整条128tokens；已执行工具和停止记录保留。
progress_bar把64px条垂直拼接到原帧下方，原RGB全部保留。条margin=min(12,width//8)，
时间映射按floor(clamp(t/duration,0,1)*(right-left))，原全视频秒坐标保持不变，cut后不重置原点。
remaining灰、已播放蓝，光标绿，highlights橙，最终当前窗红，RGB值见spec；时间文字t/duration均一位小数。
highlight真正改变当前memory条的真实区间；cut真正取被选区间的原始帧、时间均匀至最多8帧，
保留progress/highlight状态重新绘条并更新memory。空源/越界必须失败，不伪造实际执行。
动作后再生成一次96token中性反馈，看到实际最终memory；反馈、reason、queries和相关性分数
均不进入最终证据。每个实际步骤的model-visible像素、input tokens、工具状态和来源序号均记录。

最终每窗保留原全局prefix/ownstance和原视觉问题，接收本窗最多8真实LOCAL帧及最多8memory帧，
真实PTS/来源区间和图形当前窗标记；当前/最终memory重复原帧序号只保留一次、LOCAL优先。
为保留原Algorithm1的实际交互历史，另按执行顺序附每个动作前memory最多两张均匀真实帧、
当时实际条状态和执行过的typed action/真实参数。不同状态的同一原帧必须保留，因为条像素不同；
最多16最终帧+3×2历史帧=22（统一保守cap24）。原始query只作明确未核验检索请求，
reason/feedback/relevance数值不进读取。每一步的不同实际媒体及作用域进入最终V，
不是把交互结果压成一个最终静态图。该完整化在原型完成/任何GPU与GT前冻结，
属于原已审完整视觉交互功能的实现定义，没有新方法族或性能版本。
无本窗真实帧则fresh native V回退，原S逐值保持，max及固定r6不变。
图形描绘的是工具条件/真实时间，不是仇恨概率或证据认证；远程来源保留自身时间。
时间文字为固定3×5 bitmap字形、整数放大2倍，不依赖主机字体或系统font版本。
共享src/source_image_branch负责实际新图/DeepStack/位置及KV/rope恢复，生成的文字不作为事实。

## 新视频成本与完整运行

1次query生成+W个clip视觉prefill+QW个独立相关性suffix/最多16tokens+≤3个planner生成+
1次最终实际memory反馈，再W个增强V替换原V（配对实验额外保留native V）。
初始及更新memory编码/图形渲染/全视频decode/PNG IO/检索与原缓存源成本都计费；
片段视觉只在同一视频的Q条件间共享，不称离线免费处理。
原候选粗预算333个典型视频65–195 GPUmin未测量，长视频/多图模型的真实墙钟可能更长；
先fixed5测量，主实验前以实际每窗/query/tool耗时估算完整7359窗，禁止把猜测当实测。
本机原生JPEG/ASR可复用；新的来源缓存`data/temporal_time_tools/`，出处含host/date/输入路径/生成命令。
所有source及paired reader的完整333同一机器，无切片、依赖链、普通SSH GPU。

fixed5守门：各语料实际progress/highlight/cut和媒体更新至少一次，真实retrieval/newV>0，
native allraw/G/S及clone精确，完整input/pixel/token/PTS/检索/工具/成本只读重放。
不降低guard、不强迫模型执行、不把cap/UNKNOWN改成有效来源。
主门统一评测器与固定r6，215/118与within84/99完整六项，相对正式r6同指标双+.01、
其余pooled损失≤.005/within≤.01。Rule9无任一+.01归档；有信号记录实际GT error analysis后≤3修订。
所有数字development-selected，性能与机制均达标才结束当前目标。

主门通过才跑全333控制：同实际媒体集合及模型调用数的静态一次性memory（删除执行步骤角色、
把各步真实视图按原PTS合并呈现，保留每个原像素视图，不省帧；实际text token差异单列，不伪称天然精确匹配）；
去图形但保留实际切片/时间文字；
同帧数/预算均匀来源替代语义检索；循环错配高亮与实际切片来源；同长度错误真实cut区间。
每项作为novelty的部件删除都须同一主指标双语料至少下降.01，否则删除或降级。
原始V/max排序和错误绑定干预必须支持时间工具机制，不以最终r6数字或执行次数作证明。


完整prototype及作者科学CPU检查PASS：full333 raw/JPEG/ASR/native三轴预检；8组actual36layer FP32/BF16×18/20×2/24新图cached/fullreference/allKV/clone；actual10窗合成video真实1fps选帧/查询两路/三工具真正执行与更新视图/完整只读pixel/token/状态重放；实际native tokenizer/processor+36layers共享clip prefix与两个query/每个生成hidden的独立完整前向检查，FP32最大2.21e-6、BF16hidden最大.03125，在事前1e-4/.05软件容差内，所有源prefix KV逐值不变；12组actual36layer生产reader14language/5或1vision，S原调用复用计费、nativeG/allraw/clone/KV/rope及24图新V检查。都是randomweights/软件输入证据，CPU processor在sourcecache测试特意缩小到1024/2048pixel，不是真实8B/GPU/性能。初小于top8的fixture期待错误/扩展fixture宽度笔误日志保留，仅修测试，生产检索未改。来源见manifest列出的本机runs，唯一独立Rule6审查进行中。
