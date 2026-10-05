# 候选31：目标条件化空间搜索与时间绑定裁剪记忆

当前仅完成来源阅读及事前声明，代码/GPU/GT未开始。原九候选池rank4，
一次独立proposal PASS：`docs/reviews/20261005_m1_ideation_jury.md`。
本项延续同一自主目标，不重开方案审查，不重启Explorer的时间搜索预算。

来源为[V*](https://arxiv.org/html/2312.14135v2)的目标提出、区域搜索和视觉工作记忆；
已实际读§3.1、§3.3、Algorithm1/2、AppendixA.4及官方
`https://github.com/penghao-wu/vstar`的README、`visual_search.py`。
可读源码/正文/URL目录在`runs/20261005_m1_spatial_search/source_reading/`。
原文使用经过训练的定位/线索解码器、热图、置信阈值和搜索失败后的宽松救回。
本项目用同一冻结Qwen的文本框和离散区域顺序实现功能适配，不迁移训练、
额外编码器、热图祖先加权或低置信救回，不声称作者模型/数值算法复现。

## 事前机制和常数

唯一参数/提示声明为`spec.json`，两语料相同；seed0、4fps统一评测、8秒窗、
原生请求20帧（实际18–20）、完整ASR、原pixel上限、nativeG/自己的hardstance/S
及max/r6固定。没有仇恨标签、数值置信门或按语料分支。

1. 复用共享实际PTS抽取器，每窗取1/3与2/3时刻的实际解码原帧；只有实际PTS
   在当前半开窗内的帧可作LOCAL。无可用LOCAL帧则原生视觉读取，元数据保留
   缺失，不把最近但窗外的帧伪装成当前发生。日期/脚本/原视频路径/PTS/shape
   记入`data/temporal_spatial_search/PROVENANCE.md`及各视频metadata。
2. 同一Qwen用当前最多两张原帧、完整ASR及窗口正文提出最多一个需要看清的
   可观察目标，128token上限；SEARCH/NONE/UNKNOWN。目标最多12词/32内容token，
   只用于搜索条件，不是观察事实。没有目标则新鲜原生V，不复用历史分数。
3. 建立实际像素节点优先队列。先观察所选原图根节点；随后最多再观察一个节点，
   整窗最多两次96token搜索。输出FOUND框、TARGET_CUE、CONTEXT_CUE或UNKNOWN。
   未找到时以目标线索/场景线索的四子区域顺序指导搜索，而不是穷举全部区域。
   同顺序用最早节点序号；搜索预算耗尽只代表UNKNOWN，不能说对象不存在。
4. 分割遵循官方实际方向：宽>=2高为4x1，高>=2宽为1x4，其余2x2；整数floor
   stride、最后格补剩余像素，小轴<4不再分割。框采用本实现固定0–1000整数
   坐标且面积为正，逐层映射到原图：左/上floor、右/下ceil，记录整个父链。
   FOUND后从原像素直接裁剪PNG，实际像素数组必须与原图对应切片逐值相同。
   不上色、不生成图像、不用置信分数当输出。
5. 视觉工作记忆包含所选原LOCAL整帧、找到的真实裁剪、原图框位置和实际PTS，
   搜索目标明确标为请求而非事实。最终普通nativeV问题在原nativeG/ownstance
   cache后读取这一原图/裁剪图源；显式新image网格/三轴位置/DeepStack与
   当前logical继续位置，所有新key/cache在该窗结束恢复。S仍为原生独立读取。
   不跨窗复制裁决，不另生成整视频G，不组合多个模型。

目标是发现原低分辨率视图看不清、但当前时间位置可观察的细节；不是更高
像素预算本身。两个搜索节点预算、类别线索替代训练热图及无低置信救回均为
目标适配，效用未知；结构有效、FOUND或框面积并不证明定位正确。

## 成本、检查与分流

W为窗数，D<=W为有LOCAL帧的检测窗，A<=D为实际搜索窗，F<=A为找到目标窗。
新增D次检测及最多2A次搜索生成；找到时普通一次V读取新增原图/裁剪两个
视觉输入，保持一条native共享cache。没有新编码器/OCR/embedding模型。
额外读取、原始视频解码、坐标/队列/裁剪、全部图像编码和模型生成均计入
新视频成本；缓存复用仍计原始获取成本。预计333额外45–140GPU分钟仅原池
未测估计，全触发可能更长。7359窗的最大源生成次数22077、输出token预算
2354880；这是上限而非实际用量。必须报告检测/搜索/FOUND比例、各阶段实耗、
所有模型/vision调用、峰值显存和Slurm墙钟，不用“离线”掩盖新视频成本。

先实现并做完整333原视频/PTS/输入预检和实际36层科学CPU检查，独立一次
Rule6 code review；实际fixed5检查完整native读数/G/S、解析/坐标链/像素来源、
真实搜索反馈、缓存/clone/新V，再完整333统一评测。接口错误修复重跑，不
以未执行机制评价idea；不强迫出现SEARCH/FOUND，不隐去UNKNOWN。

完整主门通过后才跑机制控制：相同裁剪数/实际像素和token预算的固定空间
位置；根据初观察一次性规划、后续不使用新观察的匹配调用控制；相同真实
裁剪去位置关系。另在同视频保持完全相同裁剪像素/大小/数量但错绑时间窗，
以及错绑原图框位置。每个主张部件都需同指标双语料>=.01删除损失，否则
删除或降级；单纯新增LOCAL帧/更多像素的贡献必须单独报告。

不读取GT/预测提出本项；未来所有结果development-selected。规则9无任一主
指标+.01归档，有则真实error analysis后最多三次修订；性能与机制共同过门
才结束用户目标。当前原型、独立代码审查及真实运行都仍待完成。

原型已开始：geometry/有界target与FOUND/cue生成/真实搜索控制器及新图像suffix接口已写，尚未训练或GPU。执行前同两语料声明空文本及左去空白、大小写无关leading UNKNOWN word为不可用target/cue，记录UNKNOWN；不是事实正确保证，也不强迫搜索成功。来源抽取和完整输入/生成/像素链重放已实现，reader/科学CPU/唯一独立代码审查待完成。

## Prototype and required checks2026-10-05

完整source/reader/统一评测orchestration已实现；共享source_image_branch显式绑定newimage suffix的实际grid/DeepStack/三轴逻辑继续位置，并保持nativeG/ownstance/S。真实processor fulltoken seam/位置及实际PNG像素/奇数和极端比例分割检查PASS：`runs/20261005_m1_spatial_search/input_cpu_checks/summary.json`。完整333 rawvideoheader/原JPEG/ASR/原生prefix输入预检PASS：`full_input_preflight/summary.json`。真实36层FP32/BF16×18/20帧CPU source-image执行/clone/全部KV/native replay/像素变化进入分数PASS：`model_cpu_checks/summary.json`；fulluncached参考FP32差4.17e-7/5.96e-8，BF16 .005228/.006696，在事前1e-4/.025容差内，位置与缓存clone要求逐值相同。该数值检查没有预训练权重或真实GT，不是科学性能证据。

唯一独立Rule6审查`docs/reviews/20261005_m1_spatial_search_code.md` PASS，来源`runs/20261005_m1_spatial_search/independent_code_review/`。独立实际36层生产reader/vision和模型计数/成本/全部缓存/像素扰动、真实processor接缝、合成视频实际PTS解码→cue队列→第二节点FOUND/UNKNOWN/预算耗尽及生产input/token/metadata重放通过；篡改crop像素被拒绝。独立BF16 fullreference差.011458/.002761，保留原.025容差；没有CUDA/预训练权重/真实GT/预测/指标，实例与作者不同。实际8Bfixed5待派发，无性能结论。

实际8B固定5已在sc474399提交Slurm184。lab2完整333原视频逐项实际resolve检查通过；新增launch仅换目标分区和该机已有HateVLM激活路径，科研代码/spec不变。派发前四实验室代码一致/clean、旧foreign STRAY逐行未变，lab2空闲/568Gfree，来源`runs/20261005_m1_spatial_search/machines_before_smoke{,_note}.txt`；正常等待Slurm168/181占用的用户GPU预算。尚无本变体实际GPU或GT/性能结果。


空间搜索31固定5/sc474399/Slurm184于22:15:48 DONE，runs及sourcecache均已立即回传，本机samebackend noGT prepare PASS：native allraw/G/S/source pixels/coords及5clone精确，HMM/HCS真实FOUND和变化V为78/54；来源`runs/20261005_m1_spatial_search/r1_full_smoke_analysis/plumbing_summary.json`。计划在空闲sc474398完整重算333全部来源与读取，不复用lab2五视频source以保证本轮整体同机；172缺失原视频正由本机补齐，原帧/spec不变。原核验185仍PENDING/0秒，因较低调用空间搜索优先而取消待重新排入，不是idea裁定或目标暂停。


空间搜索31完整333已在sc474398提交Slurm192：172缺失HMM原视频已从本机补齐，全部333实际resolve/PyAV首帧PTS与shape验证PASS，不复用lab2固定5来源、完整来源与读取本轮同lab3。来源`runs/20261005_m1_spatial_search/{lab3_raw_coverage.json,machines_before_main.txt,machines_before_main_note.txt}`，四实验室0b6d7c5 clean/synced、旧foreign STRAY逐行未变。原独立核验27已在sc474397重新提交完整333 Slurm193，192优先入队；之前185取消时实际0秒，未改数据/spec或科研版本。两项正常等待168/186预算。


2026-10-06：尚未启动的Slurm192由主agent暂持有，先取得201/202/205/206固定5实际8B反馈；运行中的168/186不受影响。仅调度顺序，未GPU分配、不改科学版本/预算/来源成本；稍后由主agent释放本次实验室hold。来源`runs/20261005_m1_spatial_search/scheduling/hold_before_start.json`。


短201/202/205/206反馈全部回传后，原Slurm192的own实验室hold已释放并在sc474398完整333运行，source获取及paired reader同机。source/方法/缓存/修订次数未改；本机机器检查`runs/20261005_m1_spatial_search/machines_before_release{,_note}.txt`通过。195.46s/158窗smoke新处理低于MERIT334.08s，故先恢复本候选完整主轮，非性能优劣裁定。来源`scheduling/released_original_job.json`。

Full333 source complete onsc474398/192, paired125 atomic wholevideo records thenFAILED2:45:02/1:0: actualmodel imageprocessor rejected crop aspect366 (>intrinsic200). Allsource/current inputs andpartialrun BOTHreturned ROOT; oldlab2fixed5 inputcache retained separately data/temporal_spatial_search_smoke_sc474399 before fullreturn, no images/metadata/bounds changed. Source/model-support narrow diagnosis underway; noGT/evaluator/performance/idea verdict. Require minimally correct unsupported-source handling and independentnarrow confirmation before sameR1 resume; budgets/constants/source costs retained.
