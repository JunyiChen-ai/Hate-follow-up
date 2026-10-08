归档原因：完整333统一六指标没有任一+.01提升，固定共同性能门未过；不跑R2或控制GPU。

# 候选30：重要性质量约束的跨帧传输表示

尚未运行GPU或读取GT。完整九候选池第三顺位，一次独立proposal PASS：
`docs/reviews/20261005_m1_ideation_jury.md`。继续相同自主目标，不重开方案审查。

完整迁移 [OTT-Vid](https://arxiv.org/html/2605.11803v1) §3.1–3.4及Appendix A，
实际读取官方 `ottvid/spatial/facility_location.py`、`temporal/ot_merge.py`、
`backbones/qwen2_5_vl.py`。原代码文件及可读URL目录保存在
`runs/20261005_m1_ott/source_reading/`。来源功能包括重要性、非均匀质量、
空间语义OT、难度预算和完整链合并/删除，不能只取OT距离当分数。

目标适配：Qwen2.5的视频输入改为当前Qwen3多图前缀及全部DeepStack；固定总保留率
.50，不扫描。保留原生G、自己生成的硬立场和S，只重新构建视觉前缀读取V。
不是作者模型配置或结果复现，不把压缩效率视为定位效果。TTF28已因完整无+.01
归档；本项沿用原九池声明的完整不同机制，不改名重启TTF。

## 事前算法和常数

全部常数两语料共享，唯一 `spec.json`。原生20请求帧（实际18–20）、完整ASR、
8秒窗、原pixel上限、Qwen3-VL-8B、BF16/SDPA/fp32 head、seed0保持。

1. 捕获正常原生视觉forward的实际最终projector、全部DeepStack及最后vision
   attention中的Q/K，原生输出不改。按实际image cu_seqlens、实际vision RoPE
   计算每张图最后层attention权重，对head/query平均，再将实际spatial_merge_size
   对应的连续4个patch saliency求和，归一为每帧merged token权重。
   不用语言层熵、标签或额外视觉模型。
2. 每帧saliency-weighted coverage贪心选择K个token，K由 .50**.7 定；
   从0覆盖开始，收益为原token重要性乘新增cos覆盖。固定最早原index平局。
   对全部原token计算保留集合的最佳/次佳cos，按最佳归属累加saliency加权
   leave-one-out损失，以负softmax温度.3产生传输质量；不可替代者质量较小。
3. 相邻帧空间语义cost使用官方实际分支：语义距离 `1-max(cos,0)`，
   位置为归一merged网格，欧氏距离除sqrt2。由原完整帧同位置cos平均确定
   `alpha=1-clip(mean_cos,0,1)/2`。Log-domain Sinkhorn epsilon.01、200轮；
   保存真实边缘残差，不在运行后改收敛次数。传输总cost用于负softmax .3
   分配 K*F-round(F*N*.50) 的总时间压缩预算，按K-1容量封顶、溢出重分配、最大余数并最早平局。
4. 每相邻对按实际raw耦合降序匹配，每source最多一次、destination可接收多个。
   官方注释存在legacy mass-normalized说法，但实际 `_greedy_match` 收raw plan；
   本项使用实际实现。cost<=.3建立后帧source到前帧destination边，较高cost
   source成为pruned root，其完整后续低cost后代也删去。解析所有有向链，
   对未被删除component做均匀FP32均值；质量用于选择而非均值加权。
5. 每root保存全部原source成员，最终projector与每层DeepStack使用同一成员均值，
   转回原BF16。按原chronological root次序呈现，root的原生完整mRoPE triple
   保留，所有ASR/政策/时间文字/image边界保留，包括没有root的image块。
   这是多图坐标适配：不会用稀疏行假造dense image_grid，也不把混合token
   宣称为只包含root时刻。成员跨时刻的失真须由错来源控制检验。
6. 明确packed positions/visual mask/DeepStack调用原语言模型，新cache length
   与logical mRoPE offset分开；追加原global question和原生自己生成的hardanswer，
   不重新裁定G。每窗新V独立读取并恢复cache/rope；原S/G+新V进入原max/r6。

## 假设、成本、检查和去向

重要性约束的完整相邻传输可能保留稀少局部证据，同时压縮背景；预计within
共同改善，必须检查原始V/max排序和最终三指标。没有假设均值安全或所有背景无用。

一个实际原生vision forward供原/新prefix共用；新增三次语言prefix阶段forward/video，
新V替代原V，原G/S保留；没有caption生成。部署调用6+W+Ws，配对另加W原V。
新增最后vision saliency捕获与实际attention矩阵、CPU coverage/LOO/Sinkhorn/链聚合、
重新prefill全部计入新视频成本。预计333单5090 15–35GPU分钟仍未测；CPU时间
也计阶段wall，不以缓存已存在为免费。审计缓存和序列化另计完整Slurm墙钟。

真实源cache放 `data/temporal_optimal_transport/` 并含PROVENANCE和主机/原输入路径；
每次当前测量仍重算视觉特征，缓存只作来源重放。先完整333纯输入网格预检、
作者科学CPU和一次独立code review；真实固定5检查全native读数、G/S精确、
所有源成员/均值/坐标/DeepStack同步、新V变化、clone、手工dense-native等价，
再完整333统一评测。出现接口错误则修复，不作idea裁定。

仅完整主门通过后跑全量控制：均匀质量；均匀时间预算；去空间cost；同最终
token数均匀时空删减；原完整未压缩。错绑定保持原特征/saliency/总预算，
循环错绑OT目的帧而root按真实时间展示；另保留完全相同均值特征/成员/root
数量，仅循环错绑root的source frame mRoPE时间身份。控制内容不能只重排
等价操作；每项novelty需两语料同主指标>=.01删除损失，否则降级或删除。
机制目标没满足不结束用户任务。

规则9：无任一指标+.01归档，有则实际GT error analysis后最多三次修订。
当前未读GT/预测来设计；所有后续性能都标development-selected。

实现前复读官方预算路径确认：目标为round(F*N*.50)，实际删除预算等于已取K*F减目标，单相邻对容量K-1；已在首个GPU或CPU算法运行前修正spec，不使用近似乘法预算。

作者真实36层CPU FP32/BF16/18–20图像全部KV/dense-native/clone/组件均值/来源坐标检查PASS，完整333实际JPEG/ASR/grid预检PASS，来源`runs/20261005_m1_ott/{cpu_checks,full_input_preflight}/summary.json`。首次独立Rule6实例在完成前遭遇服务usage-limit退出，不能记为PASS；保留partial发现，作者已单列CUDA同步saliency计时，配对base扣除这个OTT专用阶段、OTT完整保留。独立同模型继续审查待完成；未GPU/GT。

同一次独立Rule6审查由另一同模型实例接续并完成PASS：`docs/reviews/20261005_m1_ott_code.md`，前usage中断记录保留。独立实际vision float64 saliency oracle最大差7.45e-9、捕获不改变native全KV；四组36层FP32/BF16/18–20图dense-native/实际稀疏组件进入DeepStack/clone、官方实际各阶段与独立图/预算oracle PASS，成本归属修复通过。没有权重/CUDA/真实GT或预测，实际8Bfixed5尚未跑。

OTT fixed5已在sc474398提交Slurm170，167/168占用实际userGPU预算所以正常排队，不绕过调度；allfourlabs ca26ac8 clean/lab3idle1.4Tfree/literalforeignSTRAYunchanged。来源`runs/20261005_m1_ott/machines_before_smoke{,_note}.txt`，提交前两项authorCPU/唯一必要独立code均PASS，无本变体GPU结果。

Actual8B fixed5 Slurm170 DONE并立即回传；输入缓存有新增则同样回传，samebackend/source/noGT/fullnative allraw/clone检查PASS，权威`runs/20261005_m1_ott/r1_full_smoke_analysis/plumbing_summary.json`。完整333ready；无性能结果。

相同R1完整333在sc474398/Slurm180于2026-10-05 21:50:39 DONE；派发前四实验室代码一致/clean及旧foreign STRAY逐行未变，来源`runs/20261005_m1_ott/machines_before_main{,_note}.txt`。完整runs和新的审计源特征立即回传本机，回传完成后严格来源重放/统一评测；当前没有主性能或机制结论。


完整OTT sourcecache/runs已全部回传后，noGT prepare在HMMnon_hate_video_137的浮点计划记录逐值重放处失败，未进入评测/GT。保存`runs/20261005_m1_ott/r1_full_main_analysis/initial_source_replay_failure.log`和`replay_thread_fix/diagnosis.json`。原GPU production measure明确CPU4threads，审计启动因MKL_NUM_THREADS=1实际为1，差异字段为masses/contributions/transport_cost/marginal_residual，实际全部结构选择相同；改回4后整个已保存plan逐值相同。唯一修复是在prepare显式设置原production4threads，不改算法/缓存/数值容差/predictions/spec。完整333作者重放和独立窄确认进行中，确认前不读主GT或评价性能。


OTT180完整333的源重放线程问题已完成作者和独立窄确认：同原production4线程下全部333份实际保存特征重算的完整plan逐值相同，不放宽任何guard。独立来源`docs/reviews/20261005_m1_ott_replay_thread_fix.md`；全部inputs/runs已回传，noGT fullprepare/currentinputs/native allraw PASS，canonical全六项/固定r6评测恢复运行。无性能或机制结论。


## 完整333结果与归档2026-10-05

sc474398/Slurm180全333完成，全部runs和20GB左右完整原projector/DeepStack/saliency源缓存回传本机。production4thread来源逐值重放、当前输入/全部source位置/原生allraw/G/S PASS，统一评测全部原生六项精确复现。权威`runs/20261005_m1_ott/r1_full_main_decoded/optimized/metrics.json`：HMM ROC/PR/within .8983402047989129/.6986361201938487/.7565903113063817（84），HCS .713884503408459/.6694852928098559/.6346518382876711（99）。within+.00580826734041473/−.002697275516427622，无任一主指标+.01，损失在既定噪声内，共同性能门FAIL。按规则9归档第27项，不跑R2或控制GPU；全部development-selected，无机制结论。

全333确实产生多成员均值/压缩，HMM384560→192280、HCS212198→106099原视觉token，7359新V改变，G/S保留；执行不代表方法有效。实际新阶段10.64006692min/native1.17416274倍，CPU传输/选择/聚合/新prefill均计入；完整Slurm墙钟另含审计cache序列化及原/新配对读数。完整成本来源`runs/20261005_m1_ott/r1_full_main_analysis/alignment.json`，全部六项来源与决定`summary.json`。实际评测报告读取本轮fullbase/new原始及固定r6预测和两个data/gt_4fps数组用于评测/逐视频排序分析；没有据此新增设计，规则9本轮直接归档。线程修复完整失败与等价证据保留，不以舍入差异判idea成败。

**2026-10-09 disk cleanup (user-approved):** the derived cache `data/temporal_optimal_transport/` (19 GB, including its PROVENANCE.md) was deleted on sc474397 and sc474398. Run outputs and metrics under `runs/` are kept.
