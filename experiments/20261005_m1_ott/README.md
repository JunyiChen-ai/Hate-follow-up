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
