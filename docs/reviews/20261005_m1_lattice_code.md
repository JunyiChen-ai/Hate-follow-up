# Candidate22 temporal speech lattice：独立代码审查

2026-10-05。**PASS（same-family provisional，独立实例）**。按规则6检查观察有效性；两项具体问题已修复确认，无剩余代码 blocker。候选22仍为备用，只有Tree结果分流允许切换后才可进行固定五视频 no-GT Slurm smoke。本结论不宣布启用、真实模型数值parity、性能或机制成立。

## 阅读与证据

完整读取实验 README、audio/lattice/extract/reader/measure/analyze/selfcheck、launch脚本、独立proposal review及新增 `src/stance_cache.py`。核对共享Judge/评测接口及保存的目标Whisper接口源码；另只读SSH检查目标lab2 HF5.15.1的mask处理、Qwen层循环和generic beam实现。未运行GPU、读取GT、下载模型或改生产代码，无哈希操作。

独立CPU证据均在 `runs/20261004_m1_lattice/code_review/`：

- `oracle_checks.log`：实际复跑独立路径枚举，验证conditional predecessor、epsilon质量、longest-path位置、单路径mask/位置退化；重构每条原beam文本；真实delayed-MKV/PyAV重采样验证起点延迟、缺失/部分crop和不压缩内部gap。fixture输出仅改到review目录。
- `tiny_qwen_mask_checks.log`：实际tiny Qwen调用生产structural函数，确认所有两层收到同一含logmass/-inf的4D bias，logical位置确实不同于physical KV长度；复制cache读数精确，原cache长度/delta恢复。此CPU HF4.57测试使用明确prefix几何，不冒称目标8B GPU验证。
- `target_mask_beam_source.txt`：目标HF5.15.1对已有4Dmask直接返回，Qwen将同一mask传给每层；generic beam logits显式转FP32，最终按num_return_sequences返回选中beams。
- `resume_binding_checks.log`：现有graph重新构造验证拒绝当前mass变化、保存token/node mass/logical/positions变化。
- `stance_length_checks.log`：实际validate_records集成fixture区分原prefix P=4、stance cache N=10、logical start=7；正确记录通过，旧的P代N混淆被拒绝。

## 具体问题及修复确认

1. 初版续跑只验证trace graph长度，未绑定当前cache的词槽/质量。重生成不同beam后，旧score可能在同路径下被复用。现 `validate_graph_trace` 用当前confusion和实际tokenizer重构完整graph，精确核对slots、IDs、节点mass/logical/physical、path_length，并重算位置；head/tail重新分词，scaffold核对当前固定文本。当前评分、续跑与prepare均调用该校验，原ASR/版本检查保留。
2. 首次修复将trace的stance cache长度N与记录的原encode_prefix长度P比较，必然导致首个有效speech记录失败。现共享helper单独记录 `stance_cache_tokens` 与 `stance_cache_logical_start`，保留原prefix_tokens语义；两臂互相核对并用N验证trace，logical起点也与记录一致。集成fixture已通过。新增共享helper没有修改Judge、统一评测器或Tree实现。

## 核心观察路径

真实音频按presentation origin构造mono16k样本时间轴，observed区间与gap分开记录。每个native非空speech窗只截自身8秒/最终剩余时段；边缘无真实样本不填造前导音频，内部gap保持原位置填零而不压缩时间。缺音频返回None，空识别文本则保留空DAG读数，两者未混淆。音频解码、language detection和每窗识别调用均收费记录。

目标Whisper public wrapper确有重复输入并把inner返回数设为1的行为；当前显式调用同模型的 `GenerationMixin.generate`，固定start/language/transcribe/no-timestamps prefix，配置beam5/return5/448总长和默认抑制器，额外抑制timestamp IDs。generic beam实现以FP32 logits进行beam评分；保存真实五序列、EOS/截断和length-normalized scores，再FP32 softmax为槽质量。没有五份仇恨预测或平均。实际GPU返回和成本仍须smoke核验，源码检查不是模型数值验证。

词级star alignment保留插入短语、删除epsilon和原大小写/标点；每槽同文本质量相加。独立槽重组合近似已声明，不冒充原beam完整相关分布。epsilon不生成KV节点，其质量仍保留，非空节点未重新归一化；FP32零质量枝不生成token，不加floor。

较早词槽key的条件质量为该alternative质量，同枝前驱/self为1，同槽异枝及future为0，tail看全部graph keys的边缘质量。实现的logmass/0/-inf矩阵与独立枚举相符。物理token始终按拓扑序占独立KV位置，RoPE按子词最长路径共享逻辑起点；tail接在最长路径之后，不拿物理长度代替逻辑距离。

reader将dense4D FP32 additive bias与三轴logical positions直接传入单次Qwen suffix forward。目标源码不重写该mask，所有语言层使用同一bias。原prefix的cache不可变，每次finally裁回原N并恢复原delta；global/native visual只算一次并供两臂共用，新speech确实进入max与固定r6。FP32 Yes/No采用共享margin helper；没有逐beam判决、标签输入或阈值拟合。

`src/stance_cache.py` 保持原native prefix/globalQ/hard answer顺序和独立分支接口，目标HF5.15增量位置从真实KV长度加multimodal delta，helper记录的logical起点取完整prefix位置最大值+1。native配对将由prepare逐global/window/curve精确检查，并由完整r6六指标精确检查；当前没有运行这些GPU验证。

真实新视频成本包含全部ASR准备、共同prefix/visual和新speech；reference native speech与三次smoke诊断另记，实际paired forwards与复用关系有独立计数断言。clone要求精确；unit-probability使用相同手工IDs比较普通serial读出，.01仅为已声明BF16 kernel操作检查，几何另由exact oracle证明。main不含诊断额外调用。

extract/reader/scoring路径无GT读取，prepare也不读GT；只有完整postscore evaluate/report调用统一4fps evaluator和固定r6 subprocess，排序分析调用共享within helper。三指标/两个语料主门方向正确，raw speech仅对共同有效帧配对，机制标记保持False。全部机制控制仅在主门后运行，其效应尚无证据。

Slurm配置为lab2 local分区、1GPU/4CPU/32G、HateVLM；Whisper抽取进程退出后才加载Qwen，输出在仓库内。真实beam API、36层8B运行、native exact、unit诊断、峰值和完整时长仍是后续固定smoke/full的验证事项，不以CPU通过替代。
