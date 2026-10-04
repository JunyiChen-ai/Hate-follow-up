# Lattice 六臂控制：独立代码审查

2026-10-05；独立实例 `/root/latents_code`，gpt-6-astra，与作者不同实例；**same-family provisional**。结论：**PASS（新增控制代码及CPU范围）**。一项成本续跑绑定缺口已修复并独立验收，无遗留本轮观察有效性阻塞项。

实际审查control_reader/control_measure/control_analyze、两个控制launch及README对应六臂定义，结合既有production structural/graph算子。未重审原方法或方案。未修改生产文件，未运行GPU、下载模型、读取GT/真实预测/真实性能summary或计算内容哈希。所有合成读数只存在CPU测试内存，没有写成实际数据集prediction文件或结果summary。

## 实现观察路径

- full实际调用production structural，不以预存margin代替；后续prepare必须与完整main/full和native的原曲线、窗口及原生状态逐项exact。onebest在相同head/tail之间一次tokenize前导空格+原onebest文本，调用普通serial cached margin。flat保留原DAG token IDs但使用serial positions/全因果mask；binary保留图位置和全部排除，仅有限bias置0。
- wrong_mass固定node顺序、IDs及位置，只旋转slot所有正质量（包含epsilon）；遇零质量整个slot不改，singleton不改并记录覆盖。wrong_audio_window只旋转available窗口，保留真实donor scaffold/actual interval，在原question前加真实destination指令；singleton保持相同。该额外指令的范围限制已明确，没有把错误来源伪装成目的窗内容。
- 所有新S从同一恢复后的native stance cache独立forward；完成及异常恢复KV长度/delta。native G/V保持共同，native S另作reference。每臂每available窗一次新S，smoke每臂首available额外clone一次；无available则无新S/clone。physical workload与每臂复用归因分列，原beam acquisition每臂standalone仍计入。
- native rendered head的image placeholders按实际processor image_counts展开；实际CPU重编码的prefix IDs加原Q/A必须等于保存展开IDs，且P原prefix长度与N stance-cache长度分别核对。当前ASR、源文件、beam/confusion、scaffold、token、source/destination、graph mass/位置、arm与margin逐项绑定。原子video bundle完成后才导出流；prepare再要求导出与bundle及完整expected IDs一致。
- analysis保持canonical evaluator/固定r6，full-minus-arm及paired bootstrap方向正确；shared speech显式取所有臂与native可用帧交集。结构门要求同一指标在双语料上同时胜过onebest和flat≥.01；其余部件门分列，mechanism_supported仍false。wrong mass/source实际干预覆盖与分数变化分开保存。
- Slurm启动门先检查完整主performance_pass、eligible84/99与333条输出覆盖，再进入控制reader；guard内容不进入方法计算。控制GT仅在完成prepare/parity后由评测/报告访问。

## 本轮具体修复

初版validate_bundle只核forward数量，未将保存preprocessing/input-forward/standalone成本重新绑定当前输入，可能把旧或改动成本当当前统计。现新增：prep与encoder/decoder counts精确等于当前metadata、peak至少覆盖输入；所有time分项finite/nonnegative；native time=prefix+V+reference S；每臂cost和record time均=当前prep+prefix+V+该臂S；physical joint至少覆盖所有分项读取及诊断之和。已实际验证改input cost、input calls、standalone cost均被拒绝；评分算法未改。

## 独立实际CPU证据

证据目录：`runs/20261004_m1_lattice/controls_code_review/`。

- `native_binding.py/log`：读取实际固定五视频原ASR/beam cache及本地Qwen CPU AutoProcessor，直接执行生产read_video/validate_bundle，模型与位置为明确stub、读数仅内存。五视频available为6/1/17/33/77。首视频raw rendered head 575 tokens，展开native stance cache 2375，原processor prefix P=2343；五视频全部与实际CPU编码一致。六臂calls/clone通过；12类native token/imagecount/file、当前ASR/beam、donor、mass、arm、count及三类cost损坏均拒绝。另合成无available边界不产生额外S/clone。**LATTICE_CONTROL_BINDING_CPU_PASS**。
- `tiny_controls.py/log`：随机初始化真实36层tiny Qwen3VLTextModel，BF16/SDPA/GQA；六臂实际forward及clone精确一致，每臂第二层注入异常后全部KV层和delta恢复。手算slot含epsilon `[.6,.3,.1]→[.3,.1,.6]`；零质量/singleton不变；flat完全因果串行，binary保留精确排除；donor/destination文字与真实区间正确。**TINY_LATTICE_CONTROLS_PASS**。普通serial adapter在CPU显式提供等价serial positions，不能据此声称目标HF5.15的8B数值parity。
- `evaluation_gate.py/log`：拦截未执行七输出流14条canonical/r6 subprocess并核参数；仅内存合成summary/覆盖检查四种启动门情形，false主门/错误eligible/332覆盖均拒绝。**LATTICE_EVALUATION_GATE_PASS**。

Tiny模型环境为本地Torch2.7.1/Transformers4.57.6，非目标Torch2.11/HF5.15 GPU。真实控制fixed5的native/full exact、六臂clone/实际算子、时间与完整333对照仍未运行或验证。本报告仅放行准备代码；控制GPU仍必须等完整主门通过，再固定五视频noGT检查，不能把本代码PASS当科学机制或晋级证据。
