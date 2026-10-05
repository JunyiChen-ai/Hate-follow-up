# M1候选32：像素跟踪的文字发生记忆

2026-10-05。原九项池C3/rank5，唯一proposal审查已PASS：
`docs/reviews/20261005_m1_ideation_jury.md`。不是已归档家族的修订预算重启。
当前仅事前规格，未实现/CPU/GPU/GT性能。后续全部development-selected。

## 来源、假设与边界

实际读[VideoAgent](https://arxiv.org/html/2403.11481v2)2.1–2.4及Algorithm1/2：
它记录物体发生、跟踪/re-ID，并用数据库查询实际出现片段。这里只借鉴发生
记忆与执行查询，不迁移其多模型工具、RT-DETR、ByteTrack或CLIP/DINOv2
ensemble，不声称完整原作者复现。此处像素文字发生/失配重观察是自研机制。

实际读[LELA](https://arxiv.org/html/2602.09637v1)3.2–3.3：目标任务已有EasyOCR、
逐帧OCR文字、与speech组合及多模态最大分数。因此不能主张首次OCR仇恨定位、
首次普通证据结构或多阶段推理。原始实际来源阅读放
`runs/20261005_m1_ideation/text_tracking_source_reading/`。

跟踪API实际查[OpenCV4.13文档](https://docs.opencv.org/4.13.0/dc/d6b/group__video__track.html)、
Shi–Tomasi及optical-flow教程。算法选定PyrLK前后向，不再保留“光流或模板”
未定选择；固定opencv-python-headless4.13.0.92，CPU纯像素，没有新训练模型。
直接urllib下载OpenCV页403，浏览工具访问成功；不声称保存了本地全文副本。

假设：全局抽样文字可能被错误延续到消失/改变之后。连续实际像素支撑及失配
重观察可以把文字事实绑定到真实发生时刻；只取本窗支持的记忆再读取一个V。
像素跟踪只支持可见性，不能证明OCR正确或文字是作者立场，原全局解释保留。

## 冻结R1规格（任何科学运行前声明）

两个语料完全相同，单Qwen3-VL-8B-Instruct/seed0/greedy/FP32 YesNo margin。
原native20请求帧（实际18–20）、完整ASR、G/自己hardstance、独立8秒V/S及
max/固定r6不变。只替换有源V的媒体/文字观察包，无源时fresh原生V，S原生。

1. 全视频按实际PTS解码，时间原点依原container start_time或首帧。4fps输入
   目标k/4取首个不早于目标的真实帧，重复index去重；另保留每8秒窗两个实际
   1/3、2/3帧（只在半开本窗，缺尾取最后本窗帧）。按原index合并排序。
   这些是真实观测，缺PTS失败、间隔超过.5秒中断，不补帧或跨缺口插值。
2. 每窗一次双帧同Qwen文字抽取，每张最多一个屏幕文字区域；384生成token，
   verbatim上限32words/64tokens，原图归一化整数框0..1000、正面积，floor左上/
   ceil右下回映。TEXT/NONE/UNKNOWN有界接口；空/leadingUNKNOWN word或整次
   达cap视UNKNOWN，无重试。不分析仇恨、身份或意图，不以ASR猜图中文字。
3. 跟踪图像RGB转GRAY，长边最多640等比例INTER_AREA。ROI内Shi–Tomasi
   max40/quality.01/minDistance3/block3；PyrLK21×21/maxLevel3/30次+.01终止/
   minEigen1e-4，forward和backward均执行。至少4且至少60%初始化角点有效，
   各点前后向L2<=1.5 tracking pixels，取有效dx/dy中位数，仅平移原区域。
   每个接受帧重新取角点，不假定语言实体重识别。出界/无角点/尺寸改变中断。
4. 当前对应crop与该发生初始crop分别resize到128×32，GRAY NCC>=.85且
   RGB均值绝对残差/255<=.12，variance epsilon1e-6；低纹理NCC不可用即失败。
   全帧64×64 RGB thumbnail均值残差/255>.30为cut，终止全部轨迹。
   任一失败先结束旧支撑（不加入失败帧），记录理由/数值，不宣称真实消失。
5. 实际失配触发同Qwen新观察：当前完整原图和旧框预测位置crop，256生成
   token；同TEXT/NONE/UNKNOWN规范，读实际当前文字和新框，不给旧OCR内容。
   每窗最多2次，按发生ID早者先，无效/到cap/超预算保持未知，无重试。
   成功文字始终新发生，绝不凭相同文本跨断点恢复旧ID；cut同样不跨镜头合并。
6. 实际anchor或重观察到达时，只有仍活跃像素轨迹、逐字内容相同且boxIoU>=.5
   才归同一发生；不活跃相同文字是新发生。同区域IoU>=.5但文字不同关闭旧
   发生，不把新事实改写到过去。最多4活跃区域；超容量结束最老发生、记录
   未知，不是可见性完整保证。发生ID是顺序counter，无内容摘要或哈希。
7. `interval_lookup`只返回有实际接受PTS落在本窗的发生，最多4条，按本窗
   首支持PTS/发生ID排序。每条提供本窗首/末实际支持原图crop（相同帧去重）、
   原pixelbox和实际PTS、verbatim及不确定性；不能叫真实onset/offset或完整
   时间覆盖。当前两个完整LOCAL帧加最多8裁剪作为source_image_branch新V
   图像suffix（原G/stance的cache下），每窗独立且恢复KV/rope。只一个margin，
   不与OCR、跟踪分数或旧V做加权，不平滑最终分数。

初始化事实来自双帧离线抽取，不声称在线因果处理；每条仍只绑定其明确的实际
源帧，不能在其首次观测前扩展时间支撑。跟踪可以跨窗连续；失败、容量和预算
都显式保留，不凭语言或内容相同跨cut/UNKNOWN补全。

## 成本、验证与去向

W窗，失配新观察C<=2W；新增W+C生成，每窗增强V替代原V。部署原生reader
3+W+Ws语言forward，另实际source自回归forwards和有源V新visionforward；
配对native extra W+Ws，clone诊断单列。最坏7359×384+14718×256=6593664
生成token预算，不把sourcecache当新视频免费输入。4fps解码/LK/失配读取/媒体
编码/CPU与GPU传输/原生及新margin都计成本，审计序列化单列完整墙钟。
初步333约60–240GPUmin另30–120CPUmin未测，文字密度/长静态片段可超估计。
复用原视频/nativeJPEG/ASR；新cache `data/temporal_text_occurrences/`，同目录
PROVENANCE包含实际生成机器、原输入、PTS、框、参数、命令、日期，无哈希。

先作者实际输入和科学CPU检查、一次独立Rule6审查、固定同5视频真实运行，
严格source像素/PTS/本窗归属/nativeallraw/G/S/clone和真实跟踪/重观察路径
验证后完整333。接口失败修复，不评价idea；不得强迫生成文字或取消UNKNOWN。

完整性能门过后才跑全量机制控制：同OCR/裁剪但最近anchor传播；同次数均匀
时刻重观察；执行本窗lookup改为同视频全相同事实（媒体预算匹配）；保持内容/
crop/数量/支持长度而循环错绑时间；保持时间而错绑实际crop。每项作为novelty
的部件须双语料同主指标删除损失>=.01；普通额外文字/像素贡献另列。性能和
机制共同成立才结束用户任务。规则9无任一+.01归档，有则实际GT error analysis
后同族最多三修订；没有新数据集、ensemble、按语料流程或输出校准。


原型开始：CPU PyrLK/前后向/固定角点/原图整数平移/初始crop外观/cut已实现，OpenCV4.13.0.92仅安装到隔离HateVLM（Torch/HF未改）。8项已知合成像素变换检查PASS，actual[3,2]位移和真实crop逐值相同、40/40角点通过，来源`runs/20261005_m1_text_tracking/tracker_cpu_checks/summary.json`。仅synthetic pixel transforms，未OCR/模型/GT/性能；来源解码、typed文字、发生控制器及新V尚待完成。原图平移以原坐标整数round位移落框、固定OpenCV CPU1thread/seed0；不是常数扫描。


2026-10-06完整source/typed文字/发生控制器/newV及canonical orchestration原型已接好。整次或单字段达cap均UNKNOWN，未完成JSONescape只关闭语法、该字段始终UNKNOWN，不作为观测；不忽略标点/引号。实际tokenizer引号/反斜杠/换行/中文/wordcap/wholecap及完整grammar replay检查PASS；已知像素/外部recognition fixture的变化不改过去支撑、cut结束、重现新ID、两次预算、真实endpoint crop检查PASS，范围明示不是OCR准确率/模型分数。完整333当前raw header/JPEG/ASR/native processor三轴输入检查PASS；实际36层FP32/BF16×18/20frame source-image cached/uncached reference/clone/所有KV/native replay和像素变化进入分数PASS，沿既有shared helper事前1e-4/.025容差，最大BF16 .006696。全部权威文件在manifest所列runs路径。独立一次Rule6审查待进行，尚未GPU/GT。


2026-10-06同一实际36层CPU科学检查扩展至来源图像2和10（本方法最大suffix），FP32/BF16×native18/20共8组全部PASS，最大cached-vs-fulluncached差FP32 6.56e-7/BF16 .003728，维持原1e-4/.025容差；完整native replay/全部KV/clone逐值相同、改变实际源像素改变margin。已将实际controller/token检查保存为可复运行脚本，未新增语料或GT/分数。


文字跟踪32一次独立Rule6代码审查PASS，来源`docs/reviews/20261006_m1_text_tracking_code.md`及`runs/20261005_m1_text_tracking/independent_code_review/`；12组actual36layer生产read_video/validate_bundle含空/首窗/次窗lookup、4occurrence/5实际图像、所有KV/nativeG/allraw/S、freshfallback/modelcalls/完整source计费通过。真实合成video PTS→跟踪/cut/gap→两次freshrepair→只读pixel/metadata replay、损坏crop拒绝、真实token转义/cap/半截escape UNKNOWN通过。same-family provisional，与作者不同，未CUDA/预训练权重/真实GT/分数。实际8Bfixed5 ready，尚无性能结论。


文字跟踪32实际8B固定5已在完整原视频及fixedopencv均齐全的sc474398提交Slurm201；派发前四实验室8ad6a25一致/clean、既有foreign STRAY逐行未变，来源`runs/20261005_m1_text_tracking/machines_before_smoke{,_note}.txt`。唯一独立代码审查及作者科学CPU/输入均PASS，正常等待168/186用户GPU预算及更早空间搜索192；尚无该候选GPU/GT/性能结果。


2026-10-06实际Slurm201固定5完成，source及runs均回传本机后才核对。Actualfixed5 BOTH returned; noGT strict source/native allraw replay reached original pixel-track execution guard FAIL; independent narrow diagnosis no observed implementation bug; interface not yet reliable, no performance verdict。来源`runs/20261005_m1_text_tracking/r1_full_smoke_analysis/`，原失败/UNKNOWN/cap未改。此轮未GT/指标，不能裁定idea优劣或算性能修订。
独立窄接口诊断`docs/reviews/20261006_m1_text_tracking_gpu_interface_diagnosis.md`：全5当前输入/原生读数/源像素只读重放通过，14个raw TEXT的13个cap被正确拒绝（4个32words、9个64tokens）；唯一有效$600框实际在地面而非字形，映射正确、下一帧0/30角点拒绝正确。没有观察到实现bug，无法把cap都归因于newline或证明combined closure logits首选。保留原guard/UNKNOWN，不salvage/换cohort/强迫TEXT；接口设计需另行明确。
