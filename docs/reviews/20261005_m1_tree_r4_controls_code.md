# Tree R4 controls：独立窄差异代码确认

2026-10-05；独立实例 `/root/latents_code`，gpt-6-astra，与作者不同实例；**same-family provisional**。**PASS（R4控制适配代码与CPU范围）**。本轮未发现观察有效性阻塞bug，无生产修复请求。

审查README最后R4 controls adaptation，control_extract revision/cache_root/binding、control_measure七臂R4路径及bundle校验、control_analyze revision/raw parity/cost/scope和两个launch；包含最后新增的visual_images/speech_images统计。主方法measure.py未改，沿用已审R4 local_content/local_visual/current-input validator；不重审既有几何或方案。

确认如下：

- 输入变换算法/CONTROL_VERSION不变；新获取写入独立r4 control cache并保存control_run_revision。validate_control同时核声明revision与当前folder==cache_root(revision)/dataset/video，不把r3缓存静默当r4使用。合成fixture只写runs并patch cache_root，无假caption进入data。
- R4七臂从同一恢复后的native G/own stance cache独立clone；完全不调用extend_observations，extension为None、extension_seconds为0。保存的observation字段不进入模型。各臂实际packet分别进入V与可用S，后接各自原生question；空S缺失。nested visual/speech traces绑定各臂source路径、当前packet、token与图像计数。
- 各臂生产调用`3+W+S`，joint是一次native reference `3+W+S` 加七臂各自`W+S`及smoke诊断。首V、首可用S分别fresh/clone；cache复制、诊断与physical joint时间分列。成本保留全部main acquisition及此前已审的实际每臂额外witness/caption/priority归因。新visual_images/speech_images逐packet核对，R4只对实际S计图像，默认R3 speech_images为0。
- resume/prepare把revision传至cache、bundle及当前local-input绑定；R4 main/native exact对照对应r4完整主结果，不借用r3评分。分析输出、decoded、rawparity、report参考与启动门均按同一revision分离。canonical evaluator/fixedr6参数未改变；wrong-link范围明确R4无shared tree inventory，但native overview/ASR可见。
- 两个launch默认r3、显式接受r4；Slurm入口读取指定revision完整主summary，必须performance_pass=True且eligible84/99才进入任何控制抽取/reader。该门不进入方法计算。代码PASS不等于运行授权。

独立实际CPU证据在 `runs/20261004_m1_tree/r4_controls_code_review/`：

1. `oracle.py/log`：实际缓存Qwen CPU AutoProcessor和原native/witness图像，模型/cache/3维positions为明确stub；实际执行生产read_controls、local_visual、standard及validate_bundle。main/smoke × none/first/middle speech六场景的joint forward为27/34、35/49、35/49；七臂无extension、native cache不变、成本和最终新增V/S图像检查PASS。same-length token、r3 source path、错revision、非法extension、speech availability、current packet、global、cost、count九类损坏拒绝；forward修改KV后注入异常仍恢复长度/delta。**R4_CONTROLS_CPU_PASS**。
2. `revision_oracle.py/log`：runs下显式PIL fixture验证r4当前folder/revision正确、wrong flag拒绝、default r3拒绝r4、即使flag匹配仍拒绝错误folder。拦截未执行8臂16条canonical/r6 subprocess；合成summary验证r3/r4各自pass/fail四个gate，并确认读取相应revision路径。**R4_CONTROL_REVISION_CPU_PASS**。
3. `r3_default.log`：复用原独立R3七臂CPU fixture，仅补新要求的input-forward元数据，实际运行当前默认R3路径通过，维持旧extension与speech语义。**DEFAULT_R3_COMPATIBILITY_PASS**。

未修改生产代码，未读取GT、真实预测或真实性能summary，未运行GPU、提交作业或写data。实际GPU full-render/数值parity、七臂main/native exact、caption与成本仍未验证。当前控制仍仅prepared；必须等完整R4主门通过，再按固定五视频noGT smoke及完整333执行，不能由本报告替代该条件。
