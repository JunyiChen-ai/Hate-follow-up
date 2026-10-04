# Lattice SDPA bias dtype：窄范围独立修复确认

2026-10-05；独立实例 `/root/latents_code`，gpt-6-astra，与作者不同实例；**same-family provisional**。结论：**PASS（本次dtype接口修复及CPU验证范围）**。

仅核对 `experiments/20261004_m1_lattice/reader.py` structural 的 attention_mask 构造改为 `torch.as_tensor(bias, device=j.device, dtype=j.model.dtype)`。已读取本地回传 Slurm121 的具体失败栈：目标 SDPA 拒绝 bias/query dtype 不同。原Judge显式以BF16加载Qwen、使用SDPA；因此这行在该既定执行环境把FP32构造的图bias转换为实际BF16 query dtype。没有改图构造、节点质量/位置、输入、读出、模型或缓存恢复逻辑；不是方法修订或效果判决。

独立实际测试：`runs/20261004_m1_lattice/bias_dtype_review/tiny_bf16.py`，完整结果 `tiny_bf16.log`。随机初始化真实 **36层 tiny Qwen3VLTextModel，BF16、SDPA、GQA**，调用生产structural（含真正cache及clone），逐层截获SDPA。forward与clone合计72次attention均收到BF16 query/key/value及BF16 4D mask，逐项等于原FP32图bias的BF16转换。clone margin精确一致；正常返回以及第二层注入异常后，全部层KV恢复原5-token前缀，rope delta恢复原值。结果 **TINY_QWEN_BF16_MASK_PASS**。

该fixture的0和−∞位置转换后完全不变；有限log-bias最大绝对量化误差 **0.0027418136596679688**，对应exp质量最大相对变化 **0.002745509147644043**。逐元素误差通过BF16舍入界检查。这些只是该fixture的实际量化数，不是全输入最大界，也不宣称有限质量仍逐位精确；FP32图构造保留，量化只发生在SDPA边界。

CPU环境为Torch **2.7.1+cu128**、Transformers **4.57.6**，没有使用CUDA、下载权重、读取GT或计算内容哈希。本测试是真实tiny模型计算，不是forward stub，但**不等于目标Torch2.11/HF5.15的8B GPU数值parity**。下一步仍须在Slurm重复相同固定五视频，验证目标接口、native exact和已声明的结构clone/退化诊断；本报告不预先声称重跑完成。
