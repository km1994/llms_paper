# Mixtral 8x7B: 稀疏专家混合语言模型

- 标题：Mixtral of Experts
- 相关领域：模型结构改进、指令微调
- 作者：Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux
-  地址：https://arxiv.org/pdf/2401.04088

这篇论文介绍了Mixtral 8x7B，一种稀疏专家混合语言模型（SMoE）。Mixtral具有与Mistral 7B相同的架构，不同之处在于每个层由8个前馈块（即专家）组成。对于每个令牌，在每个层中，路由网络选择两个专家来处理当前状态并将其输出进行组合。尽管每个令牌只能看到两个专家，但所选择的专家在每个时间步骤可以不同。结果是，每个令牌可以访问470亿个参数，但在推理过程中只使用130亿个活跃参数。Mixtral使用32k令牌的上下文尺寸进行训练，并且在所有评估基准中胜过或与Llama 2 70B和GPT-3.5相匹配。特别是，在数学、代码生成和多语言基准测试中，Mixtral远远优于Llama 2 70B。该论文还提供了一个fine-tuned的模型，Mixtral 8x7B - Instruct，在人类基准测试中超过了GPT-3.5 Turbo、Claude-2.1、Gemini Pro和Llama 2 70B - chat模型。基础模型和指令模型都是在Apache 2.0许可下发布的。



