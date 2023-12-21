# 一种新的注意力机制-System 2 Attention（System 2 Attention (is something you might need too)）

## 一、前言

- 动机：大型语言模型(LLM)非常强大，但它们仍容易出现简单的错误，这似乎显示出弱的推理能力。例如，不相关的上下文或输入提示中固有的偏好或意见，都可能使它们产生错误判断，在后一种情况下，展现了一种称为阿谀奉承的问题，即模型与输入一致同意。
- 论文方法：论文提出了一种技术方案--System 2 Attention(S2A)，可以让LLM决定输入上下文的重要部分，来生成好的响应。实现这点的方法是：**首先诱导LLM重新生成只包含相关部分的输入上下文，然后关注重新生成的上下文以引出最终响应。**
- 论文在实验中证明，S2A可以成功重写会降低最终答案质量的上下文，因此论文的方法可以同时提高事实性并减少其响应中的阿谀奉承。
- 未来的研究仍有许多空间。在论文的实验中，采用了零样本提示来实现S2A。其他方法可以通过考虑微调、强化学习或替代提示技术(alternative prompting techniques)来进一步优化论文的方法。成功的S2A还可以压缩回标准LLM生成，例如：通过使用原始提示作为输入和最终改进的S2A响应作为目标进行微调。









## 致谢

- 【LLM/大模型】一种新的注意力机制-System 2 Attention（System 2 Attention (is something you might need too)） https://zhuanlan.zhihu.com/p/669138759