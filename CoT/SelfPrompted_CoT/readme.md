# 如何提升LLMs：Self-Prompted CoT

> 论文名称：Self-prompted Chain-of-Thought on Large Language Models for Open-domain Multi-hop Reasoning
> 
> 论文地址：https://arxiv.org/pdf/2310.13552.pdf

## 一、动机

- 开放域多跳推理（ODMR） 局限性：ODMR需要通过明确的推理步骤回答多跳问题，而不依赖于任何提供的上下文。这比有上下文的多跳问答要困难得多，因为模型不能依赖于检索相关段落；
- 链式思考（CoT） 局限性：
  - 在质量或多样性上有局限性

## 二、什么是 Self-Prompted CoT

本文提出了一种自我提示的思维链（SP-CoT）自动化框架，通过大型语言模型（LLMs）自身生成高质量多样化的思维链，用于开放域多轮推理（ODMR）。关键思想是：

- 自动化流水线生成带有多跳问题和推理链的ODMR数据集
- 自适应采样选择多样化的高质量CoTs作为示范
- 通过上下文学习从生成的CoTs中学习自我引导的推理

![](img/微信截图_20231212125612.png)


## 致谢

- 如何提升LLMs：Self-Prompted CoT https://zhuanlan.zhihu.com/p/671292071

