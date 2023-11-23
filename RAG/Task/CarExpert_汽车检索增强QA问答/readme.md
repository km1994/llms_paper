# CarExpert: 汽车检索增强QA问答

> 论文名称：CarExpert: Leveraging Large Language Models for In-Car Conversational Question Answering
> 
> 论文地址：https://arxiv.org/pdf/2310.09536v1.pdf
> 
> 论文 Github地址：

## 一、论文动机

大型语言模型（LLM）通过遵循自然语言指令而无需对特定领域的任务和数据进行微调，表现出了卓越的性能。**然而，利用LLM进行特定领域的问题回答往往会产生幻觉。此外，由于缺乏对领域和预期输出的认识，LLM可能会生成不适合目标领域的错误答案。**

## 二、论文思路

论文提出了「CarExpert」，车内检索增强会话问答系统利用了LLM的不同任务。具体而言，CarExpert采用LLM来控制输入，为提取和生成回答组件提供特定领域的文档，并控制输出以确保安全和特定领域的答案。

## 三、实验结果

一项全面的实证评估显示，CarExpert在生成自然、安全和特定于汽车的答案方面优于最先进的LLM。

## 致谢

- CarExpert: Leveraging Large Language Models for In-Car Conversational Question Answering：https://arxiv.org/pdf/2310.09536v1.pdf

