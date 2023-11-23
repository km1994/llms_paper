# InstructTODS: 知识检索QA问答

> 论文名称：InstructTODS: Large Language Models for End-to-End Task-Oriented Dialogue Systems
> 
> 论文地址：https://arxiv.org/pdf/2310.08885v1.pdf
> 
> 论文 Github地址：https://github.com/WillyHC22/InstructTODS/

## 一、论文动机

当前，大语言模型(LLM)已用于各种自然语言处理(NLP)任务，但对于任务导向的对话系统（TODS），特别是端到端的TODS的探索仍然存在一定的局限性。

## 二、论文思路

论文提出了「InstructTODS，该框架可用于Zero-Shot端到端任务导向的对话系统，无需微调即可适应不同的领域」。通过利用LLM，InstructTODS生成代理信念状态(proxy belief state)，将用户意图无缝转换为动态查询，以便与任何知识库进行高效交互。

![](img/1.png)

## 三、实验结果

实验结果表明，InstructTODS 在引导对话成功完成方面达到了与完全微调的 TODS相当的性能，并且无需先验知识或特定任务数据。此外，对端到端 TODS 的严格人类评估表明，InstructTODS 产生的对话响应在有用性、信息性和人性方面明显优于黄金响应和最先进的TODS。此外，对TODS子任务（对话状态跟踪、意图分类和响应生成）的综合评估进一步支持了TODS中LLMs的有效性。

## 致谢

- InstructTODS: Large Language Models for End-to-End Task-Oriented Dialogue Systems：https://arxiv.org/pdf/2310.08885v1.pdf

