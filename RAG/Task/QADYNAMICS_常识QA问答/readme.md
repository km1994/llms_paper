# QADYNAMICS 常识QA问答

> 论文名称：QADYNAMICS: Training Dynamics-Driven Synthetic QA Diagnostic for Zero-Shot Commonsense Question Answering
> 
> 论文地址：https://arxiv.org/pdf/2310.11303v1.pdf
> 
> 论文 Github地址：https://github.com/HKUST-KnowComp/QaDynamics

## 一、论文动机

Zero-shot常识问答 (QA) 要求模型能够进行一般情况的推理。 最先进的方法一般做法是根据常识知识库 (CSKB) 构建的QA对，并对语言模型进行微调，使其能够具备更多的常识知识。但在此过程中，QA对构建过程中可能会引入来自 CSKB 的噪声，从而生成不符合预期的语法问答对，这会阻碍模型的泛化能力。

## 二、论文思路

论文提出了**「QADYNAMICS，一种用于QA诊断和改进的动态驱动框架」**。该方法分析了QA对在问答、选项两个方面上的训练动态，通过删除无信息QA对、错误标记、错误选项来简化训练检测组件。

## 三、实验结果

大量的实验证明了本文方法的有效性，仅使用33%的合成数据就超过了所有基线模型，其中包括ChatGPT等大模型。并且通过专家评估证实：该框架显着提高了 QA 合成的质量。

## 致谢

- QADYNAMICS: Training Dynamics-Driven Synthetic QA Diagnostic for Zero-Shot Commonsense Question Answering：https://arxiv.org/pdf/2310.11303v1.pdf

