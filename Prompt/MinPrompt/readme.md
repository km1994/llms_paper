# 小样本QA问答 MINPROMPT 

> 论文名称：MINPROMPT: Graph-based Minimal Prompt Data Augmentation for Few-shot Question Answering
> 
> 论文地址:https://arxiv.org/pdf/2310.05007v1.pdf
> 
> 论文 Github地址：

## 一、论文动机

小样本问答（Few-shot QA）旨在少量训练样本的情况下，让模型给出令人满意的回答。 最新的研究进展主要依赖大型语言模型（LLM）。**尽管预训练阶段已经让LLM具备了强大的推理能力，但LLM仍需要进行微调以适应特定领域，以达到最佳结果**。

## 二、论文思路

论文建议选择信息最丰富的数据进行微调，从而提高微调过程的效率。


本文研究提出了「MinPrompt」，一个基于近似图算法和无监督问题生成的开放域QA的最小数据增强框架。 作者将原始文本转换为图形结构，以在不同的事实句子之间建立联系，然后应用图形算法来识别原始文本中最多信息所需的最小句子集。然后，根据识别的句子子集生成问答对，并在选定的句子上训练模型以获得最终模型。 实证结果表明，MinPrompt 能够以高效率实现与基线相当或更好的结果。

## 三、实验结果


## 致谢

- MINPROMPT: Graph-based Minimal Prompt Data Augmentation for Few-shot Question Answering：https://arxiv.org/pdf/2310.05007v1.pdf

