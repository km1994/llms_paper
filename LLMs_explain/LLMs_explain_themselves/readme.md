# LLMs 研究 —— LLMs 自我解释性研究

> 论文名称：Can Large Language Models Explain Themselves? A Study of LLM-Generated Self-Explanations
>
> 论文地址：https://arxiv.org/pdf/2310.11207.pdf

## 一、动机

像ChatGPT这样的大型语言模型（LLM）在各种自然语言处理（NLP）任务上表现出了卓越的性能，包括情感分析、数学推理和摘要。

此外，由于这些模型是根据人类对话调整的指令，以产生“有用”的反应，它们可以而且经常会随着反应产生解释，我们称之为自我解释。

例如，在分析电影评论的情绪时，该模型不仅可以输出情绪的积极性，还可以输出解释（例如，通过在评论中列出“奇妙”和“难忘”等充满情绪的词语）。

**LLM在自我解释方面有多擅长？**

![](img/微信截图_20231027094429.png)

## 二、论文介绍

**文章评估了LLM自行生成特征归因解释的能力**。这是与LLM的可解释性相关的一个重要话题。

1. **对大型语言模型（LLMs）在自我生成特征归因解释方面的能力进行了严格评估**

> 特征归因解释：模型解释其自身预测的能力，通过说明输入中每个单词的重要性

2. **研究使用ChatGPT和SST数据集作为测试平台，构建了能可靠生成两种类型的LLM生成自我解释的提示**
3. 研究比较了这些解释与传统的解释技术（遮蔽显著性和LIME）在忠实度和一致性指标上的表现；

## 三、论文结论

1. **根据忠实度评估，无论是自动生成的解释还是其他解释，都没有明显的优势**。然而，根据一致性评估，它们之间存在很大的区别。这可能表明当前解释方法可能不是最优的，需要开发更好的方法来产生更好的自我解释；
2. 模型预测值和单词归因值都非常全面，取值如0.25，0.67，0.75等。这表明**当前的评估度量可能无法充分区分好解释和坏解释**；

## 四、论文未来研究工作

1. 需要开发更好的方式来引出自我解释；
2. 需要重新思考评估实践，可能需要采用替代评估策略，例如人类主体研究，以更好地评估解释的优劣；
3. 进一步评估其他类型的解释以及不同版本的LLMs，以深入了解这些模型如何理解自己；
4. 研究也强调了确保这些解释不被用于有害目的的重要性，并提出了可能需要解决的问题，例如解释的操纵和模型公平性；

## 致谢

- Can Large Language Models Explain Themselves? A Study of LLM-Generated Self-Explanations：https://arxiv.org/pdf/2310.11207.pdf
- 【LLM研究解读】LLM自我解释的研究  https://zhuanlan.zhihu.com/p/662109728





