# Character-LLM

> 论文名称：Character-LLM: A Trainable Agent for Role-Playing
> 
> 论文地址：https://arxiv.org/abs/2310.10158

## 一、Fine-Tuning to Role

In-Context Learning vs Fine-Tuning是一个常被讨论的议题。

Fine-Tuning的迷人之处在于：**适当的数据 + 开源LLM + Fine-Tuning 有可能超过 闭源LLM + In-Context Learning。**

Character-LLM是纯基于Fine-Tuning的，并且每一个role训练一个LLM；在RoleLLM和ChatHaruhi中则是作为一种实现方案，与in-context learning方案进行对比。

## 二、Character-LLM Dialogue数据合成方法

在Character-LLM中，作者称其合成的数据为experience。

一个experience由以下两部分内容组成：

1. scene。包括地点、背景、人物等信息；
2. dialogue。在某个scene下，目标role和其他role的对话数据，其中还额外包括目标role的思考（thinking）过程。

![](img/微信截图_20231123101341.png)

experience的获取流程如下：

1. 将目标role在wiki上的信息，作为profile；
2. 使用LLM，根据profile来生成scene，其prompt示例如下；

![](img/微信截图_20231123101510.png)

3. 基于scene + profile，让LLM生成dialogue数据。其prompt示例如下，用此种方法生成的数据可以是多轮的

![](img/微信截图_20231123101549.png)

一条用于训练的experimence如下：

![](img/微信截图_20231123101615.png)

> 需要额外提到的是，Character-LLM论文提到了角色幻觉（Character Hallucination）的概念，其举的场景是：问一个古罗马的角色关于python的问题。Character-LLM认为这种情况下，角色不应该做出回答，因此额外构建了一些protective experience，引导LLM拒绝回答那些超出角色设定的问题。论文作者还提到，角色幻觉问题的重要性在于，攻击者有可能利用这种幻觉来解锁模型的全部能力，从而带来潜在的危害；但角色幻觉也有一些益处，可以让古代角色与现代知识有更多的交融。

## 三、Character-LLM 实验结果

Character-LLM的实验结果表明：

1. fine-tuned后的LLaMa 7B模型，在4个指标上都强于Alpaca、Vicuna，并与Chatgpt可以一战；
2. 在values（价值观）维度上，fine-tuned模型效果略差，作者解释这可能与模型倾向于生成更短的文本有关。

## 致谢

- [角色扮演论文浅读：RoleLLM、Character-LLM以及ChatHaruhi](https://zhuanlan.zhihu.com/p/668207639)














