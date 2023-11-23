# LLMs 论文研读社

> 作者：杨夕
> 
> 介绍：该仓库主要记录 LLMs 算法工程师相关的顶会论文研读笔记
> 
> LLMs 论文研读社 地址：https://github.com/km1994/llms_paper
> 
> LLMs 九层妖塔 地址：https://github.com/km1994/LLMsNineStoryDemonTower
> 
> LLMs 千面郎君 地址：https://github.com/km1994/LLMs_interview_notes
> 
> LLMs 论文学习笔记：https://gitee.com/km601/llms_paper
> 
> NLP 百面百搭 地址：https://github.com/km1994/NLP-Interview-Notes
> 
> NLP论文学习笔记：https://github.com/km1994/nlp_paper_study
> 
> 推荐系统 百面百搭 地址：https://github.com/km1994/RES-Interview-Notes
> 
> 推荐系统论文学习笔记：https://github.com/km1994/RS_paper_study
> 
> 搜索引擎 百面百搭 地址：https://github.com/km1994/search-engine-Interview-Notes 【编写ing】
> 
> GCN 论文学习笔记：https://github.com/km1994/GCN_study
> 
> **推广搜 军火库**：https://github.com/km1994/recommendation_advertisement_search

> 手机版笔记，可以关注公众号 **【关于NLP那些你不知道的事】** 获取，并加入 【NLP && 推荐学习群】一起学习！！！

> 注：github 网页版 看起来不舒服，可以加入 **知识星球【关于AiGC那些你不知道的事】**

> **所有文章已经搬到 [【关于AiGC那些你不知道的事】](https://yaz1kaenukt.feishu.cn/mindnotes/OSsQbEhzomseronYdQmc6CmXnQH)，方便大家利用手机学习**
![](resource/pic/20230408151226.jpg)

## PEFT 系列篇

- [Prompt](PEFT/prompt/prompt.md)
  - 论文名称：
  - 论文地址：
  - Github 地址：
  - 会议：
  - 动机：但是**对于一个预训练的大语言模型来说，这就仿佛好像是对于每个任务都进行了定制化，十分不高效**。**是否存在一种方式，可以将预训练语言模型作为电源，不同的任务当作电器，仅需要根据不同的电器（任务），选择不同的插座，对于模型来说，即插入不同的任务特定的参数，就可以使得模型适配该下游任务**。
  - 论文方法：给 预训练语言模型 的一个线索/提示，帮助它可以更好的理解 人类的问题。

- [Instruction](PEFT/prompt/Instruction.md)
  - 论文名称：
  - 论文地址：
  - Github 地址：
  - 会议：
  - 动机：PLM 在 Few-Shot 上表现一般都很好，但是在 Zero-Shot 上就很一般了，一个潜在的原因是模型很难执行和预训练不一样格式的 prompt。
  - 论文方法：通过 激发语言模型的理解能力，利用给出更明显的指令/指示，让模型去理解并做出正确的action。

- [self-instruct](PEFT/Instruct/self_instruct)
  - 论文名称：Self-Instruct: Aligning Language Model with Self Generated Instructions
  - 论文地址：https://arxiv.org/abs/2212.10560
  - Github 地址：https://github.com/yizhongw/self-instruct
  - 会议：
  - 动机：**在训练好的LLM上进行“指令调优”具有很好的将Zero-shot设置下的指令理解能力泛化到新任务上的超凡能力**。然而，这种方法**很大程度上依赖于大型的语言模型以及人工编写的高指令数据，这需要极大的人力和物力**。
  - 论文方法：**通过在公开的LLM的接口上引导模型自己生成指令来提高LLM的指令跟随能力**。这在LLM时代是一种高效的蒸馏方法，即通过**从高质量的预训练好的LLM上接口获取有监督的数据，来调优模型，将大模型的知识蒸馏出来，部署到目标模型上**。

- [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](PEFT/LORA/)
  - 论文名称：LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS
  - 论文地址：
  - Github 地址：https://github.com/microsoft/LoRA
  - 会议：
  - 动机：
    - **增加adapter**： 主要问题**在于推理时带来的额外计算量和延迟**。
    - **优化prompt**： **前缀微调(Prefix Tuning)较难优化，而且随着参数量增长性能并非单调变化**。
  - 论文方法：
    1. 在原模型旁边增加一个旁路，通过低秩分解（先降维再升维）来模拟参数的更新量；
    2. 训练时，原模型固定，只训练降维矩阵A和升维矩阵B；
    3. 推理时，可将BA加到原参数上，不引入额外的推理延迟；
    4. 初始化，A采用高斯分布初始化，B初始化为全0，保证训练开始时旁路为0矩阵；
    5. 可插拔式的切换任务，当前任务W0+B1A1，将lora部分减掉，换成B2A2，即可实现任务切换；

- [DyLoRA：使用动态无搜索低秩适应的预训练模型的参数有效微调](PEFT/DyLoRA/)
  - 论文名称：DyLoRA: Parameter-Efficient Tuning of Pretrained Models using Dynamic Search-Free Low Rank Adaptation
  - 论文地址：https://arxiv.org/pdf/2210.07558v2.pdf
  - Github 地址：https://github.com/huawei-noah/KD-NLP/tree/main/DyLoRA
  - 会议：
  - 动机：LoRA存在的问题：
    - rank的值是固定的，训练完成后不能修改。
    - 优化rank的值需要大量的搜索和努力。
  - 论文方法：引入了一种**动态低秩适应（Dy-LoRA）技术**。通过**对适配器模块在训练期间的不同秩所学到的表示进行排序，为一系列的秩而不是单一的秩训练LoRA块**。

- [LOMO：利用有限的资源对大型语言模型进行全参数微调](PEFT/LOMO/)
  - 论文名称：FULL PARAMETER FINE-TUNING FOR LARGE LANGUAGE MODELS WITH LIMITED RESOURCES
  - 论文地址：https://arxiv.org/abs/2306.09782
  - Github 地址：https://github.com/OpenLMLab/LOMO
  - 会议：
  - 动机：LoRA存在的问题：
    1. 大型语言模型（LLMs）已经彻底改变了自然语言处理（NLP），但是**训练LLMs需要大量的GPU资源**;
    2. 虽然现有的方法着重于参数高效微调，即微调或添加少量参数，但**很少有人解决了有限资源下调整LLMs的全部参数的挑战**，而全参数微调被认为比参数高效微调更为强大;
  - 论文方法：提出了一种新的优化器LOw-Memory Optimization（LOMO），它**将梯度计算和参数更新融合在一步中以减少内存使用**。通过**将LOMO与现有的内存节省技术集成**，将内存使用降低到10.8％，与标准方法（DeepSpeed解决方案）相比。因此，该方法使单台机器上的65B模型的全参数微调成为可能，该机器配有8×RTX 3090，每个显存为24GB。

- [QLoRA](PEFT/QLoRA/)
  - 论文名称：FULL PARAMETER FINE-TUNING FOR LARGE LANGUAGE MODELS WITH LIMITED RESOURCES
  - 论文地址：https://arxiv.org/abs/2306.09782
  - Github 地址：https://github.com/OpenLMLab/LOMO
  - 会议：
  - 动机：LoRA微调中存在以下三个痛点：
    - **参数空间小**：LoRA中参与训练的参数量较少，解空间较小，效果相比全量微调有一定的差距；
    - **微调大模型成本高**：对于上百亿参数量的模型，LoRA微调的成本还是很高；
    - **精度损失**：针对第二点，可以采用int8或int4量化，进一步对模型基座的参数进行压缩。但是又会引发精度损失的问题，降低模型性能。
  - 论文方法：
    - **4-bit NormalFloat**：提出一种理论最优的4-bit的量化数据类型，优于当前普遍使用的FP4与Int4；
    - **Double Quantization**：相比于当前的模型量化方法，更加节省显存空间。每个参数平均节省0.37bit，对于65B的LLaMA模型，大约能节省3GB显存空间；
    - **Paged Optimizers**：使用NVIDIA统一内存来避免在处理小批量的长序列时出现的梯度检查点内存峰值；
    - **增加Adapter**：4-bit的NormalFloat与Double Quantization，节省了很多空间，但带来了性能损失，作者通过插入更多adapter来弥补这种性能损失。**在LoRA中，一般会选择在query和value的全连接层处插入adapter。而QLoRA则在所有全连接层处都插入了adapter，增加了训练参数，弥补精度带来的性能损失。**

- [VeRA：可调参数比LoRA小10倍的低秩微调方法](PEFT/VeRA/)
  - 论文名称：VeRA：Vector-based Random Matrix Adaptation
  - 论文地址：https://arxiv.org/pdf/2310.11454.pdf
  - Github 地址：
  - 会议：
  - 动机：LoRA微调中存在以下三个痛点：
    - LoRA：需要大量的可训练参数。基于Aghajanyan等人的研究，内在维度的上限比这种方法中通常使用的秩要小的多。因此，参数量可以进一步减少。
    - AdaLoRA：通过动态分配参数，从而进一步减少了可微调参数。但是，我们认为存在另一种可以显著减少可训练参数，且效果不会下降的方法。
  - 论文方法：
    - 低秩矩阵的重参数化。具体来说，冻结一对随机初始化的矩阵，这些矩阵在所有适配层之间共享，然后引入可以逐层自适应的可训练缩放向量。如图所示，类似于LoRA，训练的缩放向量和低秩矩阵可以合并至原始权重中，从而消除额外的推理延迟。

## GPT 系列篇

### Table 解析篇

- [小样本QA问答 MINPROMPT](GPT/Table-GPT/)
  - 论文名称：MINPROMPT: Graph-based Minimal Prompt Data Augmentation for Few-shot Question Answering
  - 论文地址：https://arxiv.org/pdf/2310.05007v1.pdf
  - 论文 Github地址：
  - 会议：
  - 动机：llm 读取表格
    - 问题一：缺失值识别
    - 问题二：缺失值识别
    - 问题三：表格问题解答
  - 论文方法：
    - 优化策略一：表调优
    - 优化策略二：创建数据集:合成增强

## RAG 系列篇

### RAG Trick篇

- [Self-RAG：一种 通过自我反思实现检索增强生成 的 RAG 策略](RAG/RAG_Trick/Active_RAG/)
  - 论文名称：Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection
  - 论文地址：https://arxiv.org/abs/2310.11511
  - 论文 Github地址：
  - 会议：
  - 动机：
    - 1. **检索文段与 query 的 不相关性**：这些方法不加区别地检索和合并了一定数量的检索文段，无论是否需要检索或文段是否相关，这会降低LLMs的多功能性或导致生成质量不佳（Shi等人，2023），因为它们不加区别地检索文段，无论事实支持是否有帮助;
    - 2. **生成的结果未必与检索的相关文段一致**（Gao等人，2023）：因为这些模型没有明确训练以利用和遵循所提供文段的事实;
  - 论文方法：
    - 通过**按需检索和自我反思来提高LLM的生成质量**，包括其事实准确性，而不损害其多功能性。
    - 论文**以端到端方式训练任意的LLM来学习反思自身的生成过程，通过生成任务输出和间歇性的特殊token**（即反思token）。反思token分为检索和评论token，分别表示检索的需求和生成的质量

- [Active RAG：一种主动判断需不需要进行检索，需要时再检索的 RAG 策略](RAG/RAG_Trick/Active_RAG/)
  - 论文名称：Active Retrieval Augmented Generation
  - 论文地址：https://arxiv.org/pdf/2305.06983.pdf
  - 论文 Github地址：https://github.com/jzbjyb/FLARE
  - 会议：
  - 动机：如果每一个step都去进行检索显然是有点冗余的问题
  - 论文方法：
    - 方法一：FLARE with Retrieval Instructions
    - 方法二：Direct FLARE

- [MINPROMPT 文档QA问答](RAG/RAG_Trick/MemSum_DQA/)
  - 论文名称：MemSum-DQA: Adapting an Efficient Long Document Extractive Summarizer for Document Question Answering
  - 论文地址：https://arxiv.org/pdf/2310.06436v1.pdf
  - 论文 Github地址：https://github.com/nianlonggu/MemSum-DQA
  - 会议：CIKM 2023
  - 动机：
  - 论文方法：论文提出了**「MemSum-DQA，这是一种高效的文档问答 (DQA) 系统」**，它利用了MemSum（一种长文档提取摘要器），通过在解析文档中的每个文本块中添加所提供的问题和问题类型的前缀，MemSum-DQA 有选择地从文档中提取文本块作为答案。 

- [PDFTriage：针对长结构化文档的问答](RAG/RAG_Trick/PDFTriage/)
  - 论文名称：PDFTriage: Question Answering over Long, Structured Documents
  - 论文地址：https://arxiv.org/pdf/2309.08872.pdf
  - 论文 Github地址：
  - 会议：
  - 动机：当文档不适合LLM的有限上下文窗口时，可以部署不同的策略来获取相关上下文。
  - 论文方法：
    - 1. **生成文档元数据**：提取文档的结构元素并将其转换为可读的元数据；
    - 2. **基于 LLM 的分类**：查询 LLM 以从文档中选择精确的内容（页面、部分、检索的内容）;
    - 3. **使用检索到的内容进行回答**：根据问题和检索到的内容，生成答案。


### RAG应用领域篇

#### 医疗领域QA问答

- [Expert-Level Medical Question-Answering 医疗可信QA问答](RAG/Task/医疗可信QA/)
  - 论文名称：Emulating Human Cognitive Processes for Expert-Level Medical Question-Answering with Large Language Models
  - 论文地址：https://arxiv.org/ftp/arxiv/papers/2310/2310.11266.pdf
  - 动机：为了满足医疗保健领域对先进临床问题解决工具的迫切需求。
  - 论文方法：论文推出了 **「BooksMed，这是一种基于大型语言模型(LLM)的新颖框架」**。BooksMed模拟人类认知过程，提供具有依据的可靠响应，利用GRADE（建议、评估、开发和评估）框架量化依据强度。

- [Medical Question-Answering by Expectation Maximization Inference over Evidence 医疗QA问答](RAG/Task/Medical_QA/)
  - 论文名称：Generating Explanations in Medical Question-Answering by Expectation Maximization Inference over Evidence
  - 论文地址:https://arxiv.org/pdf/2310.01299v1.pdf
  - 动机：医疗问答（医疗 QA）系统在协助医护人员寻找问题答案方面发挥着重要作用。然而，**仅通过医学 QA 系统提供答案是不够的，因为用户可能需要解释，即用自然语言进行更多分析性陈述，描述支持答案的元素和上下文**。
  - 论文方法：论文提出了一种新方法，**「为医学 QA 系统预测的答案生成自然语言解释」**。 由于高质量的医学解释需要额外的医学知识，因此我们的系统在解释生成过程中从医学教科书中提取知识以提高解释的质量。

#### 宗教领域QA问答

- [QASiNa 宗教领域QA问答](RAG/Task/QASiNa_宗教领域/)
  - 论文名称：QASiNa: Religious Domain Question Answering using Sirah Nabawiyah
  - 论文地址：https://arxiv.org/pdf/2310.08102v1.pdf
  - 动机：随着大型语言模型 (LLM)的发展。LLM可以应用于各个领域，但应用于伊斯兰宗教领域时却与信息传输的原则相矛盾。在伊斯兰教中，严格监管信息来源以及谁可以对该来源进行解释。LLM根据自己的解释生成答案的方法类似于tafseer的概念，LLM既不是伊斯兰专家，也不是伊斯兰教所不允许的人。鉴于LLM的影响力较高，本文作者 **「对宗教领域的LLM进行评价」**。
  - 论文方法：论文提出了问答Sirah Nabawiyah (QASiNa)数据集，这是一个根据印尼语Sirah Nabawiyah 文献编译的新颖数据集，并使用 mBERT、XLM-R和IndoBERT验证该数据集，并使用 SQuAD v2.0 的印尼语翻译进行微调。

#### 常识领域QA问答

- [QADYNAMICS 常识QA问答](RAG/Task/QADYNAMICS_常识QA问答/)
  - 论文名称：QADYNAMICS: Training Dynamics-Driven Synthetic QA Diagnostic for Zero-Shot Commonsense Question Answering
  - 论文地址：https://arxiv.org/pdf/2310.11303v1.pdf
  - 论文 Github地址：https://github.com/HKUST-KnowComp/QaDynamics
  - 动机：Zero-shot常识问答 (QA) 要求模型能够进行一般情况的推理。 最先进的方法一般做法是根据常识知识库 (CSKB) 构建的QA对，并对语言模型进行微调，使其能够具备更多的常识知识。但在此过程中，QA对构建过程中可能会引入来自 CSKB 的噪声，从而生成不符合预期的语法问答对，这会阻碍模型的泛化能力。
  - 论文方法：论文提出了**「QADYNAMICS，一种用于QA诊断和改进的动态驱动框架」**。该方法分析了QA对在问答、选项两个方面上的训练动态，通过删除无信息QA对、错误标记、错误选项来简化训练检测组件。

#### 法律领域QA问答

- [Long-Form Legal Question Answering 法律QA问答](RAG/Task/Law_QA/)
  - 论文名称：Interpretable Long-Form Legal Question Answering with Retrieval-Augmented Large Language Models
  - 论文地址:https://arxiv.org/pdf/2309.17050v1.pdf
  - 论文 Github地址：https://github.com/maastrichtlawtech/lleqa
  - 会议：CIKM 2023
  - 动机：许多人可能在一生中的某个时刻面临法律纠纷，但他们缺乏对如何解决这些复杂问题的了解，往往使他们变得脆弱。 自然语言处理的进步为通过开发自动化法律援助系统来弥合法律素养差距开辟了新途径。 然而，现**有的法律问答（LQA）方法往往范围狭窄，要么局限于特定的法律领域，要么仅限于简短、无信息的回答**。
  - 论文方法：论文提出了一种端到端的方法，**「旨在利用“先检索后阅读”的管道生成任何成文法问题的长格式答案」**。 为了支持这种方法，引入并发布了长格式法律问答 (LLeQA) 数据集，其中包含 1,868 个由专家注释的法语法律问题，以及基于相关法律条款的详细答案。

#### 知识图谱领域QA问答

- [CHATKBQA: 知识检索QA问答](RAG/Task/ChatKBQA_知识检索QA问答/)
  - 论文名称：CHATKBQA: A GENERATE-THEN-RETRIEVE FRAMEWORK FOR KNOWLEDGE BASE QUESTION ANSWERING WITH FINE-TUNED LARGE LANGUAGE MODELS
  - 论文地址:https://arxiv.org/pdf/2310.08975v1.pdf
  - 论文 Github地址：https://github.com/LHRLAB/ChatKBQA
  - 会议：
  - 动机：
    - 知识检索效率低下；
    - 检索错误影响语义解析结果；
    - 先前KBQA方法的复杂性。
  - 论文方法：论文提出首先使用微调的LLM生成逻辑形式，然后通过无监督检索方法检索和替换实体、关系，这直接地改进了生成和检索。

#### 任务型领域QA问答

- [InstructTODS: 知识检索QA问答](RAG/Task/ChatKBQA_任务导向QA问答/)
  - 论文名称：InstructTODS: Large Language Models for End-to-End Task-Oriented Dialogue Systems
  - 论文地址：https://arxiv.org/pdf/2310.08885v1.pdf
  - 论文 Github地址：https://github.com/WillyHC22/InstructTODS/
  - 会议：
  - 动机：当前，大语言模型(LLM)已用于各种自然语言处理(NLP)任务，但对于任务导向的对话系统（TODS），特别是端到端的TODS的探索仍然存在一定的局限性。
  - 论文方法：论文提出了「InstructTODS，该框架可用于Zero-Shot端到端任务导向的对话系统，无需微调即可适应不同的领域」。通过利用LLM，InstructTODS生成代理信念状态(proxy belief state)，将用户意图无缝转换为动态查询，以便与任何知识库进行高效交互。

#### 汽车领域QA问答

- [CarExpert: 汽车检索增强QA问答](RAG/Task/CarExpert_汽车检索增强QA问答/)
  - 论文名称：CarExpert: Leveraging Large Language Models for In-Car Conversational Question Answering
  - 论文地址：https://arxiv.org/pdf/2310.09536v1.pdf
  - 论文 Github地址：
  - 会议：
  - 动机：大型语言模型（LLM）通过遵循自然语言指令而无需对特定领域的任务和数据进行微调，表现出了卓越的性能。**然而，利用LLM进行特定领域的问题回答往往会产生幻觉。此外，由于缺乏对领域和预期输出的认识，LLM可能会生成不适合目标领域的错误答案。**
  - 论文方法：论文提出了「CarExpert」，车内检索增强会话问答系统利用了LLM的不同任务。具体而言，CarExpert采用LLM来控制输入，为提取和生成回答组件提供特定领域的文档，并控制输出以确保安全和特定领域的答案。

## Prompt 系列篇

- [小样本QA问答 MINPROMPT](Prompt/MinPrompt/)
  - 论文名称：MINPROMPT: Graph-based Minimal Prompt Data Augmentation for Few-shot Question Answering
  - 论文地址：https://arxiv.org/pdf/2310.05007v1.pdf
  - 论文 Github地址：
  - 会议：
  - 动机：小样本问答（Few-shot QA）旨在少量训练样本的情况下，让模型给出令人满意的回答。 最新的研究进展主要依赖大型语言模型（LLM）。**尽管预训练阶段已经让LLM具备了强大的推理能力，但LLM仍需要进行微调以适应特定领域，以达到最佳结果**。
  - 论文方法：论文提出了「MinPrompt」，一个基于近似图算法和无监督问题生成的开放域QA的最小数据增强框架。 作者将原始文本转换为图形结构，以在不同的事实句子之间建立联系，然后应用图形算法来识别原始文本中最多信息所需的最小句子集。然后，根据识别的句子子集生成问答对，并在选定的句子上训练模型以获得最终模型。 实证结果表明，MinPrompt 能够以高效率实现与基线相当或更好的结果。

## LMMs 可解释性篇

- [大模型事实性综述(Survey on Factuality in Large Language Models)](LLMs_explain/Factuality_in_LLMs/)
  - 论文名称：Survey on Factuality in Large Language Models
  - 论文地址：https://arxiv.org/pdf/2310.07521.pdf
  - 论文 Github地址：
  - 会议：
  - 动机：
    - 尽管 LLMs 具有无与伦比的能力，其**产生非事实或误导性内容的可能**也让人产生担忧；
    - **对一些特定领域知识或者实时事实知识的缺乏也极大限制了大模型的使用**；
  - 四个关键维度：
    - **事实性问题的定义及其影响**；
    - **评估事实性的技术及其定量评估**；
    - **分析 LLMs 中事实性的基本机制并确定事实错误的根本原因**；
    - **增强 LLMs 事实性的方法**。
  - 两个主要设置：
    - 没有外部知识的 LLMs，如 ChatGPT
    - 检索增强型 LLMs，如 BingChat

- [LLMs 研究 —— LLMs 自我解释性研究](LLMs_explain/LLMs_explain_themselves/)
  - 论文名称：Can Large Language Models Explain Themselves? A Study of LLM-Generated Self-Explanations
  - 论文地址：https://arxiv.org/pdf/2310.11207.pdf
  - 论文 Github地址：
  - 会议：
  - 动机：**LLM在自我解释方面有多擅长？**
  - 论文方法：
    - **对大型语言模型（LLMs）在自我生成特征归因解释方面的能力进行了严格评估**
    - **研究使用ChatGPT和SST数据集作为测试平台，构建了能可靠生成两种类型的LLM生成自我解释的提示**
    - 研究比较了这些解释与传统的解释技术（遮蔽显著性和LIME）在忠实度和一致性指标上的表现；
  - 论文结论
    - **根据忠实度评估，无论是自动生成的解释还是其他解释，都没有明显的优势**。然而，根据一致性评估，它们之间存在很大的区别。这可能表明当前解释方法可能不是最优的，需要开发更好的方法来产生更好的自我解释；
    - 模型预测值和单词归因值都非常全面，取值如0.25，0.67，0.75等。这表明**当前的评估度量可能无法充分区分好解释和坏解释**；


## LLMs4KG 篇

- [ChatKBQA](KG/ChatKBQA/)
  - 论文名称：ChatKBQA: A Generate-then-Retrieve Framework for Knowledge Base Question Answering with Fine-tuned Large Language Models
  - 论文地址：https://arxiv.org/abs/2310.08975
  - Github 地址：https://github.com/LHRLAB/ChatKBQA
  - 会议：
  - 动机：**利用微调开源大模型进行自然语言问题到逻辑形式的转换，再利用无监督实体关系检索生成图数据库查询语言，实现自然语言的知识图谱问答框架。**
  - 论文方法：提出了ChatKBQA，**这是一种基于微调开源LLMs（大型语言模型），如Llama-2-7B，ChatGLM2-6B和Baichuan2-7B等，的新型生成-检索KBQA框架**；
    - 首先微调生成逻辑形式，然后对生成的逻辑形式中的实体和关系在知识库中的实体库和关系库分别做检索，避免了以前方法存在的先检索对逻辑形式生成的影响，并提高检索效率；
    - 在生成阶段，使用指令微调技术对开源LLMs进行微调，赋予它们感知和生成逻辑形式的能力

## LLMs Agents 篇

### 角色扮演(Role-Play)

- [大语言模型的角色扮演(Role-Play with Large Language Models)](agents/RolePlay/Role_Play_LLMs/)
  - 论文名称：Role-Play with Large Language Models
  - 论文链接：https://arxiv.org/pdf/2305.1636
  - 论文动机：
    - **使用我们描述人类行为的相同语言来描述对话agents是自然的**，如：自由地使用“知道”、“理解”和“思考”等词汇。试图通过使用更科学精确的替代词来避免这样的词汇通常会导致笨拙、难以理解的文本;
    - **如果过于在字面意义上理解这种语言，会促进人格化、夸大这些AI系统与人类之间的相似之处，而掩盖其内在的区别**。
  - 论文思路：提出了两个基本的隐喻(metaphors)来描述基于LLM的对话agents：
    - 从简单的观点来看，我们可以**将对话agents视为扮演一个单一角色**；
    - 从更细微的观点来看，我们可以**将对话agents视为角色在多元宇宙中的模拟重叠**；

- [RoleLLM](agents/RolePlay/RoleLLM/)
  - 论文名称：RoleLLM: Benchmarking, Eliciting, and Enhancing Role-Playing Abilities of Large Language Models
  - 论文链接：https://arxiv.org/abs/2310.00746
  - 论文动机：
    - Few-Shot Prompting / In-Context Learning：加入few-shot examples（从该角色的历史dialogue数据中检索而来），有助于LLM获取相关知识、模仿角色的风格。
  - 论文思路：
    - RoleLLM 所用 两种Few-Shot方法：
      - single-turn prompt：在单轮对话中一次性引入examples
      - multi-turn prompt：RoleLLM称之为dialogue engineering，即将对话过程以user和assistant角色交替写入。
    - RoleLLM 数据合成方法：
      - **general domain**。收集开源的general instructions（如英文领域的Super-NaturalInstruct、中文领域的COIG等），然后给到RoleGPT，**让其在通用问题上，生成模仿某个角色的对话**；
      - **role-specific**（即论文所谓的Context-Instruct）。根据Role Profile来生成Question、Answer，这样生成的**dialogue数据更加与role本身相关，因此有role-specific的特点**；

- [Character-LLM](agents/RolePlay/CharacterLLM/)
  - 论文名称：Character-LLM: A Trainable Agent for Role-Playing
  - 论文链接：https://arxiv.org/abs/2310.10158
  - 论文动机：
    - Fine-Tuning的迷人之处在于：**适当的数据 + 开源LLM + Fine-Tuning 有可能超过 闭源LLM + In-Context Learning。**
  - 论文思路：
    - 将目标role在wiki上的信息，作为profile；
    - 使用LLM，根据profile来生成scene；
    - 基于scene + profile，让LLM生成dialogue数据。其prompt示例如下，用此种方法生成的数据可以是多轮的。

- [ChatHaruhi](agents/RolePlay/ChatHaruhi/)
  - 论文名称：ChatHaruhi: Reviving Anime Character in Reality via Large Language Model
  - 论文链接：https://arxiv.org/abs/2308.09597
  - Github 地址：https://github.com/LC1332/Chat-Haruhi-Suzumiya/tree/main
  - 优化策略：
    - 允许LLM复用few-shot examples。即允许LLM在相似场景下，直接使用角色的对话内容；
    - 在结尾额外加上一段人物性格描述，以强化生成效果。
  - 论文思路：
    - 将目标role发言前的内容作为question，给到LLM，让其继续完成这个dialogue。同时为了优化生成效果，论文也采取了few-shot prompting的策略。在实验中，有50%的概率会生成多轮对话。

