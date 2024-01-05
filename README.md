# LLMs 论文研读社

> 作者：杨夕
> 
> 介绍：该仓库主要记录 LLMs 算法工程师相关的顶会论文研读笔记（多模态、PEFT、小样本QA问答、RAG、LMMs可解释性、Agents、CoT）
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

<img src="resource/pic/20230818133801.jpg" width="50%" >

> LLMs 千面郎君 面试交流群 (注：人满 可 添加 小编wx：yzyykm666 加群！)

## 多模态篇

- Gemini：一族功能强大的多模态模
  - 论文名称：Gemini: A Family of Highly Capable Multimodal Models
  - 论文地址：https://arxiv.org/pdf/2312.11805
  - 机构：Google
  - Github 地址：
  - 会议：
  - 论文方法：该论文介绍了一种新的多模态模型系列，Gemini，在图像、音频、视频和文本理解方面具有非凡的能力。Gemini系列包括Ultra、Pro和Nano三种规模，适用于从复杂的推理任务到设备上的内存受限用例。
  - 论文实验结果：在广泛的基准测试中，该论文最先进的Gemini Ultra模型在32个基准测试中有30个取得了最新的进展，特别是首次在公认的考试基准MMLU上达到人类专家水平，并在该论文检查的20个多模态基准测试中改进了最新的技术水平。该论文相信Gemini模型在跨模态推理和语言理解方面的新能力将能够支持各种用例，并讨论了该论文在负责任地向用户部署它们方面的方法。

- 评估GPT4-V在结构化推理任务上的表现
  - 论文名称：Assessing GPT4-V on Structured Reasoning Tasks
  - 论文地址：https://arxiv.org/pdf/2312.11524
  - 机构：OpenAI
  - Github 地址：
  - 会议：
  - 论文方法：这篇论文主要评估了最新的语言模型GPT-4V和其他五个基准模型在结构化推理任务上的表现。这些任务包括数学推理、视觉数据分析和代码生成。
  - 研究结果显示，引入视觉Chain-of-Thought的多模态LLMs相比于普通模型有显著的提升。同时，论文还对模型表现良好和困难的场景进行了分类分析，突出了多模态推理中所面临的挑战。

- ProTIP: 渐进式工具检索改善规划
  - 论文名称：ProTIP: Progressive Tool Retrieval Improves Planning
  - 论文地址：https://arxiv.org/pdf/2312.10332
  - 机构：
  - Github 地址：
  - 会议：
  - 论文方法：这篇论文介绍了一种名为ProTIP的渐进式工具检索框架，用于复杂的多步骤规划任务。该框架通过对比学习的方式隐式地进行任务分解，同时保持子任务-工具的原子性。
  - 在ToolBench数据集上，ProTIP在工具检索方面超越了基于ChatGPT的任务分解方法，并且在TR的Recall@K=10方面提高了24％，在计划生成方面工具准确性提高了41％。

- LLaVA：经典的多模态大模型
  - 论文名称：Visual Instruction Tuning
  - 论文地址：https://arxiv.org/abs/2304.08485
  - 机构：微软研究院和哥伦比亚大学
  - Github 地址：https://github.com/haotian-liu/LLaVA
  - 会议：
  - 动机：像ChatGPT这种大语言模型只接受文字输入，那么如何让大语言模型接收图像输入呢？
  - 论文方法：LLaVA提出了一种方法，
    - 将Clip作为图像的编码器，在Clip后面加入一个线性映射层;
    - 将Clip编码后的图像特征 Zu 映射到语言模型特征空间中，得到视觉特征 Hv ;
    - 将其和文本的编码（语言指令的编码）一起送入到Language Model中。
  - 训练方式：
    - 第一阶段：**预训练阶段**。在这个阶段，**只训练线性映射层(Projection W)，目的是学习图像空间到语言模型词向量空间的映射**，这阶段使用的数据集为CC3M；
    - 第二阶段：**微调阶段**。在这阶段，**冻结住视觉编码器的参数，训练线性映射层和大语言模型的参数**。在这一阶段使用的数据集为ScienceQA和基于GPT-4生成的数据集。
  - 实验效果：该模型展示出了一些接近多模态 GPT-4 的图文理解能力：相对于 GPT-4 获得了 85.1% 的相对得分。当在科学问答（Science QA）上进行微调时，LLaVA 和 GPT-4 的协同作用实现了 92.53%准确率的新 SoTA。

- LLaVAR：增强的视觉指令微调
  - 论文名称：LLaVAR: Enhanced Visual Instruction Tuning for Text-Rich Image Understanding
  - 论文地址：https://arxiv.org/pdf/2306.17107.pdf
  - 机构：佐治亚理工、Adobe和斯坦福
  - Github 地址：https://github.com/SALT-NLP/LLaVAR
  - 会议：
  - 动机：
  - 论文方法：用OCR的工具从LAION数据集收集了422K包含文本信息的图片，然后用从图片中识别的文字以及图片的caption作为提示词，用text only的GPT-4生成了16K对话，每一个对话都包含和每一张图片关联的 问题-回答 pair。文中集合收集的这些对话数据集以及LLaVA的对话数据，训练了可以对图片中的场景进行细致理解的LLaVAR模型。
  - 模型结构：
    - 视觉encoder V：对于224x224分辨率的输入，采用CLIP-ViT-L/14；对于336x336分辨率的输入，采用CLIP-ViT-L/14-336。最后一层Transformer Layer输出的特征通过过一个映射矩阵 W 映射到语言Decoder的单词嵌入空间；
    - 语言Decoder D：采用基于LLAMA的Vicuna-13B
  - 训练方式：
    - 预训练：只训练视觉编码器到LLM编码器之间的映射层（采用LLaVA从CC3M中过滤的595k图文以及新构建的422k粗糙数据）；
    - 微调：训练视觉编码器到LLM编码器之间的映射层和LLM（采用LLaVA基于MSCOCO构建的158k指令数据以及新构建的16k指令数据训练模型的指令理解能力，同时微调LLM以及图文之间的映射层）；

- Vary: Scaling up the Vision Vocabulary forLarge Vision-Language Models
  - 论文名称：Vary: Scaling up the Vision Vocabulary forLarge Vision-Language Models
  - 论文地址：arxiv.org/abs/2312.06109
  - 动机：
    - PDF 类文档的难点在于，如何完整恢复图片、表格、标题、段落等内容，形成一个文字版的文档。
    - 现有开源多模态大模型的问题
      - 对中文支持较差，毕竟大部分训练数据还是以英文为主。
      - 文档级的识别程度不高，毕竟多模态大模型也不是单纯做 OCR 任务的，所以训练数据可能有欠缺，在识别文档图像时出现容易缺少内容，导致回答出现幻觉或者不准确。
  - 思路：通过收集新的数据，训练一个新的视觉编码器，然后和原有的视觉编码器合并。

- Instruct-Imagen: 多模式指导下的图像生成
  - 论文名称：Instruct-Imagen: Image Generation with Multi-modal Instruction
  - 机构：谷歌研究院、Google DeepMind
  - 相关领域：指令微调、多模态
  - 论文地址：https://arxiv.org/pdf/2401.01952
  - 作者：Hexiang Hu, Kelvin C.K. Chan, Yu-Chuan Su
  - 论文方法：篇论文介绍了instruct-imagen，一个解决异构图像生成任务并能够在未知任务上进行泛化的模型。它引入了多模式指导的图像生成，一种利用自然语言将不同模态（例如，文本、边缘、样式、主题等）综合起来的任务表示，使得丰富的图像生成意图可以以统一的格式标准化。作者通过在一个两阶段框架中对预训练的文本到图像扩散模型进行微调来构建instruct-imagen。首先，作者使用检索增强训练来使模型能够基于外部多模态上下文生成图像。随后，作者在多样的图像生成任务上对微调后的模型进行微调，这些任务需要对视觉语言进行理解（例如，基于主题的生成等），每个任务都与一个包含任务本质的多模式指导相配对。在各种图像生成数据集上进行的人工评估表明，instruct-imagen在领域内与先前的任务特定模型相媲美或超越，并展示了对未知和更复杂任务的有希望的泛化能力。

- LLaVA-φ: 高效的多模态助理与小型语言模型
  - 论文名称：LLaVA-φ: Efficient Multi-Modal Assistant with Small Language Model
  - 机构：IDEA、华东师范大学
  - 相关领域：指令微调、多模态
  - 论文地址：arxiv.org/pdf/2401.02330
  - 代码：github.com/zhuyiche/llava-phi
  - 作者：Yichen Zhu, Minjie Zhu, Ning Liu
  - 论文方法：LLaVA-φ是一个高效的多模态助理，利用最近先进的小型语言模型Phi-2的力量，促进多模态对话。LLaVA-φ标志着紧凑多模态模型领域的显著进步。它证明了即使是具有仅2.7B参数的更小的语言模型，只要它们经过高质量的语料库训练，就可以有效地参与融合文字和视觉元素的复杂对话。该论文的模型在包括视觉理解、推理和基于知识的感知在内的公开可用基准测试上具有可称赞的性能。除了在多模态对话任务中表现出色之外，该论文的模型还为在时间敏感环境和需要实时交互的系统（如具身代理）中的应用开辟了新的途径。它突显了更小的语言模型在保持更高资源效率的同时实现复杂的理解和交互水平的潜力。

- 仅使用文本训练，在零样本字幕生成中挖掘细粒度的图像-文本对齐
  - 论文名称：Mining Fine-Grained Image-Text Alignment for Zero-Shot Captioning via  Text-Only Training
  - 机构：上海科技大学
  - 相关领域：多模态
  - 论文地址：https://arxiv.org/pdf/2401.02347
  - 代码：https://github.com/Artanic30/MacCap
  - 作者：Longtian Qiu, Shan Ning, Xuming He
  - 论文方法：该论文通过对CLIP潜在空间的分析，提出了一种通过仅使用文本训练的零样本图像字幕生成框架。通过挖掘图像子区域的视觉特征和文本描述中的信息损失，可以减少模态差距，并通过引入噪声注入和重新排序策略提高字幕生成性能。

- 仅使用文本监督学习视觉-语言模型的提示学习
  - 论文名称：Learning to Prompt with Text Only Supervision for Vision-Language Models
  - 机构：Google、苏黎世联邦理工学院
  - 相关领域：预训练、多模态
  - 论文地址：https://arxiv.org/pdf/2401.02418
  - 代码：hhttps://github.com/muzairkhattak/ProText
  - 作者：Muhammad Uzair Khattak, Muhammad Ferjad Naeem, Muzammal Naseer
  - 论文方法：这篇论文通过仅使用文本数据从语言模型中学习提示，结合了视觉信息和大语言模型的优势。通过这种方法，可以实现对新类别和数据集的零样本转移，减少了大语言模型提示工程的成本。


### GPT4Video 篇

- [GPT4Video](Video/GPT4Video/readme.md)
  - 论文名称：GPT4Video: A Unified Multimodal Large Language Model for lnstruction-Followed Understanding and Safety-Aware Generation
  - 论文地址：https://arxiv.org/abs/2311.16511
  - 论文示例：https://gpt4video.github.io/
  - 论文背景：当前的多模态大语言模型（MLLM）已经验证多模态数据融合的有效性，但没有工作去探索多模态信息的生成；
  - 论文框架：
    - 视频理解模块。首先通过video feature extractor提取视频特征，然后通过video abstractor对齐视频特征和LLM；
    - 大语言模型。使用LLaMA预训练的参数，通过LoRA进行微调；
    - 视频生成模块。将LLM输出的Prompt输入到Text-Video模型，得到生成的视频。

## PEFT 系列篇

- [Prompt](PEFT/prompt/prompt.md)
  - 论文名称：Prompt Tuning
  - 论文地址：https://arxiv.org/pdf/2107.13586.pdf
  - Github 地址：
  - 会议：
  - 动机：但是**对于一个预训练的大语言模型来说，这就仿佛好像是对于每个任务都进行了定制化，十分不高效**。**是否存在一种方式，可以将预训练语言模型作为电源，不同的任务当作电器，仅需要根据不同的电器（任务），选择不同的插座，对于模型来说，即插入不同的任务特定的参数，就可以使得模型适配该下游任务**。
  - 论文方法：给 预训练语言模型 的一个线索/提示，帮助它可以更好的理解 人类的问题。

- [Instruction](PEFT/prompt/Instruction.md)
  - 论文名称：Finetuned Language Models Are Zero-Shot Learners
  - 论文地址：https://arxiv.org/abs/2109.01652
  - Github 地址：https://github.com/google-research/flan
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
  - 论文名称：QLoRA: Efficient Finetuning of Quantized LLMs
  - 论文地址：hhttps://arxiv.org/pdf/2305.14314.pdf
  - Github 地址：https://github.com/artidoro/qlora
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

- 仅用少量多语言数据即可进行多语言指令微调
  - 论文名称：Multilingual Instruction Tuning With Just a Pinch of Multilinguality
  - 相关领域：指令微调
  - 机构：谷歌研究院、特拉维夫大学
  - 作者：Uri Shaham, Jonathan Herzig, Roee Aharoni
  - 论文地址：https://arxiv.org/pdf/2401.01854
  - Github 地址：
  - 会议：
  - 分析：该论文通过研究多语言指令微调对多语言大语言模型（LLMs）的指令跟随能力的影响，发现即使在单语微调中，许多语言也能够将一些指令跟随能力转移到其他语言。此外，通过在英语微调集上仅使用40个多语言示例，可以大幅提高多语言指令跟随的性能，不论在已见或未见的语言上。尽管在这些语言中的训练示例少10倍，但总体上，与单语微调模型相比，使用多语言混合微调的模型在几种语言上表现出可比或更优的性能。最后，通过将指令微调集中的语言数量从1增加到2、3或4，可以增加跨语言通用性。实验结果表明，通过使用极小的多语言指令响应集，可以构建出大规模多语言指令微调的模型。


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

-  RAGTruth: 用于开发可靠的检索增强语言模型的幻化语料库
  - 论文名称：RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models
  - 论文地址：https://arxiv.org/pdf/2401.00396
  - 相关领域：模型评估、数据集构建
  - Github 地址：
  - 会议：
  - 论文方法：本文介绍了RAGTruth，一个专门用于在LLM应用的标准RAG框架中分析各个领域和任务中的单词级幻象的语料库。RAGTruth包括来自不同LLM使用RAG的近18000个自然生成的回复。这些回复经过精细的手动注释，包括对幻觉强度的评估。该论文不仅对不同LLM的幻觉频率进行了基准测试，还对几种现有的幻觉检测方法的有效性进行了批判性评估。此外，该论文还展示了使用高质量数据集（如RAGTruth），可以对相对较小的LLM进行微调，并在幻觉检测方面与使用GPT-4等最先进的大语言模型的现有提示式方法实现了具有竞争力的性能水平。

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

## Attention 篇

- [System 2 Attention](Attention/S2A/)
  - 论文标题：System 2 Attention (is something you might need too)
  - 论文链接：https://arxiv.org/abs/2311.11829
  - Github 地址：
  - 动机：大型语言模型(LLM)非常强大，但它们仍容易出现简单的错误，这似乎显示出弱的推理能力。例如，不相关的上下文或输入提示中固有的偏好或意见，都可能使它们产生错误判断，在后一种情况下，展现了一种称为阿谀奉承的问题，即模型与输入一致同意。
  - 论文方法：论文提出了一种技术方案--System 2 Attention(S2A)，可以让LLM决定输入上下文的重要部分，来生成好的响应。实现这点的方法是：**首先诱导LLM重新生成只包含相关部分的输入上下文，然后关注重新生成的上下文以引出最终响应。**
  - 论文在实验中证明，S2A可以成功重写会降低最终答案质量的上下文，因此论文的方法可以同时提高事实性并减少其响应中的阿谀奉承。
  - 未来的研究仍有许多空间。在论文的实验中，采用了零样本提示来实现S2A。其他方法可以通过考虑微调、强化学习或替代提示技术(alternative prompting techniques)来进一步优化论文的方法。成功的S2A还可以压缩回标准LLM生成，例如：通过使用原始提示作为输入和最终改进的S2A响应作为目标进行微调。

## 搜索 篇

- [LSM：如何用好LLMs：大型搜索模型](Search/LSM/readme.md)
  - 论文名称：Large Search Model: Redefining Search Stack in the Era of LLMs
  - 论文地址：https://arxiv.org/abs/2310.14587
  - 动机：
    - 神经网络信息检索基础 局限性：在生成长文本时它们倾向于产生不正确或不相关的信息；
    - 检索增强生成 局限性：RAG的最佳训练策略仍是一个未解之谜。人们也对模型利用检索信息的有效性表示担忧；
  - 论文框架：作者将大型搜索模型定义为一个定制的大型语言模型，它通过自然语言提示将各种搜索任务统一起来。它重新定义了由查询理解、检索、排名、摘要和问答等许多离散组件组成的传统搜索堆栈。

- SuperGen：用语言模型生成训练数据：迈向零样本语言理解
  - 论文名称：SuperGen：Generating Training Data with Language Models: Towards Zero-Shot Language Understanding
  - 论文地址：https://arxiv.org/abs/2202.04538
  - 方法：利用NLG模型生成数据质量高的优势，结合NLU模型理解能力强的优势，在多个GLUE任务上起到了不错的效果。

- DARE: 基于GPT-2的数据增强关系提取
  - 论文名称: DARE: Data Augmented Relation Extraction with GPT-2
  - 论文地址：https://arxiv.org/abs/2310.14587
  - 方法：用gpt2先在领域内数据上微调，然后用生成的训练数据来提升BERT类模型在关系抽取任务上的效果。这一思路其实是和SuperGen思路是相同的，只是gpt2的模型体量更小，在相关领域上微调后生成的数据质量可能反而更好。

### 如何 通过 大模型 构建 “query-doc”？

> 解释：对搜索数据进行数据增强就是获取更多的“query-doc”对。一种方法是根据query生成假doc，而另一种是根据doc生成假query。

- InPars: 基于大型语言模型的信息检索数据扩充
  - 论文名称: InPars: Data Augmentation for Information Retrieval using Large Language Models
  - 论文地址：https://arxiv.org/abs/2202.05144
  - 方法：InPairs利用LLM的上下文学习能力，结合给出的示例，给doc生成了大量的假query，然后通过微调后的语言模型进行结果“过滤”。

- InPars-v2: 大型语言模型作为信息检索的有效数据集生成器
  - 论文名称: InPars-v2: Large Language Models as Efficient Dataset Generators for Information Retrieval
  - 论文地址：https://arxiv.org/abs/2301.01820
  - 方法：在inPairs-V2版本中，一个较大的变化是，其利用在检索数据集上微调的T5-3B模型来过滤生成的查询，而不是简单的通过概率进行过滤，以此来提升生成数据的可靠性。

- InPairs-Light：高效排名者的成本效益无监督培训
  - 论文名称: InPairs-Light：Cost-Effective Unsupervised Training of Efficient Rankers
  - 论文地址：https://arxiv.org/abs/2301.02998
  - 方法：后续的inPairs-Light版本也对“过滤器”进行了瘦身，参数从30亿降至2亿。

- InPairs-Light：从8个例子看 Few-shot Dense Retrieval
  - 论文名称: Promptagator：Few-shot Dense Retrieval From 8 Examples
  - 论文地址：https://arxiv.org/abs/2301.02998
  - 方法：PROMPTAGATOR 利用inPairs中“生成-过滤”这一过程，在生成的样本上微调检索器，然后使用该检索器过滤生成的样本。重复这两个步骤直到收敛，以产生高质量的训练集。

- UDAPDR：基于LLM提示和重排序的无监督域自适应
  - 论文名称: UDAPDR: Unsupervised Domain Adaptation via LLM Prompting and Distillation of Rerankers
  - 论文地址：https://arxiv.org/abs/2303.00807
  - 动机：在inPairs-V2版本中，研究者意识到请求LLM如chatgpt、gpt4的API进行数据增强会带来高额的成本，开始采用开源的LLM替换API请求方式，但可能会导致增强数据的质量下降。
  - 方法：UDAPDR 针对这一问题，先用高质量LLM根据doc生成高质量query，然后用高质量doc-query送入低成本LLM扩充数量，兼顾了成本和效果问题，其过程如图所示。

### 如何 通过 大模型 标注 “query-doc” 正负样例？

> 通过上述方法 虽然 能够 构建 “query-doc”，但是 如何 辨别真假呢？这个时候可以利用LLM获取query与doc的假label，即让模型帮我判断这条数据是不是正样本，是正样本的概率是多少？

- ART：训练 Dense Passage Retriever 所需的全部问题
  - 论文名称: ART：Questions Are All You Need to Train a Dense Passage Retriever
  - 论文地址：https://arxiv.org/abs/2206.10658
  - 方法：先将query经过向量编码，然后通过向量检索器选出相关文档，再让模型给每个文档与query的相关性进行打分。这一打分被作为soft label，反馈给之前的passage encoder和question encoder进行更新训练。

- ExaRanker：Explanation-Augmented Neural Ranker
  - 论文名称: ExaRanker：Explanation-Augmented Neural Ranker
  - 论文地址：https://arxiv.org/abs/2206.10658
  - 方法：ExaRanker 使用 GPT-3.5 为检索数据集生成解释，随后训练一个 seq2seq 排名模型来生成相关标签以及给定查询-文档对的相应解释。

- ChatGPT-RetrievalQA：为交叉编码器重排器生成合成文档： ChatGPT 与人类专家的比较研究
  - 论文名称: ChatGPT-RetrievalQA：Generating Synthetic Documents for Cross-Encoder Re-Rankers: A Comparative Study of ChatGPT and Human Experts
  - 论文地址：https://arxiv.org/abs/2305.02320
  - 方法：我们研究了生成式大型语言模型(llm)在为交叉编码器重新排序器生成训练数据方面的有用性，该方向是:生成合成文档而不是合成查询。我们引入了一个新的数据集ChatGPT-RetrievalQA，并比较了在llm生成和人工生成数据上微调的模型的有效性。生成式llm生成的数据可用于增强训练数据，特别是在标记数据数量较少的领域。我们基于一个现有的数据集，人类ChatGPT比较语料库(HC3)构建ChatGPT- retrievalqa，该数据集由公共问题集合组成，其中包含来自ChatGPT的人类响应和答案。
  - 实验结果：我们在人工生成或chatgpt生成的数据上微调一系列交叉编码器重新排名。我们对MS MARCO DEV、TREC DL'19和TREC DL'20的评估表明，在ChatGPT响应上训练的交叉编码器重新排序模型比在人类响应上训练的模型更有效。在有监督的环境中，人工训练的重新排名者的表现优于法学硕士训练的重新排名者。我们的新发现表明，生成式llm在为神经检索模型生成训练数据方面具有很高的潜力。需要进一步的工作来确定在生成的响应中事实错误信息的影响，并测试我们的发现在开源法学硕士中的普遍性。我们为将来的工作发布数据、代码和交叉编码器检查点。

### 如何 通过 大模型 改写 “query-doc”？

> 让LLM作为生成模型，根据用户的query写一段文本，将其作为改写结果送入后续的检索模块，以提高最终的检索质量。

- 面向信息检索查询扩展的神经文本生成
  - 论文名称: Neural text generation for query expansion in information retrieval
  - 论文地址：https://dl.acm.org/doi/10.1145/3486622.3493957
  - 动机：在LLM时代到来之前，就有不少研究利用各种生成式模型来对query进行改写。
  - 方法：利用gpt-2，根据query生成文本作为改写结果。文章整体思路非常简单，但是可以应用在各式各样的搜索系统上，效果也不错。另外，gpt-2模型体量不大，再加上cache等手段的运用，对于搜索系统整体没有什么额外负担。

- Query2doc:使用 大语言模型 进行查询扩展
  - 论文名称: Query2doc：Query Expansion with Large Language Models
  - 论文地址：https://arxiv.org/abs/2303.07678
  - 动机：
  - 方法：通过预定的prompt，根据用户的query生成一段文本，用于辅助后续的检索。相比之下，谷歌的研究有两点不同，一是没有利用上下文学习（ICL），二是要求LLM给出其答案的思维链。

- 通过提示 大语言模型 进行查询扩展
  - 论文名称: Query Expansion by Prompting Large Language Models
  - 论文地址：https://arxiv.org/abs/2305.03653
  - 动机：查询扩展是一种广泛用于提高搜索系统查全率的技术。
  - 方法：在本文中，我们提出了一种利用大型语言模型(llm)的生成能力进行查询扩展的方法。与传统的查询扩展方法(如伪相关反馈(PRF))依赖于检索一组良好的伪相关文档来扩展查询不同，我们依赖于LLM的生成和创造能力，并利用模型中固有的知识。我们研究了各种不同的提示，包括零弹、少弹和思维链(CoT)。我们发现CoT提示对于查询扩展特别有用，因为这些提示指示模型逐步分解查询，并且可以提供与原始查询相关的大量术语。
  - 实验结果：；在MS-MARCO和BEIR上的实验结果表明，llm生成的查询扩展比传统的查询扩展方法更强大。

- LLMCS：大语言模型了解上下文搜索意图:会话搜索的提示框架
  - 论文名称: LLMCS：Large Language Models Know Your Contextual Search Intent: A Prompting Framework for Conversational Search
  - 论文地址：https://arxiv.org/abs/2303.06573
  - 动机：上述两个方法可以应用于即席搜索（ad-hoc search）场景，而现代的搜索系统很多支持会话搜索（session search），类似于多轮对话，搜索结果会考虑一个会话中的前几次搜索信息。
  - 方法：LLMCS是一个支持会话搜索的框架，且针对于会话搜索场景下LLM输入长度增加的问题使用了滑窗方法进行优化。

- GRM: 基于相关性感知样本估计的文档检索生成关联建模
  - 论文名称: GRM：Generative Relevance Modeling Using Relevance-Aware Sample Estimation for Document Retrieval
  - 论文地址：https://arxiv.org/abs/2306.09938
  - 动机：尽管LLM拥有出色的文本理解和生成能力，不可否认其还是会存在幻觉问题，导致其输出结果背离事实，引入无关噪声影响最终检索结果。
  - 方法：GRM训练了一个神经网络模型，对LLM生成的结果进行相关性打分，最后将得分作为每个生成结果的权重，以减轻无关信息对最终检索结果的影响。

### 如何 通过 大模型 综合利用PRF（伪相关反馈）+GRF（生成相关反馈）？

> 以上研究都是利用LLM的生成结果作为改写结果的主要内容，我们可以将其看作是一种生成相关反馈（GRF），而不少研究也同时在模型生成或结果后处理阶段加入伪相关反馈（PRF）的方法来改进改写结果的质量。

- HyDE:无关联标签的 精确 Zero-Shot Dense Retrieval
  - 论文名称: HyDE：Precise Zero-Shot Dense Retrieval without Relevance Labels
  - 论文地址：https://arxiv.org/abs/2212.10496
  - 动机：LLM幻觉问题
  - 方法：HyDE将LLM生成的结果进行编码，利用向量检索器，与真实的文档库中的候选文档进行相关性匹配，然后利用真实的文档作为改写的结果辅助查询。可以看出，该方法实质上就是利用LLM的输出结果而不是query去召回伪文档。
  - 优点：
    - 相比传统的PRF方法，保证了第一次检索的伪文档的相关性；
    - 相比Query2doc等方法，又通过结合PRF避免了LLM可能产生幻觉的问题，保证了结果的高度真实性。
    - 类似地，LameR则是将PRF这一过程放到了LLM输入之前。

- LameR:大型语言模型是强大的零样本检索器
  - 论文名称: LameR：Large Language Models are Strong Zero-Shot Retriever
  - 论文地址：https://arxiv.org/abs/2304.14233
  - 动机：LLM幻觉问题
  - 方法：
  - 优点：

- Rewrite-Retrieve-Read：针对检索增强的大型语言模型的查询重写
  - 论文名称: Rewrite-Retrieve-Read：Query Rewriting for Retrieval-Augmented Large Language Models
  - 论文地址：https://arxiv.org/abs/2305.14283
  - 动机：LLM幻觉问题
  - 方法：Rewrite-Retrieve-Read这一研究则是利用改写去加强检索增强LLM的效果。Rewrite-Retrieve-Read图中从左到右分别是：检索增强LLM、带有改写器的检索增强LLM、带有强化学习改写器的检索增强LLM。其中Rewrite-Retrieve-Read指的是第三个。可以看出，Rewrite-Retrieve-Read方法不仅利用LLM作为改写器增加了其检索增强的效果，还引入了强化学习，通过最终答案的反馈，来训练高质量LLM改写器。
  - 优点：

- PRF+GRF:稀疏、稠密和学习稀疏检索的生成和伪相关反馈
  - 论文名称: PRF+GRF：Generative and Pseudo-Relevant Feedback for Sparse, Dense and Learned Sparse Retrieval
  - 论文地址：https://arxiv.org/abs/2305.07477
  - 动机：LLM幻觉问题
  - 方法：PRF+GRF直接结合PRF和LLM输出的结果，然后综合加权考虑两者的结果作为改写结果。
  - 优点：

- InteR:通过搜索引擎和大型语言模型之间的交互进行知识提炼
  - 论文名称: InteR：Knowledge Refinement via Interaction Between Search Engines and Large Language Models
  - 论文地址：https://www.researchgate.net/publication/370763983_Knowledge_Refinement_via_Interaction_Between_Search_Engines_and_Large_Language_Models
  - 动机：LLM幻觉问题
  - 方法：InteR则是一种搜索系统和LLM多轮交互框架，通过多次PRF、LLM输出，达到增强两过程效果的目的。
  - 优点：

### 如何 通过 大模型 进行 召排？

#### 何为 召回？

召回（retrive）是搜索系统中的核心模块，可分为基于统计算法的稀疏检索（Sparse Retriever）和基于神经网络的密集检索（Dense Retriever）。

#### 召回 存在哪些问题？

- query短且模糊
- doc长且噪声多
- 监督数据标注成本高
- PLM模型仍存在改进空间

#### 如何 基于encoder的LLM检索器？

> 基于encoder的检索器指的是在密集检索中，使用LLM出色的语义能力获取query或doc的向量表示，用向量检索器进行检索召回。

- cpt-text:通过 Contrastive Pre-Training 嵌入文本和代码
  - 论文名称: cpt-text：Text and Code Embeddings by Contrastive Pre-Training
  - 论文地址：https://arxiv.org/abs/2201.10005
  - 动机：
  - 方法：cpt-text 在未标记的数据上使用带负采样的对比学习，将相邻的文本视为正样本，从头训练了四种参数级别的嵌入模型，用以产生文本的高质量向量表示。这种结合预训练模型初始化、大批量对比学习和大规模训练的简单配方可以产生具有广泛能力的高质量文本向量，甚至会超越在领域内数据上微调后的语言模型。
  - 优点：

- GTR：大型双编码器是可推广的检索器
  - 论文名称: GTR：Large Dual Encoders Are Generalizable Retrievers
  - 论文地址：https://arxiv.org/abs/2112.07899
  - 动机：但是对于大多数人来说，从头训练一个LLM的成本是非常高的。因此，有不少研究基于已有的LLM进行微调。
  - 方法：GTR（Generalizable T5-based dense Retrievers）使用T5家族初始化双编码器模型参数，然后在数据集上进行微调。
  - 优点：不管是cpt-text还是GTR，实验都表明，模型尺度越大，其无监督学习和文本搜索任务的迁移学习性能越好。

- TART：带指令的任务感知检索
  - 论文名称: TART：Task-aware Retrieval with Instructions
  - 论文地址：https://arxiv.org/abs/2211.09260
  - 动机：
  - 方法：TART同样基于T5，设计了一个任务感知检索模型，可以对query的信息进行初步判断，以选取跟该query高度相关的任务指令。然后将query与指令一起利用LLM进行编码后再进行检索。与改写不同的是，LLM并没有参与到TART的指令生成中，而是以检索器的身份进行指令、query、doc的编码。
  - 优点：

#### 如何 基于生成式的LLM检索器？

> 上面的研究都旨在利用LLM的强大语义编码能力对query、doc等内容进行编码。但在LLM崭露头角之前，就有不少研究致力于构建end2end式的检索模型，成为生成式检索器（Generative Retriever）。相比先编码再检索，生成式方法通过联合编码器和解码器，直接获取要检索的文档标识符

- DSI:Transformer内存作为可微分搜索索引
  - 论文名称: DSI：Transformer Memory as a Differentiable Search Index
  - 论文地址：https://arxiv.org/abs/2202.06991
  - 动机：
  - 方法：DSI就是一种典型的生成式检索模型，在检索数据集上微调T5模型，直接对query、doc进行编码，然后直接解码输出相关文档的id作为检索结果。
  - 优点：

- LLM-URL:大型语言模型内置于自回归搜索引擎中
  - 论文名称: LLM-URL：Large Language Models are Built-in Autoregressive Search Engines
  - 论文地址：https://arxiv.org/abs/2305.09612
  - 动机：
  - 方法：LLM-URL研究中发现，LLM生产的URL中包含90%以上的query的相关答案，他们利用这一点，设计合适的prompt获取LLM输出的URL，并将其作为生成式检索器的额外输入，直接获取文档相关id。这种方式相当于进行“改写”，只不过是基于生成式检索器之上的。
  - 优点：

### 如何 通过 大模型 进行 排序？

#### 微调LLM进行相似度计算

> 在gpt3等超大型参数模型出现之前，不少研究都利用PLM，将排序任务看作相似度计算任务来获得每个query和doc的相似度得分。RankT5就是这样一种模型，他基于T5直接计算查询-文档对的相关分数，并使用pairwise或listwise计算排名损失进行微调。

- RankT5: 用于具有排名损失的文本排名的微调T5
  - 论文名称: RankT5：Fine-Tuning T5 for Text Ranking with Ranking Losses
  - 论文地址：https://arxiv.org/abs/2202.06991
  - 动机：
  - 方法：RankT5有两种得分计算方法，一种是encoder-decoder结构，另一种则是不需要解码直接根据encoder编码得到排序分数。
  - 作者实验证明，两种结构效果上各有胜负，这也侧面表明decoder作用其实不大，蒸馏等操作可以直接对encoder下手。类似的研究还有很多，只是把backbone换为BERT、BART、GPT等即可。

#### 提示LLM

> 对超大规模LLM进行微调存在成本昂贵的明显问题，不少研究选择利用LLM的提示能力得到query与doc是否相似的答案。

- UPR：利用零样本问题生成改进文章检索
  - 论文名称: UPR：Improving Passage Retrieval with Zero-Shot Question Generation
  - 论文地址：https://aclanthology.org/2022.emnlp-main.249/
  - 会议：ACL2022
  - 动机：排序的实质是进行query和doc间的相似度计算，这一分数也可以看作是根据query获得doc的概率。
  - 方法：UPR利用这一过程的逆向思路，利用prompt提示LLM，针对每一个doc，逐一计算query中各个token的生成概率，并将这一概率作为query和doc的相似度分数。简单理解，就是用LLM根据prompt对每个doc生成对应的query，称为假query。然后将生成后的假query和原query送入语言模型进行打分，计算两者的一个“相似度”。这里的相似度并不是我们熟知的向量相似度，而是“假query复原原query”的概率，其过程如上面公式所示。最后，对这个得分进行排序以获取最终的排序结果。

- RankGTP：ChatGPT擅长搜索吗？作为重新排序代理的大型语言模型研究
  - 论文名称: RankGTP：Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agent
  - 论文地址：https://aclanthology.org/2023.emnlp-main.923/
  - 会议：EMNLP2023
  - 动机：
  - 方法：RankGPT和LLR都采用类似list-wise的方式来获取LLM的排序结果。相比point-wise，list-wise的场景下LLM能够关注到更多的doc信息，直接输出文档id的排序结果，且不需要打分模型的参与。为了解决list-wise场景下输入的doc过长的问题，RankGPT采用了滑动窗口的方法，指定k大小的窗口来获取最终top-k的排序结果。

- LLR:基于大型语言模型的零射击列表式文档重排序
  - 论文名称: LLR：Zero-Shot Listwise Document Reranking with a Large Language Model
  - 论文地址：https://aclanthology.org/2023.emnlp-main.923/
  - 会议：ACL2023
  - 动机：
  - 方法：

- PRP：大型语言模型是具有成对排序提示的有效文本排序器
  - 论文名称: PRP：Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting
  - 论文地址：https://arxiv.org/pdf/2306.17563.pdf
  - 会议：
  - 动机：
  - 方法：PRP的作者认为相比其他两种方式，LLM的对比理解能力更强。而且pairwise的方式既支持生成式模型，又支持打分模型，且因为要比较两个对象，可选择的排序算法较多，如堆排序、冒泡排序、快速排序等，整体方式方法较为灵活。

- Co-Prompt：通过约束生成的离散提示优化零样本重随机
  - 论文名称: Co-Prompt：Discrete Prompt Optimization via Constrained Generation for Zero-shot Re-ranker
  - 论文地址：https://aclanthology.org/2023.findings-acl.61.pdf
  - 会议：ACL2023
  - 动机：
  - 方法：Co-prompt方法将soft prompt条件生成技术应用至point-wise的LLM排序任务，将PLM作为生成器生成soft prompt，然后通过LLM作为鉴别器鉴别，来条件生成最优的prompt。这一方法可以同样被应用于其他提示LLM的任务中，有效提升LLM的提示效果。

## CoT 篇

- [如何提升LLMs：Self-Prompted CoT](CoT/SelfPrompted_CoT/readme.md)
  - 论文名称：Self-prompted Chain-of-Thought on Large Language Models for Open-domain Multi-hop Reasoning
  - 论文地址：https://arxiv.org/pdf/2310.13552.pdf
  - 动机：
    - 开放域多跳推理（ODMR） 局限性：ODMR需要通过明确的推理步骤回答多跳问题，而不依赖于任何提供的上下文。这比有上下文的多跳问答要困难得多，因为模型不能依赖于检索相关段落；
    - 链式思考（CoT） 局限性：
      - 在质量或多样性上有局限性
  - 论文框架：提出了一种自我提示的思维链（SP-CoT）自动化框架，通过大型语言模型（LLMs）自身生成高质量多样化的思维链，用于开放域多轮推理（ODMR）。关键思想是：
    - 自动化流水线生成带有多跳问题和推理链的ODMR数据集
    - 自适应采样选择多样化的高质量CoTs作为示范
    - 通过上下文学习从生成的CoTs中学习自我引导的推理

## 微调数据工程 篇

- [EMNLP'23大模型时代的数据标注——FreeAL](DataEngineering/FreeAL/readme.md)
  - 论文名称：FreeAL: Towards Human-Free Active Learning in the Era of Large Language Models[J]. 
  - 论文地址： https://arxiv.org/pdf/2311.15614
  - 思路：
    1. 数据标注依然重要，完全监督、弱监督的小模型在很多场景下比（未精调）大模型强；
    2. 利用LLM进行标注是完全可行的，小模型可以协同进行过滤、精炼大模型的标签；
    3. 弱监督学习、主动学习这两个领域，我想依然有活着的价值。

- [From Quantity to Quality：如何挑选 具有增强LLM指令调优潜力的数据样例？](DataEngineering/FromQuantityToQuality/readme.md)
  - 论文名称：From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning
  - 论文地址：https://arxiv.org/pdf/2308.12032.pdf
  - GitHub 地址：https://github.com/MingLiiii/Cherry_LLM
  - 动机：如何挑选 **具有增强LLM指令调优潜力的数据样例**？
  - 思路：
    - Learning from Brief Experience：选取有代表性的训练数据 训练 LLaMA；
    - Evaluating Based on Experience：利用训练好模型计算原始数据中所有IFD指标；
    - Retraining from Self-Guided Experience：批量跑得到每个样本的IFD得分，然后选取较高得分（prompt困难样本）的样本，paper中称为cherry samples，用其重新训练模型。

- [Active Instruction Tuning：怎么更好的选择一个新任务来提高模型泛化性？](DataEngineering/ActiveInstructionTuning/readme.md)
  - 论文名称：Active Instruction Tuning: Improving Cross-Task Generalization by Training on Prompt Sensitive Tasks
  - 论文地址：https://arxiv.org/pdf/2311.00288.pdf
  - GitHub 地址：
  - 动机：如何筛选出适合当前给定这个LLM的高质量数据，也就是说高质量是和模型深度绑定的。
  - 提出了一个Prompt Uncertainty 思路：假设有一个原始样本对<prompt, response>，然后对prompt做一些扰动得到promot_v1，其中promot_v1还是要保留大部分prompt语义，然后将prompt和promot_v1分别传给模型，分别拿到response的输出，计算得到两者之间的likelihood值，该值即为Prompt Uncertainty。

- [MoDS: 如何自动筛选高质量数据？](DataEngineering/MoDS/readme.md)
  - 论文名称：MoDS: Model-oriented Data Selection for Instruction Tuning
  - 论文地址：https://arxiv.org/pdf/2311.15653.pdf
  - GitHub 地址：https://github.com/CASIA-LM/MoDS
  - 动机：如何筛选出适合当前给定这个LLM的高质量数据，也就是说高质量是和模型深度绑定的。
  - “高质量”数据的标准是什么？
    - **质量**: **高质量的prompt**以及**对应的高质量response**可以很好的让模型学会遵循指令；
    - **覆盖率**: **prompt的多样性，越多样性越好**；
    - **必要性**: **同一条prompt对不同基座模型的重要度和必要性是不一样的**，如果一条prompt对于基座来说已经很好的输出response了，也就是说模型已经很好的遵循prompt了，不需要再训练了，相反则是模型需要的。
  - “高质量”数据的如何筛选？
    - Quality Evaluation：基于模型打分筛选出高质量的SFT数据；
    - Diverse Data Selection for Seed Instrucitons：**在这份高质量SFT数据集中继续过滤出一个子集，该子集的多样性要足够好，能表征整个数据集**；
    - Augmented Data Selection

- [符尧：别卷大模型训练了，来卷数据吧！](DataEngineering/DataEngineering/readme.md)
  - 论文名称：An Initial Exploration of Theoretical Support for Language Model Data Engineering
  - 论文地址：https://yaofu.notion.site/An-Initial-Exploration-of-Theoretical-Support-for-Language-Model-Data-Engineering-Part-1-Pretraini-dc480d9bf7ff4659afd8c9fb738086eb

- 大模型对代码的记忆痕迹
  - 论文名称：Traces of Memorisation in Large Language Models for Code
  - 论文地址：https://arxiv.org/pdf/2312.11658
  - Github 地址：
  - 会议：
  - 论文方法：该论文主要研究了大语言模型对代码的记忆问题，并比较了代码模型和自然语言模型的记忆率。研究人员构建了自然语言的基准测试集，并通过识别易受攻击的样本构建了代码的基准测试集。他们对多种模型运行了这两个测试集，并进行了数据提取攻击。研究发现，大语言模型对代码也存在数据提取攻击的风险。从可提取的训练数据中，他们成功提取了CodeGen-Mono-16B代码补全模型中的47%数据。研究还发现，随着参数数量的增加，模型记忆的内容也增加，并且模型的预训练数据也容易受到攻击。数据承载者的记忆率高于普通代码或文档，并且不同的模型架构记忆不同的样本。数据泄露具有严重后果，因此该论文敦促研究界采用更广泛的模型和提取技术来进一步调查这一现象，以建立相应的保护措施。

- 避免语言模型评估中的数据污染：动态测试构建与最新材料
  - 论文名称：Avoiding Data Contamination in Language Model Evaluation: Dynamic Test  Construction with Latest Materials
  - 论文地址：https://arxiv.org/pdf/2312.12343
  - Github 地址：
  - 会议：
  - 论文方法：这篇论文提出了最新评估方法（LatestEval），利用最新的文本创建无污染的阅读理解评估，避免数据污染带来的挑战。最新评估通过仅使用最近时间窗口内发布的文本来避免数据污染，并确保不与预训练语言模型的训练语料库重叠。论文开发了一套LatestEval自动化流程，包括：1）收集最新文本；2）识别关键信息；3）构建问题，同时从上下文中删除现有答案，鼓励模型基于剩余上下文推断答案而不是简单复制粘贴。
  - 实验结果表明，相对于先前的基准测试，语言模型在最新评估上几乎不表现出记忆行为，这表明了数据污染的风险大大降低，从而导致更可靠的评估。

- GeomVerse: 对几何推理的大型模型的系统评估
  - 论文名称：GeomVerse: A Systematic Evaluation of Large Models for Geometric  Reasoning
  - 机构：谷歌研究院、Google DeepMind
  - 论文地址：https://arxiv.org/pdf/2312.12241
  - Github 地址：
  - 会议：
  - 论文方法：这篇论文通过几何问题的视角评估了视觉语言模型（VLMs）在多个方面上的推理能力。
  - 通过在多个深度级别上构建该论文的基准测试，实验结果表明，与以前的基准测试所示的推理能力相比，这些模型在几何学（以及一般情况下需要类似推理的其他主题）方面的能力并不如人们所想的那么强大。这在解决更高深度问题时尤为明显，因为解决更高深度的问题需要较长的推理链而不是额外的记忆知识。该论文在该领域的进一步研究中发布了数据集。

-  仅用1%的数据完胜全量数据微调模型!
  - 论文名称：One Shot Learning as Instruction Data Prospector for Large Language Models
  - 机构：
  - 作者：Li, Yunshui and Hui, Binyuan and Xia, Xiaobo and Yang, Jiaxi and Yang, Min and Zhang, Lei and Si, Shuzheng and Liu, Junhao and Liu, Tongliang and Huang, Fei and others
  - 论文地址：arxiv.org/pdf/2312.10302.pdf
  - 相关领域：训练数据构建
  - Github 地址：github.com/pldlgb/nuggets/blob/master/README.md
  - 会议：
  - 论文方法：仅用1%的数据完胜全量数据微调模型!#不懂就问有问必答 论文中提出了一种名为Nuggets”的方法，意欲从堆积如山的指令微调数据中挖掘出黄金数据。这种方法利用大语言模型 (LLM)自身作为数据探索工具，通过One shot learning 或者说是Incontext learning，从庞大的指令数据集中挑选出有益的数据。直观来说，如果某个指令对于某个特定任务的少样本学习(Few shot learning)有帮助，那么这个指令就值得被用于训练。如果这个指令能对多个任务有益，那么它就应该成为主要的数据重点另外，有研究显示，In context learning通过提示(Demonstrations)来隐式微调模型，相当于语言模型在幕后以元优化器的角色进行梯度下降操作。因此，利用在In context learning下的性能来预测指令微调的效果是很有前景的。

## 高效大模型推理 篇

-  有限内存下的高效大模型推理
   - 论文名称：LLM in a flash: Efficient Large Language Model Inference with Limited  Memory
   - 论文地址：https://arxiv.org/pdf/2312.11514
   - Github 地址：
   - 会议：
   - 论文方法：这篇论文主要解决的问题是如何在有限的内存容量下高效地运行超出DRAM容量的大语言模型。通过将模型参数存储在闪存上，并根据闪存内存行为按需将其带入DRAM来解决这一挑战。论文通过构建一个与闪存内存行为相协调的推理成本模型，指导该论文在两个关键领域进行优化：减少从闪存传输的数据量和以更大、更连续的块读取数据。论文介绍了两种主要技术：窗口化策略降低数据传输量，行-列捆绑增加从闪存读取的数据块大小。这些方法使得模型可以在可用DRAM容量的两倍大小下运行，并且与CPU和GPU中的简单加载方法相比，推理速度分别增加了4-5倍和20-25倍。该论文的稀疏意识、上下文适应加载和面向硬件的设计为在内存有限的设备上高效推理大语言模型铺平了道路。

-  ComplexityNet: 通过学习任务复杂度来提高LLM推理效率
   - 论文名称：ComplexityNet: Increasing LLM Inference Efficiency by Learning Task  Complexity
   - 论文地址：https://arxiv.org/pdf/2312.11511
   - Github 地址：
   - 会议：
   - 论文方法：这篇论文主要介绍了ComplexityNet，这是一个专门用于评估任务复杂度的精简语言模型。该模型预测了不同能力的各种语言模型的输出准确性的可能性。作者的初步应用是在Mostly Basic Python Problems (MBPP)数据集上。他们首次创建了一组标签来定义任务复杂度。ComplexityNet在确定任务复杂度方面取得了显著的79%准确率，相比于原始模型的34%准确率有了显著改进。此外，与使用最高复杂度模型相比，ComplexityNet可以有效地减少90%的计算资源使用量，同时保持高达86.7%的代码生成准确率。这项研究表明，通过微调较小的模型来对任务进行分类，可以在使用大型语言模型时在准确性和效率之间取得更平衡的权衡。该论文的发现为优化LLM应用指明了一个有前景的方向，尤其是在资源受限的环境下。

-  超越Chinchilla-Optimal: 在语言模型缩放定律中考虑推理
   - 论文名称：Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws
   - 论文地址：https://arxiv.org/pdf/2401.00448
   - 相关领域：模型结构改进
   - Github 地址：
   - 会议：
   - 论文方法：本论文修改了Chinchilla缩放定律，计算了训练和部署具有给定推理需求和质量的语言模型所需的最佳参数数量和预训练数据大小。研究发现，对于预计存在相当大推理需求（约10亿次请求）的语言模型研究者来说，应该训练比Chinchilla-optimal更小更长的模型。

-  Understanding LLMs：从训练到推理的全面概述
   - 论文名称：Understanding LLMs: A Comprehensive Overview from Training to Inference
   - 论文地址：https://arxiv.org/pdf/2401.02038
   - 相关领域：模型结构改进、预训练
   - 作者：Yiheng Liu, Hao He, Tianle Han
   - Github 地址：
   - 会议：
   - 论文方法：这篇论文讨论了大语言模型（LLMs）的训练技术和推理部署技术的演变，并探讨了低成本训练和部署LLMs在未来的发展趋势。训练方面的讨论包括数据预处理、训练架构、预训练任务、并行训练以及与模型微调相关的内容。在推理方面，论文还涵盖了模型压缩、并行计算、内存调度和结构优化等主题。它还探讨了LLMs的应用，并对它们的未来发展提供了见解。


## 大模型评估篇

- Catwalk: 多数据集的统一语言模型评估框架
   - 论文名称：Catwalk: A Unified Language Model Evaluation Framework for Many Datasets
   - 论文地址：https://arxiv.org/pdf/2312.10253
   - Github 地址：https://github.com/allenai/catwalk
   - 会议：
   - 论文方法：这篇论文介绍了Catwalk，一个为了解决大规模比较NLP模型在多个任务、领域和数据集上的工程挑战而设计的统一界面。它使得在大规模实验中进行公平和可控的比较更加容易。通过一个命令，Catwalk可以在86个数据集上对64个模型进行微调和评估，而无需编写任何代码。

- KGLens: 一种参数化的知识图谱解决方案，用于评估LLM所知和不知道的内容
   - 论文名称：KGLens: A Parameterized Knowledge Graph Solution to Assess What an LLM  Does and Doesn't Know
   - 论文地址：https://arxiv.org/pdf/2312.11539
   - Github 地址：
   - 会议：
   - 论文方法：本文介绍了KGLens这一方法，通过以结构感知的方式从知识图谱中生成自然语言问题，以评估LLM。KGLens使用了参数化的知识图谱，在该图谱中，每个边都附加了一个贝塔分布，用于指导从知识图谱中进行QA测试时如何采样边。随着评估的进行，对参数化的知识图谱的不同边进行采样和评估，从而收敛到更全局的LLM在知识图谱上的性能图景。
   - 实验中，该论文构建了三个领域特定的用于知识评估的知识图谱，包含超过19,000个边，700个关系和21,000个实体。结果表明，KGLens不仅可以评估整体性能，还可以提供LLM的主题、时间和关系分析。这展示了KGLens的适应性和可定制性，强调其基于特定标准的评估能力。

- 人工智能是否能像人类一样具备创造力？
   - Can AI Be as Creative as Humans?
   - 论文地址：https://arxiv.org/pdf/2401.01623
   - 机构：斯坦福大学、罗格斯大学、微软研究院
   - 作者：Haonan Wang, James Zou, Michael Mozer
   - 相关领域：指令微调、模型评估
   - Github 地址：
   - 会议：
   - 论文方法：本文探讨了创造力的定义和评估的复杂性，介绍了一种新的概念——相对创造力。相对于试图普遍定义创造力，该论文将重点放在人工智能能否与假想的人类创造能力相匹配上。这种方法有助于通过统计量化评估AI的创造力，该论文称之为统计创造力。在此基础上，该论文讨论了统计创造力在当下的提示条件自回归模型中的应用。除了定义和分析创造力的指标外，该论文还提出了一种可行的训练指南，有效地将创造力的理论量化与实际模型训练相结合。通过这些多方面的贡献，本文建立了一个连贯、不断演变和变革性的框架，以评估和促进AI模型的统计创造力。

## 大模型预训练篇

- 大模型并不是你所需要的全部
   - 论文名称：Large Language Models aren't all that you need
   - 机构：印度理工学院
   - 作者：Kiran Voderhobli Holla, Chaithanya Kumar, Aryan Singh
   - 论文地址：arxiv.org/pdf/2401.00698
   - 相关领域：模型结构改进、预训练
   - Github 地址：
   - 会议：
   - 论文方法：这篇论文主要探讨了在解决SemEval 2023任务2：多语种复杂命名实体识别方面的架构和系统。作者评估了两种方法，一种是传统的CRF模型，另一种是经过定制头部微调的大型语言模型（LLM），并进行了比较。论文探索了一些新颖的想法，包括：1）衰减辅助损失（具有残差）- 在模型上训练粗粒度命名实体识别的辅助任务，并将该任务作为损失函数的一部分；2）三元标记混合- 在最终的命名实体识别层中，探索了混合相邻标记嵌入的方法；3）任务优化头部- 探索了各种定制头部和学习率用于LLM的最终层。作者还尝试了多个LLM，包括GPT-3，并在最终模型上进行了多种dropout和超参数设置的实验，最终在测试数据上获得了0.67/0.61的micro & macro f1分数。研究结果表明，尽管预训练的LLM相比传统模型带来了很大的性能提升，但通过上述额外的特征/损失/模型工程技术对宏观F1分数的改进是可行的。

- TinyLlama: 一个开源的小型语言模型
   - 论文名称：TinyLlama: An Open-Source Small Language Model
   - 机构：
   - 作者：Peiyuan Zhang, Guangtao Zeng, Tianduo Wang
   - 论文地址：arxiv.org/pdf/2401.02385
   - 相关领域：模型结构改进、预训练
   - Github 地址：github.com/jzhang38/TinyLlama
   - 会议：
   - 论文方法：TinyLlama是一个在大约3个时期内在大约1万亿个标记上预训练的紧凑1.1B语言模型。TinyLlama建立在Llama 2的架构和分词器之上，利用了开源社区贡献的各种进展（例如FlashAttention），实现了更好的计算效率。尽管规模相对较小，但TinyLlama在一系列下游任务中展现了显著的性能。它明显优于具有相似规模的现有开源语言模型。该论文的模型检查点和代码公开在GitHub上，网址为https://github.com/jzhang38/TinyLlama。

- LLM增强LLM：通过组合扩展能力
   - 论文名称：LLM Augmented LLMs: Expanding Capabilities through Composition
   - 机构：谷歌研究院、Google DeepMind
   - 作者：Rachit Bansal, Bidisha Samanta, Siddharth Dalmia
   - 论文地址：arxiv.org/pdf/2401.02412
   - 相关领域：模型结构改进、预训练
   - Github 地址：
   - 会议：
   - 论文方法：这篇论文主要探讨了在大语言模型的基础上如何通过组合来增强模型能力的问题。通过引入交叉注意力机制，将现有的模型与具有特定任务的模型进行组合，从而实现新的能力。作者提出的CALM方法在多个领域和设置下都适用，并通过将PaLM2-S与在低资源语言上训练的较小模型进行组合，在翻译和算术推理等任务上取得了显著的改进。

- LLaMA Pro: 带有块扩展的渐进式 LLaMA
   - 论文名称：LLaMA Pro: Progressive LLaMA with Block Expansion
   - 机构：香港大学、上海交通大学、Tencent PCG实验室
   - 作者：Chengyue Wu, Yukang Gan, Yixiao Ge
   - 论文地址：arxiv.org/pdf/2401.02415
   - 相关领域：模型结构改进、预训练
   - Github 地址：
   - 会议：
   - 论文方法：这篇论文介绍了一种新的后预训练方法，通过扩展Transformer模块，仅使用新语料库进行调整，有效提升模型的知识，避免灾难性遗忘。研究者在代码和数学语料库上进行实验，得到了LLaMA Pro-8.3B模型，该模型基于LLaMA2-7B模型初始，在通用任务、编程和数学方面有出色表现。LLaMA Pro及其指令遵循对应模型(LLaMA Pro-Instruct)在各项基准测试中取得了先进的性能，证明其在LLaMA系列和各种任务中具有卓越的优势和推理能力。该研究为融合自然语言和编程语言提供了有价值的洞见，为在不同环境中有效运行的先进语言模型的开发奠定了坚实的基础。

- 无需注释的病理定位的通用视觉语言预训练
   - 论文名称：Generalizable vision-language pre-training for annotation-free pathology  localization
   - 机构：香港大学、鹏城实验室、中国科学院大学
   - 作者：Hao Yang, Hong-Yu Zhou, Cheng Li
   - 论文地址：arxiv.org/pdf/2401.02044
   - 相关领域：预训练
   - Github 地址：
   - 会议：
   - 论文方法：该论文介绍了一种针对无需注释的病理定位的通用视觉语言预训练模型。该模型的核心优势在于其基于图像注释无关的多级语义结构对比学习，将医学报告中的多粒度医学概念与丰富的图像特征全面对齐，以适应观察到的和新出现的未知病理的多样表达。实验证明，该模型在4个不同的外部数据集上验证了其泛化能力，在定位5种不同病理方面优于6种最先进的方法，甚至超过人类基准，表明其适用于复杂的临床环境。

- ChartAssistant: 通过图表到表格预训练和多任务指令微调的通用图表多模态语言模型
   - 论文名称：ChartAssisstant: A Universal Chart Multimodal Language Model via  Chart-to-Table Pre-training and Multitask Instruction Tuning
   - 机构：香港大学、南京大学、上海交通大学
   - 作者：Fanqing Meng, Wenqi Shao, Quanfeng Lu
   - 论文地址：https://arxiv.org/pdf/2401.02384
   - 相关领域：预训练、指令微调
   - Github 地址：https://github.com/OpenGVLab/ChartAst
   - 会议：
   - 论文方法：这篇论文提出了ChartAssistant，这是一个基于图表的图像语言模型，旨在实现图表理解和推理的通用性。ChartAssistant通过图表到表格解析的预训练和多任务指令遵循的微调，解决了通用多模态模型在泛化和任务特定微调方面的挑战。实验结果显示，与最先进的UniChart方法相比，ChartAssistant在各种图表任务上取得了显著的性能提升，并在实际图表数据上优于OpenAI的GPT-4V(ision)。这篇论文的内容主要是介绍了ChartAssistant的设计与训练方法，并展示了其在图表任务上的性能优势。

- DIALIGHT: 利用大模型轻量级开发和评估任务导向对话系统
   - 论文名称：DIALIGHT: Lightweight Multilingual Development and Evaluation of  Task-Oriented Dialogue Systems with Large Language Models
   - 机构：剑桥大学
   - 作者：Fanqing Meng, Wenqi Shao, Quanfeng Lu
   - 论文地址：https://arxiv.org/pdf/2401.02208
   - 相关领域：模型结构改进、预训练
   - Github 地址：https://github.com/OpenGVLab/ChartAst
   - 会议：
   - 论文方法：



## 机器人篇

- Mobile ALOHA：低成本全身远程操作学习双手机器人移动操作
   - 论文名称：Mobile ALOHA: Learning Bimanual Mobile Manipulation with Low-Cost  Whole-Body Teleoperation
   - 机构：斯坦福大学
   - 作者：Zipeng Fu, Tony Z. Zhao, Chelsea Finn
   - 论文地址：https://arxiv.org/pdf/2401.02117
   - 相关领域：模型结构改进、预训练
   - Github 地址：
   - 会议：
   - 论文方法：本论文介绍了一种学习移动操作任务的系统，该任务需要双手协作和全身控制。使用Mobile ALOHA系统进行数据采集，通过与现有的静态ALOHA数据集联合训练，进行监督式行为克隆，提高了移动操作任务的性能，使得Mobile ALOHA能够自主完成复杂的移动操作任务。通过扩展了移动底盘和全身远程操作界面的ALOHA系统，Mobile ALOHA实现了低成本的整体身体远程操作系统。本论文解决了传统机器人学习中关注的桌面操作的局限性，使得机器人具备了移动和灵活性，可以完成更广泛实用的任务。

## 强化学习篇

- [基于表征工程的生成式语言大模型人类偏好对齐](RLHF/RAHF/)
  - 论文名称：Aligning Large Language Models with Human Preferences through Representation Engineering
  - 论文链接：https://arxiv.org/abs/2312.15997
  - 论文动机：
    - 构建类似ChatGPT生成式语言大模型一般要经过语言模型、提令精调和强化学习三个主要训练步骤，其中**第三步使用强化学习来实现人类期望对齐既有一定的技术难度，又需要多次人工标注反馈**，因而实现上有一定挑战;
    - **经过前两步语言模型和提令精调之后，语言大模型仍然会生成带有偏见、歧视或者令人不适的回答**;
    - 为了**提升大模型的安全性、可用性和可信性，与人类期望对齐是必不可少的步骤**;
    - 然而目前研究表明利用人类反馈的强化学习算法[1]（RLHF）存在训练不稳定、对超参数敏感和训练代价较高等问题。
  - 论文方法：
    - 1. 使用带偏好注释的数据集来让大型语言模型“感知”人类的偏好；
    - 2. 收集模型在不同偏好“刺激”情况下的隐层激活模式；
    - 3. 利用收集到的激活模式及差异来调整模型使其与与人类偏好对齐。

- ICE-GRT: 基于生成强化学习的指令上下文增强模型
   - 论文名称：ICE-GRT: Instruction Context Enhancement by Generative Reinforcement  based Transformers
   - 机构：字节跳动
   - 作者：Chen Zheng, Ke Sun, Da Tang
   - 论文地址：arxiv.org/pdf/2401.02072
   - 相关领域：指令微调、奖励模型、RLHF
   - Github 地址：
   - 会议：
   - 论文方法：这篇论文介绍了ICE-GRT模型，利用基于邻近策略优化（PPO）的人类反馈强化学习（RLHF）来增强大语言模型在领域特定任务中的能力。ICE-GRT在领域内场景中展示了出色的理解和推理能力，不仅能够生成强健的答案，还可以提供答案背后的详细分析。该模型在领域特定任务和12个通用语言任务中表现优秀，相比于同等规模甚至更大规模的大语言模型，取得了最先进的性能。作者对ICE-GRT进行了综合分析，突出了其对大语言模型领域的显著进展。

## 数字人

- 从音频到逼真的人体化：合成对话中的人类
   - 论文名称：From Audio to Photoreal Embodiment: Synthesizing Humans in Conversations
   - 机构：
   - 作者：
   - 论文地址：https://arxiv.org/pdf/2401.01885
   - 相关领域：
   - Github 地址：
   - 会议：
   - 论文方法：该论文提出了一个生成全身逼真的头像的框架，根据双方互动的对话动态进行手势生成。通过语音音频输入，该论文可以输出个体的多种手势动作，包括面部、身体和手部的动作。该论文的方法将向量量化的样本多样性与扩散获得的高频细节相结合，生成更具动态和表现力的动作。该论文使用高度逼真的人体化头像可视化生成的动作，可以表达手势中的重要细微之处（例如冷笑和嘲笑）。为了促进这一研究领域的发展，该论文推出了一种首个多视角对话数据集，可用于逼真重构。实验结果显示，该论文的模型生成适当且多样的手势，优于扩散和向量量化单独的方法。此外，该论文的感知评估凸显了光真度（与网格相比）在准确评估对话手势中细微动作细节方面的重要性。代码和数据集可在网上获得。




## 参考

- 文档领域多模态大模型整理 https://zhuanlan.zhihu.com/p/673470907

