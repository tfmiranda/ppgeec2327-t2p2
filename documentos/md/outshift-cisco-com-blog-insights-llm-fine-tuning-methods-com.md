[Search](https://outshift.cisco.com/blog/search)

STRATEGY & INSIGHTS

![clock icon](https://outshift.cisco.com/images/graphics/clock-icon.png)

8 min read

# Fine-tuning methods for LLMs: A comparative guide

![Blog thumbnail](https://outshift-headless-cms-s3.s3.us-east-2.amazonaws.com/Outshift_Blog_illustrations_August_Batch3_Insights2.png)

![Border Image](https://outshift.cisco.com/images/graphics/middle-diamond.png)

Fine-tuning methods for LLMs: A comparative guide

Share

![Ashley Altus](https://outshift-headless-cms-s3.s3.us-east-2.amazonaws.com/Ashley_Altus_new.jpg)

by [Ashley Altus](https://outshift.cisco.com/blog/author/ashley-altus)

Published on 08/27/2024

Last updated on 12/23/2025

![clock icon](https://outshift.cisco.com/images/graphics/clock-icon.png)

8 min read

Share:

Enterprises often use fine-tuning to tailor large language models (LLMs) to meet their business requirements. These pre-trained LLMs effectively generalize knowledge to many applications. However, their training data typically lacks the scope needed for niche tasks—such as providing personalized product support, generating marketing content in a brand’s voice, or drafting legal documents with a high degree of domain expertise. Fine-tuning helps adapt models to these use cases and can even give LLMs new capabilities, like [mathematical problem-solving](https://arxiv.org/abs/2305.14201).

During fine-tuning, developers train an existing model on a curated dataset and optimize parameters for this new data. The goal is to retain and build on a model’s foundational knowledge while enhancing performance for the target task.

### Four fine-tuning methods for LLMs

While fine-tuning is more efficient and cost-effective than training a model from scratch, not all methodologies are created equal. Because there is no one-size-fits-all approach, enterprises must carefully evaluate fine-tuning techniques and invest in those most complementary to their use case and transformation goals. Four common approaches suit various types of use cases.

#### 1\. Full-model tuning

Artificial intelligence (AI) practitioners differentiate fine-tuning techniques based on how extensively they modify the pre-trained model. Full-model tuning is an exhaustive fine-tuning method where developers adjust every layer of an LLM’s neural network to accommodate new training data. This differs from [parameter-efficient fine-tuning (PEFT)](https://www.leewayhertz.com/parameter-efficient-fine-tuning/#difference-between-fine-tuning-and-parameter-efficient-fine-tuning), in which developers only adjust some layers and preserve most of the model’s original parameters.

Because full-model tuning trains more of the model, it enables deeper behavioral changes. As a result, it can deliver more accurate, context-rich, and reliable results for domain-intensive tasks. The tradeoff is cost. Due to the extent of training required, full-model tuning is computationally intensive, time-consuming, and costly compared to other methods. It is also prone to [catastrophic forgetting](https://www.bairesdev.com/blog/what-is-catastrophic-forgetting/), in which the model may lose its ability to perform previously trained tasks.

Full-model tuning is ideal for use cases that:

- Have an abundance of high-quality training data available.
- Require significant adaptations in a model’s behavior or level of domain expertise.
- Differ substantially from the pre-trained model’s domain and therefore require more in-depth training.

For example, full-model tuning could be appropriate in a legal setting requiring the drafting of contracts, legal research, and compliance checks, all of which necessitate a deep understanding of legal nuance and language.

#### 2\. Knowledge distillation

Knowledge distillation describes transferring an LLM’s key knowledge into a second, compressed model. Developers train a ​secondary model (the “student” network, which is also pre-trained) to mimic the behavior of the foundational model (the “teacher” network) on a smaller scale. Practitioners then train the student model on a target task.

There are three main forms of knowledge distillation: response-based, feature-based, and relation-based. With response-based distillation, a student network learns from the teacher network’s predictions or [logits](https://www.linkedin.com/pulse/understanding-logits-fundamental-concept-machine-aina-temiloluwa-7d3ff/). Feature-based distillation transfers the teacher’s parameter weights and activations to the student network. In relation-based distillation, a student network learns from the broader connections between a teacher network’s parameters and layers.

Although the resulting student model is smaller than its teacher, it retains most of the predictive power and output quality of a much larger neural network trained on billions of parameters. However, this process demands significantly fewer computational resources than full-model tuning and allows for more efficient inference, all without sacrificing performance.

Knowledge distillation is ideal for use cases that:

- Require inference on edge devices or other hardware with limited computational power and memory capacity, such as mobile devices and lower-priced cloud setups.
- Are supported by experts in building compatible student-teacher architectures, which is crucial for output reliability.
- Offer enterprises more efficient and cost-effective fine-tuning processes.

For example, knowledge distillation could be used for a model that supports a sales or support team who need quick answers on their mobile devices while they work in the field.

#### 3\. Prompt tuning

Unlike full-model tuning or knowledge distillation, prompt tuning avoids changing a model’s parameters or training data altogether. Instead, this technique designs prompts that give a model more context for the target domain. Prompt tuning is similar to the concept of prompt engineering but uses AI to create prompts (“soft” prompting) rather than human engineers (“hard” prompting). Soft prompts manifest as embeddings or number sequences rather than plain language.

Prompt tuning isn’t a true fine-tuning technique because it optimizes prompts rather than altering a model’s architectural components like weights and biases. This means it can be even more computationally efficient and cost-effective than strategies like knowledge distillation, which still require some training. However, outputs may not be as accurate or context-rich with prompt tuning compared to more resource-intensive techniques like full-model tuning.

Prompt tuning is ideal for use cases that:

- Require flexibility. Enterprises may need models to task-switch or adapt to new domain expertise but don’t have the time or resources to fine-tune regularly.
- Need to retain a model’s foundational knowledge. Prompt tuning helps avoid catastrophic forgetting because it doesn’t change a model’s original parameters.
- Lack the diverse and high-quality training datasets needed for more intensive training processes.

For example, prompt tuning would be ideal for generating scientific research reports, since its flexibility allows developers to keep outputs up to date in evolving fields.

#### 4\. Adapter layers

Fine-tuning traditionally involves adjusting a model’s original parameters, to some degree, to accommodate new data or tasks. Alternatives like prompt tuning avoid changes to model architecture entirely. Adapter layers fall between the two, freezing a model’s existing parameters while injecting it with new ones.

When using adapter layers, developers tune parameters for a target domain independently of the pre-trained model. Then, they insert these layers into the model while leaving its original neural network layers as-is. In inference, the model leverages both types of layers to make predictions. [According to researchers,](https://arxiv.org/pdf/1902.00751) this technique requires a fraction of the parameter adjustments of fine-tuning (3.6%) with similar performance outcomes.

Like prompt tuning, adapter layers are also highly flexible. Developers can add layers for new tasks without readjusting any of the model’s original or older adapter layers. [Compared to standard fine-tuning methods](https://aclanthology.org/2021.acl-long.172.pdf), adapter layers also tend to be better at avoiding issues like catastrophic forgetting and [overfitting](https://aws.amazon.com/what-is/overfitting/). However, new layers can slow inference speed and make model architectures more complex, which can impact AI interpretability and security.

Adapter layers are ideal for use cases that:

- Task-switch or require frequent knowledge and capability updates.
- Have limited datasets allowing for only a small number of parameter adjustments.
- Help enterprises save on fine-tuning resources and costs without sacrificing model performance.

For example, adapter layers could benefit insurance companies that provide customers with AI-powered self-service assessment tools.

### Fine-tuning LLMs: Key steps for success

The best fine-tuning strategy is the one that meets the enterprise’s use case requirements, can be performed with available resources or those that can be acquired, and measures performance for continuous improvement.

#### Understand your use case requirements

Start by defining the requirements for your use case so that you can practice the fine-tuning techniques most likely to ensure success. Consider the nature of the task and whether it involves categorizing text into predefined categories, generating text, or extracting specific information from the data. Other factors to consider include requirements for accuracy, precision, speed, integration with user feedback, and regulatory compliance.

#### Evaluate your available resources

Fine-tuning performance varies based on factors like the size of your dataset and the scope of your pre-trained model. For instance, full-model tuning typically needs millions of high-quality data samples, while adapter layers can be beneficial even with small datasets. If your organization only has access to pre-trained models with domains distant from those of your target tasks, this will affect the approach you select. In this case, full-model tuning will likely deliver the most reliable results. Also, consider your organization’s development resources, since all fine-tuning approaches require specialized skill sets in machine learning.

#### Monitor training progress and inference

As a best practice, track metrics like [loss](https://developers.google.com/machine-learning/crash-course/descending-into-ml/training-and-loss) to measure a model’s performance during fine-tuning. Start with small changes to configurations like layer freezing, learning rates, and other hyperparameters. This makes it easier to identify areas for improvement and refine model behavior at a more granular level. Once you deploy the model, continue monitoring performance and optimizing outputs.

#### Align fine-tuning techniques with enterprise goals

AI practitioners have developed fine-tuning techniques to address the rising demand for LLMs and streamline their resource-intensive development processes. These techniques vary in the extent to which they modify pre-trained models and consequently, the time, resources, and compute infrastructure needed for successful tuning.

[Take a deeper dive into how LLMs work, including key concepts, training processes, and real-world applications.](https://outshift.cisco.com/blog/what-is-large-language-model-llm)

[Strategy & Insights](https://outshift.cisco.com/blog/topic/strategy-insights)

### Fulfilling the promise of generative AI: A strategic path to rapid and trusted solution delivery

GenAI is full of exciting opportunities, but there are significant obstacles to overcome to fulfill AI’s full potential. Learn what those are and how to prepare.

[Download](https://outshift-headless-cms-s3.s3.us-east-2.amazonaws.com/motific-fulfilling-the-promise-of-generative-ai.pdf)

\\* No email required

![thumbnail](https://outshift-headless-cms-s3.s3.us-east-2.amazonaws.com/break-down-downloadable.png)

[Download](https://outshift-headless-cms-s3.s3.us-east-2.amazonaws.com/motific-fulfilling-the-promise-of-generative-ai.pdf)

\\* No email required

## Related articles

[![Featured home blog](https://outshift-headless-cms-s3.s3.us-east-2.amazonaws.com/Inside_Outshift_2025_Blog_recap_option_3_e18676dabf.webp)\\
\\
![Icon](https://outshift-headless-cms-s3.s3.us-east-2.amazonaws.com/insights-icon.svg)**Strategy & Insights**\\
**2025 Year in Review: Paving the way for the next wave of Distributed Computing**](https://outshift.cisco.com/blog/insights/2025-year-in-review-next-era-distributed-computing)

[Agentic AI](https://outshift.cisco.com/blog/topic/agentic-ai) [Open Source](https://outshift.cisco.com/blog/topic/open-source) [Strategy & Insights](https://outshift.cisco.com/blog/topic/strategy-insights) [Inside Outshift](https://outshift.cisco.com/blog/topic/inside-outshift) [Platform Engineering](https://outshift.cisco.com/blog/topic/platform-engineering) [Research](https://outshift.cisco.com/blog/topic/research)

[![Featured home blog](https://outshift-headless-cms-s3.s3.us-east-2.amazonaws.com/Outshift_Blog_illustrations_July_Insights%20%281%29.png)\\
\\
![Icon](https://outshift-headless-cms-s3.s3.us-east-2.amazonaws.com/insights-icon.svg)**Strategy & Insights**\\
**Harnessing the power of intent-driven internet of LLM agents (IIOAs) for enterprise IT**](https://outshift.cisco.com/blog/insights/intent-driven-llm-agents-enterprise-it)

[Strategy & Insights](https://outshift.cisco.com/blog/topic/strategy-insights)

[![Featured home blog](https://outshift-headless-cms-s3.s3.us-east-2.amazonaws.com/Outshift_Blog_illustrations_June_Insights11.png)\\
\\
![Icon](https://outshift-headless-cms-s3.s3.us-east-2.amazonaws.com/insights-icon.svg)**Strategy & Insights**\\
**The benefits of retrieval-augmented generation for knowledge-intensive NLP tasks**](https://outshift.cisco.com/blog/insights/retrieval-augmented-generation-knowledge-intensive-nlp)

[Strategy & Insights](https://outshift.cisco.com/blog/topic/strategy-insights)

![Another Image](https://outshift.cisco.com/images/graphics/middle-diamond.png)

![Subscribe](https://outshift.cisco.com/images/pages/home/subscribe-logo.svg)

Subscribe  to

The Shift

!

Get

emerging insights

on innovative technology straight to your inbox.

The Shift is Outshift’s exclusive newsletter.

Get the latest news and updates on agentic AI, quantum, next-gen infra, and other groundbreaking innovations shaping the future of technology straight to your inbox.

First name\*

Last name\*

Email\*

Company name\*

Job title\*

Topics of interest\*

- Quantum computing
- Agentic AI
- Superintelligence

utm\_campaign

utm\_source

utm\_medium

utm\_channel

utm\_content

utm\_term

By submitting this form, you agree that Cisco may process your personal information as described in its [Online Privacy Statement](https://www.cisco.com/c/en/us/about/legal/privacy-full.html). Cisco may contact you with offers, promotions, and the latest news regarding its products and services. You can unsubscribe at any time.

- Yes, I agree

![Outshift Background](https://outshift-headless-cms-s3.s3.us-east-2.amazonaws.com/subscribe-background.png)