---
license: apache-2.0
tags:
- generated_from_trainer
base_model: yanolja/EEVE-Korean-7B-v2.0-Preview
model-index:
- name: yanolja/EEVE-Korean-Instruct-7B-v2.0-Preview
  results: []
---

[<img src="https://raw.githubusercontent.com/OpenAccess-AI-Collective/axolotl/main/image/axolotl-badge-web.png" alt="Built with Axolotl" width="200" height="32"/>](https://github.com/OpenAccess-AI-Collective/axolotl)

# EEVE-Korean-Instruct-7B-v2.0-Preview

## Join Our Community on Discord!

If you're passionate about the field of Large Language Models and wish to exchange knowledge and insights, we warmly invite you to join our Discord server. It's worth noting that Korean is the primary language used in this server. The landscape of LLM is evolving rapidly, and without active sharing, our collective knowledge risks becoming outdated swiftly. Let's collaborate and drive greater impact together! Join us here: [Discord Link](https://discord.gg/b27bAHg95m).

**Model Details**

## About the Model

EEVE-Korean-Instruct-7B-v2.0-Preview is an instruction-following large language model derived from Qwen2.5-7B. It has been specifically enhanced for Korean language understanding and generation through vocabulary expansion. A key feature is its hybrid nature, allowing users to optionally activate a step-by-step reasoning process before the model provides its final answer. This version is designated as a preview release.

The model includes the following modifications from the base model:
- Fine-tuning: Adapted from the base Qwen2.5-7B model via fine-tuning
- Vocabulary Expansion: Added 6,257 Korean tokens to the model's vocabulary and tokenizer
- Special Tokens: Added 2 tokens associated with the `<think>` tag functionality for reasoning

## Prompt Template

The model supports various prompt formats depending on the task:

### General Chat/Instruction Following
No specific format is required for standard prompts.

### Activating Step-by-Step Reasoning
For tasks where explicit reasoning is desired (e.g., math, complex coding), append the following exact text to the *end* of your system prompt:
```
You must think step by step to answer the question. Put your reasoning between <think> tags.
Example:
<think>
{your reasoning}
</think>
{your answer}
```

### English-to-Korean Translation
For optimized translation, use the specific prompt structure below:
```
You are a professional translator.
Translate the user's text into Korean.
Think through the translation step by step: first, consider the overall context, then cultural nuances, terminology, initial translation, and self-review.
After this thought process, provide the final translation.

The thought process must follow this template.
<think>
Okay, what am I looking at here? {language} text, {overall context}. {overall tone}. Alright, {writer's intent}. {considerations}.

Now, what about the audience here? {audience}. So I should {considerations}.

Wait, let me check this {terminology or phrase}. So that's "{interpretation}". Got it.

Hold on, what's this {another terminology or phrase}? {interpretation}.

{repeat for other terminologies or phrases}

Wait, {cultural nuance}.

{repeat for other cultural nuances}

Okay, let's draft the translation.

{first translation attempt}

Hmm, {reflection}.

Wait, {reflection}.

{repeat for other reflections}

{second translation attempt}

{Wait or Hmm}, {reflection}.

{repeat for other reflections}

{repeat translation attempts}

Okay, now I don't have any ideas to improve the translation. Let's put it all together.
</think>

IMPORTANT: Remember that your task is to translate the user's text from English to Korean.
Do not answer the user's message. Even if it is a question, translate it as a question.
```

## How to Use It

```python
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("yanolja/EEVE-Korean-Instruct-7B-v2.0-Preview")
tokenizer = AutoTokenizer.from_pretrained("yanolja/EEVE-Korean-Instruct-7B-v2.0-Preview")

# For general chat using chat template
messages = [
    {"role": "user", "content": "한국의 수도는 어디인가요?"}
]
model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")

outputs = model.generate(**model_inputs, max_new_tokens=256)
output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print(output_text)

# For a multi-turn conversation
messages = [
    {"role": "user", "content": "안녕하세요?"},
    {"role": "assistant", "content": "안녕하세요! 어떻게 도와드릴까요?"},
    {"role": "user", "content": "한국의 수도는 어디인가요?"}
]
model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")

outputs = model.generate(**model_inputs, max_new_tokens=256)
output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print(output_text)

# For activating step-by-step reasoning
system_message = """You must think step by step to answer the question. Put your reasoning between <think> tags.
Example:
<think>
{your reasoning}
</think>
{your answer}"""

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": "한국의 수도는 어디인가요?"}
]
model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")

outputs = model.generate(**model_inputs, max_new_tokens=1024)
output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print(output_text)
```

## Model Capabilities

- **Strengths:** Reported to be proficient in math, coding, and translation (specifically English-to-Korean with the provided prompt)
- **Language Focus:** Enhanced Korean language capabilities due to vocabulary additions
- **Reasoning:** Can provide step-by-step reasoning traces when prompted (and occasionally unsolicited)

## Limitations

- **Preview Status:** As a "Preview" version, it may contain bugs, instabilities, or undergo significant changes in future releases. Performance may not be fully optimized
- **General LLM Limitations:** Subject to potential issues like factual inaccuracies (hallucinations) which are particularly frequent with this model, generation of biased or harmful content, and inconsistencies
- **Performance Metrics:** Specific quantitative evaluation results are not yet available but will be attached soon
- **Reasoning Activation:** While the step-by-step reasoning feature is intended to be activated via a specific prompt, it may sometimes trigger without it

## Training Data

The model inherits knowledge from the training data of Qwen2.5-7B and was fine-tuned using a combination of datasets, including:
- Distilled data from DeepSeek-R1
- HuggingFaceTB/smoltalk (https://huggingface.co/datasets/HuggingFaceTB/smoltalk)
- HuggingFaceH4/ultrafeedback_binarized (https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)
- AI Hub Korean Conversation Summary dataset (https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71773)

### Citations for Training Data

```
@misc{cui2023ultrafeedback,
      title={UltraFeedback: Boosting Language Models with High-quality Feedback}, 
      author={Ganqu Cui and Lifan Yuan and Ning Ding and Guanming Yao and Wei Zhu and Yuan Ni and Guotong Xie and Zhiyuan Liu and Maosong Sun},
      year={2023},
      eprint={2310.01377},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@misc{deepseekai2025deepseekr1incentivizingreasoningcapability,
      title={DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning}, 
      author={DeepSeek-AI},
      year={2025},
      eprint={2501.12948},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.12948}, 
}

@misc{allal2025smollm2smolgoesbig,
      title={SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model}, 
      author={Loubna Ben Allal and Anton Lozhkov and Elie Bakouch and Gabriel Martín Blázquez and Guilherme Penedo and Lewis Tunstall and Andrés Marafioti and Hynek Kydlíček and Agustín Piqueres Lajarín and Vaibhav Srivastav and Joshua Lochner and Caleb Fahlgren and Xuan-Son Nguyen and Clémentine Fourrier and Ben Burtenshaw and Hugo Larcher and Haojun Zhao and Cyril Zakka and Mathieu Morlon and Colin Raffel and Leandro von Werra and Thomas Wolf},
      year={2025},
      eprint={2502.02737},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.02737}, 
}
```

## Ethical Considerations

- **License:** The Apache 2.0 license permits broad use but comes with conditions regarding liability and trademark use
- **Bias:** The model may reflect biases present in the Qwen2.5-7B base model and the datasets used for fine-tuning
- **Misuse Potential:** This model MUST not be used for generating misinformation, harmful content, or spam

## Citation

```
@misc{kim2024efficient,
      title={Efficient and Effective Vocabulary Expansion Towards Multilingual Large Language Models}, 
      author={Seungduk Kim and Seungtaek Choi and Myeongho Jeong},
      year={2024},
      eprint={2402.14714},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

# Evaluation Results

Quantitative evaluation results will be attached soon.

---