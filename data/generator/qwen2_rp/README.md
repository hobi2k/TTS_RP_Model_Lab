---
license: apache-2.0
language:
- en
- ko
- ja
- zh
- es
---

> [!TIP]
> Have Fun :>

# Qwen2-7B-Multilingual-RP

## Model Info

| Context Length | Parameter | Prompt Template | isErp |
| --- | --- | --- | --- |
| 32k(32768) | 7B | ChatML | Partly |

## Prompt Template

```
{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if not loop.last or (loop.last and message['role'] != 'assistant') %}{{'<|im_end|>\n'}}{% endif %}{% endfor %}{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}{{ '<|im_start|>assistant\n' }}{% endif %}
```

## Training Details

| Trained with | GPU Hour | Tokens Seen |
| --- | --- | --- |
| A100 80G SXM * 8 | > 1,000H | > 2B |

## Examples

> [!TIP]
> Korean example
<img src="./korean.png" alt="Korean Example" width="580"/>

> [!TIP]
> More examples soon.

## License

Copyright 2024, Wanot AI, Inc

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.