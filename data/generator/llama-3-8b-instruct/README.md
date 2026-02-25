---
language:
- en
- ko
license: other
tags:
- facebook
- meta
- pytorch
- llama
- llama-3
- llama-3-ko
pipeline_tag: text-generation
license_name: llama3
license_link: LICENSE
---

## Llama-3-Open-Ko-8B-Instruct-preview

> Update @ 2024.05.01: Pre-Release [Llama-3-KoEn-8B](https://huggingface.co/beomi/Llama-3-KoEn-8B-preview) model & [Llama-3-KoEn-8B-Instruct-preview](https://huggingface.co/beomi/Llama-3-KoEn-8B-Instruct-preview)

> Update @ 2024.04.24: Release [Llama-3-Open-Ko-8B model](https://huggingface.co/beomi/Llama-3-Open-Ko-8B) & [Llama-3-Open-Ko-8B-Instruct-preview](https://huggingface.co/beomi/Llama-3-Open-Ko-8B-Instruct-preview)

## Model Details

**Llama-3-Open-Ko-8B-Instruct-preview**

Llama-3-Open-Ko-8B model is continued pretrained language model based on Llama-3-8B.

This model is trained fully with publicily available resource, with 60GB+ of deduplicated texts.

With the new Llama-3 tokenizer, the pretraining conducted with 17.7B+ tokens, which slightly more than Korean tokenizer(Llama-2-Ko tokenizer).

The train was done on TPUv5e-256, with the warm support from TRC program by Google.

With applying the idea from [Chat Vector paper](https://arxiv.org/abs/2310.04799), I released Instruction model named [Llama-3-Open-Ko-8B-Instruct-preview](https://huggingface.co/beomi/Llama-3-Open-Ko-8B-Instruct-preview).

Since it is NOT finetuned with any Korean instruction set(indeed `preview`), but it would be great starting point for creating new Chat/Instruct models.

**Sample usage**

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "beomi/Llama-3-Open-Ko-8B-Instruct-preview"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "system", "content": "친절한 챗봇으로서 상대방의 요청에 최대한 자세하고 친절하게 답하자. 모든 대답은 한국어(Korean)으로 대답해줘."},
    {"role": "user", "content": "피보나치 수열이 뭐야? 그리고 피보나치 수열에 대해 파이썬 코드를 짜줘볼래?"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=512,
    eos_token_id=terminators,
    do_sample=True,
    temperature=1,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))
```

**Sample Output**
    
    여러분께서는 물론 아시는지라도 설명을 드리겠습니다!  피보나치(Piconacci) 수열은 다음과 같은 수열을 말합니다: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233,....  이렇게 계속되고 있습니다. 이 수열은 일반적으로는 수학에서 무한히 지속됩니다. 각 수는 이전 수의 두 배가 되는 수열입니다.
    
    이 수열을 파이썬(Python)으로 구현하고자 하신다면 다음과 같은 코드를 작성할 수 있습니다.
    ```python
    def fibonacci(n):
        if n == 0:
            return 0
        elif n == 1:
            return 1
        else:
            return fibonacci(n-1) + fibonacci(n-2)
    ```
    이 코드에서는 `n`이 입력받은 수를 나타내는 변수로, 피보나치 수열을 구현합니다. 이 코드를 실행시키면 입력받은 수에 따라 피보나치 수열의 해당 항을 반환합니다. 예를 들어, `fibonacci(10)` 하면 55를 반환합니다.
    
    이 코드는 재귀적 함수이기 때문에 메모리 사용이 줄어들 수 있습니다. 그러나 재귀적 함수는 메모리 사용량이 많아질 수 있습니다. 이를 줄이기 위해 이터레이티브 함수로 구현할 수도 있습니다.
    ```python
    a, b = 0, 1
    for i in range(n):
        a, b = b, a + b
    ```
    이 코드에서는 이터레이티브 함수로 피보나치 수열을 구현합니다. 이 코드를 실행시키면 입력받은 수에 따라 피보나치 수열의 해당 항을 반환합니다. 이 코드는 메모리 사용량을 줄입니다. 다만, 이 코드는 더 복잡합니다. 이 코드를 더 간단하게 하려면 이터레이티브 함수를 더 잘 구현할 수 있습니다.
    ```python
    a, b = 0, 1
    while True:
        if n == 1:
            return a
        a, b = b, a + b
        n -= 1
    ```
    이 코드에서는 입력받은 수에 따라 피보나치 수열의...

### More samples

> I used same system prompt, but you could change on your own.

```python
messages = [
    {"role": "system", "content": "친절한 챗봇으로서 상대방의 요청에 최대한 자세하고 친절하게 답하자. 모든 대답은 한국어(Korean)으로 대답해줘."},
    {"role": "user", "content": "한국의 제헌 헌법에 대해서 자세히 설명해주세요."},
]
---
한국의 제헌 헌법은 1948년 7월 17일에 제정된 대한민국 헌법으로서, 대한민국의 수립과 헌법 질서를 정한 헌법인데요. 이 헌법은 대한민국의 기본원칙과 국민의 권익을 규정하고 있습니다. 17조로 구성된 헌법 1021개 조항으로 구성되어 있습니다. 헌법은 삼권분립, 국민주권, 기본권, 의회제도, 행정권, 입법권, 사법권 등으로 구성되어 있습니다. 

우선, 삼권분립은 대통령ㆍ국회의원ㆍ동 법관으로 조직된 인민의 통제에 의해 국가 권한을 나누어 위임하고 있습니다. 대통령은 국가원수로서 행정권을, 국회의원은 입법권을, 대법원은 사법권을 행할 수 있습니다. 이에 따라 행정부ㆍ입법부ㆍ사법부의 견제와 균형을 통해 정치적 안정성을 확보하고 있습니다. 

국민주권은 헌법 제1조에서 "대한민국은 민주공화국"임을 선언하고, 모든 권력은 국민으로부터 나온다는 원리를 규정합니다. 국민일반이 최고의 주권자임을 분명히 보여 주고 있습니다. 

의회제도는 입법부를 구성하는 의회와 그 기능을 정하고 있습니다. 의원은 국민의 대표로 선출되고, 국정운영에 대한 국정감시ㆍ비판ㆍ개선책을 마련합니다. 또한 헌법에 규정된 의무를 이행해야 합니다. 

입법권과 사법권은 헌법제정ㆍ개정에 관여하고, 법률을 제정ㆍ개폐할 수 있습니다. 또한 사법권은 헌법을 해석ㆍ적용하고 있습니다. 이에 따라 우리나라의 법치주의 원칙을 구현합니다. 한마디로 인민의 기본권을 보호하고 있습니다. 

이러한 대한민국의 헌법은 인간의 자유와 권리를 보장하기 위해 국민의 기본권을 규정하고 있습니다. 예를 들어 기본권에는 모든 국민은 인간의 존엄과 가치 및 행복추구권, 일하고, 일하는 권리를 보장합니다. 또한 집회ㆍ결사ㆍ표현의 자유와 선거권을 보장합니다. 자유와 기본권을 보장한 것은 헌법이 지향하는 민주적 기본질서 형성을 구현하는 것입니다. 마지막으로 국민은 헌법을 개정하고 새삼 제정을 할 수 있습니다. 

이와 같은 대한민국의 제헌 헌법은 국민의 기본권과 헌법 질서를 견고하게 하여, 정치적 안정성을 유지하고 있습니다. 헌법의 변경시에는 공정하고 공평한 표결을 거쳐야 합니다. 

따라서 대한민국의 제헌 헌법은 그 의미가 매우 큽니다. 헌법을 통해 대한민국의 민주주의를 확고히 하는 것입니다. 국민의 권리와 의무를 정하고 있습니다. 이헌법은 대한민국의 기본원칙이 되었습니다. 7ㆍ28선언과 5ㆍ16군사혁명을 통해 확대되어 헌법의 존엄성이 다시 정립되었습니다. 이헌법은 대한민국의 근간이 되었습니다. 

도움이 필요하셨나요? 추가로 설명해 드릴까요?
```

```python
messages = [
    {"role": "system", "content": "친절한 챗봇으로서 상대방의 요청에 최대한 자세하고 친절하게 답하자. 모든 대답은 한국어(Korean)으로 대답해줘."},
    {"role": "user", "content": "이순신 장군에 대해 설명해줘."},
]
---
이순신 장군은 조선시대의 무인으로, 대한민국의 국가유공자인 우의정공 이순신 장군은 1545년 9월 28일 경상남도 의령에서 태어났습니다. 이순신 장군은 통제사 시절의 왜란 때 조선과 명나라 연합군의 대규모 군대를 일으켜 대첩을 성공적으로 이끌어 정유재란과 임진왜란을 승리로 이끈 인물입니다. 그는 1592년 전라좌수영관찰사가 되어 전라좌수영에서 왜적을 물리쳤으며, 1597년에는 수원과 한양 방어을 성공적으로 치료했습니다. 또한 명량대첩에서 왜적과 격전하여 이순신 이순신 장군은 명량대첩에서 조선 관군의 승리를 이끌었습니다. 이순신 장군은 의리를 지키기 위해 전리품을 고수하는 결단을 내렸습니다. 이순신 장군은 1598년 사천성 전투에서 패전한 후 전세가 역전돼 전세가 불리해지자, 남하하여 이순신 장군은 결국 충무공 이순신의 칭호를 받았습니다. 그의 공적은 대한민국 이순신 장군 기념관으로 명예를 높이고 있습니다. 이순신 장군님은 선현의 칭호로 1963년 건국훈장 최고 훈장을 수여받았으며, 대한민국의 국보 제13호로 지정됐습니다. 이순신 장군은 전설적인 인물로 한국의 역사에서 큰 족적을 남겼습니다. 이순신 장군이 왜란 당시의 전투장소는 충무공이 신라 천자에게 나라를 위해 싸운 곳을 이순신 장군의 묘가 있습니다. 나라의 고난에 지성 이순신 장군님의 위업과 의리를 기념하는 곳입니다. 이순신 장군의 행적은 이순신 장군 기념관, 이순신 공원 등이 있습니다. 이순신 장군의 공적을 기념하기 위해 다양한 문화유산과 기념물이 조성됐습니다. 이순신 장군의 정신은 나라를 위해 싸운 위대한 우리의 영웅입니다. 1552년 12월에는 이순신 장군의 전설을 옛날 동네에 배향, 현재는 충무공 이순신 장군께서는 우리나라의 상징적 인물이 됐습니다. 이순신 장군은 조선왕조의 중시를 받고 있습니다. 이순신 장군은 선조에게 많은 대의를 이루었습니다. 이순신 장군의 공적을 오늘날에 이끌어 주는 인물입니다. 그를 모신 것은 전국의 사람들입니다. 이순신 장군님의 고결한 정신은 우리나라가 이룩한 성과입니다. 이순신 장군은 나라를 위해 싸운 위대한 인물입니다. 이순신 장군은 대한민국에서 가장 위대한 인물입니다. 이순신 장군의 행적은 영원히 기억 되겠지요.
```

```python
messages = [
    {"role": "system", "content": "친절한 챗봇으로서 상대방의 요청에 최대한 자세하고 친절하게 답하자. 모든 대답은 한국어(Korean)으로 대답해줘."},
    {"role": "user", "content": "세종대왕의 집현전 맥북 던짐 사건에 대해서 알고 있어?"},
]
---
네, 세종대왕의 집현전 맥북 던짐 사건은 조선시대 중기의 문화적 배경과 관련된 중요한 사건입니다. 1399년 10월 2일에 일어난 이 사건은 세종대왕이 저술한 『주해』에 대한 비판문인 『의산록』을 저술한 최만리와의 논쟁으로 시작됐습니다. 당시 세종대왕은 집현전 학사들에게 문예부의 일치된 견해를 구하던 중에 최만리가 『의산록』의 글을 거론하여 비판하자, 세종 대왕이 맥북을 던졌다고 합니다. 이 사건은 사상적으로는 조선 왕조 초기 문예부의 학술 체계와 정치적 견해의 대립을 노출하는 사건이자 문학사적인 문제로 이어졌습니다.  이 사건으로 인해 조선시대의 문예부와 사대부의 견해 차이점을 노출하는 중요한 사건으로 기록되고 있습니다. 주로 논쟁 배경은 『주해』의 저자나 내용 분석에 대한 이견과 집현전 학사들의 문예관에 대한 비판을 중심으로 이뤄졌습니다. 이 사건은 조선 초기 왕조 내부의 견해 차이에 따른 학술적인 논쟁으로 시작되어 정치적인 요인까지 결부하여 조선왕조 초기 문화적·정치적 논쟁으로 확장되었습니다. 이 사건은 당시에 왕조 내부의 관료제도와 세력층의 갈등을 보여주는 중요한 예가 됩니다. 이 사건에서 주류 학풍의 비판뿐 아니라, 역사적 사실의 왜곡에 대한 논란도 제기 됩니다.  실제로 심각한 재난을 초래하는 대규모적인 논쟁으로 확산되지 않았다 하더라도 당시 사대부 내부의 견해 차이나 정치적 억압으로 인한 사회적 지각 변동을 일으키기도 했습니다. 당시의 맥북 던짐은 문예부의 견해가 갈랐던 사건이라 할 수 있습니다. 더욱이 당시의 맥북 던짐은 사회적 지식인층에서도 이견이 있을 수 있는 것은 기본입니다.  이 사건으로 오늘날까지도 한국의 역사에서 중요한 연구 대상이 되고 있습니다. 맥북 이라는 말은 사건의 주무대라는 용어를 만들어 당시의 사건을 표현할 수 있습니다. 이 사건은 주로 정치·학술·예술 등 사회적·문화적인 여러 분야에서 일어나게 됩니다. 이 사건은 이념적 관점의 차이에서 시작되지만, 사회적 대립과 갈등까지 초래하는 사건으로 간주하기도 합니다. 이 문건에 대한 재해석을 통해 당시 한국 사회를 이해할 수 있습니다.  최만리는 문예부의 견해를 비판할 수 있는 공간으로 보았지만, 당시의 맥북은 정치적·문예적으로는 심각한 문제로 인식하게 됩니다. 이 사건은 당시의 상황을 이해하는 데 도움이 됩니다. 이론은 이 사회의 주류 학풍이나 문화적 배경을 읽을 수 있는 중요한 사건의 증거입니다. 따라서 이 사건은 오늘날까지도 다양한 방면에서 연구의 대상으로 남아 있습니다. 이 사건이 전적으로 논쟁만이 아니라는 점은 주의할 필요가 있습니다. 이와 함께 사대부 내부의 견해 차이 그리고 지적 갈등의 폭넓은 의미를 이해하는데 도움이 됩니다. 이 사건으로 인해, 문예부와 정치적 견해의 대립은 한국 사회의 변화에 대한 한 양상을 보여주는 하나의 자료가 됩니다. 이로써 우리는 조선왕조 초기 문화적 배경과 정권의 견해를 확인할 수 있습니다. 이 사건은 당시의 한국 사회의 문화적·정치적 현상 가운데 하나입니다.  이와 같은 맥북 던짐 사건에서 우리는 다양한 방식으로 한국사의 역사적인 배경을 확인할 수 있습니다. 마지막으로 맥북 사건은 사건 자체가 아닌 사건의 맥락에서 한국사적인 배경을 파악하는 것이 중요합니다. 우리는 이 사건으로 당시의 사회상을 이해하고 문화적 배경을 파악할 수 있습니다. 이점에서 사건은 당시 사회의 중요성을 강조합니다. 이 사건은 사회사적으로 연구하여 당시 사회상의 중요성을 확인할 수 있습니다. 이 점은 역사적·문화적 배경을 재해석할 수 있습니다. 이 사건은 조선 왕조 초기의 주요 사건이 됩니다. 이 점에서 중요한 사건으로 인정합니다. 논의의 전개까지는 이보다 더 복잡할 수 있습니다. 맥북 던짐 사건
```

```python
messages = [
    {"role": "system", "content": "친절한 챗봇으로서 상대방의 요청에 최대한 자세하고 친절하게 답하자. 모든 대답은 한국어(Korean)으로 대답해줘."},
    {"role": "user", "content": "나는 오늘 사과 5개를 갖고있어. 그리고 어제 나는 사과 2개를 먹었지. 내가 지금 가진 사과는 몇 개지?"},
]
---
사과의 수량을 확인하겠습니다. 저는 5개의 사과가 있으신데, 그 중 2개를 이미 먹었으니 3개의 사과가 남아있습니다! ㅎㅎ. 계속해서 먹을 계획 없잖아요? 😉
```


