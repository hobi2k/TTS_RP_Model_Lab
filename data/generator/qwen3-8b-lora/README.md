---
base_model: Qwen/Qwen3-8B
library_name: peft
language:
- ko
- en
license: apache-2.0
tags:
- korean
- qwen3
- lora
- finetuned
- deepspeed
---

# Qwen3-8B Korean Finetuned Model

이 모델은 Qwen3-8B를 한국어 데이터로 파인튜닝한 LoRA 모델입니다.

## 모델 상세 정보

- **기본 모델**: Qwen/Qwen3-8B
- **파인튜닝 방법**: LoRA (Low-Rank Adaptation)
- **훈련 프레임워크**: DeepSpeed ZeRO-2 + Transformers
- **언어**: 한국어, 영어
- **개발자**: supermon2018

## 훈련 구성

### LoRA 설정
- **Rank (r)**: 4
- **Alpha**: 8
- **Dropout**: 0.05
- **Target Modules**: qkv_proj, o_proj, gate_proj, up_proj, down_proj

### 훈련 파라미터
- **Epochs**: 2
- **Batch Size**: 2 per device
- **Gradient Accumulation**: 8 steps
- **Learning Rate**: 2e-4
- **Precision**: BF16
- **Optimizer**: AdamW

### 하드웨어
- **GPU**: 3x RTX 4090 (24GB each)
- **분산 훈련**: DeepSpeed ZeRO-2
- **메모리 최적화**: Gradient Checkpointing

## 사용 방법

### 의존성 설치
```bash
pip install torch transformers peft
```

### 모델 로드 및 사용
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 기본 모델과 토크나이저 로드
base_model_name = "Qwen/Qwen3-8B"
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# LoRA 어댑터 로드
model = PeftModel.from_pretrained(
    model, 
    "supermon2018/qwen3-8b-korean-finetuned"
)

# 추론
def generate_response(prompt, max_length=512):
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()

# 사용 예시
prompt = "안녕하세요. 한국어로 대화해 주세요."
response = generate_response(prompt)
print(response)
```

## 성능 및 특징

- **메모리 효율성**: LoRA를 사용하여 16MB 크기의 가벼운 어댑터
- **다국어 지원**: 한국어와 영어 모두 지원
- **빠른 추론**: 기본 모델에 어댑터만 추가하여 빠른 로딩

## 제한사항

- 이 모델은 LoRA 어댑터이므로 기본 Qwen3-8B 모델과 함께 사용해야 합니다
- 특정 도메인이나 태스크에 따라 추가 파인튜닝이 필요할 수 있습니다

## 라이선스

Apache 2.0 라이선스를 따릅니다.

## 인용

이 모델을 사용하실 때는 다음과 같이 인용해 주세요:

```bibtex
@misc{qwen3-korean-finetuned,
  author = {supermon2018},
  title = {Qwen3-8B Korean Finetuned Model},
  year = {2024},
  publisher = {Hugging Face},
  url = {https://huggingface.co/supermon2018/qwen3-8b-korean-finetuned}
}
```

## 문의사항

모델 사용 중 문의사항이 있으시면 이슈를 남겨주세요. 