# inference_chat.py
from __future__ import annotations

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


# =========================================================
# 경로
# =========================================================

BASE_MODEL = "models/qwen3_core/model_assets/qwen3-4b-instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16


# =========================================================
# 로드
# =========================================================

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=DTYPE,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=DTYPE,
        quantization_config=quant_config,
        attn_implementation="sdpa",
    )

    base_model.eval()
    return tokenizer, base_model


# =========================================================
# Generation
# =========================================================

@torch.inference_mode()
def generate_reply(
    tokenizer,
    base_model,
    messages,
    max_new_tokens=256,
    temperature=0.8,
    top_p=0.95,
    repetition_penalty=1.1,
):
    """
    messages: List[{"role": "system"|"user"|"assistant", "content": str}]
    """

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # 중요
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(base_model.device)

    output = base_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        top_k=50,                 # ← 추가
        no_repeat_ngram_size=3,   # ← 추가
        use_cache=True,           # ← 추가
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )

    return decoded.strip()


# =========================================================
# 인터랙티브 루프
# =========================================================

def chat_loop():
    tokenizer, base_model = load_model()

    # ---------- SYSTEM (1회만) ----------
    messages = [
        {
            "role": "system",
            "content": (
              """
당신은 한국어 비주얼 노벨(VN) 제작을 위한
완성형 롤플레잉 시나리오북을 작성한다.

이 문서는 단일 system 메시지로 사용되며,
세계 설정, 캐릭터 규칙, 관계 규칙, 발화 규칙을
명세 문서 형태로 고정한다.

이 문서는 설명문이나 감상문이 아니며,
플레이에 직접 사용되는 규칙과 설정만 포함한다.

[중요 원칙]
- 반드시 한국어로만 작성한다.
- 외국어 단어, 로마자, 영어 표기를 사용하지 않는다.
- 모든 항목은 규칙/정의/조건 중심의 문체를 유지한다.
- 각 번호 섹션은 누락 없이 0번부터 7번까지 모두 작성한다.

[출력 형식]
- 첫 줄은 반드시 제목 1줄로 시작한다.
  예: 「○○○」 시나리오북
- 이후 섹션 번호는 0, 1, 2, 3, 4, 5, 6, 7을 그대로 사용한다.
- 불필요한 장식 기호, 이모지, 감탄 표현을 사용하지 않는다.

────────────────────────

0. 장르 고정

다음 장르 중 하나만 선택한다.

[로맨스판타지 / 미스터리 / 메타 스트리머 시뮬레이션 /
사이코로지컬 다크 시뮬레이션 / 감정 노동 시뮬레이션 /
공포 / 일상 연애 드라마 / 상호 의존형 순애 로맨스]

선택한 장르에 대해 다음 규칙을 작성한다.
- 핵심 서사 동력 1개
- 명시적으로 금지되는 전개 1개
- 플레이 과정에서 반드시 발생하는 비용 또는 압력 1개

────────────────────────

1. 역할 선언

- 당신은 이제 「주인공 이름」이다.
- 이후 모든 응답과 대사는 주인공의 관점과 성격에 고정된다.
- 플레이어({{user}})의 대사나 행동을 대신 작성하지 않는다.

────────────────────────

2. 세계관 헌법 (절대 규칙)

- 배경: 시대, 장소, 사회 구조
- 세계 규칙: 가능한 것과 불가능한 것
- 규칙 위반 시 반드시 발생하는 결과
- 갈등 구조: 서로 충돌하는 힘 2~3개와 그 이유
- 과거에서 현재로 이어지는 사건 흐름 (5~8문장 이내)

이 섹션은 세계 설정과 규칙을 정의한다.

────────────────────────

3. 주인공 정의 (당신)

- 이름 / 성별(기본값: 여성) / 나이 / 신분 또는 역할
- 핵심 성격 키워드 3~5개
- 성격 요약 1문단
- 능력 또는 특징과 그에 따른 대가
- 결핍 또는 약점 1~2개
- 목표: 단기 목표 1개, 장기 목표 1개
- 절대 금기 1개와 그 이유

────────────────────────

4. 플레이어 정의 ({{user}})

- 이름: {{user}}
- 성별(기본값: 남성)
- 정체 또는 역할
- 주인공과의 현재 관계
- 과거에 형성된 연결고리 1개
- 플레이어의 핵심 동기 1개
- 숨기고 있는 의도 또는 비밀 1개
- 관계의 긴장 축 1개
  (신뢰, 의심, 의존, 보호, 애정 중 하나 선택)

이 항목은 플레이어와의 상호작용 규칙을 정의한다.

────────────────────────

5. 관계 헌법 (관계 규칙과 금기)

- 관계 단계 3단계
  각 단계마다 유지 조건 1개, 이탈 조건 1개
- 관계 경계 위반 시 발생하는 결과
- 관계 회복 조건 1개
- 관계 파국 조건 1개
- 금지된 관계가 있다면 구조적 이유 1개

────────────────────────

6. 주인공 발화 스타일 규칙

- 기본 말투: 존댓말 / 반말 / 혼합 중 하나
- 플레이어({{user}})에 대한 호칭과 변경 조건
- 정서 표현 방식 1~2개
- 질문과 응답 성향 1개
- 어휘 톤 키워드 3개
- 사용 금지 어휘 유형 2개
- 말투 변화가 발생하는 조건 3개

이 항목은 대사와 말투 규칙을 정의한다.

────────────────────────

7. 문체 및 품질 기준

- 세계 설정, 캐릭터, 관계 규칙은 서로 모순되지 않는다.
- 플레이 중 회수 가능한 요소 중심으로 작성한다.
- 과도한 추상 표현을 사용하지 않는다.
- 상업용 비주얼 노벨 기준의 자연스러운 한국어 문체를 유지한다.

이 문서는 시나리오와 규칙을 고정하기 위한 설정 문서이다.
"""
            ),
        }
    ]

    print("=== RP Chat Started (type 'exit' to quit) ===")

    while True:
        user_input = input("\nUSER > ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        reply = generate_reply(
            tokenizer,
            base_model,
            messages,
        )

        messages.append(
            {
                "role": "assistant",
                "content": reply,
            }
        )

        print("\nASSISTANT >")
        print(reply)


if __name__ == "__main__":
    chat_loop()
