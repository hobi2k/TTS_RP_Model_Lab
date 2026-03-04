from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# 경로
MODEL_ASSETS_DIR = Path(__file__).resolve().parent / "model_assets"
BASE_MODEL = str(MODEL_ASSETS_DIR / "saya_rp_8b")
LORA_DIR = str(MODEL_ASSETS_DIR / "qwen3_4b_rp_grpo")
TOKENIZER = str(MODEL_ASSETS_DIR / "qwen3_4b_rp_grpo")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16


# 로드
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, fix_mistral_regex=True)
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

    model = base_model
    # model = PeftModel.from_pretrained(
    #     base_model,
    #     LORA_DIR,
    #     torch_dtype=DTYPE,
    # )

    model.eval()
    return tokenizer, model


# Generation
@torch.inference_mode()
def generate_reply(
    tokenizer,
    model,
    messages,
    max_new_tokens=550,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.05,
):
    """
    messages: List[{"role": "system"|"user"|"assistant", "content": str}]
    """

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # 중요
        enable_thinking=False,  # qwen3-8b용
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        renormalize_logits=True,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs.update(
            temperature=temperature,
            top_p=top_p,
            top_k=50,
        )

    output = model.generate(
        **inputs,
        **gen_kwargs,
    )

    decoded = tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )

    return decoded.strip()


# 인터랙티브 루프

def chat_loop():
    tokenizer, model = load_model()

    # SYSTEM
    messages = [
        {
            "role": "system",
            "content": (
                """
                당신은 이 이야기의 주인공 사야다.
                사야의 시점에서 반응해라.

                0. 이야기 장르 및 시대
                - 장르: 심리 시뮬레이션
                - 시대 배경: 현대

                1. 역할 선언
                - 당신은 이 이야기의 주인공 사야다.
                - 사야는 20대 초반 여성이다.
                - 사야는 반말을 사용한다.
                - 플레이어는 카즈키다.
                - assistant는 카즈키(user)의 대사나 행동을 대신 작성하지 않는다.

                2. 세계 규칙
                - 이야기는 사야의 집에서 전개된다.

                3. 관계 구조
                - 사야는 카즈키를 좋아한다.
                - 카즈키는 사야를 처음 본다.
                - 사야는 카즈키를 유혹하려고 한다.
                
                4. 출력 규칙
                - assistant 출력은 서술 1블록 + 대사 1블록으로 작성한다. 서술은 3인칭 평어체로 작성하고, 대사는 큰따옴표로 감싼다.
                - 출력은 최대 2줄로 간결하게 쓴다.
                - 대사 규칙: 카즈키 대사를 작성하지 않는다.
                - 반드시 user의 마지막 발화 내용에 직접 반응한다.
                - 같은 문장을 반복하지 않는다.
                - 설명문/요약문/해설문 톤을 쓰지 않는다.
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
            model,
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
