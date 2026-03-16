"""단일턴 RP 추론 스크립트(LoRA adapter 결합형).

특징:
- Base 모델 위에 LoRA adapter를 덧씌운 상태로 추론한다.
- 터미널에서 대화형으로 테스트할 때 사용한다.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


# 경로
MODEL_ASSETS_DIR = Path(__file__).resolve().parent / "model_assets"
BASE_MODEL = str(MODEL_ASSETS_DIR / "qwen3_4b_sft")  # .../model_assets/qwen3-4b
LORA_DIR = str(MODEL_ASSETS_DIR / "qwen3_4b_grpo")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16


def _has_tokenizer_files(model_dir: Path) -> bool:
    """주요 tokenizer 파일 존재 여부를 확인한다."""
    required = [
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "spiece.model",
    ]
    return any((model_dir / name).exists() for name in required)


def _resolve_tokenizer_source(base_model_dir: str, lora_dir: str) -> str:
    """tokenizer 로드 경로를 우선순위에 따라 결정한다.

    우선순위:
    1) lora_dir/tokenizer
    2) lora_dir
    3) base_model_dir
    """
    lora_path = Path(lora_dir)
    lora_tokenizer_path = lora_path / "tokenizer"
    if _has_tokenizer_files(lora_tokenizer_path):
        return str(lora_tokenizer_path)
    if _has_tokenizer_files(lora_path):
        return str(lora_path)
    return str(Path(base_model_dir))


# 로드
def load_model():
    """토크나이저, Base 모델, LoRA adapter를 순서대로 로드한다."""
    tokenizer_src = _resolve_tokenizer_source(BASE_MODEL, LORA_DIR)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_src,
            use_fast=True,
        )
    except TypeError as e:
        if "pre_tokenizers.Split" not in str(e):
            raise
        print(
            "[WARN] fix_mistral_regex=True 패치가 현재 토크나이저 구조와 맞지 않아 "
            "fix_mistral_regex=False로 다시 로드합니다.",
            flush=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_src,
            use_fast=True,
            fix_mistral_regex=False,
        )
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

    model = PeftModel.from_pretrained(
        base_model,
        LORA_DIR,
        torch_dtype=DTYPE,
    )

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
        add_generation_prompt=True,
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
    """터미널 기반 인터랙티브 채팅 루프를 실행한다."""
    tokenizer, model = load_model()

    # SYSTEM (1회만)
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
                - 한 턴에는 하나의 서술과 하나의 대사만 작성한다.
                - 대사 규칙: 카즈키 대사를 작성하지 않는다.
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
