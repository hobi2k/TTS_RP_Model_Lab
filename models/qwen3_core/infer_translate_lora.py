"""
번역 LoRA 추론 스크립트 (KO -> JA)

동작:
- 로컬 Base 모델(model_assets/qwen3-1.7b-base) 로드
- LoRA 어댑터(model_assets/qwen3_1.7_ko2ja_lora/lora_adapter) 로드
- 프롬프트(Instruction/Input/Output) 구성 후 generate
- 출력에서 "### Output:" 이후만 추출

사용 예:
uv run models/qwen3_core/infer_translate_lora.py \
  --base_dir models/qwen3_core/model_assets/qwen3-1.7b-base \
  --lora_dir models/qwen3_core/model_assets/qwen3_1.7_ko2ja_lora/lora_adapter \
  --text "이 마법은 우리 둘만의 비밀이야. 누군가 알게 된다면, 돌이킬 수 없는 일이 벌어질 거야." \
  --instruction "다음 한국어 문장을 자연스러운 일본어로 번역하시오." \
  --max_new_tokens 128 \
  --temperature 0.2 \
  --top_p 0.9
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompt(instruction: str, user_input: str) -> str:
    instruction = (instruction or "").strip()
    user_input = (user_input or "").strip()

    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Input:\n"
        f"{user_input}\n\n"
        "### Output:\n"
    )


def extract_output_text(generated_text: str) -> str:
    """
    생성 결과에서 "### Output:" 이후만 안전하게 뽑는다.
    (프롬프트를 통째로 echo 하거나, 앞부분이 섞여도 대응)
    """
    marker = "### Output:\n"
    if marker in generated_text:
        return generated_text.split(marker, 1)[1].strip()

    # 혹시 모델이 라벨을 약간 바꿔 출력한 경우의 차선책
    marker2 = "### Output:"
    if marker2 in generated_text:
        return generated_text.split(marker2, 1)[1].strip()

    return generated_text.strip()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_dir", type=str, required=True, help="Local base model directory")
    parser.add_argument("--lora_dir", type=str, required=True, help="LoRA adapter directory (save_pretrained output)")
    parser.add_argument("--instruction", type=str, default="다음 한국어 문장을 자연스러운 일본어로 번역해라.")
    parser.add_argument("--text", type=str, required=True, help="Korean source text")

    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true", help="샘플링 사용(기본: False=greedy)")
    parser.add_argument("--seed", type=int, default=42)

    # 안정적으로 끊기 위한 stop 관련
    parser.add_argument("--eos_stop", action="store_true", help="eos_token에서 멈추도록 강제(기본 True 권장)", default=True)

    args = parser.parse_args()

    base_dir = str(Path(args.base_dir).resolve())
    lora_dir = str(Path(args.lora_dir).resolve())

    torch.manual_seed(args.seed)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_dir, use_fast=True)

    # Qwen Base에서 pad_token 없는 경우가 많으므로 eos로 맞춘다.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Base + LoRA 로드
    # dtype은 GPU 지원에 맞춰 자동으로 가되, bf16 가능하면 bf16이 유리
    model = AutoModelForCausalLM.from_pretrained(
        base_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(model, lora_dir)
    model.eval()

    # 프롬프트 구성
    prompt = build_prompt(args.instruction, args.text)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )

    # device_map="auto" 환경에서 inputs를 모델 첫 디바이스로 옮기는 안전 처리
    # (단일 GPU면 보통 cuda:0)
    first_param = next(model.parameters())
    inputs = {k: v.to(first_param.device) for k, v in inputs.items()}

    # Generate
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature if args.do_sample else None,
        top_p=args.top_p if args.do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
    )

    if args.eos_stop and tokenizer.eos_token_id is not None:
        gen_kwargs["eos_token_id"] = tokenizer.eos_token_id

    with torch.no_grad():
        out_ids = model.generate(**inputs, **gen_kwargs)

    full_text = tokenizer.decode(out_ids[0], skip_special_tokens=False)

    # 프롬프트 echo 제거: prompt 길이만큼 잘라내는 방식도 가능하지만,
    # 모델이 프롬프트를 약간 변형해서 재출력하는 경우를 대비해 marker 기반 추출을 우선한다.
    translated = extract_output_text(full_text)

    # EOS 이후 잡음 제거(있으면)
    if tokenizer.eos_token and tokenizer.eos_token in translated:
        translated = translated.split(tokenizer.eos_token, 1)[0].strip()

    print("===== PROMPT =====")
    print(prompt)
    print("\n===== FULL GENERATED =====")
    print(full_text)
    print("\n===== TRANSLATION (Output-only) =====")
    print(translated)


if __name__ == "__main__":
    main()
