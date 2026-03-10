"""Kanana 전용 LoRA 단일턴 RP 추론 스크립트.

특징:
- Base Kanana VLM + LoRA adapter 결합 추론
- 학습 시와 동일하게 conv + dummy image 포맷 사용
- text-only 채팅 테스트를 위한 터미널 루프 제공
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig


def _has_any_file(dir_path: Path, candidates: list[str]) -> bool:
    """디렉토리에 후보 파일 중 하나라도 존재하는지 확인한다."""
    return any((dir_path / name).exists() for name in candidates)


def _resolve_processor_source(base_dir: Path, lora_root: Path) -> Path:
    """processor 로드 경로를 결정한다.

    우선순위:
    1) lora_root/processor
    2) lora_root
    3) base_dir
    """
    processor_files = ["processor_config.json", "preprocessor_config.json"]
    p1 = lora_root / "processor"
    if p1.exists() and _has_any_file(p1, processor_files):
        return p1
    if _has_any_file(lora_root, processor_files):
        return lora_root
    return base_dir


def _resolve_tokenizer_source(base_dir: Path, lora_root: Path) -> Path:
    """tokenizer 로드 경로를 결정한다.

    우선순위:
    1) lora_root/tokenizer
    2) lora_root
    3) base_dir
    """
    tokenizer_files = [
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "spiece.model",
    ]
    t1 = lora_root / "tokenizer"
    if t1.exists() and _has_any_file(t1, tokenizer_files):
        return t1
    if _has_any_file(lora_root, tokenizer_files):
        return lora_root
    return base_dir


def _build_kanana_conv(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """대화 히스토리를 Kanana model card 권장 conv 포맷으로 변환한다.

    포맷:
    - 첫 user 메시지: "<image>"
    - 둘째 user 메시지: 시스템+대화 히스토리 텍스트
    """
    lines: list[str] = []
    for m in messages:
        role = m.get("role", "").strip()
        content = m.get("content", "").strip()
        if not content:
            continue
        if role == "system":
            lines.append(f"[시스템]\n{content}")
        elif role == "assistant":
            lines.append(f"[어시스턴트]\n{content}")
        else:
            lines.append(f"[사용자]\n{content}")
    lines.append("[어시스턴트]\n")
    text_prompt = "\n\n".join(lines)

    return [
        {"role": "user", "content": "<image>"},
        {"role": "user", "content": text_prompt},
    ]


def load_model(
    base_model_dir: Path,
    lora_dir: Path,
    use_4bit: bool,
    attn_implementation: str,
    trust_remote_code: bool,
) -> tuple[Any, Any]:
    """Kanana base + LoRA를 로드한다."""
    dtype = torch.bfloat16
    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    # LoRA 저장 루트 추론: .../lora_adapter -> ... (stage dir)
    lora_root = lora_dir.parent if lora_dir.name == "lora_adapter" else lora_dir

    processor_src = _resolve_processor_source(base_model_dir, lora_root)
    tokenizer_src = _resolve_tokenizer_source(base_model_dir, lora_root)

    processor = AutoProcessor.from_pretrained(str(processor_src), trust_remote_code=trust_remote_code)
    # tokenizer를 별도로 우선순위 로딩해 processor에 주입
    try:
        tok = AutoProcessor.from_pretrained(str(tokenizer_src), trust_remote_code=trust_remote_code).tokenizer
        if tok is not None:
            processor.tokenizer = tok
    except Exception:
        pass

    if getattr(processor, "tokenizer", None) is not None and processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    base_model = AutoModelForVision2Seq.from_pretrained(
        str(base_model_dir),
        torch_dtype=dtype,
        device_map="auto",
        quantization_config=quant_config,
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
    )

    model = PeftModel.from_pretrained(
        base_model,
        str(lora_dir),
        torch_dtype=dtype,
    )
    model.eval()
    return processor, model


@torch.inference_mode()
def generate_reply(
    processor: Any,
    model: Any,
    messages: list[dict[str, str]],
    max_length: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    dummy_image_size: int,
) -> str:
    """Kanana conv 포맷 입력으로 assistant 응답을 생성한다."""
    conv = _build_kanana_conv(messages)
    dummy = Image.new("RGB", (dummy_image_size, dummy_image_size), color=(255, 255, 255))

    batch = processor.batch_encode_collate(
        data_list=[{"conv": conv, "image": [dummy]}],
        padding="longest",
        padding_side="right",
        max_length=max_length,
        add_generation_prompt=True,
    )

    input_ids = batch["input_ids"]

    model_device = next(model.parameters()).device
    inputs = {}
    for k, v in batch.items():
        # Kanana generate는 image_metas 같은 비텐서 메타정보도 필요하다.
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(model_device)
        else:
            inputs[k] = v

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        use_cache=True,
    )
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        if tok.eos_token_id is not None:
            gen_kwargs["eos_token_id"] = tok.eos_token_id
        if tok.pad_token_id is not None:
            gen_kwargs["pad_token_id"] = tok.pad_token_id
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=top_p, top_k=50)

    output = model.generate(**inputs, **gen_kwargs)
    out_ids = output[0]
    if tok is None:
        return ""
    text = tok.decode(out_ids, skip_special_tokens=True).strip()
    # 특수토큰 제거 후 빈 문자열이 되면 raw 디코드로 한 번 더 시도한다.
    if not text:
        text = tok.decode(out_ids, skip_special_tokens=False).strip()
    return text


def chat_loop(args: argparse.Namespace) -> None:
    """터미널 기반 대화 루프를 실행한다."""
    processor, model = load_model(
        base_model_dir=Path(args.base_dir).resolve(),
        lora_dir=Path(args.lora_dir).resolve(),
        use_4bit=args.load_in_4bit,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )

    messages: list[dict[str, str]] = [
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

    print("=== Kanana RP Chat Started (type 'exit' to quit) ===")
    while True:
        user_input = input("\nUSER > ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        reply = generate_reply(
            processor=processor,
            model=model,
            messages=messages,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            dummy_image_size=args.dummy_image_size,
        )
        messages.append({"role": "assistant", "content": reply})

        print("\nASSISTANT >")
        print(reply)


def main() -> None:
    """CLI 엔트리포인트."""
    p = argparse.ArgumentParser()
    default_assets = Path(__file__).resolve().parent / "model_assets"
    p.add_argument("--base_dir", type=str, default=str(default_assets / "kanana_3b"))
    p.add_argument("--lora_dir", type=str, default=str(default_assets / "kanana_3b_stage2/lora_adapter"))
    p.add_argument("--trust_remote_code", action="store_true", default=True)
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--attn_implementation", type=str, default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])

    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--max_new_tokens", type=int, default=220)
    p.add_argument("--do_sample", action="store_true", default=True)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.05)
    p.add_argument("--dummy_image_size", type=int, default=224)
    args = p.parse_args()

    chat_loop(args)


if __name__ == "__main__":
    main()
