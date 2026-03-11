"""Qwen 계열 병합 VLM 추론 스크립트.

기본 경로:
- merged/base: `models/qwen3_core/model_assets/qwen3.5-4b`
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig

try:
    from transformers import AutoModelForImageTextToText as AutoVLMModel
except ImportError:  # pragma: no cover
    from transformers import AutoModelForVision2Seq as AutoVLMModel


MODEL_ASSETS_DIR = Path(__file__).resolve().parent / "model_assets"
DEFAULT_MODEL_DIR = MODEL_ASSETS_DIR / "qwen3.5-4b"


def _has_any_file(dir_path: Path, candidates: list[str]) -> bool:
    return any((dir_path / name).exists() for name in candidates)


def _resolve_processor_source(merged_dir: Path) -> Path:
    processor_files = ["processor_config.json", "preprocessor_config.json"]
    p1 = merged_dir / "processor"
    if p1.exists() and _has_any_file(p1, processor_files):
        return p1
    return merged_dir


def _resolve_tokenizer_source(merged_dir: Path) -> Path:
    tokenizer_files = ["tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt", "spiece.model"]
    t1 = merged_dir / "tokenizer"
    if t1.exists() and _has_any_file(t1, tokenizer_files):
        return t1
    return merged_dir


def _load_image(image_path: str | None) -> Image.Image | None:
    if not image_path:
        return None
    try:
        return Image.open(image_path).convert("RGB")
    except Exception:
        return None


def _build_messages(messages: list[dict[str, str]], image_path: str | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    image_inserted = False
    for idx, msg in enumerate(messages):
        role = msg.get("role", "").strip()
        content = msg.get("content", "").strip()
        if not content:
            continue
        if role == "user":
            blocks: list[dict[str, str]] = []
            is_current_user = idx == len(messages) - 1
            if is_current_user and image_path and not image_inserted:
                blocks.append({"type": "image"})
                image_inserted = True
            blocks.append({"type": "text", "text": content})
            out.append({"role": "user", "content": blocks})
        else:
            out.append({"role": role, "content": content})
    return out


def load_model(
    merged_model_dir: Path,
    use_4bit: bool,
    attn_implementation: str,
    trust_remote_code: bool,
) -> tuple[Any, Any]:
    """병합된 Qwen VLM 모델을 로드한다."""
    dtype = torch.bfloat16
    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    processor_src = _resolve_processor_source(merged_model_dir)
    tokenizer_src = _resolve_tokenizer_source(merged_model_dir)
    processor = AutoProcessor.from_pretrained(str(processor_src), trust_remote_code=trust_remote_code)
    try:
        tok = AutoProcessor.from_pretrained(str(tokenizer_src), trust_remote_code=trust_remote_code).tokenizer
        if tok is not None:
            processor.tokenizer = tok
    except Exception:
        pass
    if getattr(processor, "tokenizer", None) is not None and processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = AutoVLMModel.from_pretrained(
        str(merged_model_dir),
        torch_dtype=dtype,
        device_map="auto",
        quantization_config=quant_config,
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return processor, model


@torch.inference_mode()
def generate_reply(
    processor: Any,
    model: Any,
    messages: list[dict[str, str]],
    image_path: str | None,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> str:
    """assistant 응답을 생성한다."""
    image = _load_image(image_path)
    chat_messages = _build_messages(messages, image_path if image is not None else None)
    prompt = processor.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    kwargs: dict[str, Any] = dict(text=[prompt], return_tensors="pt")
    if image is not None:
        kwargs["images"] = [image]
    inputs = processor(**kwargs)
    model_device = next(model.parameters()).device
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(model_device)
    inputs.pop("token_type_ids", None)
    prompt_len = int(inputs["input_ids"].shape[-1])

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        renormalize_logits=True,
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
    if tok is None:
        return ""
    return tok.decode(output[0][prompt_len:], skip_special_tokens=True).strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qwen 계열 merged VLM interactive inference")
    p.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--image", type=str, default=None)
    p.add_argument("--max_new_tokens", type=int, default=320)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.05)
    p.add_argument("--attn_implementation", type=str, default="sdpa")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--no_sample", action="store_true")
    return p.parse_args()


def chat_loop(args: argparse.Namespace) -> None:
    print(f"[INFO] model_dir={args.model_dir}")
    processor, model = load_model(
        merged_model_dir=Path(args.model_dir).resolve(),
        use_4bit=args.load_in_4bit,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    current_image = args.image
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                """
                당신은 이 이야기의 주인공 사야다.
                사야의 시점에서 반응해라.
                assistant는 카즈키(user)의 대사나 행동을 대신 작성하지 않는다.
                assistant 출력은 서술 1블록 + 대사 1블록으로 작성한다.
                서술은 3인칭 평어체로 쓰고, 대사는 큰따옴표로 감싼다.
                이미지가 있으면 현재 이미지에서 직접 확인되는 정보만 근거로 사용한다.
                """
            ),
        }
    ]

    print("=== Qwen VLM Merged Chat Started (type 'exit' to quit) ===")
    print("명령: /image PATH, /noimage")
    while True:
        user_input = input("\nUSER > ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if user_input.startswith("/image "):
            current_image = user_input[len("/image ") :].strip() or None
            print(f"[INFO] current image = {current_image}")
            continue
        if user_input == "/noimage":
            current_image = None
            print("[INFO] current image cleared")
            continue
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        reply = generate_reply(
            processor=processor,
            model=model,
            messages=messages,
            image_path=current_image,
            max_new_tokens=args.max_new_tokens,
            do_sample=not args.no_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        messages.append({"role": "assistant", "content": reply})
        print("\nASSISTANT >")
        print(reply)


if __name__ == "__main__":
    chat_loop(parse_args())
