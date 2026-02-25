# models/qwen3_reference/test_load_qwen3.py
from __future__ import annotations

"""
Qwen3-4B-Base 로딩/포워드/제너레이트 통합 테스트

중요:
- init_empty_weights + load_checkpoint_and_dispatch(meta) 경로는
  "체크포인트에 없는 파라미터(bias 등)가 meta로 남는 순간" 바로 터질 수 있다.
- 여기서는 transformers의 from_pretrained 경로에 로딩을 맡긴다.
  (device_map="auto", low_cpu_mem_usage=True)
- generate 테스트는 절대 삭제하지 않는다.
"""

import torch
from transformers import AutoTokenizer

from .config import SayaQwen3Config
from .causal_lm import QwenForCausalLM


def main() -> None:
    hf_model_id = "Qwen/Qwen3-4B-Base"

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"[INFO] device = {device}")

    # ------------------------------------------------------------
    # tokenizer / config
    # ------------------------------------------------------------
    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)

    print("[INFO] Loading config...")
    config = SayaQwen3Config.from_pretrained(hf_model_id)

    # ------------------------------------------------------------
    # load model weights into custom class
    # ------------------------------------------------------------
    print("[INFO] Loading model weights into custom architecture...")

    dtype = torch.float16 if use_cuda else torch.float32

    # 핵심: 이 경로가 meta->real 텐서 이동을 내부적으로 정상 처리한다.
    model = QwenForCausalLM.from_pretrained(
        hf_model_id,
        config=config,
        torch_dtype=dtype,
        device_map="auto" if use_cuda else None,
        low_cpu_mem_usage=True,
    )

    # tie_weights 경고/정합을 위해 명시 호출
    model.tie_weights()
    model.eval()

    # ------------------------------------------------------------
    # forward test
    # ------------------------------------------------------------
    print("\n[INFO] Running forward test...")
    prompt = "안녕 내 이름은"
    inputs = tokenizer(prompt, return_tensors="pt")

    # device_map을 쓸 때는 입력을 "첫 파라미터 디바이스"로 맞추는 게 안전
    first_param_device = next(model.parameters()).device
    inputs = {k: v.to(first_param_device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            use_cache=False,
        )

    print("[INFO] Forward OK")
    print("[INFO] Logits shape:", tuple(out.logits.shape))

    # ------------------------------------------------------------
    # generate test (삭제 금지)
    # ------------------------------------------------------------
    print("\n[INFO] Running generate test...")
    with torch.no_grad():
        gen_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=120,
            do_sample=False,
            use_cache=False,
        )

    print("[INFO] Generated text:")
    print(tokenizer.decode(gen_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
