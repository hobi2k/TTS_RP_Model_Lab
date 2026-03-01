from __future__ import annotations
"""

사용 예시:
uv run python -m data.rewrite.rewrite_with_local_llm \
  --input /mnt/d/rp_data/singleturn/rp_singleturn_cleaned.jsonl \
  --output /mnt/d/rp_data/rewrite/singleturn_rewrite.local.jsonl \
  --model-dir data/generator/Tri-7B
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from data.rewrite.common import (
        build_assistant_rewrite_instruction,
        build_user_rewrite_instruction,
        enforce_assistant_rp_format,
        enforce_user_rewrite_format,
        read_jsonl,
        validate_messages_sample,
    )
except ModuleNotFoundError:
    # Allow direct script execution: uv run data/rewrite/rewrite_with_local_llm.py ...
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from data.rewrite.common import (
        build_assistant_rewrite_instruction,
        build_user_rewrite_instruction,
        enforce_assistant_rp_format,
        enforce_user_rewrite_format,
        read_jsonl,
        validate_messages_sample,
    )


def generate_rewrite(
    tokenizer,
    model,
    target_role: str,
    system_context: str,
    partner_context_label: str,
    partner_context_text: str,
    recent_dialogue_context: str,
    original_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    if target_role == "assistant":
        sys_inst = build_assistant_rewrite_instruction()
        target_label = "assistant"
    else:
        sys_inst = build_user_rewrite_instruction()
        target_label = "user"
    msgs = [
        {"role": "system", "content": sys_inst},
        {
            "role": "user",
            "content": (
                f"[시스템/상황]\n{system_context}\n\n"
                f"[{partner_context_label}]\n{partner_context_text}\n\n"
                f"[최근 대화 맥락]\n{recent_dialogue_context}\n\n"
                f"[원문 {target_label}]\n{original_text}\n\n"
                f"[요청]\n{target_label} 본문만 다시 작성해라."
            ),
        },
    ]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    # Some tokenizers return token_type_ids, but many causal LMs do not accept it in generate().
    inputs.pop("token_type_ids", None)
    inputs = inputs.to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0.0,
        temperature=max(temperature, 1e-6),
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    prompt_len = inputs["input_ids"].shape[-1]
    gen_ids = output_ids[0, prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite RP dataset with local LLM.")
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--model-dir", required=True, help="Local model directory")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all")
    parser.add_argument(
        "--recent-turns",
        type=int,
        default=3,
        help="Rewrite prompt에 넣을 최근 user/assistant 턴 수(기본 3)",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    state_path = Path(f"{out_path}.state.json")
    samples = read_jsonl(in_path)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=None,
        trust_remote_code=True,
    )
    model.to(target_device)
    model.eval()
    print(f"[rewrite-local] model device: {target_device}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    skipped = 0
    resume_index = 1

    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            resume_index = max(1, int(state.get("next_index", 1)))
            skipped = int(state.get("skipped", 0))
            print(f"[rewrite-local] resume from state: sample_index={resume_index}")
        except Exception:
            resume_index = 1
    elif out_path.exists():
        # Fallback resume point when state file is missing.
        done_rows = 0
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_rows += 1
        if done_rows > 0:
            resume_index = done_rows + 1
            print(f"[rewrite-local] resume from output rows: sample_index={resume_index}")

    if resume_index > len(samples):
        print(f"[rewrite-local] already completed: input={len(samples)}")
        if state_path.exists():
            state_path.unlink()
        return

    def build_recent_dialogue_context(context_msgs: list[dict[str, str]], recent_turns: int) -> str:
        limit = max(1, recent_turns) * 2
        turns = [m for m in context_msgs if m["role"] in ("user", "assistant")]
        turns = turns[-limit:]
        if not turns:
            return "(없음)"
        return "\n".join(f"{m['role']}: {m['content']}" for m in turns).strip()

    def build_partner_context(context_msgs: list[dict[str, str]], target_role: str) -> tuple[str, str]:
        if target_role == "assistant":
            for m in reversed(context_msgs):
                if m["role"] == "user":
                    return "이번 턴 user 메시지", m["content"]
            return "이번 턴 user 메시지", "(없음)"
        for m in reversed(context_msgs):
            if m["role"] == "assistant":
                return "직전 assistant 응답", m["content"]
        return "직전 assistant 응답", "(없음)"

    def save_state(next_index: int, skipped_count: int) -> None:
        state_path.write_text(
            json.dumps(
                {
                    "next_index": next_index,
                    "skipped": skipped_count,
                    "input": str(in_path),
                    "output": str(out_path),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    written_this_run = 0
    with out_path.open("a", encoding="utf-8") as fout:
        for i, sample in enumerate(samples, start=1):
            if i < resume_index:
                continue
            if not validate_messages_sample(sample):
                skipped += 1
                save_state(i + 1, skipped)
                continue

            new_sample = deepcopy(sample)
            messages = new_sample["messages"]
            context_msgs: list[dict[str, str]] = []

            for msg in messages:
                role = msg.get("role")
                content = str(msg.get("content", ""))
                if role in ("user", "assistant"):
                    system_context = "\n\n".join(
                        m["content"] for m in context_msgs if m["role"] == "system"
                    ).strip()
                    partner_context_label, partner_context_text = build_partner_context(
                        context_msgs, role
                    )
                    recent_dialogue_context = build_recent_dialogue_context(
                        context_msgs, args.recent_turns
                    )
                    rewritten_text = generate_rewrite(
                        tokenizer=tokenizer,
                        model=model,
                        target_role=role,
                        system_context=system_context,
                        partner_context_label=partner_context_label,
                        partner_context_text=partner_context_text,
                        recent_dialogue_context=recent_dialogue_context,
                        original_text=content,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                    if role == "assistant":
                        rewritten_text = enforce_assistant_rp_format(content, rewritten_text)
                    elif role == "user":
                        rewritten_text = enforce_user_rewrite_format(content, rewritten_text)
                    msg["content"] = rewritten_text
                context_msgs.append({"role": role, "content": msg["content"]})

            fout.write(json.dumps(new_sample, ensure_ascii=False) + "\n")
            fout.flush()
            written_this_run += 1
            save_state(i + 1, skipped)
            if i % 20 == 0:
                print(f"[rewrite-local] processed {i}/{len(samples)}")

    if state_path.exists():
        state_path.unlink()
    print(
        f"[rewrite-local] done: input={len(samples)} written_this_run={written_this_run} skipped={skipped}"
    )
    print(f"[rewrite-local] wrote {out_path}")


if __name__ == "__main__":
    main()
