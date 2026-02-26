from __future__ import annotations
"""
uv run python -m data.rewrite.rewrite_with_gpt_api \
  --input /mnt/d/rp_data/singleturn/rp_singleturn_cleaned.jsonl \
  --output /mnt/d/rp_data/rewrite/singleturn_rewrite.jsonl \
  --model gpt-4o

uv run python -m data.rewrite.rewrite_with_gpt_api \
  --input /mnt/d/rp_data/v7/rp_datum_unite_cleaned.jsonl \
  --output /mnt/d/rp_data/rewrite/multiturn_rewrite.jsonl \
  --model gpt-4o
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from openai import OpenAI

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
    # Allow direct script execution: uv run data/rewrite/rewrite_with_gpt_api.py ...
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


def rewrite_with_gpt(
    client: OpenAI,
    model: str,
    target_role: str,
    system_context: str,
    partner_context_label: str,
    partner_context_text: str,
    recent_dialogue_context: str,
    original_text: str,
    temperature: float,
) -> str:
    if target_role == "assistant":
        sys_inst = build_assistant_rewrite_instruction()
        target_label = "assistant"
    else:
        sys_inst = build_user_rewrite_instruction()
        target_label = "user"
    rsp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
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
        ],
    )
    text = rsp.choices[0].message.content or ""
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite RP dataset with OpenAI GPT API.")
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all")
    parser.add_argument("--log-every", type=int, default=1, help="Progress log interval")
    parser.add_argument("--api-key", default="", help="OpenAI API key (overrides env/.env)")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--env-file", default=".env", help="dotenv file path")
    parser.add_argument("--base-url", default="", help="Optional custom API base URL")
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

    # Load .env if available.
    try:
        from dotenv import load_dotenv  # type: ignore

        env_path = Path(args.env_file)
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)
        else:
            load_dotenv(override=False)
    except Exception:
        # dotenv is optional; environment variables may already be set.
        pass

    api_key = args.api_key.strip() or os.getenv(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(
            f"API key not found. Pass --api-key or set {args.api_key_env} "
            f"(or place it in {args.env_file})."
        )

    client_kwargs: dict[str, Any] = {}
    client_kwargs["api_key"] = api_key
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    skipped = 0
    resume_index = 1

    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            resume_index = max(1, int(state.get("next_index", 1)))
            skipped = int(state.get("skipped", 0))
            print(f"[rewrite-gpt] resume from state: sample_index={resume_index}")
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
            print(f"[rewrite-gpt] resume from output rows: sample_index={resume_index}")

    if resume_index > len(samples):
        print(f"[rewrite-gpt] already completed: input={len(samples)}")
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
            t0 = time.time()
            if not validate_messages_sample(sample):
                skipped += 1
                save_state(i + 1, skipped)
                if i % max(1, args.log_every) == 0:
                    print(
                        f"[rewrite-gpt] sample {i}/{len(samples)} skipped (invalid schema) "
                        f"elapsed={time.time()-t0:.2f}s"
                    )
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
                    rewritten_text = rewrite_with_gpt(
                        client=client,
                        model=args.model,
                        target_role=role,
                        system_context=system_context,
                        partner_context_label=partner_context_label,
                        partner_context_text=partner_context_text,
                        recent_dialogue_context=recent_dialogue_context,
                        original_text=content,
                        temperature=args.temperature,
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
            if i % max(1, args.log_every) == 0:
                print(
                    f"[rewrite-gpt] sample {i}/{len(samples)} done "
                    f"written={written_this_run} skipped={skipped} elapsed={time.time()-t0:.2f}s"
                )

    if state_path.exists():
        state_path.unlink()
    print(
        f"[rewrite-gpt] done: input={len(samples)} written_this_run={written_this_run} skipped={skipped}"
    )
    print(f"[rewrite-gpt] wrote {out_path}")


if __name__ == "__main__":
    main()
