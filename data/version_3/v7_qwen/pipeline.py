"""
Scenario book -> multi-turn pipeline runner (one-by-one).

Generates one scenario book, immediately generates its multi-turn,
then repeats until samples are filled.

models/qwen3_core/model_assets/qwen-8b
data/generator/Tri-7B

사용 예시:
  uv run data/version_3/v7_qwen/pipeline.py \
    --model_path data/generator/Tri-7B \
    --scenario_out /mnt/d/rp_data/v7/v3_scenario.jsonl \
    --samples 5000 \
    --multiturn_out /mnt/d/rp_data/v7/v3_datum.jsonl \
    --fsm_path data/version_3/v7_qwen/state_fsm.yaml \
    --action_fsm_path data/version_3/v7_qwen/action_fsm.yaml \
    --turns 8 \
    --use_4bit
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from charcard import (
    SPEECH_STYLE_OPTIONS,
    SPEECH_STYLE_WEIGHTS,
    enforce_base_speech_style_line,
    extract_recent_interaction_context,
    extract_recent_protagonist_turn,
    generate_scenario_book,
    has_speech_style_mismatch,
    normalize_scenario_text,
    is_valid_scenario_book,
    invalid_reason,
    replace_recent_protagonist_turn_line,
    rewrite_recent_turn_line,
)
from generator import run_scenario


def main() -> None:
    """CLI 인자를 파싱하고 전체 실행 흐름을 시작한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--scenario_out", required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--multiturn_out", required=True)
    parser.add_argument("--fsm_path", required=True)
    parser.add_argument("--action_fsm_path", default="data/version_3/v7_qwen/action_fsm.yaml")
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--use_4bit", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=False,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            quantization_config=bnb,
            local_files_only=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )

    model.eval()

    Path(args.scenario_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.multiturn_out).parent.mkdir(parents=True, exist_ok=True)

    def count_lines(path: str) -> int:
        """
        유효 라인 수 집계

        재시작 시점 계산을 위해 jsonl 파일에서
        공백을 제외한 실제 레코드 라인 수를 센다.

        Args:
            path: 집계할 파일 경로.

        Returns:
            int: 비어있지 않은 라인 수.
        """
        if not Path(path).exists():
            return 0
        with open(path, "r", encoding="utf-8") as rf:
            return sum(1 for _ in rf if _.strip())

    scen_done = count_lines(args.scenario_out)
    multi_done = count_lines(args.multiturn_out)
    if scen_done != multi_done:
        print(
            f"[PIPELINE] resume mismatch scenario={scen_done} multiturn={multi_done}",
            flush=True,
        )

    with open(args.scenario_out, "a", encoding="utf-8") as scen_out, open(
        args.multiturn_out, "a", encoding="utf-8"
    ) as multi_out:
        if multi_done < scen_done:
            with open(args.scenario_out, "r", encoding="utf-8") as rf:
                idx = 0
                for line in rf:
                    if not line.strip():
                        continue
                    if idx < multi_done:
                        idx += 1
                        continue
                    text = json.loads(line)["messages"][0]["content"]
                    data = run_scenario(
                        system_lore=text,
                        model=model,
                        tokenizer=tokenizer,
                        turns=args.turns,
                        fsm_path=args.fsm_path,
                        action_fsm_path=args.action_fsm_path,
                    )
                    multi_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                    multi_out.flush()
                    idx += 1

            multi_done = scen_done

        accepted = min(scen_done, multi_done)
        trials = 0
        while accepted < args.samples:
            trials += 1
            print(f"[PIPELINE] try={trials} accepted={accepted}", flush=True)

            chosen_speech_style = random.choices(
                SPEECH_STYLE_OPTIONS,
                weights=SPEECH_STYLE_WEIGHTS,
                k=1,
            )[0]
            text = generate_scenario_book(
                model=model,
                tokenizer=tokenizer,
                max_tokens=1400,
                chosen_speech_style=chosen_speech_style,
            )
            text = normalize_scenario_text(text)
            text = enforce_base_speech_style_line(text, chosen_speech_style)
            # 말투가 실제로 어긋날 때만 최근 턴 rewrite를 최대 2회 수행한다.
            for _ in range(2):
                if not has_speech_style_mismatch(text):
                    break
                recent_line = extract_recent_protagonist_turn(text)
                if not recent_line:
                    break
                context = extract_recent_interaction_context(text)
                rewritten_line = rewrite_recent_turn_line(
                    model,
                    tokenizer,
                    recent_line,
                    chosen_speech_style,
                    scenario_context=context,
                ).strip()
                rewritten_line = rewritten_line.strip("\"“”'")
                text = replace_recent_protagonist_turn_line(text, rewritten_line)
                text = normalize_scenario_text(text)
                text = enforce_base_speech_style_line(text, chosen_speech_style)
            if not is_valid_scenario_book(text):
                print(f"[STEP] SCENARIO_GEN: invalid={invalid_reason(text)}", flush=True)
                continue

            scen_out.write(
                json.dumps(
                    {"messages": [{"role": "system", "content": text}]},
                    ensure_ascii=False,
                )
                + "\n"
            )
            scen_out.flush()

            data = run_scenario(
                system_lore=text,
                model=model,
                tokenizer=tokenizer,
                turns=args.turns,
                fsm_path=args.fsm_path,
                action_fsm_path=args.action_fsm_path,
            )
            multi_out.write(json.dumps(data, ensure_ascii=False) + "\n")
            multi_out.flush()

            accepted += 1


if __name__ == "__main__":
    main()
