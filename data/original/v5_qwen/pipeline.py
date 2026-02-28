"""Behavior-preserving split pipeline for original v5 Qwen.

사용 예시:
  uv run data/original/v5_qwen/pipeline.py \
    --model_path data/generator/Tri-7B \
    --scenario_out /mnt/d/rp_data/v7/rp_scenario.jsonl \
    --samples 5000 \
    --multiturn_out /mnt/d/rp_data/v7/rp_datum.jsonl \
    --fsm_path data/original/v5_qwen/state_fsm.yaml \
    --action_fsm_path data/original/v5_qwen/action_fsm.yaml \
    --turns 6 \
    --use_4bit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

if __package__ in (None, ""):
    from charcard import generate_scenario_book
    from generator import run_scenario
    from utils import normalize_scenario_text
    from validators import is_valid_scenario_book
else:
    from .charcard import generate_scenario_book
    from .generator import run_scenario
    from .utils import normalize_scenario_text
    from .validators import is_valid_scenario_book


def main() -> None:
    """v5 Qwen 파이프라인의 전체 실행 절차를 수행한다.

    처리 순서:
    1) 모델/토크나이저 초기화
    2) 시나리오 출력 파일과 멀티턴 출력 파일의 진행 상태 확인
    3) 불일치가 있으면 기존 시나리오를 이용해 멀티턴을 보충 생성
    4) 목표 샘플 수까지 시나리오 생성 + 멀티턴 생성을 반복

    산출물:
    - `scenario_out`: system lore만 포함한 JSONL
    - `multiturn_out`: user/assistant 대화 메시지 JSONL
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--scenario_out", required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--multiturn_out", required=True)
    parser.add_argument("--fsm_path", required=True)
    parser.add_argument("--action_fsm_path", default="data/original/v5_qwen/action_fsm.yaml")
    parser.add_argument("--turns", type=int, default=8)
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
        """대상 항목의 개수를 집계해 반환한다."""
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

            text = generate_scenario_book(
                model=model,
                tokenizer=tokenizer,
                max_tokens=1400,
            )
            text = normalize_scenario_text(text)
            if not is_valid_scenario_book(text):
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
