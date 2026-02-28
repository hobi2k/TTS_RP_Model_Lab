"""Behavior-preserving split pipeline for original v5 GPT."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI

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

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


def main() -> None:
    """v5 GPT 파이프라인의 전체 실행 절차를 수행한다.

    처리 순서:
    1) OpenAI 클라이언트 및 모델 이름 준비
    2) 시나리오 출력 파일과 멀티턴 출력 파일의 진행 상태 확인
    3) 불일치가 있으면 기존 시나리오를 이용해 멀티턴을 보충 생성
    4) 목표 샘플 수까지 시나리오 생성 + 멀티턴 생성을 반복

    산출물:
    - `scenario_out`: system lore만 포함한 JSONL
    - `multiturn_out`: user/assistant 대화 메시지 JSONL
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--openai_model", default="gpt-5-mini")
    parser.add_argument("--scenario_out", required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--multiturn_out", required=True)
    parser.add_argument("--fsm_path", required=True)
    parser.add_argument("--action_fsm_path", default="data/original/v5_qwen/action_fsm.yaml")
    parser.add_argument("--turns", type=int, default=8)
    args = parser.parse_args()

    if load_dotenv is not None:
        load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in .env or environment.")
    client = OpenAI(api_key=api_key)
    model_name = args.openai_model

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
                        client=client,
                        model_name=model_name,
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
                client=client,
                model_name=model_name,
                max_tokens=2600,
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
                client=client,
                model_name=model_name,
                turns=args.turns,
                fsm_path=args.fsm_path,
                action_fsm_path=args.action_fsm_path,
            )
            multi_out.write(json.dumps(data, ensure_ascii=False) + "\n")
            multi_out.flush()
            accepted += 1


if __name__ == "__main__":
    main()
