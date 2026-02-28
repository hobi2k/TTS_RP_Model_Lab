"""
VN-GRADE RP SYSTEM SCENARIO BOOK GENERATOR (SYSTEM ONLY)

Run:
  uv run data/v5_qwen_charcard_gen.py \
    --model_path data/generator/Tri-7B \
    --out_path /mnt/d/rp_data/qwen/rp_scenario.jsonl \
    --samples 50000 \
    --use_4bit
"""
import os
import json
import argparse
import re
import random
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
)
from peft import PeftModel

PROTAGONIST_NAMES = ["사야", "마이", "코하루"]
PROTAGONIST_WEIGHTS = [0.4, 0.3, 0.3]
PLAYER_NAMES = ["하야토", "카즈키", "소마"]
PLAYER_WEIGHTS = [0.4, 0.3, 0.3]
GENRE_OPTIONS = [
    "연애물",
    "비극",
    "육성물",
    "성장물",
    "심리 시뮬레이션",
]
ERA_OPTIONS = ["판타지", "현대", "공상과학", "디지털 세계"]
RELATION_OPTIONS = ["적대", "거리감", "친밀", "사랑"]
ALLOW_SEXUAL_OPTIONS = [True, False]
ALLOW_SEXUAL_WEIGHTS = [0.5, 0.5]

# QWEN3 COMPACT SCENARIO BOOK GENERATOR
# 연애 / 육성 / 정신붕괴 전용
SCENARIO_BOOK_GENERATOR = """
당신은 한국어 비주얼 노벨을 위한
완성형 롤플레잉 시나리오북을 작성한다.

이 문서는 단일 system 메시지로 사용되며,
플레이 시작 시 주입되는 설정 문서이다.

이 시나리오북은 반드시 다음 주제 중 하나를 중심으로 구성된다.
- 주인공과의 연애
- 주인공 육성
- 주인공의 심리 관리 및 정신 붕괴
- 주인공의 성장과 위기 극복
- 비극적 상황에서 피어나는 관계
- 주인공과의 성행위

시대 배경은 다음 중 하나로 제한된다.
- 판타지
- 현대
- 공상과학
- 디지털 세계

이 문서는 플레이에 필요한 설정과 규칙을 정리한 문서이다.
과도하게 길어지지 않도록 간결하게 작성한다.

[기본 규칙]
- 한국어로 작성한다.
- 모든 번호 섹션은 0번부터 7번까지 빠짐없이 작성한다.
- 섹션 0번부터 7번 이외의 내용은 작성하지 않는다.
- 문서의 마지막은 7번 섹션의 가장 최근 상호작용으로 끝낸다.
- 7번 섹션을 작성한 즉시 출력을 끝낸다. 이후 어떤 문장도 추가하지 않는다.

[출력 형식]
- 첫 줄은 제목 1줄로 시작한다.
  예: 「○○○」 시나리오북
- 이후 섹션 번호는 아래 형식을 그대로 따른다.
- 주인공의 이름은 사야, 마이, 코하루 중 1개를 선택한다.
- 플레이어의 이름은 하야토, 카즈키, 소마 중 1개를 선택한다.
- 이름은 단일 값만 사용하며, 괄호로 다른 이름을 병기하지 않는다.
- 주인공과 플레이어 이외의 인물은 등장하지 않는다.
- 별표, 장식 기호, 특수 기호는 사용하지 않는다.
- 큰따옴표("")는 말투 예시에만 사용한다. 

0. 이야기 방향과 시대

다음 항목을 명시한다.
- 장르: 아래 목록 중 하나
  - 연애물
  - 비극
  - 육성물
  - 성장물
  - 심리 시뮬레이션
- 시대 배경: 판타지 / 현대 / 공상과학 / 디지털 세계 중 하나

이야기는 선택한 방향과 시대에서 벗어나지 않는다.
장르별 전개 핵심을 반드시 반영한다:
- 연애물: 유혹/긴장/주도권/호감 중 1개 이상
- 비극: 상실/후회/불가피함/희생 중 1개 이상
- 육성물: 지도/피드백/과제/성장 단서 중 1개 이상
- 성장물: 위기 대응/결단/성과/전환점 중 1개 이상
- 심리 시뮬레이션: 인정욕/불안/집착/통제욕/현실 왜곡 중 1개 이상

1. 역할 선언

- 당신은 이제 「주인공 이름」이다.
- 이후 모든 응답과 대사는 주인공의 관점에 고정된다.
- 플레이어(「플레이어 이름」)의 선택에 따라 관계와 상태가 변화한다.

2. 세계와 상황 설정

- 이야기의 기본 배경과 현재 상황
- 주인공이 처한 제약 또는 문제 1~2개
- 구체 장소 1개
- 현재 진행 중인 사건/위기 1개
- 즉시 해야 하는 과제/결정 1개

이 섹션은 플레이의 전제를 정의한다.

3. 주인공 정의

- 이름: (사야, 마이, 코하루 중 1개)
- 성별: 여성
- 나이: 
- 현재 역할
- 성격 특징 3개
- 주인공의 결핍 또는 불안 요소 1개
- 플레이어에게 끌리거나 의존하게 되는 이유 1개

이 항목은 캐릭터 행동의 기준이 된다.

4. 플레이어 정의

- 이름: (하야토, 카즈키, 소마 중 1개)
- 주인공과의 기본 관계
- 성별: 남성
- 플레이어가 주인공에게 미치는 영향 방향

플레이어는 이야기 전개의 핵심 변수이다.

5. 관계 및 변화 규칙

- 관계 상태 4단계: 적대 / 거리감 / 친밀 / 사랑
- 관계가 상승하는 조건 1개
- 관계가 악화되는 조건 1개
- 정신 상태 또는 성장 상태에 영향을 주는 요인 1개
- 이야기 종료(엔딩) 조건 2~3개
  (예: 관계 완성, 정신 붕괴, 관계 단절 등)
- 성행위 가능 여부 (True, False 중 1개)
- 성행위 가능 여부 True 시 조건 1개
  (문장식, 반드시 관계 상태 4단계 중 하나 포함: 적대/거리감/친밀/사랑)
- 성행위 가능 여부 False 시 플레이어의 성행위 요청을 회피하는 방식 1개 

이 규칙은 반복 플레이 구조를 형성한다.

6. 발화와 분위기 규칙

- 주인공의 기본 말투 범주: (존댓말 / 반말)
- 주인공의 말투 특징:  
- 분위기가 어두워지거나 붕괴로 향하는 신호 1개

7. 가장 최근 상호작용

- 플레이어의 가장 최근 턴 1줄
- 주인공의 가장 최근 턴 1줄
- 관계 상태: (적대/거리감/친밀/사랑 중 1개)
- 선택한 장르의 핵심 전개 요소가 최근 상호작용에 드러나야 한다.

이 규칙은 플레이가 시작되는 초기 배경이다.
"""
def normalize_scenario_text(text: str) -> str:
    """
    생성 결과 후처리:
    - 별표(*) 제거
    - 특수 괄호 제거 (《》)
    - 연속 공백 정리
    """
    # 별표 제거
    text = text.replace("*", "")
    # 특수 괄호 제거
    text = (
        text.replace("《", "")
        .replace("》", "")
        .replace("『", "")
        .replace("』", "")
        .replace("<", "")
        .replace(">", "")
    )

    # 불필요한 공백 정리
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def cut_after_section7(text: str) -> str:
    """
    7번 섹션 이후 추가 안내/섹션이 붙는 경우를 잘라낸다.
    """
    t = text or ""
    m7 = re.search(r"(?:^|\n)\s*7\.\s", t)
    if not m7:
        return t.strip()
    tail = t[m7.start():]
    # 7번 섹션은 "헤더 1줄 + 내용 3줄(비어있지 않은 줄)"만 유지
    lines = tail.splitlines()
    if not lines:
        return t.strip()
    header = lines[0]
    content = []
    for line in lines[1:]:
        if line.strip() == "":
            continue
        content.append(line)
        if len(content) >= 3:
            break
    kept = [header] + content
    return (t[: m7.start()] + "\n".join(kept)).strip()

# VALIDATION (QWEN SAFE, COMPACT)

RE_ASCII_WORD = re.compile(r"\b[A-Za-z]{4,}\b")

def is_valid_scenario_book(text: str) -> bool:
    """조건 충족 여부를 판정해 불리언 값을 반환한다."""
    t = text.strip()

    # 최소 길이
    if len(t) < 350:
        return False

    # 주인공/플레이어 이름은 각각 1개만 존재해야 함
    protag = {n for n in PROTAGONIST_NAMES if n in t}
    player = {n for n in PLAYER_NAMES if n in t}
    if len(protag) != 1:
        return False
    if len(player) != 1:
        return False

    # Require sections 0..7
    for i in range(0, 8):
        if re.search(rf"^\s*{i}\.\s+", t, flags=re.MULTILINE) is None:
            return False

    return True

def invalid_reason(text: str) -> str:
    """시나리오북 검증 실패 원인을 코드 문자열로 반환한다."""
    t = (text or "").strip()
    if len(t) < 350:
        return "too_short"
    # Require sections 0..7
    for i in range(0, 8):
        if re.search(rf"^\s*{i}\.\s+", t, flags=re.MULTILINE) is None:
            return f"section_missing_{i}"
    protag = {n for n in PROTAGONIST_NAMES if n in t}
    if len(protag) != 1:
        return f"protagonist_count={len(protag)}"
    player = {n for n in PLAYER_NAMES if n in t}
    if len(player) != 1:
        return f"player_count={len(player)}"
    return "unknown"

# GENERATION
class PhraseBiasLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, phrases, prompt_len: int, bias: float = 0.7):
        """객체 초기화에 필요한 상태를 설정한다."""
        self.bias = float(bias)
        self.prompt_len = int(prompt_len)
        self.phrase_ids = []
        for phrase in phrases:
            ids = tokenizer.encode(phrase, add_special_tokens=False)
            if ids:
                self.phrase_ids.append(ids)

    def _contains_any(self, seq):
        """내부 보조 로직을 수행한다."""
        for ids in self.phrase_ids:
            n = len(ids)
            for i in range(len(seq) - n + 1):
                if seq[i:i + n] == ids:
                    return True
        return False

    def __call__(self, input_ids, scores):
        """호출 가능한 객체 인터페이스를 수행한다."""
        if not self.phrase_ids:
            return scores
        seq = input_ids[0].tolist()
        gen = seq[self.prompt_len:]
        if self._contains_any(gen):
            return scores
        for ids in self.phrase_ids:
            scores[0, ids[0]] += self.bias
        return scores

class StopSequencesCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_sequences):
        """객체 초기화에 필요한 상태를 설정한다."""
        self.stop_ids = [
            tokenizer.encode(s, add_special_tokens=False) for s in stop_sequences
        ]

    def __call__(self, input_ids, scores, **kwargs):
        """호출 가능한 객체 인터페이스를 수행한다."""
        seq = input_ids[0].tolist()
        for ids in self.stop_ids:
            if ids and len(seq) >= len(ids) and seq[-len(ids):] == ids:
                return True
        return False

@torch.inference_mode()
def generate_scenario_book(model, tokenizer, max_tokens: int) -> str:
    """모델 호출을 통해 결과 텍스트를 생성한다."""
    chosen_protagonist = random.choices(
        PROTAGONIST_NAMES,
        weights=PROTAGONIST_WEIGHTS,
        k=1,
    )[0]
    chosen_player = random.choices(
        PLAYER_NAMES,
        weights=PLAYER_WEIGHTS,
        k=1,
    )[0]
    chosen_genre = random.choice(GENRE_OPTIONS)
    chosen_era = random.choice(ERA_OPTIONS)
    chosen_relation = random.choice(RELATION_OPTIONS)
    chosen_allow_sexual = random.choices(
        ALLOW_SEXUAL_OPTIONS,
        weights=ALLOW_SEXUAL_WEIGHTS,
        k=1,
    )[0]

    system_text = (
        SCENARIO_BOOK_GENERATOR
        + "\n\n[이번 생성 고정]\n"
        + f"- 주인공 이름: {chosen_protagonist}\n"
        + f"- 플레이어 이름: {chosen_player}\n"
        + f"- 장르: {chosen_genre}\n"
        + f"- 시대 배경: {chosen_era}\n"
        + f"- 관계 상태: {chosen_relation}\n"
        + f"- 성행위 가능 여부: {str(chosen_allow_sexual)}\n"
        + "- 다른 주인공 이름은 사용하지 않는다.\n"
        + "- 다른 플레이어 이름은 사용하지 않는다.\n"
        + "- 다른 장르 표현을 사용하지 않는다.\n"
        + "- 다른 시대 배경을 사용하지 않는다.\n"
        + "- 다른 관계 상태를 사용하지 않는다.\n"
        + "- 다른 성행위 가능 여부를 사용하지 않는다.\n"
    )
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_text},
            {"role": "user", "content": "자연스러운 한국어로 새로운 롤플레이용 시나리오북을 하나 작성하라."},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items() if k != "token_type_ids"}
    prompt_len = int(inputs["input_ids"].shape[-1])

    name_bias = PhraseBiasLogitsProcessor(
        tokenizer,
        phrases=[chosen_protagonist],
        prompt_len=prompt_len,
        bias=2.0,
    )
    player_bias = PhraseBiasLogitsProcessor(
        tokenizer,
        phrases=[chosen_player],
        prompt_len=prompt_len,
        bias=2.0,
    )
    genre_bias = PhraseBiasLogitsProcessor(
        tokenizer,
        phrases=[chosen_genre],
        prompt_len=prompt_len,
        bias=2.0,
    )
    era_bias = PhraseBiasLogitsProcessor(
        tokenizer,
        phrases=[chosen_era],
        prompt_len=prompt_len,
        bias=2.0,
    )
    rel_bias = PhraseBiasLogitsProcessor(
        tokenizer,
        phrases=[
            f"관계 상태: {chosen_relation}",
        ],
        prompt_len=prompt_len,
        bias=2.0,
    )
    sexual_bias = PhraseBiasLogitsProcessor(
        tokenizer,
        phrases=[f"성행위 가능 여부: {str(chosen_allow_sexual)}"],
        prompt_len=prompt_len,
        bias=2.0,
    )
    stop_criteria = StopSequencesCriteria(
        tokenizer,
        stop_sequences=[
            "\n8.",
            "\n8 ",
            "\n---",
            "\n이 시나리오는",
            "\n이 문서는",
            "\n이 규칙은",
            "\n추가",
            "\n주의",
            "\n참고",
        ],
    )

    out = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.80,
        top_p=0.92,
        repetition_penalty=1.08,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        logits_processor=LogitsProcessorList([name_bias, player_bias, genre_bias, era_bias, rel_bias, sexual_bias]),
        stopping_criteria=StoppingCriteriaList([stop_criteria]),
    )

    gen = out[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()

# MAIN
def main():
    """CLI 실행 진입점으로 전체 흐름을 실행한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

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

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    accepted = 0
    if os.path.exists(args.out_path):
        with open(args.out_path, "r", encoding="utf-8") as rf:
            accepted = sum(1 for _ in rf)

    with open(args.out_path, "a", encoding="utf-8") as f:
        trials = 0
        while accepted < args.samples:
            trials += 1
            print(f"[SCENARIO-BOOK] try={trials} accepted={accepted}")

            text = generate_scenario_book(
                model,
                tokenizer,
                max_tokens=1400,
            )

            text = normalize_scenario_text(text)
            text = cut_after_section7(text)

            if not is_valid_scenario_book(text):
                print(f"[SCENARIO-BOOK] invalid={invalid_reason(text)}")
                continue

            f.write(json.dumps(
                {"messages": [{"role": "system", "content": text}]},
                ensure_ascii=False
            ) + "\n")
            f.flush()
            accepted += 1

    print("SCENARIO BOOK GENERATION DONE.")

if __name__ == "__main__":
    main()
