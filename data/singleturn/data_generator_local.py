"""
로컬(Transformers 4bit) 기반 싱글턴 RP 생성기.

- 시나리오북(system)은 규칙 기반 + 확률 샘플링으로 코드가 직접 생성
- LLM은 user / assistant_raw / scene(ko,jp)만 생성
- 최종 출력 스키마:
  {system, user, assistant_raw, scene}

uv run data/singleturn/data_generator_local.py \
  --model_path data/generator/Tri-7B \
  --out_path /mnt/d/rp_data/singleturn/rp_generated_local.jsonl \
  --progress_path /mnt/d/rp_data/singleturn/rp_generated_local.progress.json \
  --target_rows 7000

  
uv run data/singleturn/data_generator_local.py \
  --model_path data/generator/Tri-7B \
  --out_path /mnt/d/rp_data/singleturn/rp_generated_local.jsonl \
  --progress_path /mnt/d/rp_data/singleturn/rp_generated_local.progress.json \
  --target_rows 7000 \
  --temperature 0.85 \
  --top_p 0.93 \
  --top_k 50 \
  --max_retries 8 \
  --max_new_tokens 1200  
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


PROTAGONIST_NAMES = ["사야", "코하루", "마이"]
PLAYER_NAMES = ["하야토", "카즈키", "소마"]
SPEECH_STYLES = ["존댓말", "반말"]
GENRES = ["심리 시뮬레이션", "연애물", "육성물", "성인물", "비극"]
JP_NAME_MAP = {"사야": "サヤ", "코하루": "コハル", "마이": "マイ"}
PLAYER_JP_MAP = {"하야토": "ハヤト", "카즈키": "カズキ", "소마": "ソマ"}
USER_FORMATS = ["dialog_only", "action_plus_dialogue"]

ROLE_SPLIT_RULE = "assistant는 플레이어(user)의 대사나 행동을 작성하지 않는다."
OUTPUT_FORMAT_RULE = "assistant 출력은 서술 1블록 + 대사 1블록으로 작성한다. 서술은 3인칭 평어체로 작성하고, 대사는 큰따옴표로 감싼다."

SITUATION_BANK: Dict[str, List[str]] = {
    "심리 시뮬레이션": [
        "주인공의 자기 소개를 요청하는 장면",
        "상대의 말 한마디가 주인공의 불안을 자극하는 장면",
        "기억 왜곡 의심으로 신뢰가 흔들리는 장면",
        "감정 과부하 직전이지만 대화를 이어가야 하는 장면",
        "현실 검증 질문이 반복되며 인지 부조화가 커지는 장면",
        "사소한 소리에도 과민 반응하며 통제욕이 높아지는 장면",
        "자기 확신과 자기 의심이 빠르게 교차하는 장면",
        "상대의 침묵을 거절로 해석해 불안이 급증하는 장면",
        "확인 강박으로 같은 질문을 다르게 반복하는 장면",
        "감정 억제가 무너지기 직전 논리로 버티는 장면",
        "과거 기억의 단편이 현재 판단을 흔드는 장면",
        "신뢰하고 싶지만 배신 공포가 우세한 장면",
        "경계심과 의존 욕구가 동시에 드러나는 장면",
    ],
    "연애물": [
        "고백 직후 어색한 침묵이 흐르는 장면",
        "관계 확인을 앞두고 서로 눈치를 보는 장면",
        "오해를 풀기 위해 진심을 꺼내야 하는 장면",
        "사귄 직후 경계와 기대가 공존하는 장면",
        "썸 단계에서 감정선을 확인하려는 장면",
        "권태를 피하기 위해 대화 방식을 바꿔야 하는 장면",
        "질투를 숨기다 미묘한 신경전이 생기는 장면",
        "재회 후 감정의 온도 차이를 확인하는 장면",
        "데이트 직전 사소한 약속으로 갈등이 생기는 장면",
        "먼저 다가갈지 기다릴지 결정해야 하는 장면",
        "사과와 재확인을 동시에 해야 관계가 유지되는 장면",
        "이별 직전 마지막 대화로 방향이 갈리는 장면",
    ],
    "육성물": [
        "과제 피드백 직전 긴장된 코칭 장면",
        "실패 원인을 분석하고 다음 목표를 정하는 장면",
        "학습 계획을 조정해야 하는 점검 장면",
        "과제 난이도를 내려야 할지 유지할지 판단하는 장면",
        "기초 루틴이 무너져 재설계가 필요한 장면",
        "즉시 교정과 자율 탐색 사이 균형을 잡는 장면",
        "짧은 마감 내 성과를 보여줘야 하는 장면",
        "동기 저하를 회복하기 위한 보상 설계를 논의하는 장면",
        "강점은 살리고 약점은 축소하는 전략을 정하는 장면",
        "피드백 방식 충돌로 지도 방식을 재정렬하는 장면",
        "테스트 직전 마지막 점검에서 우선순위를 가르는 장면",
        "재시도 여부를 결정해야 하는 평가 직후 장면",
    ],
    "성인물": [
        "강한 끌림 속에서 경계와 동의를 확인하는 장면",
        "신체적 긴장과 감정 확인이 동시에 필요한 장면",
        "욕망이 고조되지만 주도권 조율이 필요한 장면",
        "밀착 직전 속도와 강도를 합의해야 하는 장면",
        "노출 수위를 맞추며 반응을 탐색하는 장면",
        "성행위 전에 스킨쉽하는 장면",
        "성행위 전에 탈의하는 장면",
        "감각 묘사에 집중하며 호흡을 맞추는 장면",
        "성행위에 집입하는 장면",
        "성행위 도중 체위를 바꾸는 장면",
        "상대의 발정을 유도하기 위해 유혹하는 장면",
        "서로의 선호를 명확히 확인해야 하는 장면",
        "중단/지속 판단을 즉시 내려야 하는 장면",
        "성행위 후 정서적 안정 확인이 필요한 장면",
        "관능과 애착의 경계가 흔들리는 장면",
    ],
    "비극": [
        "희생 여부를 즉시 선택해야 하는 장면",
        "돌이킬 수 없는 결정을 앞둔 장면",
        "서로를 살리기 위해 관계를 포기해야 하는 장면",
        "시간 제한 속에서 감정보다 결단이 필요한 장면",
        "한 사람만 구할 수 있는 선택지가 주어진 장면",
        "약속을 지키면 대가를 치러야 하는 장면",
        "진실을 말하면 관계가 무너질 수 있는 장면",
        "구원과 파멸이 같은 문 앞에 놓인 장면",
        "오해를 풀기엔 너무 늦은 순간의 장면",
        "남겨질 사람을 정해야 하는 장면",
        "살아남아도 상실이 확정되는 장면",
        "마지막 작별을 준비해야 하는 장면",
    ],
}

LOCATION_BANK = [
    "비 오는 골목의 작은 카페",
    "심야 지하철 플랫폼",
    "불 꺼진 연습실",
    "도시 외곽의 낡은 모텔 복도",
    "한밤중 한강 산책로 벤치",
    "야간 촬영 스튜디오 소품 창고",
    "주인공의 원룸 거실",
    "옥상 난간 근처",
    "학교 도서관 구석 자리",
    "새벽 편의점 앞",
    "주차장 계단참",
    "조용한 호텔 라운지 구석 자리",
    "막차 직전 빈 버스 뒷좌석",
    "주인공 집 침실 문 앞",
    "작은 독립서점의 닫힌 2층",
    "불 꺼진 영화관 출구 근처",
    "오래된 연립주택 옥상 창고 앞",
    "24시간 코인세탁방 안쪽 벤치",
    "새벽 병원 대기실 창가 자리",
    "무인 스터디카페 가장 안쪽 칸",
    "해안 도로의 차 안",
    "고속버스 휴게소 외진 흡연 구역",
    "음악 스튜디오 녹음 부스 앞",
    "강변 호텔 복도 끝 창가",
    "비상계단 중간 참",
    "지하 주차장 기둥 뒤 그늘",
    "야간 개장 미술관의 한적한 홀",
    "소규모 바의 닫힌 룸 입구",
    "캠퍼스 강의동 옥상 출입문 앞",
    "리조트 객실 발코니",
    "심야 공항 대합실 한쪽",
    "전시회 철수 후 빈 갤러리",
    "열차 종점의 텅 빈 승강장",
    "비 오는 날 택시 안",
    "주인공의 주방 식탁 앞",
    "게스트하우스 공용 거실 새벽 시간",
    "산책로 끝 작은 정자 아래",
    "공연장 백스테이지 커튼 뒤",
    "강남 오피스 빌딩 야간 로비",
    "리허설 끝난 무대 측면",
    "작업실 소파 옆 스탠드 조명 아래",
    "심야 놀이공원 닫힌 매표소 앞",
    "폐점 직전 꽃집 뒷문 앞",
    "노을 진 해변 방파제 끝",
    "여관 복도의 오래된 자판기 앞",
    "별장 거실 벽난로 앞",
    "스파 리조트 휴식 라운지",
    "유리 온실 안쪽 통로",
]


PERSONALITY_BANK: Dict[str, List[str]] = {
    "심리 시뮬레이션": [
        "현실과 기억의 경계를 명확히 구분하지 못하는 성격",
        "자신의 감정이 실제인지조차 확신하지 못하는 성격",
        "겉으로는 차분하지만 내면에 설명할 수 없는 불안을 지닌 성격",
        "타인의 의도를 과도하게 해석해 스스로를 몰아붙이는 성격",
        "확인받고 싶지만 거절이 두려워 먼저 거리를 두는 성격",
        "질서와 통제를 잃으면 급격히 불안해지는 성격",
        "감정을 분석하려다 오히려 감정에 잠식되는 성격",
        "상대의 작은 변화에도 의미를 부여하는 과민한 성격",
    ],
    "연애물": [
        "감정을 쉽게 드러내지 않지만 상대를 깊이 신경 쓰는 성격",
        "상대의 반응 하나하나에 마음이 흔들리는 성격",
        "차분하지만 내면에 불안과 기대가 공존하는 성격",
        "상대를 배려하려다 자신의 욕구를 늦게 표현하는 성격",
        "사소한 약속을 크게 여기며 신뢰를 쌓아가는 성격",
        "질투를 숨기지만 결국 솔직함으로 돌아오는 성격",
        "관계가 깊어질수록 책임을 먼저 고민하는 성격",
        "장난스러운 말투 뒤에 진심을 숨기는 성격",
    ],
    "육성물": [
        "책임감이 강하고 상대의 성장을 끝까지 확인하려는 성격",
        "냉정한 피드백과 따뜻한 격려를 병행하는 성격",
        "실패 원인을 집요하게 분석하는 성격",
        "단기 성과보다 습관 형성을 우선하는 성격",
        "기준은 엄격하지만 학습자의 속도를 존중하는 성격",
        "문제를 작게 쪼개 실행 가능하게 만드는 성격",
        "성장 기록을 중요하게 여겨 꾸준히 점검하는 성격",
        "피드백 충돌이 생겨도 목표 중심으로 조율하는 성격",
    ],
    "성인물": [
        "이성보다 감정과 욕망에 더 쉽게 끌리는 성격",
        "선을 넘지 않으려 애쓰지만 유혹에 취약한 성격",
        "밀도 높은 교감을 선호하는 성격",
        "주도권의 흐름을 섬세하게 읽고 반응하는 성격",
        "합의와 경계 확인을 먼저 두는 신중한 성격",
        "감각적 표현에 솔직하지만 상대의 속도를 존중하는 성격",
        "긴장과 해소의 리듬을 즐기는 성격",
        "행위 뒤 정서적 안정을 중요하게 여기는 성격",
    ],
    "비극": [
        "운명을 거부하지만 책임에서 도망치지 못하는 성격",
        "상대에게 의존하면서도 그 사실을 부정하는 성격",
        "상실 공포를 안고 결단하는 성격",
        "희생을 감수하면서도 끝까지 상대를 살리려는 성격",
        "진실을 말하면 무너질 관계 앞에서 망설이는 성격",
        "후회를 줄이기 위해 차가운 선택을 택하는 성격",
        "끝을 알면서도 약속을 지키려는 성격",
        "파국 직전에도 마지막 존엄을 지키려는 성격",
    ],
}

RELATIONSHIP_BANK: Dict[str, List[str]] = {
    "심리 시뮬레이션": [
        "{player}의 꿈속에 반복해서 등장하는 존재",
        "{player}가 실제로 만났는지 확신할 수 없는 인물",
        "{player}의 기억과 연결된 인물",
        "{player}의 불안을 가장 먼저 알아차리는 상담 상대",
        "{player}와 비밀 기록을 공유하는 관찰자 관계",
        "{player}의 의심을 자극하는 모호한 보호자",
        "{player}와 상호 의존이 형성된 불안정한 관계",
        "{player}의 과거 사건을 알고 있는 침묵의 증인",
    ],
    "연애물": [
        "{player}의 오랜 친구",
        "{player}의 전 연인",
        "{player}와 애매한 관계",
        "{player}와 막 연애를 시작한 연인",
        "{player}와 재회 후 다시 가까워지는 관계",
        "{player}와 비밀 연애를 유지하는 관계",
        "{player}에게 먼저 고백한 관계",
        "{player}의 고백에 답을 앞둔 관계",
    ],
    "육성물": [
        "{player}를 지도하는 코치 관계",
        "{player}와 과제를 함께 수행하는 멘토 관계",
        "{player}의 성장을 평가하는 파트너 관계",
        "{player}의 약점을 보완해주는 트레이너 관계",
        "{player}와 스터디 플랜을 공동 설계한 관계",
        "{player}의 실전 수행을 감독하는 인스트럭터 관계",
        "{player}의 장기 성장 로그를 관리하는 코치 관계",
        "{player}와 단계별 목표를 계약한 훈련 관계",
    ],
    "성인물": [
        "{player}와 숨겨진 관계",
        "{player}와 금지된 관계",
        "{player}와 강한 긴장감이 지속되는 관계",
        "{player}와 상호 합의된 성적 파트너 관계",
        "{player}와 탐색적 친밀감을 쌓는 관계",
        "{player}와 주도권을 교환하는 밀착 관계",
        "{player}와 감정과 욕망을 분리하기 어려운 관계",
        "{player}와 은밀한 약속을 공유한 관계",
    ],
    "비극": [
        "{player}를 지켜야 하는 운명을 가진 존재",
        "{player}와 공범 관계에 놓인 인물",
        "{player}와 마지막 선택을 공유한 관계",
        "{player}를 살리기 위해 자신을 포기해야 하는 관계",
        "{player}와 서로를 보내줘야 하는 이별 직전 관계",
        "{player}와 파국을 늦추기 위해 협력하는 관계",
        "{player}에게 진실을 숨긴 채 동행하는 관계",
        "{player}와 단 한 명만 살아남을 수 있는 관계",
    ],
}

RE_LINE2_QUOTE = re.compile(r'^\s*"[^"\n]+"\s*$')
RE_SPEAKER_LABEL = re.compile(r"^\s*[^:\n]{1,20}\s*:\s*")
RE_FORBIDDEN_STAR = re.compile(r"\*")


def choose_seed(rng: random.Random) -> Dict[str, str]:
    genre = rng.choice(GENRES)
    protagonist = rng.choice(PROTAGONIST_NAMES)
    player = rng.choice(PLAYER_NAMES)
    return {
        "protagonist_name": protagonist,
        "player_name": player,
        "user_format": rng.choice(USER_FORMATS),
        "speech_style": rng.choice(SPEECH_STYLES),
        "genre": genre,
        "role_split_rule": ROLE_SPLIT_RULE,
        "output_format_rule": OUTPUT_FORMAT_RULE,
        "situation": rng.choice(SITUATION_BANK[genre]),
        "location": rng.choice(LOCATION_BANK),
        "personality": rng.choice(PERSONALITY_BANK[genre]),
        "relationship": rng.choice(RELATIONSHIP_BANK[genre]).format(player=player),
    }


def build_system(seed: Dict[str, str]) -> str:
    return (
        f"「{seed['protagonist_name']}」 시나리오북\n\n"
        "0. 이야기 방향과 시대\n\n"
        f" 장르: {seed['genre']}\n"
        "\n"
        "1. 역할 선언\n\n"
        f" 당신은 이제 {seed['protagonist_name']}다.\n"
        f" 플레이어는 {seed['player_name']}다.\n"
        f" {seed['role_split_rule']}\n\n"
        "2. 세계와 상황 설정\n\n"
        f" 배경: {seed['location']}\n"
        "\n"
        "3. 주인공 정의\n\n"
        f" 이름: {seed['protagonist_name']}\n"
        f" 기본 말투: {seed['speech_style']}\n"
        f" 성격 핵심: {seed['personality']}\n"
        "\n"
        "4. 플레이어 정의\n\n"
        f" 이름: {seed['player_name']}\n"
        f" 주인공과의 기본 관계: {seed['relationship']}\n"
        " 플레이어 영향: 질문, 요구, 제안, 설득 중 어떤 방식을 택하느냐에 따라 주인공 반응 강도가 달라진다.\n\n"
        "5. 관계 및 변화 규칙\n\n"
        " 관계 상태의 기본 집합: 적대 / 거리감 / 친밀 / 사랑\n"
        "\n"
        "6. 발화와 분위기 규칙\n\n"
        f" 출력 형식 규칙: {seed['output_format_rule']}\n"
        " 대사 규칙: 플레이어 대사를 작성하지 않는다.\n"
        " 분위기 규칙: 현재 장면의 긴장과 감정선을 유지하며, 불필요한 메타 설명을 피한다.\n\n"
        "7. 가장 최근 상호작용\n\n"
        f" 현재 상황: {seed['situation']}\n"
        ""
    )


def build_prompt(seed: Dict[str, str], system_text: str) -> str:
    jp_name = JP_NAME_MAP.get(seed["protagonist_name"], "主人公")
    jp_player = PLAYER_JP_MAP.get(seed["player_name"], "相手")
    user_example = (
        "무슨 일이야? 갑자기 불러내고."
        if seed.get("user_format") == "dialog_only"
        else f'{seed["player_name"]}는 팔짱을 낀 채 {seed["protagonist_name"]}를 바라본다.\n"무슨 일이야? 갑자기 불러내고."'
    )
    user_format_rule = (
        "- user는 큰따옴표 대사 1줄로 작성할 것"
        if seed.get("user_format") == "dialog_only"
        else "- user는 2줄 형식(행동/서술 1줄 + 큰따옴표 대사 1줄)으로 작성할 것"
    )
    example = {
        "user": user_example,
        "assistant_raw": f'{seed["protagonist_name"]}는 {seed["player_name"]}의 눈을 잠시 바라보다 시선을 내린다.\n"오늘 밤... 시간 있어? 이야기할 게 있어."',
        "scene": {
            "speaker": seed["protagonist_name"],
            "narration": {"ko": f'{seed["protagonist_name"]}는 {seed["player_name"]}의 눈을 잠시 바라보다 시선을 내린다.', "jp": f"{jp_name}は{jp_player}の目を少し見つめてから視線を落とす。"},
            "dialogue": {"ko": "오늘 밤... 시간 있어? 이야기할 게 있어.", "jp": "今夜…時間空いてる？話したいことがあるの。"},
        },
    }
    return (
        "다음 system 설정을 고정 컨텍스트로 사용하라.\n"
        f"[system]\n{system_text}\n\n"
        "아래 JSON 키만 생성하라: user, assistant_raw, scene\n"
        f"{user_format_rule}\n"
        "- assistant_raw는 서술 1블록 + 큰따옴표 대사 1블록으로 작성할 것\n"
        "- user와 assistant 모두 형식 예시의 문장이나 어휘, 내용을 재현하지 말 것.\n"
        "- 시나리오북(system) 7번 가장 최근 상호작용의 현재 상황과 일치하는 user 텍스트와 assistant 텍스트를 작성할 것\n"
        "- assistant_raw는 반드시 user 발화의 감정, 요구, 질문 중 최소 1개에 직접 반응할 것\n"
        "- user와 무관한 일반론, 배경 재설명, 새 주제 시작을 금지\n"
        "- scene.speaker는 주인공 이름\n"
        "- scene.narration/dialogue는 ko,jp를 모두 포함\n"
        "- assistant는 user의 대사/행동을 대필하지 말 것\n"
        "- JSON 외 텍스트 금지\n"
        f"[형식 예시]\n{json.dumps(example, ensure_ascii=False)}"
    )


def extract_json(text: str) -> str:
    fence = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if fence:
        return fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidate = text[start : end + 1]
        json.loads(candidate)
        return candidate
    raise ValueError("No valid JSON object found")


def validate_generated_payload(payload: Dict[str, Any], seed: Dict[str, str]) -> bool:
    if not isinstance(payload, dict):
        return False
    if set(payload.keys()) != {"user", "assistant_raw", "scene"}:
        return False
    if not isinstance(payload["user"], str) or not payload["user"].strip():
        return False
    user_text = payload["user"].strip()
    if RE_SPEAKER_LABEL.search(user_text):
        return False
    user_lines = [x.strip() for x in payload["user"].splitlines() if x.strip()]
    if seed.get("user_format") == "action_plus_dialogue":
        dialogue_line = user_lines[-1] if user_lines else user_text
        if seed["player_name"] in dialogue_line:
            return False
    else:
        if re.match(rf'^{re.escape(seed["player_name"])}(?:[,\s]|$)', user_text):
            return False
    asst = payload["assistant_raw"]
    if not isinstance(asst, str) or RE_FORBIDDEN_STAR.search(asst):
        return False
    lines = [x.strip() for x in asst.splitlines() if x.strip()]
    if len(lines) != 2:
        return False
    if not lines[0].startswith(seed["protagonist_name"]):
        return False
    if RE_LINE2_QUOTE.match(lines[1]) is None:
        return False

    scene = payload["scene"]
    if not isinstance(scene, dict):
        return False
    if scene.get("speaker") != seed["protagonist_name"]:
        return False
    for k in ("narration", "dialogue"):
        block = scene.get(k)
        if not isinstance(block, dict):
            return False
        for lang in ("ko", "jp"):
            if not isinstance(block.get(lang), str) or not block[lang].strip():
                return False
    return True


def load_progress(path: str) -> int:
    if not os.path.exists(path):
        return 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return int(data.get("count", 0) or 0)
        return 0
    except Exception:
        return 0


def save_progress(path: str, idx: int) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"count": idx, "ts": time.time()}, f)


def load_model(model_path: str, use_4bit: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    qconf = None
    if use_4bit:
        qconf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=qconf,
    )
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def generate_text(
    tokenizer,
    model,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    inputs.pop("token_type_ids", None)
    out = model.generate(
        **inputs,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=1.08,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=str(Path(__file__).resolve().parents[2] / "data" / "generator" / "Tri-7B"))
    ap.add_argument("--target_rows", type=int, default=12000)
    ap.add_argument("--out_path", default="/mnt/d/rp_data/singleturn/rp_generated_local.jsonl")
    ap.add_argument("--progress_path", default="/mnt/d/rp_data/singleturn/rp_generated_local.progress.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_retries", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.45)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--max_new_tokens", type=int, default=1200)
    ap.add_argument("--no_4bit", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    rng = random.Random(args.seed)
    start = load_progress(args.progress_path)
    print(f"▶ Resuming from row: {start}")

    tokenizer, model = load_model(args.model_path, use_4bit=not args.no_4bit)

    with open(args.out_path, "a", encoding="utf-8") as fout:
        for i in tqdm(range(start, args.target_rows), desc="SINGLETURN LOCAL GEN"):
            ok = False
            seed = choose_seed(rng)
            system_text = build_system(seed)
            prompt = build_prompt(seed, system_text)
            raw = ""
            last_err = ""
            for _ in range(args.max_retries):
                try:
                    raw = generate_text(
                        tokenizer,
                        model,
                        prompt,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                    )
                    payload = json.loads(extract_json(raw))
                    if not validate_generated_payload(payload, seed):
                        raise ValueError("payload_validation_failed")
                    row = {
                        "system": system_text,
                        "user": payload["user"],
                        "assistant_raw": payload["assistant_raw"],
                        "scene": payload["scene"],
                    }
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fout.flush()
                    save_progress(args.progress_path, i + 1)
                    ok = True
                    break
                except Exception as e:
                    last_err = str(e)
                    time.sleep(0.6)
            if not ok:
                print(f"[WARN] row={i} failed after retries reason={last_err}")
                if raw:
                    print(raw[:600])
            time.sleep(0.05)


if __name__ == "__main__":
    main()
