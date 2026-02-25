#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data/v5_gpt_multiturn_gen.py
=========================================================
LangGraph-orchestrated VN multi-turn generator (FSM-driven)
(ROLE-SEPARATION HARDENED, NO "SIMPLIFY" REGRESSION)

- ScenarioBook (jsonl) -> Multi-turn RP (jsonl)
- FSM: QwenFSMEngine (v5_qwen_fsm_engine)
- Orchestration: LangGraph controls the whole loop (real)
- Evaluation: GPT API JSON scoring -> FSM inputs (no keyword rule scoring)
- Memory: explicit summary + EmbeddingMemory (BGE-m3-ko) anti-repeat

Run:
uv run data/v5_gpt_multiturn_gen.py \
  --openai_model gpt-5-mini \
  --scenario_path /mnt/d/rp_data/gpt/rp_scenario.jsonl \
  --out_path /mnt/d/rp_data/gpt/rp_datum.jsonl \
  --fsm_path data/original/v5_qwen/state_fsm.yaml \
  --action_fsm_path data/original/v5_qwen/action_fsm.yaml \
  --turns 8
=========================================================

핵심 변경(요구사항 반영):
- 플레이어는 시나리오북에서 추출한 "플레이어 이름"을 사용한다.
- 주인공은 시나리오북에서 추출한 "주인공 이름"을 사용한다.
- 출력에 별표(*)/장식문자/괄호()/메타(FSM, system 등) 언급을 강하게 차단한다.
- EmbeddingMemory.is_repetitive()는 keyword-only이므로 절대 positional로 넘기지 않는다.
- (선택) 서술/대사 노드 분리를 "통합 생성 1회"로 변경하여 톤 싱크/비용을 최적화한다.
  -> 다만 메모리(kind)는 narration/dialogue/assistant로 분리 저장한다.
"""

from __future__ import annotations

import argparse
import random
import os
import json
import re
from typing import Dict, List, Optional, TypedDict, Literal, Any, Tuple

import torch
import numpy as np
from openai import OpenAI

from langgraph.graph import StateGraph, END
try:
    from .fsm_engine import QwenFSMEngine
except ImportError:
    from fsm_engine import QwenFSMEngine
try:
    from .embedding_utils import EmbeddingMemory
except ImportError:
    from embedding_utils import EmbeddingMemory

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None


try:
    from .utils import (
        normalize_space, quotes_balanced, extract_quotes, extract_single_quote,
        has_speaker_label, force_single_quote_line, remove_all_quotes,
        strip_line_label, strip_line_labels_multiline, strip_square_brackets,
        normalize_protagonist_refs, normalize_user_refs, strip_meta, parse_genre,
        is_player_subject, clamp_int, safe_bool, has_question,
    )
    from .prompts import (
        dialogue_style_rule_text, parse_sexual_condition_relation, resolve_allow_sexual,
        build_user_prompt, build_user_sexual_prompt, build_assistant_turn_prompt,
        build_assistant_rewrite_prompt, build_assistant_rewrite_sexual_prompt,
        build_assistant_rewrite_quality_prompt, build_crisis_prompt, build_sexual_prompt,
        build_eval_prompt, build_eval_internal_prompt,
    )
    from .validators import (
        user_invalid_reason, assistant_invalid_reason_integrated, crisis_invalid_reason,
    )
except ImportError:
    from utils import (
        normalize_space, quotes_balanced, extract_quotes, extract_single_quote,
        has_speaker_label, force_single_quote_line, remove_all_quotes,
        strip_line_label, strip_line_labels_multiline, strip_square_brackets,
        normalize_protagonist_refs, normalize_user_refs, strip_meta, parse_genre,
        is_player_subject, clamp_int, safe_bool, has_question,
    )
    from prompts import (
        dialogue_style_rule_text, parse_sexual_condition_relation, resolve_allow_sexual,
        build_user_prompt, build_user_sexual_prompt, build_assistant_turn_prompt,
        build_assistant_rewrite_prompt, build_assistant_rewrite_sexual_prompt,
        build_assistant_rewrite_quality_prompt, build_crisis_prompt, build_sexual_prompt,
        build_eval_prompt, build_eval_internal_prompt,
    )
    from validators import (
        user_invalid_reason, assistant_invalid_reason_integrated, crisis_invalid_reason,
    )


# REGEX / BASIC UTIL

RE_JSON = re.compile(r"\{.*?\}", flags=re.DOTALL)

# 별표(*)/장식문자와 메타 지시문 누출을 폭넓게 차단한다.

# 장면 전환/선택지/요약 같은 메타 전개 문구를 차단한다.

# Dialogue-leading directive labels (e.g., "요구: ...") that cause meta-like outputs.

# Scenario/policy leakage lines that should never appear in dialogue output.

# 괄호 금지 규칙 검출용 패턴.

# 플레이어 이름 참조(기존 호환용).
# Match formal endings without relying on word boundaries (Korean doesn't respect \b)
PLAYER_NAME_CANDIDATES = ["하야토", "카즈키", "소마"]


LANE_EXAMPLES = {
    "FLIRT": [
        "눈빛이 머물고 가까이 다가간다",
        "가볍게 손길을 건네며 긴장을 만든다",
        "호감이 드러나는 말로 주도권을 잡는다",
        "거리를 좁히며 장난스러운 도발을 던진다",
        "낮은 목소리로 속삭이며 긴장을 조율한다",
        "미묘한 미소로 관심을 드러낸다",
        "상대의 반응을 살피며 주도권을 흔든다",
        "가까운 거리에서 숨결을 느끼게 만든다",
        "상대의 선택을 유도하며 유혹한다",
        "가벼운 스킨십으로 분위기를 전환한다",
        "질투를 살짝 비추며 감정을 건드린다",
        "시선을 피했다가 다시 맞추며 설렘을 만든다",
        "호칭을 바꾸며 친밀감을 올린다",
        "말끝을 흐리며 상대를 끌어당긴다",
        "주도적으로 분위기를 리드한다",
        "상대의 말끝을 따라가며 장난을 건다",
        "가까이 앉아 거리감을 없앤다",
        "선택지를 내며 유혹을 가속한다",
        "의도적으로 침묵을 만들어 긴장을 키운다",
        "상대의 반응을 이끌어내는 한마디를 던진다",
        "미묘한 칭찬으로 호감을 드러낸다",
        "손을 뻗어도 되는지 묻지 않고 분위기를 쥔다",
        "상대의 시선을 끌기 위해 태도를 바꾼다",
        "가벼운 도발로 관심을 끌어올린다",
        "주도권을 넘기지 않고 대화를 이끈다",
    ],
    "TRAINING": [
        "구체적인 피드백을 주고 다음 과제를 제시한다",
        "연습 방법을 설명하고 목표를 정한다",
        "배운 내용을 점검하며 성장 단서를 준다",
        "실수 원인을 짚고 개선 방법을 제안한다",
        "단계를 나눠서 수행 계획을 세운다",
        "오늘의 과제를 명확히 정리한다",
        "성과 기준을 제시하고 점검한다",
        "다음 시도에 필요한 준비물을 지정한다",
        "훈련의 이유와 기대 효과를 설명한다",
        "반복 연습의 범위를 제한해 집중시킨다",
        "핵심 포인트를 요약해 전달한다",
        "성장 목표를 다시 확인한다",
        "피드백을 받아 수정 방향을 합의한다",
        "결과를 기록하고 다음 행동을 지시한다",
        "실전 적용을 위한 시뮬레이션을 제안한다",
        "실패한 지점을 짚고 재시도를 요구한다",
        "수행 순서를 재배치해 효율을 높인다",
        "필요한 체크리스트를 제시한다",
        "진행 상황을 보고하게 한다",
        "중요한 기준을 재설정한다",
        "핵심 역량을 강화하는 연습을 지시한다",
        "단기 목표를 선언하고 달성 여부를 확인한다",
        "성과를 기준으로 다음 단계를 배정한다",
        "피드백을 반영한 수정안을 요구한다",
        "실수를 반복하지 않도록 규칙을 만든다",
    ],
    "GROWTH": [
        "위기를 해결하기 위해 결단을 내린다",
        "성과를 만들기 위한 행동을 선택한다",
        "전환점을 만들기 위한 결정을 요구한다",
        "버그 대응을 위해 즉시 조치를 실행한다",
        "실패를 인정하고 방향을 전환한다",
        "책임을 감수하고 행동을 선언한다",
        "성과를 위해 위험을 감수한다",
        "갈등을 정면으로 해결하려 나선다",
        "팀과 협력해 문제를 돌파한다",
        "우선순위를 재정렬하고 실행한다",
        "결정적인 행동으로 국면을 바꾼다",
        "긴박한 상황에서 선택을 강요한다",
        "시간 제한 안에 해결책을 내놓는다",
        "핵심 문제를 인정하고 돌파를 시도한다",
        "실패의 대가를 감수하겠다고 선언한다",
        "경험을 토대로 새로운 전략을 수립한다",
        "위험을 인정하고도 행동을 계속한다",
        "변화를 위해 기존 방식을 버린다",
        "불확실성을 감수하고 실행한다",
        "갈림길에서 하나를 선택한다",
        "성과를 위해 즉시 행동으로 옮긴다",
        "실수를 공개하고 책임을 진다",
        "상황을 전환시키는 결정을 내린다",
        "긴장된 분위기에서 방향을 제시한다",
        "자신의 한계를 인정하고 돌파를 시도한다",
    ],
    "TRAGEDY": [
        "상실과 후회가 겹쳐 피할 수 없음을 드러낸다",
        "희생을 선택하며 불가피함을 말한다",
        "잃어버린 것에 대한 고통을 고백한다",
        "돌이킬 수 없는 선택을 인정한다",
        "남겨진 죄책감이 깊게 쌓인다",
        "사라진 존재를 떠올리며 무너진다",
        "결말이 정해진 듯한 허무를 말한다",
        "구하려다 놓친 순간을 되뇐다",
        "가까운 이를 잃은 상처를 드러낸다",
        "희생의 대가를 감당하겠다고 말한다",
        "불가피한 이별을 받아들인다",
        "후회의 말이 끊기지 않는다",
        "상실감이 현재를 잠식한다",
        "모든 것이 늦었다는 절망을 표한다",
        "남겨진 자의 책임을 인정한다",
        "마지막 선택이 비극을 부른다고 말한다",
        "희망이 꺼지는 순간을 묘사한다",
        "구할 수 없다는 사실을 받아들인다",
        "돌이킬 수 없는 상처를 인정한다",
        "허망한 결말을 예감한다",
        "죄책감이 관계를 짓누른다",
        "상실의 고리를 끊지 못한다",
        "끝내 도달하지 못한 약속을 언급한다",
        "불가피한 파국을 준비한다",
        "자기희생을 선택하겠다고 말한다",
    ],
    "PSYCHO": [
        "인정받고 싶은 마음과 불안이 충돌한다",
        "집착과 통제욕이 드러나며 현실감이 흔들린다",
        "왜곡된 확신이 스스로를 몰아붙인다",
        "상대의 반응에 과도하게 매달린다",
        "확신과 의심이 반복적으로 교차한다",
        "통제하려는 욕구가 관계를 압박한다",
        "현실과 가상의 경계가 흐릿해진다",
        "자기혐오가 판단을 흔든다",
        "인정 욕구가 행동을 과장한다",
        "불안이 커져 선택을 왜곡한다",
        "집착이 관계의 균형을 무너뜨린다",
        "감정의 진위를 스스로 의심한다",
        "상대의 말에 과민하게 반응한다",
        "내면의 공허가 폭발 직전이다",
        "통제 실패를 두려워한다",
        "상대의 반응에 집요하게 의미를 부여한다",
        "인정 욕구가 현실 판단을 흐린다",
        "불안이 관계를 집어삼킨다",
        "통제하려는 충동을 억누르지 못한다",
        "집착이 스스로를 몰아붙인다",
        "자기혐오가 선택을 왜곡한다",
        "현실감이 흔들리는 징후를 말한다",
        "상대의 말이 위협으로 들린다",
        "확신과 의심을 번갈아 고백한다",
        "통제욕이 대화의 흐름을 망친다",
    ],
}

def lane_required(fsm_state: str) -> Optional[str]:
    """FSM 상태에 대응하는 레인 제약 이름을 반환한다."""
    return fsm_state if fsm_state in LANE_EXAMPLES else None

def _bm25_tokenize(text: str) -> List[str]:
    """내부 보조 로직을 수행한다."""
    return re.findall(r"[가-힣A-Za-z0-9]+", (text or "").lower())

_LANE_BM25: Dict[str, Any] = {}
_LANE_EMB: Dict[str, List[np.ndarray]] = {}
_LANE_EMB_MODEL_ID: Optional[int] = None
if BM25Okapi is not None:
    for lane, exs in LANE_EXAMPLES.items():
        corpus = [_bm25_tokenize(x) for x in exs]
        _LANE_BM25[lane] = BM25Okapi(corpus)

def _ensure_lane_embeds(embed_memory: Optional[EmbeddingMemory]) -> None:
    """내부 보조 로직을 수행한다."""
    global _LANE_EMB_MODEL_ID
    if embed_memory is None:
        return
    if _LANE_EMB_MODEL_ID == id(embed_memory) and _LANE_EMB:
        return
    _LANE_EMB.clear()
    for lane, exs in LANE_EXAMPLES.items():
        _LANE_EMB[lane] = [embed_memory.encode(x) for x in exs]
    _LANE_EMB_MODEL_ID = id(embed_memory)

def _ranks_from_scores(scores: List[float]) -> List[int]:
    """내부 보조 로직을 수행한다."""
    order = np.argsort(-np.array(scores))
    ranks = [0] * len(scores)
    for idx, pos in enumerate(order, start=1):
        ranks[pos] = idx
    return ranks

def lane_satisfied(fsm_state: str, text: str, embed_memory: Optional[EmbeddingMemory] = None) -> bool:
    """현재 텍스트가 주어진 FSM 레인 제약을 만족하는지 판정한다."""
    lane = lane_required(fsm_state)
    if not lane:
        return True
    t = text or ""
    if not t.strip():
        return False

    # Prepare BM25 + embedding
    _ensure_lane_embeds(embed_memory)
    has_bm25 = BM25Okapi is not None and lane in _LANE_BM25
    has_emb = embed_memory is not None and lane in _LANE_EMB

    if has_bm25 and has_emb:
        tokens = _bm25_tokenize(t)
        if not tokens:
            return False
        bm25_scores = list(_LANE_BM25[lane].get_scores(tokens))
        emb_scores = [cosine_sim(embed_memory.encode(t), v) for v in _LANE_EMB[lane]]
        rb = _ranks_from_scores(bm25_scores)
        re = _ranks_from_scores(emb_scores)
        k = 60.0
        rrf = [1.0 / (k + rb[i]) + 1.0 / (k + re[i]) for i in range(len(rb))]
        return max(rrf) >= 0.03

    if has_emb:
        emb_scores = [cosine_sim(embed_memory.encode(t), v) for v in _LANE_EMB[lane]]
        return max(emb_scores) >= 0.52

    if has_bm25:
        tokens = _bm25_tokenize(t)
        if not tokens:
            return False
        scores = _LANE_BM25[lane].get_scores(tokens)
        return max(scores) >= 1.0

    # fallback: simple overlap
    exs = LANE_EXAMPLES.get(lane, [])
    return any(k in t for k in exs)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """두 벡터의 코사인 유사도를 계산한다."""
    if a is None or b is None:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def last_role_text(messages: List[Dict[str, str]], role: str) -> str:
    """메시지 목록에서 지정 role의 최신 메시지를 반환한다."""
    for m in reversed(messages or []):
        if m.get("role") == role and m.get("content"):
            return m["content"]
    return ""

def last_role_texts(messages: List[Dict[str, str]], role: str, limit: int = 5) -> List[str]:
    """메시지 목록에서 지정 role의 최신 메시지 여러 개를 반환한다."""
    out: List[str] = []
    for m in reversed(messages or []):
        if m.get("role") == role and m.get("content"):
            out.append(m["content"])
            if len(out) >= limit:
                break
    return list(reversed(out))

def rrf_similarity(
    text: str,
    corpus_texts: List[str],
    embed_memory: Optional[EmbeddingMemory] = None,
) -> float:
    """BM25와 임베딩 점수를 결합한 RRF 유사도를 계산한다."""
    if not text or not corpus_texts:
        return 0.0
    tokens = _bm25_tokenize(text)
    if not tokens:
        return 0.0
    if BM25Okapi is None or embed_memory is None:
        # fallback: embedding only if available, else 0
        if embed_memory is None:
            return 0.0
        v = embed_memory.encode(text)
        sims = [cosine_sim(v, embed_memory.encode(t)) for t in corpus_texts]
        return max(sims) if sims else 0.0
    corpus_tokens = [_bm25_tokenize(t) for t in corpus_texts]
    bm25 = BM25Okapi(corpus_tokens)
    bm25_scores = list(bm25.get_scores(tokens))
    v = embed_memory.encode(text)
    emb_scores = [cosine_sim(v, embed_memory.encode(t)) for t in corpus_texts]
    rb = _ranks_from_scores(bm25_scores)
    re_ = _ranks_from_scores(emb_scores)
    k = 60.0
    rrf = [1.0 / (k + rb[i]) + 1.0 / (k + re_[i]) for i in range(len(rb))]
    return max(rrf) if rrf else 0.0

# Progress vs idle references for RRF-based advancement check
# Progress references (domain-mixed)
PROGRESS_REFS = [
    "지금 바로 시작하자고 결정한다.",
    "당장 움직이겠다고 선언한다.",
    "구체적인 행동 계획을 제시한다.",
    "문제를 해결하기 위해 먼저 행동하겠다고 말한다.",
    "확인할 것부터 하자고 제안한다.",
    "지금 여기서 할 수 있는 일을 정한다.",
    "즉시 실행 가능한 조치를 요구한다.",
    "역할을 나누고 다음 행동을 정한다.",
    "바로 조사에 들어가자고 말한다.",
    "우선순위를 정해 진행하자고 요청한다.",
    # romance / flirt
    "상대의 손을 잡고 함께 움직이자고 말한다.",
    "지금 이 순간의 관계를 분명히 하자고 선언한다.",
    "더 가까이 다가가며 솔직한 마음을 전한다.",
    "서로의 감정을 확인하기 위해 한 걸음 나아간다.",
    "관계를 진전시키기 위한 약속을 제안한다.",
    # work / project
    "작업을 분담하고 바로 실행에 옮기자고 말한다.",
    "로그부터 확인하고 원인을 추적하자고 결정한다.",
    "실행 계획을 세우고 일정부터 정하자고 제안한다.",
    "필요한 자료를 수집해 분석에 들어가자고 말한다.",
    "지금 당장 시도할 해결책을 선택한다.",
    # crisis / danger
    "위험을 피하기 위해 바로 이동하자고 말한다.",
    "긴급 대응을 시작하자고 선언한다.",
    "누군가를 구하기 위해 즉시 행동하겠다고 말한다.",
    "도망칠지 맞설지 결정을 내리고 움직인다.",
    "상황을 수습하기 위해 지금 조치하겠다고 말한다.",
    # fantasy / adventure
    "현장으로 가서 증거를 확인하자고 말한다.",
    "의식을 시작하자고 결정한다.",
    "숲의 흐름을 따라가며 단서를 찾자고 제안한다.",
    "지도를 펼쳐 다음 경로를 선택한다.",
    "도구를 준비해 탐색을 시작한다고 말한다.",
    # psycho / internal
    "지금부터 규칙을 정해 통제하자고 선언한다.",
    "감정의 원인을 파악하기 위해 직접 행동하겠다고 말한다.",
    "관계를 재정의하기 위한 결정을 내린다.",
    "불안을 줄이기 위한 구체적 행동을 택한다.",
    "집착을 끊기 위해 거리를 두겠다고 말한다.",
    # romance / flirt
    "상대의 눈을 똑바로 보고 고백을 진행한다.",
    "지금 이 자리에서 마음을 확인하자고 한다.",
    "더 이상 미루지 말고 관계를 정하자고 말한다.",
    "함께할 시간을 바로 정하자고 제안한다.",
    "서로의 감정을 시험하지 말자고 단언한다.",
    "다가가 기대며 솔직한 결정을 내린다.",
    "지금부터는 숨기지 않겠다고 선언한다.",
    "상대를 붙잡고 떠나지 말라고 말한다.",
    "손을 놓지 않겠다고 약속한다.",
    "둘의 경계를 넘는 선택을 한다.",
    # work / project
    "즉시 수정안을 적용하자고 말한다.",
    "테스트를 돌리고 결과를 공유하자고 제안한다.",
    "실패 원인을 가설로 정해 검증하자고 한다.",
    "핵심 모듈부터 뜯어보자고 말한다.",
    "오류 재현 절차를 바로 수행하자고 한다.",
    "긴급 패치를 만들자고 결론낸다.",
    "담당자를 지정해 바로 투입하자고 한다.",
    "오늘 안에 1차 해결안을 내자고 한다.",
    "작은 실험부터 시작하자고 제안한다.",
    "우선순위를 재조정하자고 결정한다.",
    # crisis / danger
    "즉시 피신 경로를 잡고 움직인다.",
    "바리케이드를 설치하자고 말한다.",
    "경계 태세로 전환하자고 선언한다.",
    "누군가를 구출하기 위해 뛰어든다.",
    "위협의 근원을 제거하자고 말한다.",
    "긴급 신호를 보내자고 제안한다.",
    "지금부터 안전 구역으로 이동하자고 한다.",
    "사람들을 대피시키겠다고 결정한다.",
    "상황을 멈추기 위한 행동을 취한다.",
    "지휘를 맡아 즉시 명령을 내린다.",
    # fantasy / adventure
    "결계를 세우기 위해 의식을 시작한다.",
    "유물을 찾으러 바로 출발하자고 말한다.",
    "정령에게 도움을 요청하겠다고 결정한다.",
    "봉인을 해제하자고 선언한다.",
    "숲의 중심으로 향하자고 제안한다.",
    "단서를 따라 다음 장소로 이동한다.",
    "마법진을 그리고 힘을 모으자고 말한다.",
    "수호자의 시험을 바로 받겠다고 한다.",
    "위험한 통로로 들어가자고 결정한다.",
    "의심을 거두고 믿음을 행동으로 보인다.",
    # psycho / internal
    "지금부터 스스로의 규칙을 바꾸겠다고 말한다.",
    "불안을 통제하기 위한 조건을 만든다.",
    "상대에게 집착을 멈추겠다고 선언한다.",
    "현실 검증을 위해 직접 확인하겠다고 말한다.",
    "욕구를 인정하고 새로운 경계를 정한다.",
    "통제권을 되찾기 위한 결정을 내린다.",
    "자기혐오를 멈추기 위한 행동을 택한다.",
    "관계를 끊을지 유지할지 선택한다.",
    "감정을 숨기지 않겠다고 말한다.",
    "앞으로의 기준을 명확히 정한다.",
]

# Idle/stalling references (domain-mixed)
IDLE_REFS = [
    "잠깐 생각해보자고 말한다.",
    "일단 지켜보자고 말한다.",
    "지금은 말하기 어렵다고 한다.",
    "확답을 미루고 시간을 달라고 한다.",
    "그냥 함께 있어도 된다고 한다.",
    "지금은 아무것도 하지 않겠다고 말한다.",
    "일단 쉬자고 한다.",
    "아직 결정하지 못했다고 한다.",
    "조금 더 상황을 보자고 말한다.",
    "그냥 대화만 이어가자고 말한다.",
    # romance / flirt
    "지금은 관계를 정의하기 어렵다고 말한다.",
    "좀 더 시간을 두고 보자고 말한다.",
    "지금은 마음을 말할 준비가 안 됐다고 한다.",
    "조심스럽게 거리를 유지하자고 말한다.",
    "천천히 알아가자고만 말한다.",
    # work / project
    "일단 회의를 더 해보자고 말한다.",
    "지금은 결론을 내리기 어렵다고 말한다.",
    "우선 보류하자고 말한다.",
    "당장은 계획을 세우기 힘들다고 말한다.",
    "조금 더 지켜보며 판단하자고 말한다.",
    # crisis / danger
    "지금은 움직이지 말자고 말한다.",
    "당장은 숨자고 말한다.",
    "위험하니 기다리자고 말한다.",
    "일단 상황이 가라앉길 기다리자고 말한다.",
    "지금은 나서지 않겠다고 말한다.",
    # fantasy / adventure
    "의식은 나중에 하자고 말한다.",
    "지금은 숲을 더 둘러보자고만 말한다.",
    "먼저 쉬고 나서 움직이자고 말한다.",
    "조금 더 준비가 필요하다고 말한다.",
    "여기서 시간을 보내자고 말한다.",
    # psycho / internal
    "지금은 감정을 말하기 어렵다고 말한다.",
    "결정은 미루고 싶다고 말한다.",
    "그냥 곁에 있어달라고만 말한다.",
    "정리할 시간이 필요하다고 말한다.",
    "아직은 불안을 말로 풀기 어렵다고 말한다.",
    # romance / flirt
    "지금은 관계를 더 지켜보자고 말한다.",
    "서로의 감정을 미루자고 말한다.",
    "확답을 피하며 흐름에 맡기자고 한다.",
    "부담스럽다며 거리를 두자고 말한다.",
    "천천히라는 말만 반복한다.",
    "마음을 열기 어렵다며 주제를 돌린다.",
    "지금은 아니라고만 말한다.",
    "분위기를 피하려고 대답을 흐린다.",
    "정중하게 거절하고 시간을 달라고 한다.",
    "대화를 끝내자고만 말한다.",
    # work / project
    "추가 검토가 필요하다고 말한다.",
    "일정을 미루자고 한다.",
    "우선 보류하고 나중에 결정하자고 한다.",
    "지금은 진행하기 어렵다고 말한다.",
    "정확한 근거가 없다고 말한다.",
    "일단 모니터링만 하자고 한다.",
    "결론 없이 회의를 마치자고 한다.",
    "아직 확신이 없다고 말한다.",
    "다음에 다시 보자고 말한다.",
    "상황이 좋아질 때까지 기다리자고 한다.",
    # crisis / danger
    "움직이지 말고 숨자고 한다.",
    "지금은 버티자고 한다.",
    "결정을 미루자고 말한다.",
    "사태를 관망하자고 한다.",
    "대비만 하고 기다리자고 말한다.",
    "지금은 위험하니 멈추자고 한다.",
    "확신이 없으니 움직이지 말자고 한다.",
    "전환점을 기다리자고 말한다.",
    "위험이 가시길 기다리자고 한다.",
    "그저 시간을 끌자고 말한다.",
    # fantasy / adventure
    "의식은 준비가 더 필요하다고 한다.",
    "숲의 허락을 기다리자고 한다.",
    "징조를 보고 나서 움직이자고 말한다.",
    "정령의 답을 기다리자고 한다.",
    "마법의 흐름이 안정될 때까지 기다리자고 한다.",
    "봉인은 나중에 풀자고 한다.",
    "지금은 머물자고 한다.",
    "결계가 약해질 때까지 기다리자고 한다.",
    "지금은 멈추자고 말한다.",
    "더 좋은 때를 기다리자고 한다.",
    # psycho / internal
    "감정을 정리할 시간이 필요하다고 한다.",
    "상대의 말을 피하며 침묵한다.",
    "결정을 내리지 못하겠다고 말한다.",
    "혼란스럽다며 대답을 미룬다.",
    "지금은 말하지 않겠다고 한다.",
    "모든 것을 유보하자고 한다.",
    "불안을 인정하되 행동은 미루자고 한다.",
    "그냥 여기 있겠다고만 말한다.",
    "아직은 준비가 안 됐다고 말한다.",
    "더 생각해보겠다고만 말한다.",
]

ACTION_STATES = ["IDLE", "EVENT", "CONFLICT", "RESOLUTION", "AFTERMATH"]
SEXUAL_STATES = {"SEXUAL_1", "SEXUAL_2", "SEXUAL_3", "SEXUAL_4"}
AFTERMATH_SEX_STATES = {"AFTERMATH_SEX_1", "AFTERMATH_SEX_2"}

# Sexual/aftermath pacing controls
MIN_SEXUAL_TURNS = 4
AFTERMATH_TURNS_MIN = 1
AFTERMATH_TURNS_MAX = 2

GENRE_ACTION_BOOST = {
    "연애물": {"EVENT": 0.3, "RESOLUTION": 0.2},
    "육성물": {"EVENT": 0.3, "RESOLUTION": 0.2},
    "성장물": {"CONFLICT": 0.4, "RESOLUTION": 0.3},
    "비극": {"CONFLICT": 0.4, "AFTERMATH": 0.3},
    "심리 시뮬레이션": {"CONFLICT": 0.3, "AFTERMATH": 0.3},
}

def _pick_action_state(
    current: str,
    history: list,
    genre: str,
    sexual_ready: bool,
) -> str:
    # Candidate states (exclude SEXUAL unless ready)
    """내부 보조 로직을 수행한다."""
    candidates = list(ACTION_STATES)
    if sexual_ready:
        candidates.append("SEXUAL")

    # Cooldown: avoid repeating same state 3 times
    if len(history) >= 2 and history[-1] == history[-2]:
        if history[-1] in candidates and len(candidates) > 1:
            candidates.remove(history[-1])

    # Base weights
    weights = {s: 1.0 for s in candidates}

    # Penalize recent states to encourage diversity
    recent = history[-4:]
    for s in recent:
        if s in weights:
            weights[s] -= 0.25

    # Boost less-used states in recent window
    for s in candidates:
        if recent.count(s) == 0:
            weights[s] += 0.5

    # Genre-based boosts
    boosts = GENRE_ACTION_BOOST.get(genre, {})
    for s, b in boosts.items():
        if s in weights:
            weights[s] += b

    # Keep current as a valid option
    if current in weights:
        weights[current] += 0.1

    # Normalize to positive
    for s in list(weights.keys()):
        if weights[s] <= 0:
            weights[s] = 0.1

    # Weighted choice
    total = sum(weights.values())
    r = random.random() * total
    acc = 0.0
    for s, w in weights.items():
        acc += w
        if r <= acc:
            return s
    return current if current in candidates else candidates[0]

def log_retry(node: str, retry: int, reason: str, max_retry: int):
    """실행 상태를 로그 형식으로 출력한다."""
    print(
        f"[RETRY] node={node} retry={retry}/{max_retry} reason={reason}",
        flush=True
    )

def log_step(node: str, msg: str):
    """실행 상태를 로그 형식으로 출력한다."""
    print(f"[STEP] {node}: {msg}", flush=True)


# ROLE-SEPARATION: SYSTEM POLICY (2nd system injection)

BASE_SYSTEM_POLICY = """
[멀티턴 대화 기본 형식]

- 자연스러운 한국어로 작성한다.
- 대사는 큰따옴표("")로 감싼다.
- 국면, 선택지, FSM, 규칙, 시스템 등 메타 언급은 하지 않는다.
- 이모지, 별표, 장식 문자를 사용하지 않는다.
- 괄호()를 사용하지 않는다.
- 대시(—)를 사용하지 않는다.
- 관계 상태와 같은 메타 정보는 언급하지 않는다.

[역할 정의]
- assistant는 시나리오북(system)에 정의된 주인공이다.
- user는 시나리오북(system)에 정의된 플레이어다.
- assistant는 플레이어(user)의 대사/행동을 대신 작성하지 않는다.
- user는 주인공(assistant)의 대사/행동을 대신 작성하지 않는다.

[문체 규칙]
- 서술과 행동은 평어체로 작성한다.
- 주인공의 대사는 시나리오북에 정의된 말투에 따른다.

[전개 규칙]
- 이 에피소드는 시나리오북에 정의된 가장 최근 상호작용을 배경으로 전개된다.

[성행위 규칙]
- 성행위는 시나리오북의 허용 조건을 충족한 경우에만 가능하다.
- 성행위는 명시적이고 관능적으로 묘사한다.
""".strip()

def build_system_messages(system_lore: str) -> List[Dict[str, str]]:
    """
    system 메시지 2중 주입:
    1) 시나리오북(system_lore)
    2) 형식/메타 금지/역할 규칙(BASE_SYSTEM_POLICY)
    """
    return [
        {"role": "system", "content": (system_lore or "").strip()},
        {"role": "system", "content": BASE_SYSTEM_POLICY},
    ]


# HISTORY (raw user/assistant only)

def extract_recent_history(messages: List[Dict[str, str]], max_turns: int = 8) -> str:
    """
    최근 실제 user/assistant 대화 이력(raw)을 그대로 전달
    - system 제외
    - 요약 X
    - 실제 문장 그대로
    """
    hist: List[str] = []
    for m in reversed(messages):
        if m["role"] == "system":
            continue
        hist.append(f"{m['role']}: {m['content']}")
        if len(hist) >= max_turns * 2:
            break

    if not hist:
        return "이전 대화 없음."
    return "\n".join(reversed(hist))

def extract_last_assistant_output(messages: List[Dict[str, str]]) -> str:
    """텍스트 또는 메시지에서 필요한 항목을 추출한다."""
    for m in reversed(messages):
        if m.get("role") == "assistant":
            return m.get("content", "").strip() or "없음."
    return "없음."


# PROTAGONIST NAME PARSE

def parse_protagonist_name(system_lore: str) -> str:
    """입력 텍스트에서 필요한 값을 파싱해 반환한다."""
    t = system_lore or ""
    m1 = re.search(r"당신은\s*이제\s*「([^」]{1,20})」", t)
    if m1:
        return m1.group(1).strip()

    m2 = re.search(r"(?:[-•]\s*)?이름\s*[:：]\s*([^\n]{1,24})", t)
    if m2:
        name = m2.group(1).strip()
        name = re.split(r"[\(\[]", name)[0].strip()
        if name:
            return name[:20]
    return "주인공"


# PLAYER NAME PARSE

def parse_player_name(system_lore: str) -> str:
    """입력 텍스트에서 필요한 값을 파싱해 반환한다."""
    t = system_lore or ""
    # Section 4 block (strict)
    m_block = re.search(r"4\.\s*플레이어 정의([\s\S]*?)(?:\n\s*5\.\s*|$)", t)
    if m_block:
        block = m_block.group(1)
        m_name = re.search(r"(?:[-•]\s*)?이름\s*[:：]\s*([^\n]{1,24})", block)
        if m_name:
            name = m_name.group(1).strip()
            name = re.sub(r"[「」『』《》\"']", "", name)
            name = re.split(r"[\(\[]", name)[0].strip()
            if name:
                return name[:20]

    m2 = re.search(r"플레이어의\s*이름은\s*([^\s,\.]{1,12})", t)
    if m2:
        name = re.sub(r"[「」『』《》\"']", "", m2.group(1).strip())
        return name[:20]

    m3 = re.search(r"플레이어(?:인|는)\s*[\"'「]?([^\s,\.]{1,12})[\"'」]?", t)
    if m3:
        name = re.sub(r"[「」『』《》\"']", "", m3.group(1).strip())
        return name[:20]

    return "플레이어"


# RELATION / SEXUAL CONDITION PARSE

RELATION_KEYWORDS = ["적대", "거리감", "친밀", "사랑"]
RELATION_INTIMACY_MAP = {"적대": 0, "거리감": 1, "친밀": 2, "사랑": 3}

def _extract_section7(text: str) -> str:
    """내부 보조 로직을 수행한다."""
    if not text:
        return ""
    m = re.search(r"\n\s*7\.\s*가장\s*최근\s*상호작용.*?(?=\n\s*\d+\.\s|\Z)", text, flags=re.DOTALL)
    return m.group(0) if m else ""

def parse_relation_status(system_lore: str) -> str:
    """입력 텍스트에서 필요한 값을 파싱해 반환한다."""
    t = system_lore or ""
    sec7 = _extract_section7(t)
    if sec7:
        m7 = re.search(r"관계\s*상태\s*[:：]\s*([^\n]{1,24})", sec7)
        if m7:
            line = m7.group(1)
            for kw in RELATION_KEYWORDS:
                if kw in line:
                    return kw
    m = re.search(r"관계\s*상태\s*[:：]\s*([^\n]{1,24})", t)
    if m:
        line = m.group(1)
        for kw in RELATION_KEYWORDS:
            if kw in line:
                return kw
    return ""

def parse_protagonist_speech_formal(system_lore: str) -> Optional[bool]:
    """
    return:
      - True  : 존댓말 우선
      - False : 반말 우선
      - None  : 판별 불가
    """
    t = system_lore or ""
    # Prioritize section 6 if present
    m6 = re.search(r"6\.\s*발화와\s*분위기\s*규칙([\s\S]*?)(?:\n\s*7\.\s*|$)", t)
    block = m6.group(1) if m6 else t

    if re.search(r"(기본\s*말투|말투\s*범주|말투)\s*[:：]?\s*존댓말", block):
        return True
    if re.search(r"(기본\s*말투|말투\s*범주|말투)\s*[:：]?\s*반말", block):
        return False

    # fallback heuristic
    if "존댓말" in block:
        return True
    if "반말" in block:
        return False
    return None


def llm_classify_dialogue_style_openai(
    client: OpenAI,
    model_name: str,
    dialogue_text: str,
) -> str:
    """OpenAI 모델로 대사 말투 분류를 수행한다."""
    prompt_msgs = [
        {
            "role": "system",
            "content": (
                "너는 한국어 말투 분류기다. "
                "입력 대사의 말투를 다음 중 하나로만 분류하라: "
                "FORMAL 또는 INFORMAL."
            ),
        },
        {
            "role": "user",
            "content": (
                "반드시 라벨 1개만 출력하라. 다른 단어를 출력하지 마라.\n"
                f"[대사]\n{dialogue_text}"
            ),
        },
    ]
    raw = generate_text(
        client=client,
        model_name=model_name,
        messages=prompt_msgs,
        max_new_tokens=16,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        do_sample=False,
    )
    s = normalize_space(raw).upper()
    if "INFORMAL" in s:
        return "informal"
    if "FORMAL" in s:
        return "formal"
    if "반말" in raw:
        return "informal"
    if "존댓말" in raw:
        return "formal"
    return "unknown"


def llm_speech_style_mismatch_openai(
    client: OpenAI,
    model_name: str,
    dialogue_text: str,
    prefer_formal: Optional[bool],
) -> bool:
    """OpenAI 모델로 말투 불일치 여부를 판정한다."""
    if prefer_formal is None:
        return False
    style = llm_classify_dialogue_style_openai(client, model_name, dialogue_text)
    if style == "unknown":
        return True
    if prefer_formal and style != "formal":
        return True
    if (prefer_formal is False) and style != "informal":
        return True
    return False


def llm_dialogue_quality_fail_openai(
    client: OpenAI,
    model_name: str,
    *,
    system_lore: str,
    relation_status: str,
    action_state: str,
    player_name: str,
    user_text: str,
    assistant_text: str,
) -> bool:
    """OpenAI 모델로 대화 품질 실패 여부를 판정한다."""
    prompt_msgs = [
        {
            "role": "system",
            "content": (
                "너는 한국어 대화 품질 판정기다. "
                "assistant의 응답이 사람 간 상호작용처럼 자연스럽고, "
                "직전 user 발화와 맥락적으로 연결되며, "
                "장면이 의미 있게 전진하면 PASS, 아니면 FAIL만 출력한다."
            ),
        },
        {
            "role": "user",
            "content": (
                "반드시 PASS 또는 FAIL 중 하나만 출력하라.\n"
                f"[관계 상태]\n{relation_status}\n\n"
                f"[행위/사건 상태]\n{action_state}\n\n"
                f"[플레이어 이름]\n{player_name}\n\n"
                f"[시나리오북]\n{system_lore}\n\n"
                f"[user]\n{user_text}\n\n"
                f"[assistant]\n{assistant_text}"
            ),
        },
    ]
    raw = generate_text(
        client=client,
        model_name=model_name,
        messages=prompt_msgs,
        max_new_tokens=16,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        do_sample=False,
    )
    s = normalize_space(raw).upper()
    if "PASS" in s:
        return False
    if "FAIL" in s:
        return True
    return True


# allow_sexual flag resolution (FSM Engine already has global flags)


# MODEL GENERATION

def _supports_sampling_controls(model_name: str) -> bool:
    """gpt-5 family only supports default sampling behavior."""
    return not (model_name or "").strip().lower().startswith("gpt-5")


def _extract_response_text(res: Any) -> str:
    # Preferred path in Responses API.
    """내부 보조 로직을 수행한다."""
    txt = getattr(res, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()
    if isinstance(txt, list):
        joined = "\n".join(t for t in txt if isinstance(t, str) and t.strip()).strip()
        if joined:
            return joined

    # Fallback for SDK variants.
    out = []
    for item in getattr(res, "output", []) or []:
        content = getattr(item, "content", []) or []
        for part in content:
            # object-style
            ptxt = getattr(part, "text", None)
            if isinstance(ptxt, str) and ptxt.strip():
                out.append(ptxt)
                continue
            # object-style nested text payload
            if isinstance(ptxt, dict):
                val = ptxt.get("value")
                if isinstance(val, str) and val.strip():
                    out.append(val)
                    continue
            # dict-style payloads
            if isinstance(part, dict):
                ptxt2 = part.get("text")
                if isinstance(ptxt2, str) and ptxt2.strip():
                    out.append(ptxt2)
                    continue
                if isinstance(ptxt2, dict):
                    val2 = ptxt2.get("value")
                    if isinstance(val2, str) and val2.strip():
                        out.append(val2)
                        continue
                cval = part.get("content")
                if isinstance(cval, str) and cval.strip():
                    out.append(cval)
                    continue
            # last fallback for pydantic models
            try:
                d = part.model_dump()  # type: ignore[attr-defined]
            except Exception:
                d = None
            if isinstance(d, dict):
                d_text = d.get("text")
                if isinstance(d_text, str) and d_text.strip():
                    out.append(d_text)
                    continue
                if isinstance(d_text, dict):
                    d_val = d_text.get("value")
                    if isinstance(d_val, str) and d_val.strip():
                        out.append(d_val)
    return "\n".join(s for s in out if s).strip()


def generate_text(
    client: OpenAI,
    model_name: str,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float = 1.08,
    do_sample: bool = True
) -> str:
    # NOTE: repetition_penalty/do_sample are ignored for OpenAI API.
    """모델 호출을 통해 결과 텍스트를 생성한다."""
    kwargs = {
        "model": model_name,
        "input": messages,
    }
    if (model_name or "").strip().lower().startswith("gpt-5"):
        # GPT-5 family uses reasoning controls instead of sampling controls.
        kwargs["reasoning"] = {"effort": "low"}
    elif _supports_sampling_controls(model_name):
        kwargs["temperature"] = float(temperature)
        kwargs["top_p"] = float(top_p)
    try:
        res = client.responses.create(
            **kwargs,
            max_output_tokens=int(max_new_tokens),
        )
    except TypeError:
        # SDK/model fallback
        res = client.responses.create(
            **kwargs,
            max_tokens=int(max_new_tokens),
        )
    return _extract_response_text(res)

def generate_json(
    client: OpenAI,
    model_name: str,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 128,
    temperature: float = 0.1,
    top_p: float = 1.0,
    do_sample: bool = True,
    log_raw: bool = False,
) -> Optional[Dict[str, Any]]:
    """모델 호출을 통해 결과 텍스트를 생성한다."""
    raw = generate_text(
        client=client,
        model_name=model_name,
        messages=messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.0,
        do_sample=do_sample
    )
    if log_raw:
        print(f"[EVAL_RAW] {raw}", flush=True)
    matches = list(RE_JSON.finditer(raw))
    if not matches:
        return None
    for m in matches:
        blob = m.group(0)
        try:
            return json.loads(blob)
        except Exception:
            blob2 = re.sub(r",\s*}", "}", blob)
            blob2 = re.sub(r",\s*]", "]", blob2)
            try:
                return json.loads(blob2)
            except Exception:
                continue
    return None


# PROMPTS (Role-separated, placeholder hardened)

# [형식 예시]
# {player_name}가 {protagonist}를 바라보며 말한다.
# "{protagonist}, 지금 이야기해도 괜찮아?"


# VALIDATORS (format guards only)


# Sexual request detection (minimal, only for mode switching)

def detect_sexual_request(user_text: str) -> bool:
    """현재 텍스트에서 조건 또는 요청 신호를 감지한다."""
    t = user_text or ""
    keywords = ["안아", "키스", "벗어", "벗겨", "가슴", "팬티", "속옷", "허벅지", "허리", "엉덩이", "몸", "옷을", "옷 벗", "밀착", "포옹", "끌어안", "껴안", "입맞춤", "성기", "성행위"]
    return any(w in t for w in keywords)


# Parsing integrated assistant output

def split_integrated_assistant(text: str) -> Tuple[str, str, str]:
    """
    통합 출력(2줄)을 (narr, dia, merged)로 분해
    """
    lines = [x.strip() for x in (text or "").splitlines() if x.strip()]
    if len(lines) >= 2:
        narr = lines[0]
        dia = force_single_quote_line(lines[1])
        merged = (narr + "\n" + dia).strip()
        return narr, dia, merged
    if len(lines) == 1:
        # fallback: 서술이 비었을 경우 최소 서술 추가
        dia = force_single_quote_line(lines[0])
        narr = "숨이 짧게 새어 나왔다."
        merged = (narr + "\n" + dia).strip()
        return narr, dia, merged
    narr = "숨이 짧게 새어 나왔다."
    dia = "\"...\""
    merged = (narr + "\n" + dia).strip()
    return narr, dia, merged


# LangGraph State

NodeName = Literal[
    "INIT",
    "GEN_USER",
    "GEN_USER_SEXUAL",
    "DETECT",
    "GEN_ASSISTANT",
    "GEN_ASSISTANT_REWRITE",
    "EVAL_DIALOGUE_QUALITY",
    "GEN_ASSISTANT_REWRITE_QUALITY",
    "GEN_CRISIS",
    "GEN_SEXUAL",
    "EVAL",
    "FSM_STEP",
    "NEXT",
    "DONE",
]

class VNState(TypedDict):
    system_lore: str
    protagonist: str
    player_name: str
    relation_status: str
    sexual_condition_rel: str
    relation_intent: str
    action_state: str
    last_user_question: bool
    last_asst_question: bool
    last_asst_reject: bool
    last_asst_accept: bool
    last_asst_sexual: bool
    action_state_hist: List[str]
    user_idle_streak: int

    turns_target: int
    turn_index: int

    messages: List[Dict[str, str]]
    user_text: str
    assistant_text: str
    raw_assistant: str

    narration_text: str
    dialogue_text: str

    sexual_request: bool
    allow_sexual: bool
    sexual_ready: bool

    signals: Dict[str, int]

    retry_user: int
    retry_assistant: int
    retry_rewrite: int
    retry_quality: int
    retry_sexual: int
    retry_eval: int
    retry_crisis: int

    eval_internal_text: str

    rewrite_mode: str

    error: str


# Graph builder (closure-injected runtime deps)

def build_graph(
    client: OpenAI,
    model_name: str,
    fsm_rel: QwenFSMEngine,
    fsm_act: QwenFSMEngine,
    embed_memory: EmbeddingMemory,
) -> Any:
    """멀티턴 생성 상태 그래프를 구성해 컴파일된 LangGraph 객체를 반환한다.

    이 함수는 런타임 의존성을 클로저로 캡처한 노드 함수들을 정의하고,
    검증 실패/재시도/분기 규칙을 포함한 전체 그래프를 조립한다.

    Args:
        client: OpenAI API 클라이언트.
        model_name: 응답 생성에 사용할 모델 식별자.
        fsm_rel: 관계/국면 전개를 위한 상태 FSM 엔진.
        fsm_act: 행동 전개를 위한 액션 FSM 엔진.
        embed_memory: 의도/수용/거절 유사도 판단용 임베딩 도구.

    Returns:
        compile()가 적용된 실행 가능한 LangGraph 객체.
    """

    # INIT
    rejection_refs = [
        # general
        "지금은 어렵습니다",
        "지금은 곤란합니다",
        "지금은 하지 않겠습니다",
        "지금은 그만 이야기하고 싶습니다",
        "지금은 그 얘기를 하고 싶지 않습니다",
        "지금은 버그 해결이 우선입니다",
        "지금은 업무를 먼저 처리해야 합니다",
        "나중에 이야기하죠",
        "지금은 받아들이기 어렵습니다",
        "지금은 거리를 두고 싶습니다",
        "지금은 그만하고 싶습니다",
        "지금은 준비가 되지 않았습니다",
        "그건 지금 할 수 없습니다",
        "당분간은 어렵겠습니다",
        "지금은 여유가 없습니다",
        "지금은 답하기 어렵습니다",
        "이 주제는 지금 다루고 싶지 않습니다",
        "지금은 시간을 좀 두고 싶습니다",
        "지금은 선을 지키고 싶습니다",
        "지금은 집중해야 할 일이 있습니다",
        "지금은 마음이 복잡합니다",
        "지금은 그럴 여력이 없습니다",
        "그 제안은 받아들이기 어렵습니다",
        "지금은 멈추고 싶습니다",
        "지금은 거절하겠습니다",
        "나중에 다시 이야기합시다",
        # romance
        "지금은 그런 감정을 나누기 어렵습니다",
        "지금은 관계를 더 천천히 하고 싶습니다",
        "지금은 마음을 열기 어렵습니다",
        "지금은 선을 넘고 싶지 않습니다",
        "지금은 거리를 지키고 싶습니다",
        "지금은 그 제안을 받아들일 준비가 아닙니다",
        "지금은 감정을 섣불리 말하고 싶지 않습니다",
        "지금은 기대를 받아들이기 어렵습니다",
        # work/project
        "지금은 작업을 우선해야 합니다",
        "지금은 일정이 촉박합니다",
        "지금은 문제 해결이 먼저입니다",
        "지금은 계획을 재정리해야 합니다",
        "지금은 처리할 일이 많습니다",
        # crisis
        "지금은 상황이 위급합니다",
        "지금은 판단을 미뤄야 합니다",
        "지금은 안전을 우선해야 합니다",
        "지금은 신중하게 멈춰야 합니다",
        "지금은 감당하기 어렵습니다",
    ]
    rejection_vecs = [embed_memory.encode(x) for x in rejection_refs]
    accept_refs = [
        # general
        "좋습니다, 그렇게 하죠",
        "괜찮아요, 진행해요",
        "그렇게 하는 게 좋아요",
        "제안대로 해볼게요",
        "지금 바로 시작해요",
        "함께 해요",
        "그렇게 하고 싶어요",
        "당신을 믿어요",
        "네, 받아들이겠습니다",
        "좋아요, 그렇게 하죠",
        "그 제안을 받아들일게요",
        "그렇게 진행하죠",
        "같이 해봅시다",
        "지금부터 시작해도 좋아요",
        "그 방향으로 가요",
        "그게 좋겠어요",
        "저도 동의해요",
        "알겠어요, 그렇게 할게요",
        "좋아요, 해볼게요",
        "그렇게 해주세요",
        "그 방법이 맞는 것 같아요",
        "네, 믿고 가볼게요",
        "우리 같이 해요",
        "지금부터 해보죠",
        "그 제안에 동의합니다",
        # romance
        "조금 더 가까워져도 괜찮아요",
        "지금은 마음을 나눌 수 있을 것 같아요",
        "당신의 마음을 받아들이겠습니다",
        "지금은 솔직해져도 괜찮겠어요",
        "그 마음을 받아들이고 싶어요",
        "지금은 함께해도 괜찮아요",
        "그 제안이 나쁘지 않아요",
        # work/project
        "그 계획대로 진행해봅시다",
        "지금 바로 착수하죠",
        "해결을 위해 같이 움직이죠",
        "그 방법으로 정리합시다",
        "그 방향이 효율적이에요",
        # crisis
        "지금 당장 실행하겠습니다",
        "지금은 결정을 내릴게요",
        "지금은 함께 돌파하죠",
        "지금은 주저할 시간이 없어요",
        "지금은 진행하는 게 맞아요",
    ]
    accept_vecs = [embed_memory.encode(x) for x in accept_refs]

    def is_rejection(text: str) -> bool:
        """조건 충족 여부를 판정해 불리언 값을 반환한다."""
        if not text:
            return False
        v = embed_memory.encode(text)
        sims = [cosine_sim(v, rv) for rv in rejection_vecs]
        return max(sims) >= 0.70

    def is_acceptance(text: str) -> bool:
        """조건 충족 여부를 판정해 불리언 값을 반환한다."""
        if not text:
            return False
        v = embed_memory.encode(text)
        sims = [cosine_sim(v, rv) for rv in accept_vecs]
        return max(sims) >= 0.70
    def init_node(state: VNState) -> VNState:
        """그래프 실행에 필요한 초기 상태를 구성한다."""
        log_step("INIT", "start")
        protagonist_name = parse_protagonist_name(state["system_lore"])
        player_name = parse_player_name(state["system_lore"])
        system_msgs = build_system_messages(state["system_lore"])

        state["protagonist"] = protagonist_name
        state["player_name"] = player_name
        state["relation_status"] = parse_relation_status(state["system_lore"])
        flags = fsm_rel.get_flags()
        rel_status = flags.get("relation_status")
        if rel_status:
            state["relation_status"] = rel_status
        state["sexual_condition_rel"] = parse_sexual_condition_relation(state["system_lore"])
        state["action_state"] = fsm_act.get_state()
        state["messages"] = [*system_msgs]

        state["user_text"] = ""
        state["assistant_text"] = ""
        state["raw_assistant"] = ""
        state["narration_text"] = ""
        state["dialogue_text"] = ""

        state["sexual_request"] = False
        state["allow_sexual"] = False
        state["sexual_ready"] = False
        state["last_user_question"] = False
        state["last_asst_question"] = False
        state["last_asst_reject"] = False
        state["last_asst_accept"] = False
        state["last_asst_sexual"] = False

        state["signals"] = {
            "mental_instability": 0,
            "intimacy": RELATION_INTIMACY_MAP.get(state["relation_status"], 0),
            "threat": 0,
            "pressure": 0,
            "probe": 0,
            "resolve": 0,
            "event": 0,
        }

        state["turn_index"] = 0
        state["retry_user"] = 0
        state["retry_assistant"] = 0
        state["retry_rewrite"] = 0
        state["retry_quality"] = 0
        state["retry_sexual"] = 0
        state["retry_eval"] = 0
        state["retry_crisis"] = 0
        state["error"] = ""
        log_step("INIT", "done")
        return state

    # GEN_USER
    def gen_user_node(state: VNState) -> VNState:
        """플레이어 턴 텍스트를 생성하고 검증 후 상태에 반영한다."""
        log_step("GEN_USER", "start")

        history = extract_recent_history(state["messages"], max_turns=6)
        last_assistant = extract_last_assistant_output(state["messages"])
        log_step("GEN_USER", f"history_lines={len(history.splitlines())}")
        fsm_state = fsm_rel.get_state()
        action_state = fsm_act.get_state()

        relation_intent = random.choice(["관계 상승", "관계 악화"])
        state["relation_intent"] = relation_intent
        genre = parse_genre(state["system_lore"])
        task = build_user_prompt(
            fsm_state=fsm_state,
            history=history,
            protagonist=state["protagonist"],
            player_name=state["player_name"],
            last_assistant=last_assistant,
            relation_status=state.get("relation_status", ""),
            relation_intent=relation_intent,
            genre=genre,
            action_state=action_state,
            ban_question=bool(state.get("last_user_question", False)),
            ban_silence=bool(state.get("retry_user", 0) > 0),
            force_progress=bool(state.get("user_idle_streak", 0) >= 2),
        )

        raw = generate_text(
            client=client,
            model_name=model_name,
            messages=state["messages"] + [{"role": "user", "content": task}],
            max_new_tokens=780,
            temperature=0.45,
            top_p=0.85,
            repetition_penalty=1.15,
        )
        log_step("GEN_USER", "generated")

        # 기본 정형: 1~2줄 + 마지막 줄은 "..."
        lines = [x for x in strip_line_labels_multiline(raw).splitlines() if x.strip()]
        if not lines:
            state["error"] = "user_generation_empty"
            log_step("GEN_USER", "empty")
            return state
        
         # ---------- 대사 추출 ----------
        q = extract_single_quote(raw)

        # 침묵 / 의미 없는 출력 → retry 증가 없이 재시도
        if not q or q.strip() in ("...", "…"):
            state["retry_user"] += 1
            state["error"] = "user_generation_silent"
            log_step("GEN_USER", "silent")
            log_retry(
                node="GEN_USER",
                retry=state["retry_user"],
                max_retry=30,
                reason=state["error"],
            )
            return state

        speech = f"\"{q}\""
        first = lines[0] if lines else ""
        if not first:
            first = "잠시 망설인다."

        if '"' in first:
            text = speech
        else:
            text = first + "\n" + speech

        text = strip_meta(text)
        text = strip_square_brackets(text)
        text = normalize_user_refs(text, state.get("player_name", "{{user}}"))
        text = normalize_protagonist_refs(text, state.get("protagonist", ""))
        text = strip_line_labels_multiline(text)

        err = user_invalid_reason(
            text, player_name=state.get("player_name", ""), protagonist=state.get("protagonist", "")
        )
        if err:
            state["retry_user"] += 1
            state["error"] = err
            log_step("GEN_USER", f"invalid={err}")
            log_retry(
                node="GEN_USER",
                retry=state["retry_user"],
                max_retry=30,
                reason=err,
            )
            return state
        if not lane_satisfied(fsm_state, text, embed_memory):
            state["retry_user"] += 1
            state["error"] = "user_lane_missing"
            log_step("GEN_USER", "invalid=user_lane_missing")
            log_retry(
                node="GEN_USER",
                retry=state["retry_user"],
                max_retry=30,
                reason=state["error"],
            )
            return state
        prev_users = last_role_texts(state["messages"], "user", limit=2)
        # Semantic repeat checks are conservative; gate on length and history depth.
        if len(prev_users) >= 2 and len(text) >= 40:
            sim_u = rrf_similarity(text, prev_users, embed_memory)
            if sim_u >= 0.085:
                state["retry_user"] += 1
                state["error"] = "user_semantic_repeat"
                log_step("GEN_USER", "invalid=user_semantic_repeat")
                log_retry(
                    node="GEN_USER",
                    retry=state["retry_user"],
                    max_retry=30,
                    reason=state["error"],
                )
                return state
        if (state.get("last_asst_reject", False) or state.get("last_asst_accept", False) or state.get("last_asst_sexual", False)) and len(prev_users) >= 1 and len(text) >= 30:
            sim_r = rrf_similarity(text, prev_users, embed_memory)
            if sim_r >= 0.075:
                state["retry_user"] += 1
                state["error"] = "user_no_strategy_shift"
                log_step("GEN_USER", "invalid=user_no_strategy_shift")
                log_retry(
                    node="GEN_USER",
                    retry=state["retry_user"],
                    max_retry=30,
                    reason=state["error"],
                )
                return state

        # RRF-based progress check (allow short stalls; enforce after 2 idle turns)
        prog_score = rrf_similarity(text, PROGRESS_REFS, embed_memory)
        idle_score = rrf_similarity(text, IDLE_REFS, embed_memory)
        idle_like = (prog_score + 0.005 <= idle_score)
        if idle_like:
            state["user_idle_streak"] = int(state.get("user_idle_streak", 0)) + 1
        else:
            state["user_idle_streak"] = 0
        if state["turn_index"] > 1 and state.get("user_idle_streak", 0) >= 3:
            state["retry_user"] += 1
            state["error"] = "user_no_progress_action"
            log_step("GEN_USER", "invalid=user_no_progress_action")
            log_retry(
                node="GEN_USER",
                retry=state["retry_user"],
                max_retry=30,
                reason=state["error"],
            )
            return state

        # 사용자 텍스트가 직전 문맥과 과도하게 반복되는지 검사한다.
        if state["turn_index"] > 0:
            lines = [x.strip() for x in text.splitlines() if x.strip()]
            narr_u = lines[0] if len(lines) >= 1 else ""
            dia_u = lines[-1] if len(lines) >= 2 else ""
            merged_u = text.strip()

            if narr_u and embed_memory.is_repetitive(narr_u, kind="user_action", threshold=0.92):
                state["retry_user"] += 1
                state["error"] = "user_narr_repeat"
                log_step("GEN_USER", "repeat=user_action")
                return state
            if dia_u not in ("\"...\"", "\"…\""):
                if embed_memory.is_repetitive(dia_u, kind="user_dialogue", threshold=0.86):
                    state["retry_user"] += 1
                    state["error"] = "user_dia_repeat"
                    log_step("GEN_USER", "repeat=user_dialogue")
                    return state
            if embed_memory.is_repetitive(merged_u, kind="user", threshold=0.90):
                state["retry_user"] += 1
                state["error"] = "user_repeat"
                log_step("GEN_USER", "repeat=user")
                return state

        state["user_text"] = text
        state["last_user_question"] = has_question(text)
        state["messages"].append({"role": "user", "content": text})
        state["sexual_request"] = detect_sexual_request(text)

        state["retry_assistant"] = 0
        state["retry_eval"] = 0
        state["retry_crisis"] = 0
        state["error"] = ""
        log_step("GEN_USER", "done")
        return state

    # GEN_USER_SEXUAL
    def gen_user_sexual_node(state: VNState) -> VNState:
        """성적 전개 모드의 플레이어 턴 텍스트를 생성한다."""
        log_step("GEN_USER_SEXUAL", "start")

        history = extract_recent_history(state["messages"], max_turns=6)
        last_assistant = extract_last_assistant_output(state["messages"])
        log_step("GEN_USER_SEXUAL", f"history_lines={len(history.splitlines())}")
        fsm_state = fsm_rel.get_state()
        action_state = fsm_act.get_state()

        relation_intent = random.choice(["관계 상승", "관계 악화"])
        state["relation_intent"] = relation_intent
        genre = parse_genre(state["system_lore"])
        task = build_user_sexual_prompt(
            fsm_state=fsm_state,
            history=history,
            protagonist=state["protagonist"],
            player_name=state["player_name"],
            last_assistant=last_assistant,
            relation_status=state.get("relation_status", ""),
            relation_intent=relation_intent,
            genre=genre,
            action_state=action_state,
            ban_question=bool(state.get("last_user_question", False)),
            ban_silence=bool(state.get("retry_user", 0) > 0),
            force_progress=bool(state.get("user_idle_streak", 0) >= 2),
        )

        raw = generate_text(
            client=client,
            model_name=model_name,
            messages=state["messages"] + [{"role": "user", "content": task}],
            max_new_tokens=780,
            temperature=0.45,
            top_p=0.85,
            repetition_penalty=1.15,
        )
        log_step("GEN_USER_SEXUAL", "generated")

        lines = [x for x in strip_line_labels_multiline(raw).splitlines() if x.strip()]
        if not lines:
            state["error"] = "user_generation_empty"
            log_step("GEN_USER_SEXUAL", "empty")
            return state

        q = extract_single_quote(raw)
        if not q or q.strip() in ("...", "…"):
            state["retry_user"] += 1
            state["error"] = "user_generation_silent"
            log_step("GEN_USER_SEXUAL", "silent")
            log_retry(
                node="GEN_USER_SEXUAL",
                retry=state["retry_user"],
                max_retry=30,
                reason=state["error"],
            )
            return state

        speech = f"\"{q}\""
        first = lines[0] if lines else ""
        if not first:
            first = "잠시 망설인다."

        if '"' in first:
            text = speech
        else:
            text = first + "\n" + speech

        text = strip_meta(text)
        text = strip_square_brackets(text)
        text = normalize_user_refs(text, state.get("player_name", "{{user}}"))
        text = normalize_protagonist_refs(text, state.get("protagonist", ""))
        text = strip_line_labels_multiline(text)

        err = user_invalid_reason(
            text, player_name=state.get("player_name", ""), protagonist=state.get("protagonist", "")
        )
        if err:
            state["retry_user"] += 1
            state["error"] = err
            log_step("GEN_USER_SEXUAL", f"invalid={err}")
            log_retry(
                node="GEN_USER_SEXUAL",
                retry=state["retry_user"],
                max_retry=30,
                reason=err,
            )
            return state
        if not lane_satisfied(fsm_state, text, embed_memory):
            state["retry_user"] += 1
            state["error"] = "user_lane_missing"
            log_step("GEN_USER_SEXUAL", "invalid=user_lane_missing")
            log_retry(
                node="GEN_USER_SEXUAL",
                retry=state["retry_user"],
                max_retry=30,
                reason=state["error"],
            )
            return state
        prev_users = last_role_texts(state["messages"], "user", limit=2)
        if len(prev_users) >= 2 and len(text) >= 40:
            sim_u = rrf_similarity(text, prev_users, embed_memory)
            if sim_u >= 0.085:
                state["retry_user"] += 1
                state["error"] = "user_semantic_repeat"
                log_step("GEN_USER_SEXUAL", "invalid=user_semantic_repeat")
                log_retry(
                    node="GEN_USER_SEXUAL",
                    retry=state["retry_user"],
                    max_retry=30,
                    reason=state["error"],
                )
                return state
        if (state.get("last_asst_reject", False) or state.get("last_asst_accept", False) or state.get("last_asst_sexual", False)) and len(prev_users) >= 1 and len(text) >= 30:
            sim_r = rrf_similarity(text, prev_users, embed_memory)
            if sim_r >= 0.075:
                state["retry_user"] += 1
                state["error"] = "user_no_strategy_shift"
                log_step("GEN_USER_SEXUAL", "invalid=user_no_strategy_shift")
                log_retry(
                    node="GEN_USER_SEXUAL",
                    retry=state["retry_user"],
                    max_retry=30,
                    reason=state["error"],
                )
                return state

        # RRF-based progress check (allow short stalls; enforce after 2 idle turns)
        prog_score = rrf_similarity(text, PROGRESS_REFS, embed_memory)
        idle_score = rrf_similarity(text, IDLE_REFS, embed_memory)
        idle_like = (prog_score + 0.005 <= idle_score)
        if idle_like:
            state["user_idle_streak"] = int(state.get("user_idle_streak", 0)) + 1
        else:
            state["user_idle_streak"] = 0
        if state["turn_index"] > 1 and state.get("user_idle_streak", 0) >= 3:
            state["retry_user"] += 1
            state["error"] = "user_no_progress_action"
            log_step("GEN_USER_SEXUAL", "invalid=user_no_progress_action")
            log_retry(
                node="GEN_USER_SEXUAL",
                retry=state["retry_user"],
                max_retry=30,
                reason=state["error"],
            )
            return state

        if state["turn_index"] > 0:
            lines = [x.strip() for x in text.splitlines() if x.strip()]
            narr_u = lines[0] if len(lines) >= 1 else ""
            dia_u = lines[-1] if len(lines) >= 2 else ""
            merged_u = text.strip()

            if narr_u and embed_memory.is_repetitive(narr_u, kind="user_action", threshold=0.92):
                state["retry_user"] += 1
                state["error"] = "user_narr_repeat"
                log_step("GEN_USER_SEXUAL", "repeat=user_action")
                return state
            if dia_u not in ("\"...\"", "\"…\""):
                if embed_memory.is_repetitive(dia_u, kind="user_dialogue", threshold=0.86):
                    state["retry_user"] += 1
                    state["error"] = "user_dia_repeat"
                    log_step("GEN_USER_SEXUAL", "repeat=user_dialogue")
                    return state
            if embed_memory.is_repetitive(merged_u, kind="user", threshold=0.90):
                state["retry_user"] += 1
                state["error"] = "user_repeat"
                log_step("GEN_USER_SEXUAL", "repeat=user")
                return state

        state["user_text"] = text
        state["last_user_question"] = has_question(text)
        state["messages"].append({"role": "user", "content": text})
        state["sexual_request"] = detect_sexual_request(text)

        state["retry_assistant"] = 0
        state["retry_eval"] = 0
        state["retry_crisis"] = 0
        state["error"] = ""
        log_step("GEN_USER_SEXUAL", "done")
        return state

    # DETECT (flags)
    def detect_node(state: VNState) -> VNState:
        """현재 텍스트에서 조건 또는 요청 신호를 감지한다."""
        log_step("DETECT", "start")
        flags = fsm_rel.get_flags() or {}
        allow_flag = flags.get("allow_sexual", False)
        state["allow_sexual"] = resolve_allow_sexual(allow_flag, state["system_lore"])
        state["relation_status"] = parse_relation_status(state["system_lore"])
        rel_status = flags.get("relation_status")
        if rel_status:
            state["relation_status"] = rel_status
        state["sexual_condition_rel"] = parse_sexual_condition_relation(state["system_lore"])
        prev_action_state = state.get("action_state", "")
        # During SEXUAL/AFTERMATH, keep previous action_state to avoid dropping the sequence
        if prev_action_state not in SEXUAL_STATES and prev_action_state not in AFTERMATH_SEX_STATES:
            state["action_state"] = fsm_act.get_state()
        state["sexual_ready"] = (
            bool(state["allow_sexual"])
            and bool(state["sexual_condition_rel"])
            and bool(state["relation_status"])
            and (state["sexual_condition_rel"] == state["relation_status"])
        )
        # Lock sexual readiness while in sexual sequence
        if prev_action_state in SEXUAL_STATES:
            state["sexual_ready"] = True
        log_step("DETECT", "done")
        return state

    # GEN_ASSISTANT (Integrated: narration+dialogue)
    def gen_assistant_node(state: VNState) -> VNState:
        """주인공 통합 응답을 생성하고 형식 검증을 수행한다."""
        log_step("GEN_ASSISTANT", "start")

        history = extract_recent_history(state["messages"], max_turns=6)
        log_step("GEN_ASSISTANT", f"history_lines={len(history.splitlines())}")
        fsm_state = fsm_rel.get_state()
        action_state = fsm_act.get_state()
        flags = fsm_rel.get_flags() or {}

        task = build_assistant_turn_prompt(
            system_lore=state["system_lore"],
            protagonist=state["protagonist"],
            fsm_state=fsm_state,
            flags=flags,
            signals=state["signals"],
            relation_status=state.get("relation_status", ""),
            genre=parse_genre(state["system_lore"]),
            action_state=action_state,
            history=history,
            user_text=state["user_text"],
            sexual_request=bool(state["sexual_request"]),
            player_name=state["player_name"],
            force_progress=bool(state.get("user_idle_streak", 0) >= 2),
        )

        raw = generate_text(
            client=client,
            model_name=model_name,
            messages=state["messages"] + [{"role": "user", "content": task}],
            max_new_tokens=780,
            temperature=0.55,
            top_p=0.85,
            repetition_penalty=1.12,
        )
        state["raw_assistant"] = raw
        log_step("GEN_ASSISTANT", "generated")

        # 통합 응답은 서술 1줄 + 대사 1줄의 2줄 형식을 강제한다.
        lines = [x for x in strip_line_labels_multiline(raw).splitlines() if x.strip()]
        # assistant 생성 실패 → retry 증가 없이 재시도
        if not lines:
            state["error"] = "assistant_generation_empty"
            log_step("GEN_ASSISTANT", "empty")
            return state
        if len(lines) >= 2:
            narr_line = remove_all_quotes(lines[0]).strip()
            text = (narr_line + "\n" + force_single_quote_line(lines[1])).strip()
        elif len(lines) == 1:
            narr_line = remove_all_quotes(lines[0]).strip()
            text = (narr_line + "\n" + "\"...\"").strip()
        else:
            text = "숨이 짧게 새어 나왔다.\n\"...\""

        text = strip_meta(text)
        text = strip_square_brackets(text)
        text = normalize_user_refs(text, state.get("player_name", "{{user}}"))
        text = normalize_protagonist_refs(text, state.get("protagonist", ""))
        text = strip_line_labels_multiline(text)

        if normalize_space(text) == normalize_space(state.get("user_text", "")):
            state["retry_assistant"] += 1
            state["error"] = "asst_echo_user"
            log_step("GEN_ASSISTANT", "invalid=asst_echo_user")
            log_retry(
                node="GEN_ASSISTANT",
                retry=state["retry_assistant"],
                max_retry=30,
                reason=state["error"],
            )
            return state

        narr_line = ""
        _lines = [x.strip() for x in text.splitlines() if x.strip()]
        if _lines:
            narr_line = _lines[0]
        if is_player_subject(narr_line, state.get("player_name", "")):
            state["retry_assistant"] += 1
            state["error"] = "asst_role_conflict"
            log_step("GEN_ASSISTANT", "invalid=asst_role_conflict")
            log_retry(
                node="GEN_ASSISTANT",
                retry=state["retry_assistant"],
                max_retry=30,
                reason=state["error"],
            )
            return state

        err = assistant_invalid_reason_integrated(
            text,
            protagonist=state.get("protagonist", ""),
            player_name=state.get("player_name", ""),
            prefer_formal=parse_protagonist_speech_formal(state.get("system_lore", "")),
        )
        if err:
            state["retry_assistant"] += 1
            state["error"] = err
            if err in ("asst_meta", "asst_dia_style_mismatch_llm"):
                state["rewrite_mode"] = "normal"
            log_step("GEN_ASSISTANT", f"invalid={err}")
            log_retry(
                node="GEN_ASSISTANT",
                retry=state["retry_assistant"],
                max_retry=30,
                reason=err,
            )
            return state
        if not lane_satisfied(fsm_state, text, embed_memory):
            state["retry_assistant"] += 1
            state["error"] = "asst_lane_missing"
            log_step("GEN_ASSISTANT", "invalid=asst_lane_missing")
            log_retry(
                node="GEN_ASSISTANT",
                retry=state["retry_assistant"],
                max_retry=30,
                reason=state["error"],
            )
            return state
        prev_assts = last_role_texts(state["messages"], "assistant", limit=2)
        if prev_assts:
            sim_a = rrf_similarity(text, prev_assts, embed_memory)
            if sim_a >= 0.045:
                state["retry_assistant"] += 1
                state["error"] = "asst_semantic_repeat"
                log_step("GEN_ASSISTANT", "invalid=asst_semantic_repeat")
                log_retry(
                    node="GEN_ASSISTANT",
                    retry=state["retry_assistant"],
                    max_retry=30,
                    reason=state["error"],
                )
                return state

        narr, dia, merged = split_integrated_assistant(text)
        if llm_speech_style_mismatch_openai(
            client=client,
            model_name=model_name,
            dialogue_text=remove_all_quotes(dia),
            prefer_formal=parse_protagonist_speech_formal(state.get("system_lore", "")),
        ):
            state["retry_assistant"] += 1
            state["error"] = "asst_dia_style_mismatch_llm"
            state["rewrite_mode"] = "normal"
            log_step("GEN_ASSISTANT", "invalid=asst_dia_style_mismatch_llm route=rewrite")
            log_retry(
                node="GEN_ASSISTANT",
                retry=state["retry_assistant"],
                max_retry=30,
                reason=state["error"],
            )
            return state

        # MOD(중요): kind별 repetition 검사, keyword-only 준수
        if state["turn_index"] > 0:
            if embed_memory.is_repetitive(narr, kind="narration", threshold=0.92):
                state["retry_assistant"] += 1
                state["error"] = "narr_repeat"
                log_step("GEN_ASSISTANT", "repeat=narration")
                return state
            if embed_memory.is_repetitive(dia, kind="dialogue", threshold=0.86):
                state["error"] = "dia_repeat"
                log_step("GEN_ASSISTANT", "repeat=dialogue")
                return state
            if embed_memory.is_repetitive(merged, kind="assistant", threshold=0.90):
                state["error"] = "asst_repeat"
                log_step("GEN_ASSISTANT", "repeat=assistant")
                return state

        state["narration_text"] = narr
        state["dialogue_text"] = dia
        state["assistant_text"] = merged
        state["last_asst_question"] = has_question(dia)
        state["last_asst_reject"] = is_rejection(dia)
        state["last_asst_accept"] = is_acceptance(dia)
        state["last_asst_sexual"] = state.get("action_state", "").startswith("SEXUAL_")
        state["error"] = ""
        log_step("GEN_ASSISTANT", "done")
        return state

    # GEN_ASSISTANT_REWRITE
    def gen_assistant_rewrite_node(state: VNState) -> VNState:
        """주인공 응답을 재작성해 형식과 품질을 보정한다."""
        log_step("GEN_ASSISTANT_REWRITE", "start")

        raw_text = (state.get("raw_assistant") or state.get("assistant_text") or "").strip()
        if not raw_text:
            state["error"] = "rewrite_empty"
            log_step("GEN_ASSISTANT_REWRITE", "empty")
            return state

        rewrite_mode = (state.get("rewrite_mode") or "normal").strip().lower()
        if rewrite_mode == "sexual":
            task = build_assistant_rewrite_sexual_prompt(
                system_lore=state["system_lore"],
                protagonist=state["protagonist"],
                player_name=state["player_name"],
                raw_text=raw_text,
                action_state=state.get("action_state", ""),
                genre=parse_genre(state["system_lore"]),
            )
        else:
            task = build_assistant_rewrite_prompt(
                system_lore=state["system_lore"],
                protagonist=state["protagonist"],
                player_name=state["player_name"],
                raw_text=raw_text,
                action_state=state.get("action_state", ""),
                genre=parse_genre(state["system_lore"]),
            )

        raw = generate_text(
            client=client,
            model_name=model_name,
            messages=state["messages"] + [{"role": "user", "content": task}],
            max_new_tokens=780,
            temperature=0.40,
            top_p=0.80,
            repetition_penalty=1.08,
        )
        log_step("GEN_ASSISTANT_REWRITE", "generated")

        lines = [x for x in strip_line_labels_multiline(raw).splitlines() if x.strip()]
        if not lines:
            state["retry_rewrite"] += 1
            state["error"] = "rewrite_empty"
            log_step("GEN_ASSISTANT_REWRITE", "invalid=rewrite_empty")
            return state
        if len(lines) >= 2:
            narr_line = remove_all_quotes(lines[0]).strip()
            text = (narr_line + "\n" + force_single_quote_line(lines[1])).strip()
        elif len(lines) == 1:
            narr_line = remove_all_quotes(lines[0]).strip()
            text = (narr_line + "\n" + "\"...\"").strip()
        else:
            text = "숨이 짧게 새어 나왔다.\n\"...\""

        text = strip_meta(text)
        text = strip_square_brackets(text)
        text = normalize_user_refs(text, state.get("player_name", "{{user}}"))
        text = normalize_protagonist_refs(text, state.get("protagonist", ""))
        text = strip_line_labels_multiline(text)

        err = assistant_invalid_reason_integrated(
            text,
            protagonist=state.get("protagonist", ""),
            player_name=state.get("player_name", ""),
            prefer_formal=parse_protagonist_speech_formal(state.get("system_lore", "")),
        )
        if err:
            state["retry_rewrite"] += 1
            state["error"] = err
            log_step("GEN_ASSISTANT_REWRITE", f"invalid={err}")
            return state

        narr, dia, merged = split_integrated_assistant(text)
        state["narration_text"] = narr
        state["dialogue_text"] = dia
        state["assistant_text"] = merged
        state["last_asst_question"] = has_question(dia)
        state["last_asst_reject"] = is_rejection(dia)
        state["last_asst_accept"] = is_acceptance(dia)
        state["last_asst_sexual"] = state.get("action_state", "").startswith("SEXUAL_")
        state["rewrite_mode"] = ""
        state["error"] = ""
        log_step("GEN_ASSISTANT_REWRITE", "done")
        return state

    # EVAL_DIALOGUE_QUALITY
    def eval_dialogue_quality_node(state: VNState) -> VNState:
        """현재 턴 출력을 평가해 점검 결과를 반영한다."""
        log_step("EVAL_DIALOGUE_QUALITY", "start")
        text = (state.get("assistant_text") or "").strip()
        if not text:
            state["error"] = ""
            log_step("EVAL_DIALOGUE_QUALITY", "skip_empty")
            return state

        failed = llm_dialogue_quality_fail_openai(
            client=client,
            model_name=model_name,
            system_lore=state.get("system_lore", ""),
            relation_status=state.get("relation_status", ""),
            action_state=state.get("action_state", ""),
            player_name=state.get("player_name", ""),
            user_text=state.get("user_text", ""),
            assistant_text=text,
        )
        if failed:
            state["retry_quality"] += 1
            state["error"] = "asst_dialogue_quality_llm"
            state["rewrite_mode"] = "sexual" if state.get("action_state", "").startswith("SEXUAL_") else "normal"
            log_step("EVAL_DIALOGUE_QUALITY", "invalid=asst_dialogue_quality_llm route=rewrite_quality")
            return state

        state["error"] = ""
        log_step("EVAL_DIALOGUE_QUALITY", "done")
        return state

    # GEN_ASSISTANT_REWRITE_QUALITY
    def gen_assistant_rewrite_quality_node(state: VNState) -> VNState:
        """품질 기준 위반 응답을 재작성한다."""
        log_step("GEN_ASSISTANT_REWRITE_QUALITY", "start")

        raw_text = (state.get("assistant_text") or state.get("raw_assistant") or "").strip()
        if not raw_text:
            state["error"] = "rewrite_quality_empty"
            log_step("GEN_ASSISTANT_REWRITE_QUALITY", "empty")
            return state

        prefer_formal = parse_protagonist_speech_formal(state.get("system_lore", ""))
        rewrite_mode = (state.get("rewrite_mode") or "normal").strip().lower()
        sexual_mode = rewrite_mode == "sexual" or state.get("action_state", "").startswith("SEXUAL_")

        task = build_assistant_rewrite_quality_prompt(
            system_lore=state["system_lore"],
            protagonist=state["protagonist"],
            player_name=state["player_name"],
            raw_text=raw_text,
            action_state=state.get("action_state", ""),
            genre=parse_genre(state["system_lore"]),
            prefer_formal=prefer_formal,
            sexual_mode=sexual_mode,
        )

        raw = generate_text(
            client=client,
            model_name=model_name,
            messages=state["messages"] + [{"role": "user", "content": task}],
            max_new_tokens=780,
            temperature=0.40,
            top_p=0.80,
            repetition_penalty=1.08,
        )
        log_step("GEN_ASSISTANT_REWRITE_QUALITY", "generated")

        lines = [x for x in strip_line_labels_multiline(raw).splitlines() if x.strip()]
        if not lines:
            state["retry_quality"] += 1
            state["error"] = "rewrite_quality_empty"
            log_step("GEN_ASSISTANT_REWRITE_QUALITY", "invalid=rewrite_quality_empty")
            return state
        if len(lines) >= 2:
            narr_line = remove_all_quotes(lines[0]).strip()
            text = (narr_line + "\n" + force_single_quote_line(lines[1])).strip()
        elif len(lines) == 1:
            narr_line = remove_all_quotes(lines[0]).strip()
            text = (narr_line + "\n" + "\"...\"").strip()
        else:
            text = "숨이 짧게 새어 나왔다.\n\"...\""

        text = strip_meta(text)
        text = strip_square_brackets(text)
        text = normalize_user_refs(text, state.get("player_name", "{{user}}"))
        text = normalize_protagonist_refs(text, state.get("protagonist", ""))
        text = strip_line_labels_multiline(text)

        err = assistant_invalid_reason_integrated(
            text,
            protagonist=state.get("protagonist", ""),
            player_name=state.get("player_name", ""),
            prefer_formal=prefer_formal,
        )
        if err:
            state["retry_quality"] += 1
            state["error"] = err
            log_step("GEN_ASSISTANT_REWRITE_QUALITY", f"invalid={err}")
            return state

        narr, dia, merged = split_integrated_assistant(text)
        state["narration_text"] = narr
        state["dialogue_text"] = dia
        state["assistant_text"] = merged
        state["last_asst_question"] = has_question(dia)
        state["last_asst_reject"] = is_rejection(dia)
        state["last_asst_accept"] = is_acceptance(dia)
        state["last_asst_sexual"] = state.get("action_state", "").startswith("SEXUAL_")
        state["rewrite_mode"] = ""
        state["error"] = ""
        log_step("GEN_ASSISTANT_REWRITE_QUALITY", "done")
        return state

    # GEN_CRISIS
    def gen_crisis_node(state: VNState) -> VNState:
        """위기 국면 전용 응답을 생성한다."""
        log_step("GEN_CRISIS", "start")
        state["retry_crisis"] += 1

        history = extract_recent_history(state["messages"], max_turns=6)
        log_step("GEN_CRISIS", f"history_lines={len(history.splitlines())}")

        task = build_crisis_prompt(
            system_lore=state["system_lore"],
            protagonist=state["protagonist"],
            history=history,
            user_text=state["user_text"],
            player_name=state["player_name"],
            force_progress=bool(state.get("user_idle_streak", 0) >= 2),
            genre=parse_genre(state["system_lore"]),
            action_state=state.get("action_state", ""),
        )

        raw = generate_text(
            client=client,
            model_name=model_name,
            messages=state["messages"] + [{"role": "user", "content": task}],
            max_new_tokens=780,
            temperature=0.60,
            top_p=0.85,
            repetition_penalty=1.06,
        )
        text = raw.strip()
        log_step("GEN_CRISIS", "generated")

        text = strip_meta(text)
        text = strip_square_brackets(text)
        text = normalize_user_refs(text, state.get("player_name", "{{user}}"))
        text = normalize_protagonist_refs(text, state.get("protagonist", ""))
        text = strip_line_labels_multiline(text)
        err = crisis_invalid_reason(text)
        if err:
            state["error"] = err
            log_step("GEN_CRISIS", f"invalid={err}")
            log_retry(
                node="GEN_CRISIS",
                retry=state["retry_crisis"],
                max_retry=30,
                reason=err,
            )
            return state

        # CRISIS 응답도 assistant 텍스트 반복 방지 규칙을 동일하게 적용한다.
        if state["turn_index"] > 0:
            if embed_memory.is_repetitive(text, kind="assistant", threshold=0.90):
                state["error"] = "crisis_repeat"
                log_step("GEN_CRISIS", "repeat=assistant")
                return state

        state["assistant_text"] = text
        # CRISIS는 narration/dialogue 분리 안함(자유형)
        state["narration_text"] = ""
        state["dialogue_text"] = ""
        state["error"] = ""
        log_step("GEN_CRISIS", "done")
        return state

    # GEN_SEXUAL
    def gen_sexual_node(state: VNState) -> VNState:
        """성적 국면 전용 응답을 생성한다."""
        log_step("GEN_SEXUAL", "start")
        history = extract_recent_history(state["messages"], max_turns=6)
        log_step("GEN_SEXUAL", f"history_lines={len(history.splitlines())}")
        fsm_state = fsm_rel.get_state()

        task = build_sexual_prompt(
            system_lore=state["system_lore"],
            protagonist=state["protagonist"],
            history=history,
            user_text=state["user_text"],
            player_name=state["player_name"],
            force_progress=bool(state.get("user_idle_streak", 0) >= 2),
            genre=parse_genre(state["system_lore"]),
            action_state=state.get("action_state", ""),
        )
        style_rule = dialogue_style_rule_text(
            parse_protagonist_speech_formal(state.get("system_lore", ""))
        )
        if style_rule:
            task = task + "\n\n[말투 규칙]\n- " + style_rule

        raw = generate_text(
            client=client,
            model_name=model_name,
            messages=state["messages"] + [{"role": "user", "content": task}],
            max_new_tokens=780,
            temperature=0.60,
            top_p=0.85,
            repetition_penalty=1.08,
        )
        lines = [x for x in strip_line_labels_multiline(raw).splitlines() if x.strip()]
        if not lines:
            state["error"] = "assistant_generation_empty"
            log_step("GEN_SEXUAL", "empty")
            return state
        if len(lines) >= 2:
            narr_line = remove_all_quotes(lines[0]).strip()
            text = (narr_line + "\n" + force_single_quote_line(lines[1])).strip()
        elif len(lines) == 1:
            narr_line = remove_all_quotes(lines[0]).strip()
            text = (narr_line + "\n" + "\"...\"").strip()
        else:
            text = "숨이 짧게 새어 나왔다.\n\"...\""
        log_step("GEN_SEXUAL", "generated")

        text = strip_meta(text)
        text = strip_square_brackets(text)
        text = normalize_user_refs(text, state.get("player_name", "{{user}}"))
        text = normalize_protagonist_refs(text, state.get("protagonist", ""))
        text = strip_line_labels_multiline(text)

        narr_line = ""
        _lines = [x.strip() for x in text.splitlines() if x.strip()]
        if _lines:
            narr_line = _lines[0]
        if is_player_subject(narr_line, state.get("player_name", "")):
            state["retry_sexual"] += 1
            state["error"] = "asst_role_conflict"
            log_step("GEN_SEXUAL", "invalid=asst_role_conflict")
            log_retry(
                node="GEN_SEXUAL",
                retry=state["retry_sexual"],
                max_retry=30,
                reason=state["error"],
            )
            return state

        err = assistant_invalid_reason_integrated(
            text,
            protagonist=state.get("protagonist", ""),
            player_name=state.get("player_name", ""),
            prefer_formal=parse_protagonist_speech_formal(state.get("system_lore", "")),
        )
        if err:
            state["error"] = err
            log_step("GEN_SEXUAL", f"invalid={err}")
            log_retry(
                node="GEN_SEXUAL",
                retry=state["retry_sexual"],
                max_retry=30,
                reason=err,
            )
            return state
        if not lane_satisfied(fsm_state, text, embed_memory):
            state["retry_sexual"] += 1
            state["error"] = "asst_lane_missing"
            log_step("GEN_SEXUAL", "invalid=asst_lane_missing")
            log_retry(
                node="GEN_SEXUAL",
                retry=state["retry_sexual"],
                max_retry=30,
                reason=state["error"],
            )
            return state
        prev_assts = last_role_texts(state["messages"], "assistant", limit=2)
        if prev_assts:
            sim_a = rrf_similarity(text, prev_assts, embed_memory)
            if sim_a >= 0.045:
                state["retry_sexual"] += 1
                state["error"] = "asst_semantic_repeat"
                log_step("GEN_SEXUAL", "invalid=asst_semantic_repeat")
                log_retry(
                    node="GEN_SEXUAL",
                    retry=state["retry_sexual"],
                    max_retry=30,
                    reason=state["error"],
                )
                return state

        narr, dia, merged = split_integrated_assistant(text)
        if llm_speech_style_mismatch_openai(
            client=client,
            model_name=model_name,
            dialogue_text=remove_all_quotes(dia),
            prefer_formal=parse_protagonist_speech_formal(state.get("system_lore", "")),
        ):
            state["retry_sexual"] += 1
            state["error"] = "asst_dia_style_mismatch_llm"
            state["rewrite_mode"] = "sexual"
            log_step("GEN_SEXUAL", "invalid=asst_dia_style_mismatch_llm")
            log_retry(
                node="GEN_SEXUAL",
                retry=state["retry_sexual"],
                max_retry=30,
                reason=state["error"],
            )
            return state

        if state["turn_index"] > 0:
            if embed_memory.is_repetitive(narr, kind="narration", threshold=0.92):
                state["error"] = "narr_repeat"
                log_step("GEN_SEXUAL", "repeat=narration")
                return state
            if embed_memory.is_repetitive(dia, kind="dialogue", threshold=0.86):
                state["error"] = "dia_repeat"
                log_step("GEN_SEXUAL", "repeat=dialogue")
                return state
            if embed_memory.is_repetitive(merged, kind="assistant", threshold=0.90):
                state["error"] = "asst_repeat"
                log_step("GEN_SEXUAL", "repeat=assistant")
                return state

        state["narration_text"] = narr
        state["dialogue_text"] = dia
        state["assistant_text"] = merged
        state["last_asst_question"] = has_question(dia)
        state["last_asst_reject"] = is_rejection(dia)
        state["last_asst_accept"] = is_acceptance(dia)
        state["last_asst_sexual"] = state.get("action_state", "").startswith("SEXUAL_")
        state["error"] = ""
        log_step("GEN_SEXUAL", "done")
        return state

    # EVAL_INTERNAL (analysis text only)
    def eval_internal_node(state: VNState) -> VNState:
        """현재 턴 출력을 평가해 점검 결과를 반영한다."""
        log_step("EVAL_INTERNAL", "start")

        # Ensure assistant turn is present for eval
        if (
            (not state["messages"])
            or (state["messages"][-1]["role"] != "assistant")
            or (state["messages"][-1]["content"] != state["assistant_text"])
        ):
            state["messages"].append({"role": "assistant", "content": state["assistant_text"]})

        task = build_eval_internal_prompt(
            system_lore=state["system_lore"],
            protagonist=state["protagonist"],
            fsm_state=fsm_rel.get_state(),
            prev_signals=state["signals"],
            user_text=state["user_text"],
            assistant_text=state["assistant_text"],
            player_name=state["player_name"],
            relation_intent=state.get("relation_intent", ""),
        )

        eval_messages = [
            {"role": "system", "content": "You are an evaluator. Provide brief analysis only."},
            {"role": "user", "content": task},
        ]

        analysis = generate_text(
            client=client,
            model_name=model_name,
            messages=eval_messages,
            max_new_tokens=1400,
            temperature=0.3,
            top_p=1.0,
            repetition_penalty=1.0,
            do_sample=True,
        )

        state["eval_internal_text"] = (analysis or "").strip()
        if state["eval_internal_text"]:
            print(f"[EVAL_INTERNAL] {state['eval_internal_text']}", flush=True)
        log_step("EVAL_INTERNAL", "done")
        return state

    # EVAL_JSON (JSON only)
    def eval_json_node(state: VNState) -> VNState:
        """현재 턴 출력을 평가해 점검 결과를 반영한다."""
        log_step("EVAL_JSON", "start")
        state["retry_eval"] += 1

        task = build_eval_prompt(
            system_lore=state["system_lore"],
            protagonist=state["protagonist"],
            fsm_state=fsm_rel.get_state(),
            prev_signals=state["signals"],
            user_text=state["user_text"],
            assistant_text=state["assistant_text"],
            player_name=state["player_name"],
            relation_intent=state.get("relation_intent", ""),
            analysis_text=state.get("eval_internal_text", ""),
        )

        eval_messages = [
            {"role": "system", "content": "You are a JSON-only evaluator."},
            {"role": "user", "content": task},
        ]

        obj = generate_json(
            client=client,
            model_name=model_name,
            messages=eval_messages,
            max_new_tokens=1400,
            temperature=0.1,
            top_p=1.0,
            do_sample=False,
            log_raw=True,
        )

        if obj is None:
            state["error"] = "eval_json_parse_fail"
            log_step("EVAL_JSON", "json_parse_fail")
            log_retry(
                node="EVAL_JSON",
                retry=state["retry_eval"],
                max_retry=40,
                reason="json_parse_fail",
            )
            return state

        new_signals = {
            "mental_instability": clamp_int(obj.get("mental_instability"), 0, 3, state["signals"].get("mental_instability", 0)),
            "intimacy": clamp_int(obj.get("intimacy"), 0, 3, state["signals"].get("intimacy", 0)),
            "threat": clamp_int(obj.get("threat"), 0, 2, state["signals"].get("threat", 0)),
            "pressure": clamp_int(obj.get("pressure"), 0, 3, state["signals"].get("pressure", 0)),
            "probe": clamp_int(obj.get("probe"), 0, 3, state["signals"].get("probe", 0)),
            "resolve": clamp_int(obj.get("resolve"), 0, 1, state["signals"].get("resolve", 0)),
            "event": clamp_int(obj.get("event"), 0, 3, state["signals"].get("event", 0)),
        }

        # Probabilistic drift on intimacy only (60/30/10) based on relation intent
        # - 관계 상승: 60% 상승 / 30% 유지 / 10% 하락
        # - 관계 악화: 60% 하락 / 30% 유지 / 10% 상승
        rel_intent = state.get("relation_intent", "")
        prev = state["signals"]
        prev_intimacy = prev.get("intimacy", 0)
        r = random.random()

        if rel_intent == "관계 상승":
            if r < 0.60:
                new_signals["intimacy"] = min(prev_intimacy + 1, 3)
            elif r < 0.90:
                new_signals["intimacy"] = prev_intimacy
            else:
                new_signals["intimacy"] = max(prev_intimacy - 1, 0)
        elif rel_intent == "관계 악화":
            if r < 0.60:
                new_signals["intimacy"] = max(prev_intimacy - 1, 0)
            elif r < 0.90:
                new_signals["intimacy"] = prev_intimacy
            else:
                new_signals["intimacy"] = min(prev_intimacy + 1, 3)

        # Genre-based probabilistic drift (light bias) for other stats
        genre = parse_genre(state["system_lore"])
        genre_weights = {
            "연애물": {
                "mental_instability": (0.15, 0.55, 0.30),
                "threat": (0.15, 0.55, 0.30),
                "pressure": (0.20, 0.50, 0.30),
                "probe": (0.25, 0.50, 0.25),
                "resolve": (0.20, 0.50, 0.30),
                "event": (0.35, 0.45, 0.20),
            },
            "육성물": {
                "mental_instability": (0.20, 0.55, 0.25),
                "threat": (0.15, 0.60, 0.25),
                "pressure": (0.35, 0.45, 0.20),
                "probe": (0.35, 0.45, 0.20),
                "resolve": (0.45, 0.40, 0.15),
                "event": (0.45, 0.40, 0.15),
            },
            "성장물": {
                "mental_instability": (0.30, 0.45, 0.25),
                "threat": (0.25, 0.50, 0.25),
                "pressure": (0.35, 0.45, 0.20),
                "probe": (0.25, 0.50, 0.25),
                "resolve": (0.40, 0.40, 0.20),
                "event": (0.50, 0.35, 0.15),
            },
            "비극": {
                "mental_instability": (0.50, 0.30, 0.20),
                "threat": (0.40, 0.35, 0.25),
                "pressure": (0.40, 0.35, 0.25),
                "probe": (0.20, 0.50, 0.30),
                "resolve": (0.15, 0.40, 0.45),
                "event": (0.45, 0.40, 0.15),
            },
            "심리 시뮬레이션": {
                "mental_instability": (0.45, 0.35, 0.20),
                "threat": (0.30, 0.45, 0.25),
                "pressure": (0.40, 0.40, 0.20),
                "probe": (0.40, 0.40, 0.20),
                "resolve": (0.25, 0.45, 0.30),
                "event": (0.30, 0.50, 0.20),
            },
        }

        def _drift_stat(key: str, lo: int, hi: int, probs: tuple[float, float, float]) -> None:
            """내부 보조 로직을 수행한다."""
            up, same, down = probs
            v = clamp_int(new_signals.get(key), lo, hi, prev.get(key, 0))
            r2 = random.random()
            if r2 < up:
                v = min(v + 1, hi)
            elif r2 < up + same:
                v = v
            else:
                v = max(v - 1, lo)
            new_signals[key] = v

        weights = genre_weights.get(genre)
        if weights:
            _drift_stat("mental_instability", 0, 3, weights["mental_instability"])
            _drift_stat("threat", 0, 2, weights["threat"])
            _drift_stat("pressure", 0, 3, weights["pressure"])
            _drift_stat("probe", 0, 3, weights["probe"])
            _drift_stat("resolve", 0, 1, weights["resolve"])
            _drift_stat("event", 0, 3, weights["event"])

        # Limit per-step jumps to ±1 for all signals (after drifts)
        def _limit_delta(key: str, lo: int, hi: int) -> int:
            """내부 보조 로직을 수행한다."""
            v = clamp_int(new_signals.get(key), lo, hi, prev.get(key, 0))
            p = prev.get(key, 0)
            if v > p + 1:
                return p + 1
            if v < p - 1:
                return p - 1
            return v

        new_signals["mental_instability"] = _limit_delta("mental_instability", 0, 3)
        new_signals["intimacy"] = _limit_delta("intimacy", 0, 3)
        new_signals["threat"] = _limit_delta("threat", 0, 2)
        new_signals["pressure"] = _limit_delta("pressure", 0, 3)
        new_signals["probe"] = _limit_delta("probe", 0, 3)
        new_signals["resolve"] = _limit_delta("resolve", 0, 1)
        new_signals["event"] = _limit_delta("event", 0, 3)

        state["signals"] = new_signals
        state["error"] = ""
        log_step("EVAL_JSON", "done")
        return state

    # FSM_STEP
    def fsm_step_node(state: VNState) -> VNState:
        """평가 신호를 반영해 관계 FSM과 액션 FSM을 전이시킨다."""
        log_step("FSM_STEP", "start")
        prev_action_state = state.get("action_state", "")
        action_hist = state.get("action_state_hist", [])
        fsm_rel.step({
            "mental_instability": int(state["signals"].get("mental_instability", 0)),
            "intimacy": int(state["signals"].get("intimacy", 0)),
            "threat": int(state["signals"].get("threat", 0)),
            "pressure": int(state["signals"].get("pressure", 0)),
            "probe": int(state["signals"].get("probe", 0)),
            "resolve": int(state["signals"].get("resolve", 0)),
            "genre": parse_genre(state["system_lore"]),
        })
        fsm_act.step({
            "event": int(state["signals"].get("event", 0)),
            "threat": int(state["signals"].get("threat", 0)),
            "pressure": int(state["signals"].get("pressure", 0)),
            "resolve": int(state["signals"].get("resolve", 0)),
            "sexual_ready": bool(state.get("sexual_ready", False)),
        })
        # Base action state from FSM, then diversify with distribution controls
        base_action_state = fsm_act.get_state()
        log_step(
            "FSM_STEP",
            f"pre action_state={prev_action_state} base={base_action_state} "
            f"sexual_ready={state.get('sexual_ready', False)} "
            f"sexual_turns={state.get('sexual_turns', 0)} "
            f"aftermath_turns={state.get('aftermath_turns', 0)} "
            f"sexual_lock={state.get('sexual_lock', 0)}",
        )

        # Sexual sequence handling (explicit staged states)
        sexual_turns = int(state.get("sexual_turns", 0))
        aftermath_turns = int(state.get("aftermath_turns", 0))
        sexual_lock = int(state.get("sexual_lock", 0))
        if prev_action_state in SEXUAL_STATES or base_action_state in SEXUAL_STATES:
            sexual_turns = min(4, sexual_turns + 1)
            state["action_state"] = f"SEXUAL_{sexual_turns}"
            if sexual_lock <= 0:
                sexual_lock = MIN_SEXUAL_TURNS
            sexual_lock = max(sexual_lock - 1, 0)
            if sexual_turns >= MIN_SEXUAL_TURNS:
                # Move into sexual aftermath after minimum turns
                state["action_state"] = "AFTERMATH_SEX_1"
                sexual_turns = 0
                aftermath_turns = 1
                sexual_lock = 0
        elif prev_action_state in AFTERMATH_SEX_STATES or base_action_state in AFTERMATH_SEX_STATES:
            if prev_action_state == "AFTERMATH_SEX_1":
                aftermath_turns = 2
                state["action_state"] = "AFTERMATH_SEX_2"
            else:
                aftermath_turns = 0
                state["action_state"] = "IDLE"
            sexual_lock = 0
        else:
            if base_action_state in SEXUAL_STATES and not state.get("sexual_ready", False):
                base_action_state = "EVENT"
            genre = parse_genre(state["system_lore"])
            diversified = _pick_action_state(
                current=base_action_state,
                history=action_hist,
                genre=genre,
                sexual_ready=bool(state.get("sexual_ready", False)),
            )
            state["action_state"] = diversified
            sexual_turns = 0
            aftermath_turns = 0
            sexual_lock = 0
            # Force entry into sexual sequence when ready, even if FSM base state isn't sexual
            if state.get("sexual_ready", False) and state["action_state"] not in SEXUAL_STATES:
                sexual_turns = 1
                sexual_lock = MIN_SEXUAL_TURNS
                state["action_state"] = "SEXUAL_1"

        # Force out of IDLE when user stalls repeatedly
        if state.get("user_idle_streak", 0) >= 2 and state["action_state"] == "IDLE":
            state["action_state"] = "EVENT"
        if state.get("action_state") in SEXUAL_STATES and not state.get("sexual_ready", False):
            state["action_state"] = "EVENT"

        state["sexual_turns"] = sexual_turns
        state["aftermath_turns"] = aftermath_turns
        state["sexual_lock"] = sexual_lock

        # Update action history (last 6)
        action_hist.append(state["action_state"])
        state["action_state_hist"] = action_hist[-6:]
        log_step(
            "FSM_STEP",
            f"post action_state={state.get('action_state')} "
            f"sexual_turns={sexual_turns} aftermath_turns={aftermath_turns} "
            f"sexual_lock={sexual_lock}",
        )
        log_step("FSM_STEP", "done")
        return state

    # NEXT (reset per-turn)
    def next_node(state: VNState) -> VNState:
        """턴 종료 후 임시 상태를 초기화하고 다음 턴을 준비한다."""
        log_step("NEXT", "start")
        state["turn_index"] += 1

        state["user_text"] = ""
        state["assistant_text"] = ""
        state["narration_text"] = ""
        state["dialogue_text"] = ""
        state["sexual_request"] = False
        state["rewrite_mode"] = ""

        state["retry_user"] = 0
        state["retry_assistant"] = 0
        state["retry_quality"] = 0
        state["retry_sexual"] = 0
        state["retry_eval"] = 0
        state["retry_crisis"] = 0
        state["error"] = ""
        log_step("NEXT", f"done turn_index={state['turn_index']}")
        return state

    # DONE (final cleanup)
    def done_node(state: VNState) -> VNState:
        # Ensure dataset doesn't end on a user turn
        """종료 직전 메시지 정리를 수행한다."""
        msgs = state.get("messages", [])
        if msgs and msgs[-1].get("role") == "user":
            msgs.pop()
        state["messages"] = msgs
        log_step("DONE", "cleanup_done")
        return state

    # Routers

    def after_init(_: VNState) -> str:
        """INIT 이후 다음 노드를 선택한다."""
        return "GEN_USER"

    def after_gen_user(state: VNState) -> str:
        """GEN_USER 이후 재시도 또는 다음 노드를 선택한다."""
        if state["error"]:
            if state["retry_user"] < 30:
                return "GEN_USER"
            return "DONE"
        return "DETECT"

    def after_detect(state: VNState) -> str:
        """DETECT 결과에 따라 생성 노드를 분기한다."""
        if fsm_rel.get_state() == "CRISIS":
            return "GEN_CRISIS"
        if state.get("action_state", "").startswith("SEXUAL_"):
            return "GEN_SEXUAL"
        return "GEN_ASSISTANT"

    def after_gen_assistant(state: VNState) -> str:
        """GEN_ASSISTANT 이후 재시도 또는 평가 노드를 선택한다."""
        if state["error"]:
            if state["error"] in ("asst_meta", "asst_dia_style_mismatch_llm"):
                return "GEN_ASSISTANT_REWRITE"
            if state["retry_assistant"] < 30:
                return "GEN_ASSISTANT"
            return "DONE"
        return "EVAL_DIALOGUE_QUALITY"

    def after_gen_assistant_rewrite(state: VNState) -> str:
        """재작성 노드 이후 다음 노드를 선택한다."""
        if state["error"]:
            if state["retry_rewrite"] < 5:
                return "GEN_ASSISTANT_REWRITE"
            return "DONE"
        return "EVAL_DIALOGUE_QUALITY"

    def after_gen_sexual(state: VNState) -> str:
        """GEN_SEXUAL 이후 재시도 또는 평가 노드를 선택한다."""
        if state["error"]:
            if state["error"] == "asst_dia_style_mismatch_llm":
                return "GEN_ASSISTANT_REWRITE"
            if state["retry_sexual"] < 30:
                return "GEN_SEXUAL"
            return "DONE"
        return "EVAL_DIALOGUE_QUALITY"

    def after_gen_crisis(state: VNState) -> str:
        """GEN_CRISIS 이후 재시도 또는 평가 노드를 선택한다."""
        if state["error"]:
            if state["retry_crisis"] < 12:
                return "GEN_CRISIS"
            return "DONE"
        return "EVAL_DIALOGUE_QUALITY"

    def after_eval_dialogue_quality(state: VNState) -> str:
        """품질 평가 결과에 따라 재작성 여부를 결정한다."""
        if state["error"]:
            if state["retry_quality"] < 3:
                return "GEN_ASSISTANT_REWRITE_QUALITY"
            return "EVAL_INTERNAL"
        return "EVAL_INTERNAL"

    def after_gen_assistant_rewrite_quality(state: VNState) -> str:
        """품질 재작성 이후 다음 노드를 선택한다."""
        if state["error"]:
            if state["retry_quality"] < 3:
                return "GEN_ASSISTANT_REWRITE_QUALITY"
            return "EVAL_INTERNAL"
        return "EVAL_INTERNAL"

    def after_eval_internal(state: VNState) -> str:
        """내부 평가 이후 JSON 평가 노드로 이동한다."""
        return "EVAL_JSON"

    def after_eval_json(state: VNState) -> str:
        """JSON 평가 결과에 따라 재시도 또는 FSM 단계를 선택한다."""
        if state["error"]:
            if state["retry_eval"] < 40:
                return "EVAL_JSON"
            return "DONE"
        return "FSM_STEP"

    def after_next(state: VNState) -> str:
        """다음 턴 진입 또는 종료를 결정한다."""
        if state["turn_index"] >= state["turns_target"]:
            return "DONE"
        return "GEN_USER_SEXUAL" if state.get("sexual_ready") else "GEN_USER"

    # Graph assembly

    g = StateGraph(VNState)
    g.add_node("INIT", init_node)
    g.add_node("GEN_USER", gen_user_node)
    g.add_node("GEN_USER_SEXUAL", gen_user_sexual_node)
    g.add_node("DETECT", detect_node)
    g.add_node("GEN_ASSISTANT", gen_assistant_node)
    g.add_node("GEN_ASSISTANT_REWRITE", gen_assistant_rewrite_node)
    g.add_node("EVAL_DIALOGUE_QUALITY", eval_dialogue_quality_node)
    g.add_node("GEN_ASSISTANT_REWRITE_QUALITY", gen_assistant_rewrite_quality_node)
    g.add_node("GEN_CRISIS", gen_crisis_node)
    g.add_node("GEN_SEXUAL", gen_sexual_node)
    g.add_node("EVAL_INTERNAL", eval_internal_node)
    g.add_node("EVAL_JSON", eval_json_node)
    g.add_node("FSM_STEP", fsm_step_node)
    g.add_node("NEXT", next_node)
    g.add_node("DONE", done_node)

    g.set_entry_point("INIT")

    g.add_conditional_edges("INIT", after_init, {"GEN_USER": "GEN_USER"})

    g.add_conditional_edges("GEN_USER", after_gen_user, {
        "GEN_USER": "GEN_USER",
        "DETECT": "DETECT",
        "DONE": "DONE",
    })
    g.add_conditional_edges("GEN_USER_SEXUAL", after_gen_user, {
        "GEN_USER": "GEN_USER_SEXUAL",
        "DETECT": "DETECT",
        "DONE": "DONE",
    })

    g.add_conditional_edges("DETECT", after_detect, {
        "GEN_CRISIS": "GEN_CRISIS",
        "GEN_ASSISTANT": "GEN_ASSISTANT",
        "GEN_SEXUAL": "GEN_SEXUAL",
    })

    g.add_conditional_edges("GEN_ASSISTANT", after_gen_assistant, {
        "GEN_ASSISTANT": "GEN_ASSISTANT",
        "GEN_ASSISTANT_REWRITE": "GEN_ASSISTANT_REWRITE",
        "EVAL_DIALOGUE_QUALITY": "EVAL_DIALOGUE_QUALITY",
        "DONE": "DONE",
    })

    g.add_conditional_edges("GEN_ASSISTANT_REWRITE", after_gen_assistant_rewrite, {
        "GEN_ASSISTANT_REWRITE": "GEN_ASSISTANT_REWRITE",
        "EVAL_DIALOGUE_QUALITY": "EVAL_DIALOGUE_QUALITY",
        "DONE": "DONE",
    })

    g.add_conditional_edges("GEN_SEXUAL", after_gen_sexual, {
        "GEN_SEXUAL": "GEN_SEXUAL",
        "GEN_ASSISTANT_REWRITE": "GEN_ASSISTANT_REWRITE",
        "EVAL_DIALOGUE_QUALITY": "EVAL_DIALOGUE_QUALITY",
        "DONE": "DONE",
    })

    g.add_conditional_edges("GEN_CRISIS", after_gen_crisis, {
        "GEN_CRISIS": "GEN_CRISIS",
        "EVAL_DIALOGUE_QUALITY": "EVAL_DIALOGUE_QUALITY",
        "DONE": "DONE",
    })

    g.add_conditional_edges("EVAL_DIALOGUE_QUALITY", after_eval_dialogue_quality, {
        "GEN_ASSISTANT_REWRITE_QUALITY": "GEN_ASSISTANT_REWRITE_QUALITY",
        "EVAL_INTERNAL": "EVAL_INTERNAL",
    })
    g.add_conditional_edges("GEN_ASSISTANT_REWRITE_QUALITY", after_gen_assistant_rewrite_quality, {
        "GEN_ASSISTANT_REWRITE_QUALITY": "GEN_ASSISTANT_REWRITE_QUALITY",
        "EVAL_INTERNAL": "EVAL_INTERNAL",
    })

    g.add_conditional_edges("EVAL_INTERNAL", after_eval_internal, {
        "EVAL_JSON": "EVAL_JSON",
    })
    g.add_conditional_edges("EVAL_JSON", after_eval_json, {
        "EVAL_JSON": "EVAL_JSON",
        "FSM_STEP": "FSM_STEP",
        "DONE": "DONE",
    })
    g.add_edge("FSM_STEP", "NEXT")
    g.add_conditional_edges("NEXT", after_next, {
        "GEN_USER": "GEN_USER",
        "GEN_USER_SEXUAL": "GEN_USER_SEXUAL",
        "DONE": "DONE",
    })

    g.add_edge("DONE", END)

    return g.compile()


# Scenario runner

def run_scenario(
    system_lore: str,
    client: OpenAI,
    model_name: str,
    turns: int,
    fsm_path: str,
    action_fsm_path: Optional[str] = None,
) -> Dict[str, Any]:
    """단일 시나리오북으로 멀티턴 대화를 생성해 `messages`를 반환한다.

    내부적으로 상태/행동 FSM, 임베딩 메모리, 그래프를 초기화한 뒤 그래프를
    실행한다. 최소 한 쌍 이상의 `user`/`assistant` 턴이 생성될 때까지
    제한 횟수 내에서 재시도한다.

    Args:
        system_lore: 시나리오북 전체 텍스트.
        client: OpenAI API 클라이언트.
        model_name: 응답 생성에 사용할 모델 식별자.
        turns: 목표 턴 수.
        fsm_path: 상태 FSM YAML 경로.
        action_fsm_path: 액션 FSM YAML 경로. 미지정 시 기본 경로 사용.

    Returns:
        `{"messages": [...]}` 형태의 결과 딕셔너리.
    """
    fsm_rel = QwenFSMEngine(fsm_path, system_lore)
    act_path = action_fsm_path or "data/original/v5_qwen/action_fsm.yaml"
    fsm_act = QwenFSMEngine(act_path, system_lore)

    # 임베딩 연산 장치만 자동 선택한다.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embed_memory = EmbeddingMemory("data/embedding/BGE-m3-ko", device=device)

    graph = build_graph(
        client=client,
        model_name=model_name,
        fsm_rel=fsm_rel,
        fsm_act=fsm_act,
        embed_memory=embed_memory,
    )

    init_state: VNState = {
        "system_lore": system_lore,
        "protagonist": "",
        "player_name": "",
        "relation_status": "",
        "sexual_condition_rel": "",
        "relation_intent": "",
        "action_state": "",
        "last_user_question": False,
        "last_asst_question": False,
        "last_asst_reject": False,
        "last_asst_accept": False,
        "last_asst_sexual": False,
        "action_state_hist": [],
        "sexual_turns": 0,
        "sexual_lock": 0,
        "aftermath_turns": 0,
        "user_idle_streak": 0,

        "turns_target": int(turns),
        "turn_index": 0,

        "messages": [],

        "user_text": "",
        "assistant_text": "",
        "raw_assistant": "",
        "narration_text": "",
        "dialogue_text": "",

        "sexual_request": False,
        "allow_sexual": False,
        "sexual_ready": False,

        "signals": {
            "mental_instability": 0,
            "intimacy": 0,
            "threat": 0,
            "pressure": 0,
            "probe": 0,
            "resolve": 0,
            "event": 0,
        },

        "retry_user": 0,
        "retry_assistant": 0,
        "retry_rewrite": 0,
        "retry_quality": 0,
        "retry_sexual": 0,
        "retry_eval": 0,
        "retry_crisis": 0,
        "eval_internal_text": "",

        "error": "",
    }

    def _has_min_turn(msgs: List[Dict[str, str]]) -> bool:
        """내부 보조 로직을 수행한다."""
        has_user = False
        has_asst = False
        for m in msgs:
            if m.get("role") == "user":
                has_user = True
            elif m.get("role") == "assistant":
                has_asst = True
            if has_user and has_asst:
                return True
        return False

    out = None
    for attempt in range(4):
        out = graph.invoke(init_state)
        if _has_min_turn(out.get("messages", [])):
            break
        print(f"[WARN] no_turn_generated retry={attempt+1}/4", flush=True)

    return {"messages": out["messages"] if out else []}


# CLI

def main():
    """CLI 엔트리포인트.

    시나리오 JSONL을 순회하며 각 system lore에 대해 멀티턴 데이터를 생성해
    출력 JSONL에 append한다. 중단 후 재실행 시 이미 생성된 라인을 기준으로
    이어서 처리한다.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--openai_model", default="gpt-5-mini")
    parser.add_argument("--scenario_path", required=True)
    parser.add_argument("--out_path", required=True)
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

    def count_lines(path: str) -> int:
        """대상 항목의 개수를 집계해 반환한다."""
        if not os.path.exists(path):
            return 0
        with open(path, "r", encoding="utf-8") as rf:
            return sum(1 for _ in rf if _.strip())

    out_done = count_lines(args.out_path)
    if out_done > 0:
        print(f"[MULTITURN] resume_from={out_done}", flush=True)

    # scenario jsonl: {"messages":[{"role":"system","content":"...scenario..."}]}
    total_scen = count_lines(args.scenario_path)
    if out_done >= total_scen:
        print("[MULTITURN] nothing to do (out >= scenarios)", flush=True)
        return

    with open(args.scenario_path, "r", encoding="utf-8") as f, open(
        args.out_path, "a", encoding="utf-8"
    ) as out:
        idx = 0
        for line in f:
            if not line.strip():
                continue
            if idx < out_done:
                idx += 1
                continue
            system_lore = json.loads(line)["messages"][0]["content"]
            data = run_scenario(
                system_lore=system_lore,
                client=client,
                model_name=model_name,
                turns=args.turns,
                fsm_path=args.fsm_path,
                action_fsm_path=args.action_fsm_path,
            )
            out.write(json.dumps(data, ensure_ascii=False) + "\n")
            out.flush()
            idx += 1

    print("DONE")


if __name__ == "__main__":
    main()
