"""
Japanese G2P (Style-Bert-VITS2 호환 래퍼)

목표
- Style-Bert-VITS2가 "실제로" 사용하는 일본어 전처리(정규화/G2P/악센트/word2ph)를
  그대로 호출해서,
- 학습(train)과 추론(infer) 모두에서 동일한 입력 텐서(phoneme_ids/tone_ids/lang_ids/word2ph)를
  만들 수 있게 한다.

왜 래퍼로 감싸나?
- SBV2 내부 구현은 디렉토리/버전별로 함수명이 바뀌거나(리팩토링),
  JP-Extra 전용 분기 등이 존재한다.
- 이 코드는 "호출 시그니처"를 고정해두고,
  내부에서 SBV2 함수를 찾아 연결하도록 만들면
  나중에 모델/버전이 바뀌어도 교체 포인트가 명확해진다.

참고
- Style-Bert-VITS2는 일본어 처리(g2p.py 등)를 별도 모듈로 보유한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any


# 출력 포맷(우리 프로젝트 고정)
@dataclass(frozen=True)
class JapaneseG2PResult:
    """
    Style-Bert-VITS2의 clean_text(...)가 결국 만들어내는 핵심 결과를
    우리 파이프라인에서 쓰기 좋은 형태로 고정한다.

    phones:
        - 음소(phoneme) 문자열 시퀀스
        - 예: ["k", "o", "N", "n", "i", "ch", "i", "w", "a", ...]
    tones:
        - 각 phone에 대응하는 pitch-accent tone(또는 악센트 관련) 시퀀스
        - 길이는 phones와 같아야 한다.
    word2ph:
        - "원 단위 token(단어/표기 단위)"가 phones에서 몇 개의 phone을 차지하는지 나타내는 배열
        - 예: [2, 3, 1, ...]  (합(sum)이 len(phones)와 일치)
        - 이후 word2ph_to_duration 같은 로직에서 duration(프레임 길이)의 초기 구조로 자주 사용된다.
    normalized_text:
        - SBV2 내부 정규화(기호 치환/숫자 처리/공백 처리 등) 이후 텍스트
        - 디버깅 때 “왜 phone이 이상해졌지?”를 빠르게 추적하기 위해 반드시 같이 보관한다.
    """
    phones: List[str]
    tones: List[int]
    word2ph: List[int]
    normalized_text: str


# SBV2 함수 로딩 유틸
def _import_sbv2_japanese_frontend() -> Dict[str, Any]:
    """
    Style-Bert-VITS2 패키지에서 일본어 G2P/정규화/텍스트 클린 관련 함수를 찾아 반환한다.

    여기서 중요한 설계 포인트:
    - SBV2는 리팩토링이 잦아서 '정확한 import 경로'가 버전에 따라 달라질 수 있다.
    - 따라서 "후보 경로를 여러 개" 두고, 실제 설치된 패키지에서 성공하는 경로를 채택한다.

    반환:
        dict에 callable들을 담아서 반환한다.
        - normalize(text) -> str
        - g2p(text) -> (phones, tones, word2ph) 혹은 유사 구조
        - clean_text(text, ...) -> (phones, tones, word2ph, normalized_text) 형태일 수도 있음
    """
    errors: List[str] = []

    # 1. 일본어 전용 g2p/normalizer 모듈이 있는 경우
    candidates = [
        # 자주 쓰이는 형태: style_bert_vits2.nlp.japanese.g2p / normalizer
        ("style_bert_vits2.nlp.japanese.g2p", "style_bert_vits2.nlp.japanese.normalizer"),
        # 일부 포크/버전에서 nlp 하위가 바뀌는 경우를 대비
        ("style_bert_vits2.nlp.japanese.g2p", None),
    ]

    for g2p_mod, norm_mod in candidates:
        try:
            g2p_module = __import__(g2p_mod, fromlist=["*"])
            normalizer_module = __import__(norm_mod, fromlist=["*"]) if norm_mod else None

            # g2p 함수명은 버전마다 다를 수 있어서 가능한 후보를 탐색한다.
            g2p_fn = None
            for name in ("g2p", "text_to_phonemes", "japanese_g2p"):
                if hasattr(g2p_module, name):
                    g2p_fn = getattr(g2p_module, name)
                    break

            # normalizer도 마찬가지로 후보를 탐색한다.
            norm_fn = None
            if normalizer_module is not None:
                for name in ("normalize", "normalize_text", "text_normalize"):
                    if hasattr(normalizer_module, name):
                        norm_fn = getattr(normalizer_module, name)
                        break

            # 최소 조건: g2p_fn은 있어야 “SBV2 기반”이라고 말할 수 있다.
            if g2p_fn is None:
                raise ImportError(f"Cannot find g2p callable in {g2p_mod}")

            return {"g2p": g2p_fn, "normalize": norm_fn}

        except Exception as e:
            errors.append(f"[{g2p_mod}] {repr(e)}")
            continue

    # 여기까지 왔다는 건, 현재 venv에 style-bert-vits2가 없거나 구조가 많이 달라진 것.
    raise ImportError(
        "Failed to import Style-Bert-VITS2 Japanese frontend.\n"
        "Tried:\n- " + "\n- ".join(errors) + "\n\n"
        "Check that `style-bert-vits2` is installed (in the same venv) "
        "and that its package layout matches expected modules."
    )


# 외부 공개 API (우리 파이프라인이 호출)
class JapaneseG2PFrontend:
    """
    일본어 텍스트를 '모델 입력'으로 바꾸는 유일한 관문.

    사용:
        frontend = JapaneseG2PFrontend()
        out = frontend(text)

    이 클래스를 두는 이유:
    - 학습/추론/전처리 파이프라인 어디서든 동일 동작을 보장
    - 디버깅 로그를 중앙에서 통제 가능
    - SBV2 버전 변경 시 내부만 수정하면 됨
    """
    def __init__(self) -> None:
        self._sbv2 = _import_sbv2_japanese_frontend()

    def __call__(self, text: str) -> JapaneseG2PResult:
        """
        text -> (normalized_text, phones, tones, word2ph)

        주의:
        - SBV2 내부 g2p 함수의 반환 포맷이 (phones, tones, word2ph) 형태가 “대체로” 맞지만,
          버전에 따라 dict/tuple 구조가 달라질 수 있다.
        - 그래서 여기서 결과를 “정규화”해서 고정 포맷(JapaneseG2PResult)으로 만든다.
        """
        # 1. SBV2 normalizer를 먼저 태운다.
        #    normalizer가 없다면, 원문 그대로 g2p에 넣는다.
        normalize_fn = self._sbv2.get("normalize")
        normalized = normalize_fn(text) if callable(normalize_fn) else text

        # 2. SBV2 g2p 실행
        g2p_fn = self._sbv2["g2p"]
        raw = g2p_fn(normalized)

        # 3. 반환 포맷 정리
        phones, tones, word2ph = _coerce_g2p_output(raw)

        # 4. 길이/합 검증(디버깅 지옥 예방)
        if len(phones) != len(tones):
            raise ValueError(
                f"[G2P] phones/tones length mismatch: {len(phones)} vs {len(tones)}\n"
                f"normalized_text={normalized!r}"
            )
        if sum(word2ph) != len(phones):
            raise ValueError(
                f"[G2P] sum(word2ph) must equal len(phones): sum={sum(word2ph)} len={len(phones)}\n"
                f"normalized_text={normalized!r}"
            )

        return JapaneseG2PResult(
            phones=phones,
            tones=tones,
            word2ph=word2ph,
            normalized_text=normalized,
        )


def _coerce_g2p_output(raw: Any) -> Tuple[List[str], List[int], List[int]]:
    """
    SBV2 g2p 결과를 (phones, tones, word2ph)로 강제 변환한다.

    가능한 입력 예시:
    - tuple/list: (phones, tones, word2ph)
    - dict: {"phones":..., "tones":..., "word2ph":...}
    - 기타: 버전 차이로 생기는 형태

    여기서 “강제 변환 계층”을 둬야,
    상위 코드(데이터셋/전처리/학습)가 SBV2 내부 변화에 덜 흔들린다.
    """
    # case 1) (phones, tones, word2ph) 형태
    if isinstance(raw, (tuple, list)) and len(raw) >= 3:
        phones = list(raw[0])
        tones = list(raw[1])
        word2ph = list(raw[2])
        return _cast_phones_tones_word2ph(phones, tones, word2ph)

    # case 2) dict 형태
    if isinstance(raw, dict):
        # 키 이름은 버전/포크에 따라 다를 수 있어 후보를 둔다.
        phones = raw.get("phones") or raw.get("phone") or raw.get("phonemes")
        tones = raw.get("tones") or raw.get("tone") or raw.get("accents")
        word2ph = raw.get("word2ph") or raw.get("w2p")

        if phones is None or tones is None or word2ph is None:
            raise TypeError(f"Unsupported g2p dict keys: {list(raw.keys())}")

        return _cast_phones_tones_word2ph(list(phones), list(tones), list(word2ph))

    raise TypeError(f"Unsupported g2p output type: {type(raw)} / value={raw!r}")


def _cast_phones_tones_word2ph(
    phones: List[Any],
    tones: List[Any],
    word2ph: List[Any],
) -> Tuple[List[str], List[int], List[int]]:
    """
    타입을 확실히 고정:
    - phones: List[str]
    - tones: List[int]
    - word2ph: List[int]
    """
    phones_out = [str(p) for p in phones]
    tones_out = [int(t) for t in tones]
    word2ph_out = [int(x) for x in word2ph]
    return phones_out, tones_out, word2ph_out


# CLI 디버깅(단독 실행용)
def debug_run() -> None:
    """
    이 파일 단독으로 실행했을 때:
    - SBV2 g2p 연결이 정상인지
    - phones/tones/word2ph의 길이/합이 맞는지
    빠르게 확인하는 용도
    """
    frontend = JapaneseG2PFrontend()

    test_text = "こんにちは。私はサヤです。"
    out = frontend(test_text)

    print("[JapaneseG2PFrontend] OK")
    print("input           :", test_text)
    print("normalized_text :", out.normalized_text)
    print("phones(len)     :", len(out.phones))
    print("tones(len)      :", len(out.tones))
    print("word2ph(len)    :", len(out.word2ph))
    print("sum(word2ph)    :", sum(out.word2ph))
    print("phones(head)    :", out.phones[:30])
    print("tones(head)     :", out.tones[:30])
    print("word2ph(head)   :", out.word2ph[:30])


if __name__ == "__main__":
    debug_run()
