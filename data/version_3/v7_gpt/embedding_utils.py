import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List


class EmbeddingMemory:
    """
    BGE-m3-ko 기반 멀티-카인드 임베딩 메모리

    - kind별 벡터 풀 분리
      (narration / dialogue / assistant)
    - is_repetitive: 검사만 수행
    - add: 명시적으로 벡터 추가
    """

    DEFAULT_THRESHOLDS = {
        "narration": 0.92,   # 서술은 느슨
        "dialogue": 0.86,    # 대사는 엄격
        "assistant": 0.90,   # 최종 응답 중간
        "user_action": 0.92,
        "user_dialogue": 0.86,
        "user": 0.90,
    }

    def __init__(self, model_path: str, device: str = "cuda"):
        """객체 초기화에 필요한 상태를 설정한다."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.model.eval()
        self.device = device

        # kind별 벡터 저장소
        self.vectors: Dict[str, List[np.ndarray]] = {
            "narration": [],
            "dialogue": [],
            "assistant": [],
            "user_action": [],
            "user_dialogue": [],
            "user": [],
        }

    # Encoding
    @torch.inference_mode()
    def encode(self, text: str) -> np.ndarray:
        """입력 텍스트를 정규화된 임베딩 벡터로 인코딩한다."""
        if not text or not text.strip():
            # 빈 텍스트는 더미 벡터 (항상 반복으로 처리)
            return np.zeros(768, dtype=np.float32)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        out = self.model(**inputs)
        emb = out.last_hidden_state[:, 0]  # CLS
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.cpu().numpy()[0]

    # Repetition Check
    def is_repetitive(
        self,
        text: str,
        *,
        kind: str = "assistant",
        threshold: float | None = None,
        window: int = 8,
    ) -> bool:
        """
        반복 여부만 검사 (벡터 추가 안 함)

        Args:
            text: 검사할 텍스트
            kind: narration | dialogue | assistant
            threshold: None이면 kind별 기본 threshold 사용
            window: 최근 몇 개 벡터만 비교할지
        """
        if not text or len(text.strip()) < 4:
            return True

        if kind not in self.vectors:
            kind = "assistant"

        vecs = self.vectors[kind]
        if not vecs:
            return False

        thr = threshold if threshold is not None else self.DEFAULT_THRESHOLDS[kind]

        v = self.encode(text)
        recent = vecs[-window:]
        sims = cosine_similarity([v], recent)[0]

        return sims.max() >= thr

    # Add Memory

    def add(self, text: str, *, kind: str = "assistant"):
        """
        벡터를 명시적으로 저장
        """
        if not text or len(text.strip()) < 4:
            return

        if kind not in self.vectors:
            kind = "assistant"

        v = self.encode(text)
        self.vectors[kind].append(v)

        # 무한 증가 방지
        if len(self.vectors[kind]) > 32:
            self.vectors[kind] = self.vectors[kind][-32:]
