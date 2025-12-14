"""
Style-Bert-VITS2의 "BERT 경로"는 TTS에서 매우 중요한 역할을 한다.

- 일반적인 TTS는 텍스트(또는 phoneme)만 보고 발음을 만든다.
- Style-Bert-VITS2는 여기에 "문맥(context)"를 추가한다.
  즉, 같은 문장이라도 상황에 따라:
    - 망설이는 톤
    - 확신하는 톤
    - 문장 부호(……, …)의 의미
  를 다르게 반영할 수 있다.

이 문맥 신호를 만들어 주는 모듈이 BERT이고,
TextEncoder는 이 BERT 출력(특징)을 받아서 "발화 설계도"를 만든다.

출력: text(str) -> bert_hidden_states(torch.FloatTensor)

반드시 확인해야 하는 것:
  1. 출력 shape: (B, T_bert, D_bert)
  2. dtype: float32
  3. device: cuda/cpu 일치
  4. pooler/cls 같은 요약 벡터를 쓰지 않고 "토큰별(hidden_states)"를 유지하는지
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModel


@dataclass
class BertOutput:
    """
    BERT 출력 래퍼.

    - feats: last_hidden_state
      shape: (B, T_bert, D_bert)
      예) (1, 34, 1024) 또는 (1, 50, 768) 등

    왜 래퍼로 감싸나?
    - 추후 "길이 정렬", "mask", "투영(conv1d)" 등
      부가 정보가 필요해지는 경우가 많다.
    - 출력 텐서만 던지면 나중에 shape 추적이 힘들어지므로
      dataclass로 구조를 고정해 두면 디버깅이 쉬워진다.
    """
    feats: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None  # (B, T_bert) int64/bool 등


class StyleBertModel:
    """
    BERT 모델 래퍼

    이 클래스는 "원본 style_bert_vits2/nlp/bert_models.py의 의도"를 반영한다.
    - 추론에서 필요한 건 BERT의 last_hidden_state뿐
    - tokenizer/model 설정을 원본 전제와 최대한 동일하게 맞춘다

    작업 시 가장 중요한 포인트:
    - hidden_size(D_bert)가 TextEncoder의 bert_proj 입력 채널과 맞아야 한다.
      예: D_bert = 1024 라면 bert_proj = Conv1d(1024 -> hidden_channels)
    """

    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = torch.device(device)

        # 1. Tokenizer 로딩
        # Style-Bert-VITS2 쪽은 일본어 DeBERTa char WWM을 전제로 한다.
        # add_prefix_space=True 는 RoBERTa/DeBERTa 계열 토크나이저에서
        # 단어 경계 처리에 영향을 줄 수 있어 종종 켠다.
        #
        # "왜 토크나이저가 중요하냐?"
        # - BERT가 뽑는 hidden state는 토큰 단위로 나오므로
        #   토큰화 방식이 달라지면 T_bert가 달라지고,
        #   텍스트 토큰 길이(T_text)와 정렬시키는 로직이 흔들린다.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            add_prefix_space=True,
        )

        # 2. Model 로딩
        # AutoModel은 "인코더 본체"만 로드한다.
        # 우리는 MLM 헤드나 분류 헤드가 필요 없고,
        # last_hidden_state만 필요하다.
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # dropout 등 비활성화

        # 3. BERT 출력 차원(hidden_size)
        # 이후 TextEncoder의 bert_proj 입력 채널이 이 값과 일치해야 한다.
        self.hidden_size: int = int(self.model.config.hidden_size)

    def _tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """
        text -> tokenizer outputs

        Returns:
            dict with keys like:
              input_ids: (B, T_bert)
              attention_mask: (B, T_bert)
              token_type_ids: (B, T_bert)  # 모델에 따라 없을 수도

        주의:
        - 일본어는 띄어쓰기가 거의 없어서 토큰화가 길어질 수 있다.
        - truncation=True는 너무 긴 문장에 대해 길이를 자른다.
          (캐릭터 대사가 길어질 때, 이게 문맥 손실로 이어질 수 있어
           이후에 max_length 전략을 따로 세울 수 있다.)
        """
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,    # 단일 문장 추론이면 padding 불필요
            truncation=True,
        )
        # tokenizer는 CPU 텐서를 반환하므로 device로 이동
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        return encoded

    @torch.inference_mode()
    def forward(self, text: str) -> BertOutput:
        """
        [가장 중요한 함수]
        - 입력: raw text
        - 출력: last_hidden_state

        Output 설명:
        - feats: (B, T_bert, D_bert)
          B=1 (단일 문장), T_bert는 토큰 길이, D_bert는 hidden size.
        """
        # 1. tokenize
        inputs = self._tokenize(text)

        # 2. forward
        # outputs는 BaseModelOutputWithPoolingAndCrossAttentions 같은 형태일 수 있음.
        # 우리가 사용하는 건 last_hidden_state 뿐.
        outputs = self.model(**inputs)

        # 3. last_hidden_state 추출
        feats = outputs.last_hidden_state

        # 4. dtype 통일
        # 일부 환경/모델에서 fp16/bf16로 나올 수 있으므로 float32로 고정한다.
        feats = feats.float()

        # 5. attention_mask도 같이 넘겨두면 이후 정렬/마스킹할 때 유용하다.
        attn_mask = inputs.get("attention_mask")

        return BertOutput(feats=feats, attention_mask=attn_mask)

    def debug_print(self, text: str) -> None:
        """
        디버깅용 출력 함수.

        - shape, dtype, device를 명확히 찍어준다.
        - 모델이 기대하는 텐서 형태를 머리에 박는 데 도움됨.
        """
        out = self.forward(text)
        feats = out.feats
        print(f"[BERT] model_name = {self.model_name}")
        print(f"[BERT] hidden_size = {self.hidden_size}")
        print(f"[BERT] feats.shape = {tuple(feats.shape)}  # (B, T_bert, D_bert)")
        print(f"[BERT] feats.dtype = {feats.dtype}")
        print(f"[BERT] feats.device = {feats.device}")
        if out.attention_mask is not None:
            print(f"[BERT] mask.shape = {tuple(out.attention_mask.shape)}  # (B, T_bert)")
            print(f"[BERT] mask.dtype = {out.attention_mask.dtype}")
