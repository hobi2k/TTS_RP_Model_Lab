"""
"TTS 전체"가 아니라, Style-Bert-VITS2의 BERT 경로를 정확히 고정하는 단계

즉 지금은:
  text -> bert_hidden_states

까지만 만든다.

왜 이렇게 쪼개나?
- TTS는 모듈이 많아서 한 번에 다 붙이면
  어디서 shape가 틀렸는지 찾기 지옥이 열린다.
- Qwen 아키텍처 구현할 때도
  Attention/MLP/Block을 분리해 하나씩 검증했듯,
  여기서도 BERT부터 검증하고 TextEncoder로 넘어간다.
"""
from dataclasses import dataclass
import torch

from .config import SayaTTSConfig
from .bert.bert_model import StyleBertModel, BertOutput


@dataclass
class Part1Result:
    """
    결과를 고정된 형태로 반환한다.
    - 나중에 text_ids와 결합할 때도 같은 구조로 확장 가능.
    """
    bert: BertOutput


class SayaTTSPipeline:
    """
    엔진의 '유일한 통로'를 여기로 모은다.
    - synthesize(text)까지 확장하더라도,
      BERT 추출은 여기서 동일한 방식으로 유지된다.
    """
    def __init__(self, cfg: SayaTTSConfig):
        self.cfg = cfg
        self.device = cfg.device

        # BERT 로더: 원본이 전제하는 모델명과 설정을 사용
        self.bert = StyleBertModel(cfg.bert_model_name, device=cfg.device)

    @torch.inference_mode()
    def run_part1(self, text: str) -> Part1Result:
        """
        공식 API.
        - raw text 입력
        - bert hidden states 반환
        """
        bert_out = self.bert.forward(text)
        return Part1Result(bert=bert_out)

    def debug_part1(self, text: str) -> None:
        """
        실제로 어떤 텐서가 나오는지 1회 출력
        """
        print("Part1 Debug")
        print(f"input text: {text}")
        self.bert.debug_print(text)
