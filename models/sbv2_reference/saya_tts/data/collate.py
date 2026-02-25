"""
saya_tts/data/collate.py

Batch Collate 함수 (Style-Bert-VITS2 / JP-Extra 입력 규약 확정판)

이 파일의 책임은 명확하다.
- Dataset.__getitem__이 반환한 "가변 길이 샘플(dict)"들을
- Trainer / model(enc_p)이 바로 사용할 수 있는
  "배치 텐서"로 정렬(padding)하고, 마스크를 재생성한다.

중요 원칙(절대 바꾸지 말 것):
1) padding 값은 0
2) x_mask는 (phoneme_id != 0) 규칙으로 생성
3) 모든 텍스트 관련 길이 축은 T_text로 정렬
4) mel은 T_mel로 따로 정렬 (텍스트와 길이 다름)

이 규칙은:
- enc_p
- duration predictor(dp/sdp)
- infer_after_enc_p
전부에서 전제하고 있다.
"""
from __future__ import annotations

from typing import List, Dict, Any, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_saya_tts(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    SayaTTSDataset 전용 collate_fn.

    Args:
        batch:
            List[dict], 각 dict는 Dataset.__getitem__의 반환값

    Returns:
        batch_dict:
            Trainer / model.forward가 바로 사용할 수 있는 배치 텐서 묶음

    반환되는 주요 텐서 shape 요약:

    텍스트 계열 (T_text 기준):
    - phoneme_ids : [B, T_text]
    - tone_ids    : [B, T_text]
    - language_ids: [B, T_text]
    - x_mask      : [B, 1, T_text]
    - bert_feats  : [B, D_bert, T_text]

    오디오 계열 (T_mel 기준):
    - mel         : [B, n_mels, T_mel]

    기타:
    - word2ph     : List[Tensor] (가변 길이, padding 안 함)
    - style_id    : [B]
    - speaker_id  : [B]
    """

    # 0. batch size
    B = len(batch)

    # 1. 텍스트 관련 시퀀스 모으기
    # NOTE:
    # - pad_sequence는 "길이 축이 0번 차원"일 때 가장 쓰기 편하다.
    # - 따라서 [T] 형태 텐서를 그대로 넘긴다.
    phoneme_ids_list = [item["phoneme_ids"] for item in batch]   # List[[T_i]]
    tone_ids_list = [item["tone_ids"] for item in batch]         # List[[T_i]]
    language_ids_list = [item["language_ids"] for item in batch] # List[[T_i]]

    # 2. padding (텍스트)
    # padding_value=0 은 <pad> 규칙과 일치
    #
    # 결과 shape:
    # - phoneme_ids : [B, T_text]
    # - tone_ids    : [B, T_text]
    # - language_ids: [B, T_text]
    phoneme_ids = pad_sequence(
        phoneme_ids_list,
        batch_first=True,
        padding_value=0,
    )

    tone_ids = pad_sequence(
        tone_ids_list,
        batch_first=True,
        padding_value=0,
    )

    language_ids = pad_sequence(
        language_ids_list,
        batch_first=True,
        padding_value=0,
    )

    # 3. x_mask 재생성 (중요!)
    # 규칙:
    # - padding 토큰 == 0
    # - 유효 토큰 != 0
    #
    # x_mask shape:
    #   [B, 1, T_text]
    #
    # NOTE:
    # - Dataset에서 만들어온 x_mask는 "샘플 단위" 마스크였고,
    #   padding 이후에는 길이가 맞지 않으므로 여기서 다시 만든다.
    # - enc_p, dp/sdp, generate_path 전부 이 규칙을 전제한다.
    x_mask = (phoneme_ids != 0).float().unsqueeze(1)

    # 4. BERT feature padding
    # bert_feats는 [D_bert, T_i] 형태이므로,
    # 먼저 [T_i, D_bert]로 transpose한 뒤 pad_sequence를 쓴다.
    bert_list = [item["bert_feats"].transpose(0, 1) for item in batch]  # [T_i, D_bert]

    bert_padded = pad_sequence(
        bert_list,
        batch_first=True,
        padding_value=0.0,
    )  # [B, T_text, D_bert]

    # 다시 enc_p 입력 규약에 맞게 [B, D_bert, T_text]로 transpose
    bert_feats = bert_padded.transpose(1, 2)

    # 5. mel padding (오디오)
    # mel은 [n_mels, T_mel_i]
    # 텍스트 길이와 독립적이므로 별도로 padding
    mel_list = [item["mel"].transpose(0, 1) for item in batch]  # [T_mel_i, n_mels]

    mel_padded = pad_sequence(
        mel_list,
        batch_first=True,
        padding_value=0.0,
    )  # [B, T_mel, n_mels]

    # [B, n_mels, T_mel]로 복원
    mel = mel_padded.transpose(1, 2)

    # 6. word2ph 처리 (padding 안 함)
    # word2ph는 길이가 "문자/word 단위"라서
    # 배치 내에서도 길이가 다를 수 있다.
    #
    # Style-Bert-VITS2 학습에서는:
    # - word2ph를 그대로 리스트로 들고 다니며
    # - 필요할 때만 순회하거나 alignment loss 계산에 사용한다.
    #
    # 따라서 여기서는 padding하지 않고 그대로 둔다.
    word2ph = [item["word2ph"] for item in batch]

    # 7. style_id / speaker_id
    # Dataset에서 [1] shape로 들어왔으므로
    # 여기서 squeeze해서 [B]로 만든다.
    style_id = torch.cat([item["style_id"] for item in batch], dim=0)
    speaker_id = torch.cat([item["speaker_id"] for item in batch], dim=0)

    # 8. 기타 메타 (디버깅용)
    utt_ids = [item["utt_id"] for item in batch]
    raw_texts = [item["raw_text"] for item in batch]

    # 9. 최종 batch dict
    return {
        # 텍스트 입력
        "phoneme_ids": phoneme_ids,      # [B, T_text]
        "tone_ids": tone_ids,            # [B, T_text]
        "language_ids": language_ids,    # [B, T_text]
        "x_mask": x_mask,                # [B, 1, T_text]
        "bert_feats": bert_feats,        # [B, D_bert, T_text]

        # 오디오 입력
        "mel": mel,                      # [B, n_mels, T_mel]

        # 정렬/보조 정보
        "word2ph": word2ph,              # List[Tensor], padding 안 함

        # 조건 정보
        "style_id": style_id,            # [B]
        "speaker_id": speaker_id,        # [B]

        # 디버깅/로그용
        "utt_id": utt_ids,
        "raw_text": raw_texts,
    }
