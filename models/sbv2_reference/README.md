text (str)
 ├─ normalize
 ├─ text → symbols → text_ids        (T_text)
 ├─ bert(text) → bert_feats          (T_bert)
 ├─ align bert_feats → T_text
 ├─ TextEncoder(
 │     text_ids,
 │     tone_ids,
 │     bert_feats_aligned,
 │     style_vec
 │   ) → prior stats
 ├─ SynthesizerTrn.infer(...)
 └─ wav (44100 Hz)

 net_g (SynthesizerTrn / SynthesizerTrnJPExtra)
├── enc_p      ← Text Encoder / Prior Encoder  \
├── dp         ← deterministic duration predictor
├── sdp        ← stochastic duration predictor
├── flow       ← normalizing flow
├── dec        ← decoder (latent → waveform)
├── emb_g      ← speaker embedding table
└── (enc_q)    ← posterior encoder (훈련 전용)

enc_p
├── InputEmbeddingBlock   ★ 지금 구현할 부분
│   ├── phoneme embedding
│   ├── tone embedding
│   ├── language embedding
│   ├── bert projection (1024 → H)
│   └── embedding fusion (합산)
│
├── TransformerEncoderStack
│
└── PriorHead
    ├── m_p
    └── logs_p
    
0) 모듈 맵 (무엇을 조절하는지부터)

JP-Extra 계열에서 대략 다음 블록이 있다(이름은 repo에 따라 다를 수 있으나 역할은 동일).

Text Frontend

BERT(ja DeBERTa/…)

tokenizer / clean_text / phone/tone/lang 생성
→ 텍스트를 “특징”으로 만드는 부분

enc_p (TextEncoder/Prior)

input embedding(phoneme/tone/lang + bert)

transformer stack

prior head → m_p, logs_p
→ 텍스트 조건부 prior 분포를 만드는 부분 (발음/억양의 뼈대)

Duration predictors

dp(결정적) / sdp(확률적)
→ 말 속도/리듬/토큰별 길이

Flow
→ latent 변환(음색/질감에 영향 큼, 잘못 건드리면 불안정)

Decoder (vocoder/decoder)
→ 파형 생성(음색/해상도/노이즈/호흡감)

Conditioning

style_vec

g(speaker embedding or ref enc)
→ “사야”를 규정하는 핵심 컨트롤

1) 추천: 3단계 Progressive Unfreeze (가장 안전)
Stage 0: “사야로 말하게 만들기” 최소 업데이트 (안전, 첫 시도용)

목표: 원본 음질/발음 유지 + 사야 스타일만 살짝

Freeze:

BERT(전부)

enc_p(전부)

dp/sdp(전부)

flow(전부)

decoder(전부)

Unfreeze:

speaker embedding (emb_g): ON

style projection / style embedding 경로(있다면): ON

(옵션) decoder의 very last layer / output proj만 ON (아주 소량)

학습률:

emb_g / style 관련: 높게(예: 1e-3 ~ 5e-4)

decoder 마지막: 낮게(예: 1e-5 ~ 5e-6)

언제 성공이라고 보나:

텍스트는 잘 읽고(깨짐 없음)

음질이 크게 무너지지 않는데

“사야 톤”이 조금이라도 묻어나오면 Stage 1로

Stage 1: “사야 음색” 붙이기 (가장 실전적인 단계)

목표: 음색/질감/발성 습관을 사야로 만들기

Freeze:

BERT(전부 유지)

enc_p(대부분 유지: transformer stack은 일단 freeze)

duration(dp/sdp) 대부분 유지(리듬 깨질 위험)

flow: 대부분 freeze, 마지막 1~2 block만 옵션으로 ON 가능

Unfreeze:

decoder (상위/후반 블록 위주): ON

전체를 풀지 말고 “뒤쪽부터” 푼다(음색 담당 비중 큼)

prior_head: ON (m_p/logs_p 쪽은 음색/프로소디에도 영향)

emb_g / style 경로: 계속 ON

학습률:

decoder: 1e-5 ~ 2e-5

prior_head: 5e-5 ~ 1e-4

emb_g/style: 5e-4 ~ 1e-3

언제 Stage 2로 가나:

사야 음색이 확실해졌는데

리듬/속도/억양이 “데이터”에 비해 어색하거나

길이 예측이 원본과 부조화가 나는 경우

Stage 2: “리듬/속도/억양”까지 사야화 (최후, 신중)

목표: 사야의 말버릇(빠르기/쉬는 타이밍/억양)을 더 맞춤

Freeze:

BERT: 계속 freeze 권장 (과적합/불안정 방지)

flow: 대부분 freeze 유지(불안정의 진원지)

Unfreeze:

dp: ON (결정적 길이 예측부터)

(옵션) sdp: 아주 조심히 ON (확률성 때문에 튀기 쉬움)

enc_p transformer stack: “상단 1~2 layer만” ON (필요할 때만)

decoder: 계속 ON

학습률:

dp: 1e-5 ~ 5e-5

sdp: 1e-6 ~ 1e-5 (아주 낮게)

enc_p 상단: 1e-5 ~ 2e-5

2) “한 방에” 하고 싶다면: 단일 단계 설계 (리스크 있음)

데이터가 충분하고(수 시간 이상), 품질이 높고, 발음 깨짐이 적다면

Unfreeze:

emb_g/style

prior_head

decoder 후반 50%

dp(선택)

Freeze:

BERT 전체

enc_p transformer 대부분

flow 전체(또는 마지막 1 block만)

3) 사야(단일 화자) 기준 “가장 추천” 조합

너 상황(사야 캐릭터 보이스, 품질 유지 중요) 기준으로는 이게 제일 안전하다.

BERT: freeze

enc_p: freeze (prior_head만 unfreeze)

dp/sdp: freeze

flow: freeze

decoder: 후반만 unfreeze

emb_g/style: unfreeze (항상)

이 조합은:

발음 깨짐 최소

음질 급락 최소

사야 음색 전이 최대

4) 체크포인트/실험 설계 (실무 팁)

각 Stage에서 반드시 남겨라:

동일 문장 10개 고정(짧은/긴/감탄/웃음 포함)

매 N 스텝마다 wav 저장

“깨짐(발음/속도/노이즈)”이 보이면 즉시 이전 Stage로 롤백