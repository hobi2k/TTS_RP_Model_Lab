# SBV2 Core Study & Refactor Roadmap

본 문서는 `litagin02/Style-Bert-VITS2`를 기반으로 구성한 `sbv2_core` 디렉토리를  
**코드 수준에서 완전히 이해하고**, 이후 **내 프로젝트(saya_char_qwen2.5)에 맞게 점진적으로 개조**하기 위한 로드맵이다.

목표는 다음과 같다.

- Style-Bert-VITS2 엔진의 **훈련 / 추론 / 전처리 흐름을 정확히 이해**
- WebUI, 배포 스크립트, 보조 툴을 완전히 제거
- 이해 이후 리팩터링 및 커스터마이징 수행

---

## 0. 현재 디렉토리 기준 상태

```
sbv2_core/
├─ style_bert_vits2/ # 엔진 코어 (절대 기준점)
├─ bert/ # 언어별 BERT 자산 (JP / EN / ZH)
├─ slm/ # WavLM 등 음성 관련 자산
├─ dict_data/ # 사용자 사전 / 발음 사전
├─ configs/ # 원본 설정 파일
│
├─ preprocess_all.py
├─ preprocess_text.py
├─ data_utils.py
├─ resample.py
├─ slice.py
│
├─ train_ms_jp_extra.py # 주 훈련 엔트리
├─ train_ms.py # 비교/참고용
│
├─ losses.py
├─ mel_processing.py
│
├─ config.py
├─ default_config.yml
├─ gen_yaml.py
├─ default_style.py
│
├─ pyproject.toml
├─ requirements.txt
├─ LICENSE
└─ LGPL_LICENSE
```

---

## 1. 학습 단계에서의 절대 원칙

- 파일명 변경 금지
- 디렉토리 구조 변경 금지
- config 통합 금지
- 변수명 리팩터링 금지
- 기능 추가 금지

이 단계의 목적은 **개선이 아니라 “정확한 이해”**다.  
수정은 이해가 끝난 뒤에만 수행한다.

---

## 2. 권장 코드 읽기 순서

### A. 전체 흐름 파악 (엔트리포인트)

#### `train_ms_jp_extra.py`
- 학습 시작 지점
- config 로딩 방식
- Dataset / DataLoader 생성
- `style_bert_vits2`로 진입하는 지점 확인

#### `train_ms.py`
- JP extra 없는 기본 구조 비교용
- 공통/차이점 구분

**목표**
- “학습은 어디서 시작해서 어디로 들어가는가?”
- “JP extra는 어디에서 분기되는가?”

---

### B. 데이터 전처리 파이프라인

#### `preprocess_all.py`
- 전체 전처리 오케스트레이터
- wav / text / meta 생성 흐름

#### `preprocess_text.py`
- 텍스트 정규화
- 언어별 분기(JP/EN/ZH)

#### `data_utils.py`
- filelist / metadata 구조
- 훈련 입력 포맷의 실체

#### `resample.py`
- 샘플레이트 통일

#### `slice.py`
- 긴 음성 분할 로직

### `gradio_tabs` 내 함수들

**목표**
- “훈련에 실제로 들어가는 데이터 형식은 무엇인가?”
- “wav / text가 어떤 규칙으로 묶이는가?”

---

### C. 음향 처리 / 손실 함수

#### `mel_processing.py`
- mel spectrogram 생성
- STFT / window / hop 설정

#### `losses.py`
- duration / alignment / mel / KL 등 loss 구성

**목표**
- 음향 손실과 정렬 손실의 역할 구분
- 어느 부분이 스타일/억양에 영향을 주는지 파악

---

### D. 엔진 코어 (가장 중요)

#### `style_bert_vits2/models/models_jp_extra.py`
- `SynthesizerTrn` (JP extra 버전)
- 스타일, BERT feature, 언어 조건 결합 지점

#### `style_bert_vits2/models/models.py`
- 기본 구조 비교

#### `style_bert_vits2/models/infer.py`
- 추론 전용 forward 경로

#### `style_bert_vits2/tts_model.py`
- 외부에서 엔진을 호출하는 인터페이스

#### `style_bert_vits2/voice.py`
- 음성 출력 / 후처리

**목표**
- `SynthesizerTrn.forward()` 전체 데이터 흐름 설명 가능
- 스타일은 “생성”이 아니라 “조건/주입”임을 명확히 이해
- BERT feature가 언제 결합되는지 파악

---

## 3. 이해가 끝났다는 판단 기준

아래 4가지를 **말로 설명할 수 있으면**, 다음 단계로 이동한다.

1. 학습 시작 -> `SynthesizerTrn`까지의 호출 경로
2. 텍스트 -> G2P -> BERT -> style 조건 -> mel 예측 흐름
3. JP extra가 구조적으로 추가되는 정확한 지점

---

## 4. 이해 이후의 개조 단계

1. `train_ms_jp_extra.py`를 단일 `train.py`로 단순화
2. config를 하나의 명시적 설정 파일로 통합
3. 데이터 경로를 프로젝트 규칙에 맞게 수정
4. 코드 기반 `infer.py` 작성

---

## 5. 핵심 결론

- `sbv2_core`는 **엔진 연구용 코어**다
- WebUI는 절대 섞지 않는다
- 이해 -> 단순화 -> 커스터마이즈 순서를 지킨다


1. resample            (선택, but 보통 먼저)
2. slice               (orig.wav → orig-0.wav, orig-1.wav …)
3. transcription 생성 / 수정
   (slice된 wav 기준으로 utt 작성)
4. preprocess_text     (slice된 utt 기준)
5. bert_gen             
6. style_gen            
7. train_ms_jp_extra