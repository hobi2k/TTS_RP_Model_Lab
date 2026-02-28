# Compatibility Shim Notes

이 문서는 RTX 5080 대응을 위해 PyTorch nightly (`cu128`)를 사용하는 현재 환경에서, 상위 라이브러리와의 API 불일치를 임시로 흡수하기 위해 넣은 호환 shim을 정리한 문서다.

대상 프로젝트:

- `/home/ahnhs2k/pytorch-demo/saya_char_qwen2.5/models/Style-BERT-VITS2`

현재 shim은 아래 파일에 들어 있다.

- [style_gen.py](/home/ahnhs2k/pytorch-demo/saya_char_qwen2.5/models/Style-BERT-VITS2/style_gen.py)

## 1. 왜 shim이 필요한가

현재 환경은 RTX 5080(`sm_120`) 때문에 구버전 `torch`를 사용할 수 없고, 다음 방식으로 PyTorch nightly를 사용한다.

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

이때 설치되는 `torchaudio` nightly는 과거 버전에서 노출하던 일부 레거시 API를 더 이상 최상위 모듈에 제공하지 않는다. 또한 최근 PyTorch nightly는 `torch.load`의 기본 보안 정책이 강화되어, 과거 체크포인트를 그대로 읽지 못하는 경우가 있다.

반면 `pyannote.audio`는 import 시점과 런타임에 여전히 아래 항목들을 기대한다.

- `torchaudio.AudioMetaData`
- `torchaudio.list_audio_backends`
- `torchaudio.get_audio_backend`
- `torchaudio.set_audio_backend`
- `torchaudio.info`
- `torchaudio.load`가 legacy 스타일로 오디오를 직접 읽는 동작
- `torch.load`가 `weights_only=False`에 가깝게 동작하던 과거 체크포인트 로딩 관행

그 결과 `style_gen.py` 실행 시, 실제 모델 추론 전에 import 단계에서 아래와 같은 오류가 발생했다.

- `AttributeError: module 'torchaudio' has no attribute 'AudioMetaData'`
- `AttributeError: module 'torchaudio' has no attribute 'list_audio_backends'`

즉, 현재 shim의 목적은 다음 두 가지다.

1. `pyannote.audio`가 기대하는 구 `torchaudio` API를 최소한으로 복원해 `style_gen.py`를 계속 실행 가능하게 만드는 것
2. PyTorch nightly의 강화된 기본 보안 정책 아래에서도 신뢰 가능한 `pyannote` 체크포인트를 로드할 수 있게 하는 것

## 2. 현재 적용된 shim

`style_gen.py`에서 `from pyannote.audio import Inference, Model`보다 먼저 `torchaudio`를 import한 뒤, 필요한 심볼이 없으면 동적으로 채워 넣는다.

현재 보강하는 항목은 아래와 같다.

### 2.1 `AudioMetaData`

nightly `torchaudio`에서 `torchaudio.AudioMetaData`가 사라진 경우를 대비해, 동일 이름의 간단한 dataclass를 주입한다.

필드:

- `sample_rate`
- `num_frames`
- `num_channels`
- `bits_per_sample`
- `encoding`

이 shim은 `pyannote.audio`의 타입 참조와 기본 메타데이터 전달을 위한 최소 구조만 제공한다.

### 2.2 `list_audio_backends`

`torchaudio.list_audio_backends()`가 없으면 다음을 반환하도록 한다.

```python
["soundfile"]
```

의미:

- 현재 호환 계층은 `soundfile` 기반 오디오 메타데이터 접근을 전제로 한다.
- `pyannote.audio`가 백엔드 목록 확인 시 import 단계에서 죽지 않게 한다.

### 2.3 `get_audio_backend`

`torchaudio.get_audio_backend()`가 없으면 항상 `"soundfile"`을 반환한다.

### 2.4 `set_audio_backend`

`torchaudio.set_audio_backend()`가 없으면 no-op 함수로 대체한다.

의미:

- `pyannote.audio`가 백엔드를 설정하려고 해도 즉시 실패하지 않게 한다.
- 현재 구현은 실제 백엔드 전환 기능을 제공하지 않는다.

### 2.5 `info`

`torchaudio.info()`가 없으면 `soundfile.info()`를 사용해 메타데이터를 읽고, 위에서 만든 `AudioMetaData` 형태로 변환해 반환한다.

현재 변환 항목:

- `sample_rate <- soundfile.info(...).samplerate`
- `num_frames <- soundfile.info(...).frames`
- `num_channels <- soundfile.info(...).channels`
- `bits_per_sample <- subtype 문자열에서 숫자 추출`
- `encoding <- subtype 문자열 또는 `"UNKNOWN"`

즉, 현재 shim은 메타데이터 조회를 `torchaudio` 대신 `soundfile`로 우회한다.

### 2.6 `load`

nightly `torchaudio`에서 `torchaudio.load()`가 내부적으로 `torchcodec`를 강제하는 경우가 있다.

현재 환경에서는 다음 오류가 발생했다.

- `ImportError: TorchCodec is required for load_with_torchcodec. Please install torchcodec to use this function.`

이를 피하기 위해 `style_gen.py`는 `torchaudio.load`를 `soundfile.read()` 기반 호환 함수로 덮어쓴다.

동작:

- `soundfile.read(path, always_2d=True, dtype="float32")`로 WAV를 읽음
- `(frames, channels)` 배열을 `(channels, frames)` 형태로 전치
- `torch.Tensor`로 변환
- `(waveform, sample_rate)` 형식으로 반환

의미:

- `pyannote.audio`가 `torchaudio.load(...)`를 호출해도 `torchcodec` 경로를 타지 않는다.
- Step 5 스타일 특징 추출에 필요한 WAV 로드는 계속 가능하다.

## 3. PyTorch Safe Globals Shim

PyTorch 2.6+ 및 nightly에서는 `torch.load` 기본값이 사실상 `weights_only=True` 기준으로 강화되었다.

그 결과 `pyannote` 체크포인트 로딩 중 아래와 같은 오류가 순차적으로 발생했다.

- `Unsupported global: GLOBAL torch.torch_version.TorchVersion`
- `Unsupported global: GLOBAL pyannote.audio.core.task.Specifications`
- `Unsupported global: GLOBAL pyannote.audio.core.task.Problem`
- `Unsupported global: GLOBAL pyannote.audio.core.task.Resolution`

이를 해결하기 위해 `style_gen.py`는 `Model.from_pretrained(...)` 호출 전에 `torch.serialization.add_safe_globals(...)`를 사용해 필요한 클래스들을 allowlist에 등록한다.

현재 허용하는 항목:

- `torch.torch_version.TorchVersion`
- `pyannote.audio.core.task.Specifications`
- `pyannote.audio.core.task.Problem`
- `pyannote.audio.core.task.Resolution`

의미:

- 현재 사용 중인 `pyannote` 체크포인트가 PyTorch nightly 보안 정책에 막히지 않게 한다.
- 체크포인트 로딩 자체만 통과시키는 조치이며, 추론 커널 성능이나 메모리 사용량을 높이지는 않는다.

## 4. 적용 범위

이 shim은 현재 [style_gen.py](/home/ahnhs2k/pytorch-demo/saya_char_qwen2.5/models/Style-BERT-VITS2/style_gen.py) 실행 경로에만 적용된다.

직접적인 목적:

- `pyannote.audio` import 성공
- `Model.from_pretrained(...)`의 체크포인트 로딩 진입
- `Inference(...)` 초기화까지 진입
- `torchaudio.load(...)` 호출 시 `torchcodec` 경로 우회
- Step 5 스타일 특징 파일 생성 단계 진행

이 shim은 프로젝트 전체의 `torchaudio` 호환성을 보장하지 않는다.

특히 아래는 별도 보장이 없다.

- 다른 라이브러리가 `torchaudio`의 더 많은 레거시 API를 요구하는 경우
- 실제 오디오 디코딩 경로가 `soundfile`로 우회했을 때와 원본 `torchaudio` 구현 사이 차이에 민감한 경우
- `pyannote.audio`가 이후 단계에서 추가로 제거된 API를 참조하는 경우
- 체크포인트 내부에 아직 allowlist되지 않은 추가 글로벌 심볼이 들어 있는 경우

## 5. 한계와 주의점

현재 shim은 호환성 확보를 위한 최소 우회다. 다음 한계가 있다.

### 4.1 타입/구조 호환 위주

`AudioMetaData`는 실제 `torchaudio` 원본 클래스와 완전히 동일하지 않다.

따라서:

- 타입 힌트
- 단순 속성 접근

에는 대응하지만, 원본 클래스의 더 복잡한 동작까지 재현하지는 않는다.

### 4.2 백엔드 전환 미지원

`set_audio_backend()`는 no-op이다.

즉:

- 백엔드를 바꾸는 호출이 와도 실제 동작은 바뀌지 않는다.
- 현재 구현은 사실상 `soundfile` 고정이다.

### 4.3 `soundfile` 의존

`torchaudio.info()`와 `torchaudio.load()` shim은 모두 `soundfile`에 의존한다.

따라서 `soundfile`이 없거나 대상 파일 포맷을 못 읽으면 이 shim도 실패한다.

### 4.4 임시 우회 성격

이 shim은 상위 라이브러리 버전 정렬이 불가능한 상황에서 넣은 임시 호환층이다.

근본 해결은 아래 중 하나다.

1. `pyannote.audio`가 최신 `torchaudio` API를 공식 지원하는 버전으로 올라가기
2. 현재 nightly가 아닌, `pyannote.audio`와 API 호환되는 `torchaudio` 버전으로 내리기
3. `style_gen.py` 구현을 `pyannote.audio` 의존 없이 다른 스타일 임베딩 방식으로 교체하기
4. `torchcodec`를 포함한 정식 오디오 스택과 `pyannote`가 nightly에서 완전히 호환되도록 버전 정렬하기

## 6. 언제 제거할 수 있는가

아래 조건 중 하나가 충족되면 shim 제거를 검토할 수 있다.

1. `pyannote.audio` 최신 버전이 현재 nightly `torchaudio`에서 import 에러 없이 동작함이 확인될 때
2. RTX 5080 지원과 `pyannote.audio` 호환을 동시에 만족하는 안정 버전의 `torch/torchaudio` 조합으로 돌아갈 때
3. `style_gen.py`가 더 이상 `pyannote.audio`를 사용하지 않게 될 때
4. `Model.from_pretrained(...)`가 별도 `add_safe_globals(...)` 없이도 현재 체크포인트를 읽을 수 있을 때

제거 전 확인 방법:

```bash
python -c "import torchaudio; print(hasattr(torchaudio, 'AudioMetaData')); print(hasattr(torchaudio, 'list_audio_backends')); print(hasattr(torchaudio, 'info'))"
python -c "from pyannote.audio import Inference, Model; print('ok')"
```

둘 다 별도 shim 없이 성공하면 제거 후보가 된다.

`torch.load` 경로 확인:

```bash
python -c "from pyannote.audio import Model; Model.from_pretrained('pyannote/wespeaker-voxceleb-resnet34-LM'); print('ok')"
```

이것도 별도 allowlist 없이 성공해야 safe globals shim 제거 후보가 된다.

## 7. 운영 지침

현재 shim을 유지하는 동안 권장 사항은 다음과 같다.

- `style_gen.py` 실행 전에는 [style_gen.py](/home/ahnhs2k/pytorch-demo/saya_char_qwen2.5/models/Style-BERT-VITS2/style_gen.py)의 shim 코드가 제거되지 않았는지 확인한다.
- `torch/torchaudio` nightly를 업데이트한 뒤에는 Step 5를 바로 재실행하기 전에 import 테스트를 먼저 한다.
- `pyannote.audio` 관련 새 `AttributeError`가 나오면, 먼저 제거된 `torchaudio` 레거시 API인지 확인한 뒤 같은 방식으로 최소 shim을 추가한다.
- `_pickle.UnpicklingError`가 나오면, 에러 메시지에 표시된 글로벌 심볼을 보고 `add_safe_globals(...)` allowlist에 추가할지 검토한다.
- `TorchCodec is required for load_with_torchcodec`가 다시 나오면 `torchaudio.load` shim이 제거됐거나 덮어써졌는지 확인한다.

간단 확인:

```bash
python style_gen.py --help
```

이 단계에서 import가 통과하면, 최소한 shim은 현재 버전에 맞게 동작 중인 것이다.
