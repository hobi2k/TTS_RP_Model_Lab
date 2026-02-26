# Dataset Rewriter

멀티턴/싱글턴 공통 JSONL(`{"messages":[...]}`)을 대상으로 `user + assistant`를 모두 리라이트한다.

규칙:
- `user`: 원문 말투 레벨(존댓말/반말) 유지
- `assistant`: 시나리오북(`system`) 규칙/말투/출력 형식 우선 준수

## 1) GPT API 리라이터

```bash
uv run python data/rewrite/rewrite_with_gpt_api.py \
  --input data/raw/train.jsonl \
  --output data/processed/train.rewrite.gpt.jsonl \
  --model gpt-4o-mini
```

옵션:
- `--temperature` (default: `0.3`)
- `--max-samples` (default: `0`, 전체)
- `--base-url` (OpenAI 호환 게이트웨이 사용 시)

환경변수:
- `OPENAI_API_KEY` 필요

## 2) 로컬 LLM 리라이터

```bash
uv run python data/rewrite/rewrite_with_local_llm.py \
  --input data/raw/train.jsonl \
  --output data/processed/train.rewrite.local.jsonl \
  --model-dir models/qwen3_core/model_assets/qtranslator_1.7b
```

옵션:
- `--max-new-tokens` (default: `192`)
- `--temperature` (default: `0.2`)
- `--top-p` (default: `0.9`)
- `--max-samples` (default: `0`, 전체)

## 입력 형식

각 row는 아래 형태:

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

`user/assistant` 메시지가 여러 개인 멀티턴도 처리한다.
