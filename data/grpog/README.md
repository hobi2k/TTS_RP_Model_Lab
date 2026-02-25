# GRPOG Dataset Pipeline

`data/grpog/build_grpo_dataset.py`는 기존 RP 데이터(jsonl)에서 GRPO 학습용 `prompt/reference` 샘플을 만든다.

## 지원 입력 형식

1. Chat 형식
```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

2. Legacy singleturn 형식
```json
{"system":"...","user":"...","assistant":"..."}
```

## 출력 형식

각 row는 아래 형태다.

```json
{
  "id": "source.jsonl:123:45:0",
  "prompt": "SYSTEM: ...\nUSER: ...",
  "prompt_messages": [
    {"role":"system","content":"..."},
    {"role":"user","content":"..."}
  ],
  "reference": "assistant target text",
  "meta": {
    "source_path": "...",
    "source_line": 123,
    "assistant_turn_index": 2
  }
}
```

## 실행 예시

```bash
uv run data/grpog/build_grpo_dataset.py \
  --inputs /mnt/d/rp_data/singleturn/rp_singleturn_cleaned.jsonl /mnt/d/rp_data/v7/rp_datum_unite_cleaned.jsonl \
  --out_train /mnt/d/rp_data/grpo/grpo_train.jsonl \
  --out_eval /mnt/d/rp_data/grpo/grpo_eval.jsonl \
  --eval_ratio 0.05 \
  --seed 42 \
  --max_context_messages 12 \
  --min_prompt_chars 8 \
  --min_reference_chars 4 \
  --tokenizer_path models/qwen3_core/model_assets/qwen3_4b_rp \
  --max_prompt_tokens 1024 \
  --max_reference_tokens 220
```

## 주요 옵션

- `--max_context_messages`: prompt 메시지 최대 개수(뒤쪽 유지).
- `--eval_ratio`: eval 분할 비율.
- `--min_prompt_chars`: 너무 짧은 prompt 제거.
- `--min_reference_chars`: 너무 짧은 assistant 타깃 제거.
- `--max_prompt_tokens`: prompt 토큰 상한. 초과 row는 drop.
- `--max_reference_tokens`: reference 토큰 상한. 초과 row는 drop.
- `--tokenizer_path`: 토큰 길이 필터링 시 사용할 tokenizer 경로.
