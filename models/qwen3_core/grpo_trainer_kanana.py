"""Kanana 전용 RP GRPO 학습 스크립트.

참고 페이지:
https://huggingface.co/learn/cookbook/fine_tuning_vlm_grpo_trl

목표:
- Kanana 계열 모델을 텍스트 RP 보상함수로 GRPO 학습
- 불필요한 복잡도(임베딩 보상 등)를 제거한 최소 실행형

예시:
uv run models/qwen3_core/grpo_trainer_kanana.py \
  --model_name models/qwen3_core/model_assets/saya_vlm_3b_sft \
  --train_data /mnt/d/rp_data/grpo/grpo3_train.jsonl \
  --eval_data /mnt/d/rp_data/grpo/grpo3_eval.jsonl \
  --output_dir models/qwen3_core/model_assets/kanana_3b_grpo \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 2 \
  --learning_rate 1e-6 \
  --max_prompt_length 2048 \
  --num_generations 2 \
  --temperature 0.7 \
  --top_p 0.9 \
  --top_k 40 \
  --repetition_penalty 1.05 \
  --use_lora \
  --load_in_4bit \
  --bf16 \
  --trust_remote_code \
  --debug_data_preview 5
"""

from __future__ import annotations

import argparse
import inspect
import re
from pathlib import Path
from types import MethodType
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from PIL import Image
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig, ProcessorMixin
from trl import GRPOConfig, GRPOTrainer


PLAYER_NAMES = ("카즈키", "하야토", "소마", "유저", "플레이어")
ROLE_MARKERS = ("SYSTEM:", "USER:", "ASSISTANT:", "role:", "<|im_start|", "<|assistant|")


class KananaGRPOProcessor(ProcessorMixin):
    """TRL GRPO에서 KananaVProcessor를 사용할 수 있게 해주는 어댑터.

    TRL은 Processor를 `processor(images=..., text=..., return_tensors='pt')` 형태로 호출한다.
    KananaVProcessor는 `batch_encode_collate(data_list=...)` 경로를 쓰므로 이를 맞춰준다.
    """

    attributes = ["tokenizer"]

    def __init__(self, base_processor: Any, max_length: int, dummy_image_size: int = 224) -> None:
        self.base_processor = base_processor
        self.tokenizer = getattr(base_processor, "tokenizer", None)
        self.chat_template = getattr(base_processor, "chat_template", None)
        if self.chat_template is None and self.tokenizer is not None:
            self.chat_template = getattr(self.tokenizer, "chat_template", None)
        self.max_length = max_length
        self.dummy_image_size = dummy_image_size
        self._debug_printed = False
        if self.tokenizer is None:
            raise ValueError("Kanana processor에 tokenizer가 없습니다.")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _dummy_image(self) -> Image.Image:
        return Image.new("RGB", (self.dummy_image_size, self.dummy_image_size), color=(255, 255, 255))

    def _to_conv(self, text: str) -> list[dict[str, str]]:
        # Kanana 인코더는 텍스트 내 <image> 개수와 image_meta 개수가 일치해야 한다.
        # 여기서 image 슬롯은 첫 user 메시지로 고정 주입하므로 본문의 <image>는 제거한다.
        text = (text or "").replace("<image>", "").strip()
        return [
            {"role": "user", "content": "<image>"},
            {"role": "user", "content": text},
        ]

    def __call__(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        texts = kwargs.get("text")
        images = kwargs.get("images")
        padding = kwargs.get("padding", "longest")
        max_length = kwargs.get("max_length", self.max_length)
        # Kanana tokenizer는 padding을 bool이 아닌 문자열로만 받는다.
        if padding is True:
            padding = "longest"
        elif padding is False or padding is None:
            padding = "longest"
        elif isinstance(padding, str):
            if padding not in {"longest", "max_length"}:
                padding = "longest"
        else:
            padding = "longest"

        if texts is None:
            # TRL 경로가 아닌 호출은 base processor에 위임
            return self.base_processor(*args, **kwargs)

        if isinstance(texts, str):
            texts = [texts]

        norm_images: list[list[Any]] = []
        if images is None:
            norm_images = [[self._dummy_image()] for _ in texts]
        else:
            for img in images:
                if img is None or img == []:
                    norm_images.append([self._dummy_image()])
                elif isinstance(img, list):
                    fixed = []
                    for one in img:
                        if isinstance(one, Image.Image):
                            fixed.append(one)
                        else:
                            fixed.append(self._dummy_image())
                    norm_images.append(fixed if fixed else [self._dummy_image()])
                else:
                    norm_images.append([img if isinstance(img, Image.Image) else self._dummy_image()])

        data_list = []
        for txt, imgs in zip(texts, norm_images, strict=True):
            data_list.append({"conv": self._to_conv(str(txt)), "image": imgs})

        try:
            batch = self.base_processor.batch_encode_collate(
                data_list=data_list,
                padding=padding,
                padding_side="left",
                max_length=max_length,
                add_generation_prompt=True,
            )
        except AssertionError as e:
            if "Length exceeded" in str(e):
                raise ValueError(
                    f"{e}\n"
                    "Kanana GRPO prompt 길이가 max_prompt_length를 초과했습니다. "
                    "--max_prompt_length를 더 크게 주거나(예: 2048/3072/4096), "
                   "데이터의 대화 길이를 줄이세요."
                ) from e
            raise
        # Kanana tokenizer collate는 list를 반환할 수 있어 GRPO generate에서 shape 접근 시 실패한다.
        # generate 입력은 텐서가 필요하므로 여기서 강제 변환한다.
        if isinstance(batch.get("input_ids"), list):
            batch["input_ids"] = torch.as_tensor(batch["input_ids"], dtype=torch.long)
        if isinstance(batch.get("attention_mask"), list):
            batch["attention_mask"] = torch.as_tensor(batch["attention_mask"], dtype=torch.long)
        if not self._debug_printed and isinstance(batch.get("input_ids"), torch.Tensor):
            in_ids = batch["input_ids"]
            attn = batch.get("attention_mask")
            non_pad = attn.sum(dim=1).tolist() if isinstance(attn, torch.Tensor) else []
            print(
                "[DEBUG] processor batch:",
                f"shape={tuple(in_ids.shape)}",
                f"non_pad={non_pad[:4]}",
                f"max_length={max_length}",
            )
            self._debug_printed = True
        return batch

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> str:
        # TRL은 generate 단계에서 tokenize=True, return_tensors='pt', return_dict=True로 호출한다.
        tokenize = kwargs.get("tokenize", False)
        if tokenize:
            add_generation_prompt = kwargs.get("add_generation_prompt", True)
            padding = kwargs.get("padding", True)
            padding_side = kwargs.get("padding_side", "left")
            max_length = kwargs.get("max_length", self.max_length)

            # conversation: list[list[{"role","content"}]] 형태(batch) 혹은 단일 대화
            conv_batch = conversation
            if isinstance(conv_batch, list) and conv_batch and isinstance(conv_batch[0], dict):
                conv_batch = [conv_batch]
            if not isinstance(conv_batch, list):
                conv_batch = [[{"role": "user", "content": str(conv_batch)}]]

            def _conv_to_plain_text(conv: Any) -> str:
                if not isinstance(conv, list):
                    return str(conv)
                lines: list[str] = []
                for m in conv:
                    if not isinstance(m, dict):
                        continue
                    role = str(m.get("role", "user")).strip().lower()
                    content = _as_text(m.get("content", "")).strip()
                    if not content:
                        continue
                    content = content.replace("<image>", "").strip()
                    if not content:
                        continue
                    lines.append(f"{role.upper()}: {content}")
                return "\n".join(lines).strip()

            texts: list[str] = []
            for conv in conv_batch:
                txt = _conv_to_plain_text(conv)
                texts.append(txt if isinstance(txt, str) else str(txt))

            batch = self(
                text=texts,
                images=None,
                padding=padding,
                padding_side=padding_side,
                max_length=max_length,
                return_tensors="pt",
            )
            return batch

        # 비토크나이즈 경로는 문자열 템플릿 반환
        try:
            return self.base_processor.apply_chat_template(conversation, **kwargs)
        except Exception:
            return self.tokenizer.apply_chat_template(conversation, **kwargs)

    def batch_decode(self, *args: Any, **kwargs: Any) -> Any:
        if not args:
            return self.tokenizer.batch_decode(*args, **kwargs)

        sequences = args[0]
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()
        if not isinstance(sequences, list):
            return self.tokenizer.batch_decode(*args, **kwargs)

        vocab_cap = int(getattr(self.tokenizer, "vocab_size", 0) or 0)
        unk_id = getattr(self.tokenizer, "unk_token_id", None)
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        fallback_id = unk_id if isinstance(unk_id, int) and unk_id >= 0 else (pad_id if isinstance(pad_id, int) and pad_id >= 0 else 0)

        def _norm_one(tok: Any) -> int:
            try:
                v = int(tok)
            except Exception:
                return fallback_id
            if v < 0:
                return fallback_id
            if vocab_cap > 0 and v >= vocab_cap:
                return fallback_id
            return v

        normalized: list[list[int]] = []
        changed = 0
        for seq in sequences:
            if isinstance(seq, torch.Tensor):
                seq = seq.tolist()
            if not isinstance(seq, list):
                seq = [seq]
            fixed = [_norm_one(t) for t in seq]
            if len(fixed) != len(seq) or any((not isinstance(t, int)) for t in seq):
                changed += 1
            normalized.append(fixed)

        try:
            return self.tokenizer.batch_decode(normalized, **kwargs)
        except OverflowError:
            if not getattr(self, "_decode_debug_printed", False):
                lengths = [len(s) for s in normalized[:4]]
                max_id = max((max(s) for s in normalized if s), default=-1)
                min_id = min((min(s) for s in normalized if s), default=-1)
                print(
                    "[DEBUG] batch_decode overflow after normalize:",
                    f"seqs={len(normalized)}",
                    f"lens_head={lengths}",
                    f"min_id={min_id}",
                    f"max_id={max_id}",
                    f"vocab_size={vocab_cap}",
                    f"changed={changed}",
                )
                self._decode_debug_printed = True
            return self.tokenizer.batch_decode(
                [[fallback_id] for _ in normalized],
                **kwargs,
            )

    def decode(self, *args: Any, **kwargs: Any) -> Any:
        return self.tokenizer.decode(*args, **kwargs)

    def save_pretrained(self, save_directory: str | Path, **kwargs: Any) -> Any:
        """Trainer 저장 단계에서 wrapper 대신 base processor를 저장한다."""
        return self.base_processor.save_pretrained(save_directory, **kwargs)

    @property
    def pad_token(self) -> str | None:
        return self.tokenizer.pad_token

    @property
    def pad_token_id(self) -> int | None:
        return self.tokenizer.pad_token_id

    @property
    def eos_token(self) -> str | None:
        return self.tokenizer.eos_token

    @property
    def eos_token_id(self) -> int | None:
        return self.tokenizer.eos_token_id


def _as_text(x: Any) -> str:
    """보상 입력을 문자열로 정규화한다.

    Args:
        x: 임의의 completion/prompt 객체
    Returns:
        str: 평탄화된 문자열
    """
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        if isinstance(x.get("content"), str):
            return x["content"]
        return str(x)
    if isinstance(x, list):
        parts = [_as_text(v) for v in x]
        return "\n".join([p for p in parts if p])
    return str(x)


def _normalize_role(role: Any) -> Optional[str]:
    """role 표기를 표준 role로 정규화한다.

    Args:
        role: 원본 role
    Returns:
        Optional[str]: system/user/assistant 또는 None
    """
    if not isinstance(role, str):
        return None
    role_map = {
        "system": "system",
        "user": "user",
        "human": "user",
        "assistant": "assistant",
        "gpt": "assistant",
        "bot": "assistant",
        "ai": "assistant",
        "model": "assistant",
    }
    return role_map.get(role.strip().lower())


def _extract_last_user(prompt: str) -> str:
    """평탄 prompt에서 마지막 user 발화를 뽑는다.

    Args:
        prompt: ROLE prefix 문자열
    Returns:
        str: 마지막 USER 블록
    """
    if isinstance(prompt, list):
        for msg in reversed(prompt):
            if not isinstance(msg, dict):
                continue
            role = _normalize_role(msg.get("role", msg.get("from", msg.get("speaker"))))
            if role == "user":
                content = _as_text(msg.get("content", msg.get("value", msg.get("text", "")))).strip()
                if content:
                    return content
        return ""

    if not isinstance(prompt, str) or not prompt:
        return ""
    matches = re.findall(
        r"USER:\s*(.*?)(?=\n(?:SYSTEM|USER|ASSISTANT):|\Z)",
        prompt,
        flags=re.DOTALL,
    )
    return matches[-1].strip() if matches else ""


def _normalize_completion_for_scoring(raw: str) -> tuple[str, str, bool]:
    """completion을 채점 가능한 형태로 정규화한다.

    기대 형식:
    - 1줄: 서술
    - 2줄: 큰따옴표 대사
    """
    txt = (raw or "").strip()
    if not txt:
        return "", "", False
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if len(lines) < 2:
        return "", "", False
    narration = lines[0]
    quote = ""
    for ln in lines[1:]:
        if re.fullmatch(r'"[^"\n]{2,300}"', ln):
            quote = ln
            break
    if not quote:
        return "", "", False
    valid = len(lines) == 2
    return narration, quote, valid


def _rough_token_len(text: str) -> int:
    """간이 길이 측정."""
    if not text:
        return 0
    return len(re.findall(r"[가-힣A-Za-z0-9]+|[^\s]", text))


def _ngram_overlap_ratio(a: str, b: str, n: int = 3) -> float:
    """두 문자열 n-gram 중복률을 계산한다."""
    if not a or not b:
        return 0.0
    ta = re.findall(r"[가-힣A-Za-z0-9]+", a.lower())
    tb = re.findall(r"[가-힣A-Za-z0-9]+", b.lower())
    if len(ta) < n or len(tb) < n:
        return 0.0
    a_ngrams = {" ".join(ta[i : i + n]) for i in range(len(ta) - n + 1)}
    b_ngrams = {" ".join(tb[i : i + n]) for i in range(len(tb) - n + 1)}
    if not a_ngrams:
        return 0.0
    return len(a_ngrams & b_ngrams) / float(len(a_ngrams))


def reward_format(prompts: List[Any], completions: List[Any], **kwargs: Any) -> List[float]:
    """형식 보상.

    - 서술 1줄 + 대사 1줄
    - role marker/메타 출력 방지
    """
    scores: List[float] = []
    for comp in completions:
        raw = _as_text(comp)
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        _, quote, valid = _normalize_completion_for_scoring(raw)
        has_role_marker = any(m in raw.upper() for m in ROLE_MARKERS)

        score = 0.0
        if lines and not lines[0].startswith('"'):
            score += 0.30
        if quote:
            score += 0.40
        if valid and not has_role_marker:
            score += 0.30
        scores.append(max(0.0, min(1.0, score)))
    return scores


def reward_role_split(prompts: List[Any], completions: List[Any], **kwargs: Any) -> List[float]:
    """assistant가 player 화법을 대필하지 않도록 보상한다."""
    scores: List[float] = []
    player_re = "|".join(PLAYER_NAMES)
    for comp in completions:
        raw = _as_text(comp)
        penalty = 0.0
        if re.search(rf"(?:^|\n)\s*(?:{player_re})\s*[:：]", raw):
            penalty += 0.6
        if re.search(rf"(?:^|\n)\s*(?:{player_re}).{{0,16}}(?:말했|말한다|묻는다|대답했)", raw):
            penalty += 0.5
        scores.append(max(0.0, 1.0 - penalty))
    return scores


def reward_grounded(prompts: List[Any], completions: List[Any], **kwargs: Any) -> List[float]:
    """직전 user 발화와의 표면 정합 보상."""
    scores: List[float] = []
    for prompt, comp in zip(prompts, completions):
        prompt_text = _as_text(prompt)
        user_text = _extract_last_user(prompt_text)
        raw = _as_text(comp)
        narr, quote, valid = _normalize_completion_for_scoring(raw)
        if not valid or not user_text:
            scores.append(0.0)
            continue
        out = f"{narr} {quote}"
        overlap = _ngram_overlap_ratio(user_text, out, n=2)
        if overlap >= 0.25:
            s = 1.0
        elif overlap >= 0.12:
            s = 0.7
        elif overlap >= 0.05:
            s = 0.4
        else:
            s = 0.1
        scores.append(s)
    return scores


def reward_length(prompts: List[Any], completions: List[Any], **kwargs: Any) -> List[float]:
    """길이 안정성 보상."""
    scores: List[float] = []
    for comp in completions:
        raw = _as_text(comp)
        n = _rough_token_len(raw)
        if 35 <= n <= 140:
            score = 1.0
        elif 20 <= n < 35 or 140 < n <= 200:
            score = 0.7
        elif 200 < n <= 260:
            score = 0.35
        else:
            score = 0.1
        scores.append(score)
    return scores


def load_grpo_dataset(path: str) -> Dataset:
    """GRPO 데이터셋을 로드해 prompt/reference 컬럼으로 정규화한다.

    지원 입력 예:
    - {"prompt":"...", "reference":"..."}
    - {"messages":[...], "reference":"..."}
    """
    ds = load_dataset("json", data_files=path)["train"]

    def _trim_to_last_user_turn(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """GRPO prompt를 '마지막 user 턴'으로 끝나게 정리한다.

        TRL GRPO는 prompt 뒤에 completion을 생성해야 하므로, prompt 마지막 role이 assistant면
        모델이 즉시 종료하거나 빈 completion을 반환할 수 있다.
        """
        if not messages:
            return []
        last_user_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user" and _as_text(messages[i].get("content")).strip():
                last_user_idx = i
                break
        if last_user_idx < 0:
            return []
        trimmed = messages[: last_user_idx + 1]
        return [m for m in trimmed if _as_text(m.get("content", "")).strip()]

    def _map(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt: list[dict[str, str]] = []

        raw_prompt = ex.get("prompt")
        if isinstance(raw_prompt, list):
            for m in raw_prompt:
                if not isinstance(m, dict):
                    continue
                role = _normalize_role(m.get("role", m.get("from", m.get("speaker"))))
                if role is None:
                    continue
                content = _as_text(m.get("content", m.get("value", m.get("text", "")))).strip()
                if content:
                    prompt.append({"role": role, "content": content})

        if not prompt and isinstance(ex.get("messages"), list):
            for m in ex["messages"]:
                if not isinstance(m, dict):
                    continue
                role = _normalize_role(m.get("role", m.get("from", m.get("speaker"))))
                if role is None:
                    continue
                content = _as_text(m.get("content", m.get("value", m.get("text", "")))).strip()
                if content:
                    prompt.append({"role": role, "content": content})

        if not prompt:
            plain = _as_text(ex.get("prompt", "")).strip()
            if plain:
                prompt = [{"role": "user", "content": plain}]

        prompt = _trim_to_last_user_turn(prompt)
        reference = _as_text(ex.get("reference", ex.get("output", ""))).strip()
        # TRL 멀티모달 경로를 타도록 image 필드를 항상 채운다.
        # 실제 이미지는 KananaGRPOProcessor에서 더미 PIL 이미지로 대체한다.
        return {"prompt": prompt, "reference": reference, "image": "__dummy__"}

    ds = ds.map(_map, remove_columns=ds.column_names)
    ds = ds.filter(
        lambda ex: (
            isinstance(ex.get("prompt"), list)
            and len(ex["prompt"]) > 0
            and ex["prompt"][-1].get("role") == "user"
            and bool(_as_text(ex["prompt"][-1].get("content", "")).strip())
        )
    )
    if len(ds) == 0:
        raise ValueError(f"No valid GRPO samples from: {path}")
    return ds


def build_lora_config(args: argparse.Namespace) -> LoraConfig:
    """LoRA 설정을 만든다."""
    targets = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    if not targets:
        raise ValueError("--lora_target_modules is empty.")
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=None,
        target_modules=targets,
    )


def patch_kanana_input_embeddings(model: Any) -> None:
    """Kanana 모델의 input embedding getter/setter를 보강한다.

    TRL/PEFT 경로에서 `enable_input_require_grads()` 호출 시
    `get_input_embeddings()` 미구현으로 터지는 문제를 우회한다.
    """

    def _kanana_get_input_embeddings(self):
        lm = getattr(self, "language_model", None)
        if lm is not None and hasattr(lm, "get_input_embeddings"):
            return lm.get_input_embeddings()
        raise NotImplementedError("Kanana model has no accessible language_model.get_input_embeddings().")

    def _kanana_set_input_embeddings(self, value):
        lm = getattr(self, "language_model", None)
        if lm is not None and hasattr(lm, "set_input_embeddings"):
            return lm.set_input_embeddings(value)
        raise NotImplementedError("Kanana model has no accessible language_model.set_input_embeddings().")

    try:
        _ = model.get_input_embeddings()
    except Exception:
        model.get_input_embeddings = _kanana_get_input_embeddings.__get__(model, model.__class__)
        model.set_input_embeddings = _kanana_set_input_embeddings.__get__(model, model.__class__)


def patch_kanana_generate_output(model: Any) -> None:
    """Kanana generate 반환을 TRL 기대 포맷(prompt+completion)으로 보정한다.

    일부 remote-code 모델은 generate가 completion-only 텐서를 반환한다.
    TRL GRPO는 prompt+completion을 가정하고 `[:, prompt_len:]`을 하므로
    completion-only 반환 시 빈 텐서가 되어 크래시가 난다.
    """
    orig_generate = model.generate

    def _wrapped_generate(self, *args: Any, **kwargs: Any):
        input_ids = kwargs.get("input_ids")
        out = orig_generate(*args, **kwargs)
        if (
            isinstance(out, torch.Tensor)
            and out.dim() == 2
            and isinstance(input_ids, torch.Tensor)
            and input_ids.dim() == 2
            and out.size(0) == input_ids.size(0)
        ):
            in_len = int(input_ids.size(1))
            out_len = int(out.size(1))
            if out_len < in_len:
                # completion-only 반환으로 간주하고 prompt를 앞에 붙여 TRL 가정에 맞춘다.
                out = torch.cat([input_ids, out.to(input_ids.device)], dim=1)
                if not getattr(self, "_grpo_generate_debug_printed", False):
                    print(
                        f"[DEBUG] patched generate output: completion_only_len={out_len}, "
                        f"prompt_len={in_len}, merged_len={out.size(1)}"
                    )
                    self._grpo_generate_debug_printed = True
            elif out_len == in_len and not getattr(self, "_grpo_generate_debug_printed", False):
                print(f"[DEBUG] generate returned no new tokens: out_len={out_len}, prompt_len={in_len}")
                self._grpo_generate_debug_printed = True
        return out

    model.generate = MethodType(_wrapped_generate, model)


def patch_kanana_forward_kwargs(model: Any) -> None:
    """TRL/Transformers 공통 kwargs 중 Kanana forward 미지원 항목을 제거한다."""
    orig_forward = model.forward
    allowed = set(inspect.signature(orig_forward).parameters.keys())
    # 실전에서 자주 섞여 들어오는 항목들
    drop_candidates = {
        "use_cache",
        "cache_position",
        "position_ids",
        "past_key_values",
        "output_attentions",
        "output_hidden_states",
        "return_dict_in_generate",
    }

    def _wrapped_forward(self, *args: Any, **kwargs: Any):
        removed = []
        for k in list(kwargs.keys()):
            if (k in drop_candidates and k not in allowed) or (k not in allowed and k.startswith("decoder_")):
                kwargs.pop(k, None)
                removed.append(k)
        # TRL의 per-token logprob 계산 경로는 text-only 입력을 전달할 수 있다.
        # Kanana forward는 pixel_values/image_metas를 필수 인자로 선언하므로,
        # 누락 시 텍스트 전용 경로로 강제 정규화한다.
        has_pixel = "pixel_values" in kwargs and kwargs.get("pixel_values") is not None
        has_meta = "image_metas" in kwargs and kwargs.get("image_metas") is not None
        if has_pixel and not has_meta:
            kwargs["pixel_values"] = None
            kwargs["image_metas"] = None
            removed.append("pixel_values(no_image_metas)")
        else:
            kwargs.setdefault("pixel_values", None)
            kwargs.setdefault("image_metas", None)
        if removed and not getattr(self, "_forward_kwargs_debug_printed", False):
            print(f"[DEBUG] stripped unsupported forward kwargs: {sorted(set(removed))}")
            self._forward_kwargs_debug_printed = True
        return orig_forward(*args, **kwargs)

    model.forward = MethodType(_wrapped_forward, model)


def main() -> None:
    """GRPO 학습 실행 엔트리포인트."""
    parser = argparse.ArgumentParser(description="Kanana RP GRPO trainer")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--eval_data", type=str, default=None)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_completion_length", type=int, default=220)
    parser.add_argument("--dummy_image_size", type=int, default=224)
    parser.add_argument("--num_generations", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--eval_strategy", type=str, default="no", choices=["no", "steps", "epoch"])
    parser.add_argument("--eval_steps", type=int, default=0)

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4")
    parser.add_argument("--bnb_4bit_use_double_quant", action="store_true")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2"],
    )
    parser.add_argument("--trust_remote_code", action="store_true")

    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug_data_preview", type=int, default=3)

    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.trust_remote_code:
        raise ValueError("Kanana 전용 스크립트는 --trust_remote_code가 필요합니다.")

    base_processor = AutoProcessor.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = getattr(base_processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("AutoProcessor에서 tokenizer를 찾지 못했습니다.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    template_path = Path(args.model_name) / "chat_template.jinja"
    if template_path.exists():
        chat_template = template_path.read_text(encoding="utf-8")
        tokenizer.chat_template = chat_template
        if hasattr(base_processor, "chat_template"):
            base_processor.chat_template = chat_template

    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.bfloat16

    quant_config = None
    if args.load_in_4bit:
        compute_dtype = torch.bfloat16 if args.bnb_4bit_compute_dtype == "bfloat16" else torch.float16
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        )

    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=dtype,
        quantization_config=quant_config,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.bos_token_id = tokenizer.bos_token_id

    patch_kanana_input_embeddings(model)
    patch_kanana_generate_output(model)
    patch_kanana_forward_kwargs(model)

    peft_config = None
    if args.use_lora:
        if args.load_in_4bit:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        peft_config = build_lora_config(args)

    train_ds = load_grpo_dataset(args.train_data)
    eval_ds = load_grpo_dataset(args.eval_data) if args.eval_data else None
    if args.debug_data_preview > 0:
        n = min(args.debug_data_preview, len(train_ds))
        print(f"[DEBUG] train samples: {len(train_ds)} (preview {n})")
        for i in range(n):
            p = train_ds[i]["prompt"]
            last_role = p[-1]["role"] if p else "none"
            last_text = _as_text(p[-1].get("content", ""))[:120] if p else ""
            print(f"[DEBUG] sample#{i} turns={len(p)} last_role={last_role} last_text={last_text!r}")
        if eval_ds is not None:
            print(f"[DEBUG] eval samples: {len(eval_ds)}")
    eval_strategy = args.eval_strategy if eval_ds is not None else "no"
    eval_steps = args.eval_steps if args.eval_steps > 0 else args.save_steps

    grpo_kwargs = {
        "output_dir": str(out_dir),
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "eval_strategy": eval_strategy,
        "eval_steps": eval_steps if eval_strategy == "steps" else None,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "seed": args.seed,
        "report_to": "none",
        "remove_unused_columns": False,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "num_generations": args.num_generations,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "reward_weights": [0.35, 0.25, 0.20, 0.20],
    }
    supported = set(inspect.signature(GRPOConfig.__init__).parameters.keys())
    grpo_args = GRPOConfig(**{k: v for k, v in grpo_kwargs.items() if k in supported})

    trainer = GRPOTrainer(
        model=model,
        processing_class=KananaGRPOProcessor(
            base_processor=base_processor,
            max_length=args.max_prompt_length,
            dummy_image_size=args.dummy_image_size,
        ),
        reward_funcs=[
            reward_format,
            reward_role_split,
            reward_grounded,
            reward_length,
        ],
        args=grpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(out_dir / "tokenizer")
    base_processor.save_pretrained(out_dir / "processor")
    print(f"[DONE] GRPO saved to {out_dir}")


if __name__ == "__main__":
    main()
