# models/qwen3_reference/causal_lm.py
from __future__ import annotations

from typing import Optional, Union, Dict, Any

import torch
from torch import nn

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.utils import GenerationMixin

from .config import SayaQwen3Config
from .model import Qwen3Model


class QwenForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = SayaQwen3Config
    base_model_prefix = "model"
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: SayaQwen3Config):
        super().__init__(config)

        self.model = Qwen3Model(config)

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        self.tie_weights()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if use_cache is None:
            use_cache = bool(getattr(self.config, "use_cache", True))

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        if isinstance(logits_to_keep, int):
            if logits_to_keep > 0:
                hidden_states = hidden_states[:, -logits_to_keep:, :]
        else:
            hidden_states = hidden_states[:, logits_to_keep, :]

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values=None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if past_key_values is not None:
            if inputs_embeds is not None:
                inputs_embeds = inputs_embeds[:, -1:, :]
                input_ids = None
            else:
                input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if position_ids is None and attention_mask is not None and attention_mask.dim() == 2:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids.clamp(min=0)
            if past_key_values is not None:
                position_ids = position_ids[:, -1:].contiguous()

        use_cache = kwargs.get("use_cache", None)
        if use_cache is None:
            use_cache = bool(getattr(self.config, "use_cache", True))

        model_inputs: Dict[str, Any] = {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

        for k, v in kwargs.items():
            if k not in model_inputs:
                model_inputs[k] = v

        return model_inputs
