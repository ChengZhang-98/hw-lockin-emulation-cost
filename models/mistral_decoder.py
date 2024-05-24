import logging
import inspect
import math
from copy import deepcopy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.mistral.modeling_mistral import (
    ACT2FN,
    MistralConfig,
    MistralForCausalLM,
    MistralForSequenceClassification,
    MistralRotaryEmbedding,
    MistralRMSNorm,
    MistralDecoderLayer,
    _get_unpad_data,
    apply_rotary_pos_emb,
    repeat_kv,
    is_flash_attn_2_available,
    Cache,
)


from ..quantize import get_quantized_func, get_quantized_layer_cls
from ..utils import find_matched_pattern, get_layer_name

logger = logging.getLogger(__name__)


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)


class MistralQuantizedMLP(nn.Module):
    def __init__(self, config, loqer_config: dict):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        # fmt: off
        self.gate_proj = get_quantized_layer_cls("linear", q_config=loqer_config["gate_proj"])(self.hidden_size, self.intermediate_size, bias=False, q_config=loqer_config["gate_proj"])
        self.up_proj = get_quantized_layer_cls("linear", q_config=loqer_config["up_proj"])(self.hidden_size, self.intermediate_size, bias=False, q_config=loqer_config["up_proj"])
        self.down_proj = get_quantized_layer_cls("linear", q_config=loqer_config["down_proj"])(self.intermediate_size, self.hidden_size, bias=False, q_config=loqer_config["down_proj"])
        self.loqer_config = loqer_config
        # fmt: on
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MistralQuantizedAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: MistralConfig, layer_idx: Optional[int], loqer_config: dict):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        if layer_idx is None:
            logger.warning(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        # fmt: off
        self.q_proj = get_quantized_layer_cls("linear", q_config=loqer_config["q_proj"])(self.hidden_size, self.num_heads * self.head_dim, bias=False, q_config=loqer_config["q_proj"])
        self.k_proj = get_quantized_layer_cls("linear", q_config=loqer_config["k_proj"])(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, q_config=loqer_config["k_proj"])
        self.v_proj = get_quantized_layer_cls("linear", q_config=loqer_config["v_proj"])(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, q_config=loqer_config["v_proj"])
        self.o_proj = get_quantized_layer_cls("linear", q_config=loqer_config["o_proj"])(self.num_heads * self.head_dim, self.hidden_size, bias=False, q_config=loqer_config["o_proj"])
        self.loqer_config = loqer_config
        # fmt: on

        self.rotary_emb = MistralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            logger.warning(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # *: matmul_0
        # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        query_states = query_states.reshape(bsz * self.num_heads, q_len, self.head_dim)
        key_states = key_states.reshape(bsz * self.num_heads, kv_seq_len, self.head_dim)
        attn_weights = get_quantized_func("matmul", q_config=self.loqer_config["matmul_0"])(
            query_states, key_states.transpose(1, 2), q_config=self.loqer_config["matmul_0"]
        ) / math.sqrt(self.head_dim)
        attn_weights = attn_weights.reshape(bsz, self.num_heads, q_len, kv_seq_len)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        # *: matmul_1
        # attn_output = torch.matmul(attn_weights, value_states)
        attn_weights = attn_weights.reshape(bsz * self.num_heads, q_len, kv_seq_len)
        value_states = value_states.reshape(bsz * self.num_heads, kv_seq_len, self.head_dim)
        attn_output = get_quantized_func("matmul", q_config=self.loqer_config["matmul_1"])(
            attn_weights, value_states, q_config=self.loqer_config["matmul_1"]
        )
        attn_output = attn_output.reshape(bsz, self.num_heads, q_len, self.head_dim)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


MISTRAL_ATTENTION_CLASSES = {
    "eager": MistralQuantizedAttention,
}


class MistralQuantizedDecoderLayer(nn.Module):
    def __init__(self, config: MistralConfig, layer_idx: int, loqer_config: dict):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = MISTRAL_ATTENTION_CLASSES[config._attn_implementation](
            config, layer_idx, loqer_config=loqer_config["self_attn"]
        )

        self.mlp = MistralQuantizedMLP(config, loqer_config=loqer_config["mlp"])
        self.input_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            logger.warning(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


def build_loqer_config_mistral(model: MistralForCausalLM, loqer_config: dict):
    parsed_config = {}

    decoder_layer_i: MistralDecoderLayer
    for i, decoder_layer_i in enumerate(model.model.layers):
        parsed_config[f"model_layer_{i}"] = {"self_attn": {}, "mlp": {}}
        for fc_short_name in ["k_proj", "q_proj", "v_proj", "o_proj"]:
            fc_name = get_layer_name(model, getattr(decoder_layer_i.self_attn, fc_short_name))
            matched_entry = find_matched_pattern(fc_name, loqer_config.keys())
            assert matched_entry is not None, f"Cannot find matched entry for {fc_name}"
            if isinstance(loqer_config[matched_entry], str):
                matched_entry = loqer_config[matched_entry]
            parsed_config[f"model_layer_{i}"]["self_attn"][fc_short_name] = deepcopy(loqer_config[matched_entry])
        for matmul_short_name in ["matmul_0", "matmul_1"]:
            matmul_name = fc_name.replace("o_proj", matmul_short_name)
            matched_entry = find_matched_pattern(matmul_name, loqer_config.keys())
            assert matched_entry is not None, f"Cannot find matched entry for {matmul_name}"
            if isinstance(loqer_config[matched_entry], str):
                matched_entry = loqer_config[matched_entry]
            parsed_config[f"model_layer_{i}"]["self_attn"][matmul_short_name] = deepcopy(loqer_config[matched_entry])
        for fc_short_name in ["gate_proj", "up_proj", "down_proj"]:
            fc_name = get_layer_name(model, getattr(decoder_layer_i.mlp, fc_short_name))
            matched_entry = find_matched_pattern(fc_name, loqer_config.keys())
            assert matched_entry is not None, f"Cannot find matched entry for {fc_name}"
            if isinstance(loqer_config[matched_entry], str):
                matched_entry = loqer_config[matched_entry]
            parsed_config[f"model_layer_{i}"]["mlp"][fc_short_name] = deepcopy(loqer_config[matched_entry])

    return parsed_config


def quantize_mistral_model(
    model: MistralForCausalLM | MistralForSequenceClassification,
    loqer_config: dict,
):
    loqer_config = build_loqer_config_mistral(model, loqer_config)

    for layer_id, ori_decoder_layer in enumerate(model.model.layers):
        layer_entry = f"model_layer_{layer_id}"
        layer_q_config = loqer_config[layer_entry]

        # replace the decoder layer with quantized decoder layer
        new_decoder_layer = MistralQuantizedDecoderLayer(model.config, layer_id, layer_q_config)
        new_decoder_layer.to(next(iter(ori_decoder_layer.parameters())).dtype)
        new_decoder_layer.to(next(iter(ori_decoder_layer.parameters())).device)
        new_decoder_layer.load_state_dict(ori_decoder_layer.state_dict(), strict=False)
        model.model.layers[layer_id] = new_decoder_layer
        # remove the original layer
        del ori_decoder_layer

    model._no_split_modules = ["MistralDecoderLayer", "MistralQuantizedDecoderLayer"]

    return model

def find_layers_to_register_scale_hook_mistral(model: MistralForCausalLM) -> list[dict[str, str | list[str]]]:
    """
    return a list of dict, each dict contains the following keys:

    ```
    {
        "target_layer": ...,
        "layers_sharing_scale": [...],
    }
    ```
    """

    assert model.config._attn_implementation == "eager"
    layers_to_register = []

    for decoder_layer in model.model.layers:
        k_name = get_layer_name(model, decoder_layer.self_attn.k_proj)
        q_name = get_layer_name(model, decoder_layer.self_attn.q_proj)
        v_name = get_layer_name(model, decoder_layer.self_attn.v_proj)
        layers_to_register.append(
            dict(target_layer=k_name, layers_sharing_scale=[q_name, v_name]),
        )

        o_name = get_layer_name(model, decoder_layer.self_attn.o_proj)
        layers_to_register.append(
            dict(target_layer=o_name, layers_sharing_scale=[]),
        )

        gate_name = get_layer_name(model, decoder_layer.mlp.gate_proj)
        up_name = get_layer_name(model, decoder_layer.mlp.up_proj)
        layers_to_register.append(
            dict(target_layer=gate_name, layers_sharing_scale=[up_name]),
        )

        down_name = get_layer_name(model, decoder_layer.mlp.down_proj)
        layers_to_register.append(
            dict(target_layer=down_name, layers_sharing_scale=[]),
        )

    return layers_to_register


def find_layers_to_approximate_mistral(model: MistralForCausalLM):
    layers_to_approximate = []

    for layer_name, layer in model.named_modules():
        if not isinstance(layer, nn.Linear):
            continue
        if "lm_head" in layer_name:
            continue

        layers_to_approximate.append(layer_name)

    return layers_to_approximate
