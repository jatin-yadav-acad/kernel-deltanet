r"""
Kernel-DeltaNet Engine
Module: models.hf_compat
==========================================
HuggingFace Transformers Native Integration Scaffolding.
Wraps KernelTTT into full PretrainedConfig and PreTrainedModel classes,
enabling drop-in compatibility with AutoModelForCausalLM, HF Trainer, and generation pipelines.
"""

from typing import Optional, Tuple, Union, Dict, Any
import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .kernel_deltanet_lm import KernelDeltaNetLM
from layers.hybrid_kernel_deltanet_block import RMSNorm


class KernelTTTConfig(PretrainedConfig):
    r"""
    Configuration class for KernelDeltaNetLM.
    Compatible with HuggingFace Hub JSON serialization.
    """
    model_type = "kernel_ttt"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        rkhs_factor: int = 4,
        chunk_size: int = 64,
        d_ff: Optional[int] = None,
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        tie_word_embeddings: bool = True,
        initial_decay: float = 0.99,
        initial_eta: float = 0.5,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        **kwargs
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.rkhs_factor = rkhs_factor
        self.chunk_size = chunk_size
        self.d_ff = d_ff if d_ff is not None else int(4 * d_model * 2 / 3)
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.initial_decay = initial_decay
        self.initial_eta = initial_eta


class KernelTTTPreTrainedModel(PreTrainedModel):
    r"""
    Abstract base class handling weights initialization and HF PreTrainedModel interface.
    """
    config_class = KernelTTTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module: nn.Module):
        """Standard LeCun normal initialization for deep recurrent/state models."""
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)


class KernelTTTForCausalLM(KernelTTTPreTrainedModel, GenerationMixin):
    r"""
    Complete HuggingFace Causal LM implementation with native generate() support.
    """
    _tied_weights_keys = {"model.lm_head.weight": "model.embed.weight"}

    def __init__(self, config: KernelTTTConfig):
        super().__init__(config)
        self.model = KernelDeltaNetLM(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            rkhs_factor=config.rkhs_factor,
            chunk_size=config.chunk_size,
            d_ff=config.d_ff,
            tie_weights=config.tie_word_embeddings
        )
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed

    def set_input_embeddings(self, value: nn.Embedding):
        self.model.embed = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.model.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.model.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        HuggingFace standard causal forward method.
        """
        return_dict = return_dict if return_dict is not None else getattr(self.config, "return_dict", True)

        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify either input_ids or inputs_embeds")

        logits, loss = self.model(
            input_ids=input_ids,
            targets=labels
        )

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None, # In training mode, states are not cached
            hidden_states=None,
            attentions=None
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        r"""
        Prepares inputs for HuggingFace generate() loop.
        """
        # If recurrent past states are present, only pass the last token
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": True
        }
