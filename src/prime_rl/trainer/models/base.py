import torch
import torch.nn as nn
from torch import Tensor
from transformers.modeling_utils import PreTrainedModel


class PreTrainedModelPrimeRL(PreTrainedModel):
    """
    Base class for all PrimeRL models that extends HuggingFace PreTrainedModel.

    Provides a unified interface for state dict conversion between different formats
    (e.g., HuggingFace format vs. training-optimized format) and buffer initialization
    after loading with meta device.

    Subclasses that support Multi-Token Prediction should override the ``mtp_*``
    properties and ``mtp_layer_forward`` to expose their MTP components.
    """

    # ------------------------------------------------------------------
    # MTP interface — override in subclasses that have MTP layers
    # ------------------------------------------------------------------

    @property
    def mtp_layers(self) -> nn.ModuleList | nn.ModuleDict | None:
        """Return the MTP layer container, or None if this model has no MTP support."""
        return getattr(self.model, "mtp_layers", None)

    @property
    def mtp_embed_tokens(self) -> nn.Module:
        """Return the token embedding module used for MTP input."""
        for attr in ("embed_tokens", "embeddings"):
            mod = getattr(self.model, attr, None)
            if mod is not None:
                return mod
        raise AttributeError(f"Cannot find token embedding module on {type(self).__name__}")

    @property
    def mtp_rotary_emb(self) -> nn.Module | None:
        """Return the rotary embedding module, or None."""
        return getattr(self.model, "rotary_emb", None)

    @property
    def mtp_num_prediction_steps(self) -> int:
        """Number of MTP prediction depths to train.

        For non-shared weights this equals len(mtp_layers).
        For shared weights this may be larger (same layer applied multiple times).
        """
        layers = self.mtp_layers
        return len(layers) if layers is not None else 0

    @property
    def mtp_shared_weights(self) -> bool:
        """Whether MTP layers share parameters across prediction steps."""
        return False

    def mtp_layer_forward(
        self,
        mtp_layer: nn.Module,
        hidden_states: Tensor,
        token_embeds: Tensor,
        position_ids: Tensor | None,
        position_embeddings: tuple[Tensor, Tensor] | None,
    ) -> Tensor:
        """Run a single MTP layer. Override for model-specific forward signatures.

        Default assumes mtp_layer has enorm/hnorm/eh_proj/block sub-modules
        following the DeepSeek-V3/GLM-4 pattern (per-layer fusion).
        """
        e = mtp_layer.enorm(token_embeds)
        h = mtp_layer.hnorm(hidden_states)
        combined = mtp_layer.eh_proj(torch.cat([e, h], dim=-1))
        out = mtp_layer.block(combined, position_embeddings=position_embeddings)
        if hasattr(mtp_layer, "norm"):
            out = mtp_layer.norm(out)
        return out

    @classmethod
    def from_config(cls, config, **kwargs):
        """Public from_config that mirrors the Auto class API."""
        return cls._from_config(config, **kwargs)

    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        """PrimeRL models use custom MoE implementations and don't support dynamic experts implementation."""
        return False

    def get_correct_experts_implementation(self, requested_experts: str | None) -> str:
        """PrimeRL models always use eager experts implementation."""
        return "eager"

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        """
        Check if the state dict is in HuggingFace format.

        Args:
            state_dict: The state dict to check.

        Returns:
            True if the state dict is in HuggingFace format, False otherwise.
        """
        raise NotImplementedError(f"is_hf_state_dict is not implemented for {cls.__name__}")

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        """
        Check if the state dict is in PrimeRL training format.

        Args:
            state_dict: The state dict to check.

        Returns:
            True if the state dict is in PrimeRL format, False otherwise.
        """
        raise NotImplementedError(f"is_prime_state_dict is not implemented for {cls.__name__}")

    @classmethod
    def convert_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Convert state dict from PrimeRL training format to HuggingFace format in-place.

        This is used when saving checkpoints or broadcasting weights to inference engines
        that expect HuggingFace-compatible format.

        Args:
            state_dict: The state dict to convert (modified in-place).
        """
        raise NotImplementedError(f"convert_to_hf is not implemented for {cls.__name__}")

    @classmethod
    def convert_to_prime(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Convert state dict from HuggingFace format to PrimeRL training format in-place.

        This is used when loading pretrained HuggingFace models for training with
        PrimeRL-specific optimizations.

        Args:
            state_dict: The state dict to convert (modified in-place).
        """
        raise NotImplementedError(f"convert_to_prime is not implemented for {cls.__name__}")

    @classmethod
    def convert_layer_to_hf(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        """
        Convert a single layer's state dict from PrimeRL format to HuggingFace format in-place.

        This is used for layer-by-layer conversion during NCCL broadcast to reduce memory usage.

        Args:
            state_dict: The state dict containing the layer to convert (modified in-place).
            layer_idx: The index of the layer to convert.
        """
        raise NotImplementedError(f"convert_layer_to_hf is not implemented for {cls.__name__}")

    @classmethod
    def convert_layer_to_prime(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        """
        Convert a single layer's state dict from HuggingFace format to PrimeRL format in-place.

        This is used for layer-by-layer conversion during loading.

        Args:
            state_dict: The state dict containing the layer to convert (modified in-place).
            layer_idx: The index of the layer to convert.
        """
        raise NotImplementedError(f"convert_layer_to_prime is not implemented for {cls.__name__}")

    @classmethod
    def convert_layer_to_vllm_kernel(
        cls,
        state_dict: dict[str, Tensor],
        layer_idx: int,
        quantize_fp8: bool = False,
    ) -> dict[str, Tensor]:
        """
        Convert a single layer's state dict from PrimeRL format to vLLM kernel format.

        Args:
            state_dict: Layer weights in PrimeRL format.
            layer_idx: Layer index to convert.
            quantize_fp8: Whether to emit FP8 (e4m3) kernel weights with per-block scales.
        """
        raise NotImplementedError(f"convert_layer_to_vllm_kernel is not implemented for {cls.__name__}")

    def init_buffers_post_meta(self) -> None:
        """
        Initialize buffers that are not in the state dict after loading with meta device.

        Some models have buffers (non-trainable tensors) that are not saved in the state dict
        but need to be properly initialized after loading the model on meta device and then
        moving to the actual device. This method should initialize such buffers.

        This is called after loading the model from a checkpoint with meta device.
        """
        raise NotImplementedError(f"init_buffers_post_meta is not implemented for {self.__class__.__name__}")


__all__ = ["PreTrainedModelPrimeRL"]
