from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.trainer.models.base import PreTrainedModelPrimeRL


def _shift_left(t: Tensor, position_ids: Tensor | None = None) -> Tensor:
    """Shift tensor one position left, zero-padding the end of each packed sequence."""
    shifted = F.pad(t[..., 1:], (0, 1))
    if position_ids is not None:
        boundary = position_ids[..., :-1] >= position_ids[..., 1:]
        shifted[..., :-1][boundary] = 0
    return shifted


def compute_mtp_mask(loss_mask: Tensor, num_steps: int = 1, position_ids: Tensor | None = None) -> Tensor:
    """Compute MTP loss mask: position t is valid iff loss_mask[t+1..t+K+1] are all valid."""
    result = torch.ones_like(loss_mask, dtype=torch.float)
    shifted = loss_mask.float()
    for _ in range(num_steps + 1):
        shifted = _shift_left(shifted, position_ids)
        result *= shifted
    return result


def compute_mtp_token_losses(
    model: PreTrainedModelPrimeRL,
    hidden_states: Tensor,
    input_ids: Tensor,
    position_ids: Tensor | None,
    chunk_size: int = 512,
) -> tuple[Tensor, int]:
    """Compute per-token MTP CE losses averaged across prediction steps.

    Returns (token_loss [B, S], num_steps).
    """
    B, S, _ = hidden_states.shape
    V = model.lm_head.weight.shape[0]
    num_steps = model.mtp_num_prediction_steps
    layers = list(model.mtp_layers.values() if isinstance(model.mtp_layers, nn.ModuleDict) else model.mtp_layers)

    h = hidden_states.detach()
    detached_weight = model.lm_head.weight.detach()
    total_loss: Tensor | None = None
    current_ids, current_pos = input_ids, position_ids

    for step in range(num_steps):
        shifted_ids = _shift_left(current_ids, current_pos)
        labels = _shift_left(shifted_ids, current_pos)
        shifted_pos = _shift_left(current_pos, current_pos) if current_pos is not None else None

        with torch.no_grad():
            embeds = model.mtp_embed_tokens(shifted_ids)
            rotary = model.mtp_rotary_emb
            pos_emb = rotary(h, shifted_pos) if rotary is not None and shifted_pos is not None else None

        layer = layers[0] if model.mtp_shared_weights else layers[step]
        mtp_out = model.mtp_layer_forward(layer, h, embeds, shifted_pos, pos_emb)

        # Chunked CE to avoid materializing full [B, S, V] logits
        chunks = []
        for start in range(0, S, chunk_size):
            end = min(start + chunk_size, S)
            logits = F.linear(mtp_out[:, start:end, :], detached_weight)
            chunk_loss = F.cross_entropy(logits.view(-1, V), labels[:, start:end].reshape(-1), reduction="none")
            chunks.append(chunk_loss.view(B, -1))
            del logits
        step_loss = torch.cat(chunks, dim=1)
        total_loss = step_loss if total_loss is None else total_loss + step_loss

        h, current_ids, current_pos = mtp_out, shifted_ids, shifted_pos

    if num_steps > 1:
        total_loss = total_loss / num_steps
    return total_loss, num_steps


def setup_mtp_training(model: nn.Module, mtp_config: object) -> None:
    """Validate model has MTP support and store config for use during forward."""
    from prime_rl.trainer.models.base import PreTrainedModelPrimeRL

    if not isinstance(model, PreTrainedModelPrimeRL):
        raise TypeError(
            f"MTP training requires a PreTrainedModelPrimeRL subclass, got {type(model).__name__}. "
            "Create a custom model implementation with the MTP interface."
        )
    if model.mtp_layers is None:
        raise ValueError(f"MTP training enabled but {type(model).__name__}.mtp_layers returned None.")

    model._mtp_config = mtp_config
    num_params = sum(p.numel() for p in model.mtp_layers.parameters())
    get_logger().info(
        f"MTP training enabled: {len(model.mtp_layers)} layer(s), {model.mtp_num_prediction_steps} prediction step(s)"
        f"{' (shared weights)' if model.mtp_shared_weights else ''}, {num_params:,} parameters"
    )
